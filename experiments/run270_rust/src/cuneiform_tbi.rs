//! Cuneiform 6D/8D (H8 -> S^7) sparse pattern bank with ARMv8 TBI [63:56] flat pointers.
//!
//! Binary container (`C8DB`, v1) designed for zero-copy `MAP_SHARED` `mmap` beside
//! RAM-resident `.t16c` weights. Each expert payload is 64-byte aligned and referenced
//! by a 64-bit TBI-tagged flat pointer handle:
//!   - bits `[63:56]`: Cuneiform 6D domain/octant metadata tag (ignored by AArch64 EL0 MMU)
//!   - bits `[55:0]`:  flat virtual address inside the mapped pattern bank
//!
//! Both an accelerated C/NEON/AVX2 path and a bit-accurate pure-Rust fallback are
//! implemented so fail-closed parity (`--no-native-ffi`) can always be checked.

use std::collections::HashMap;
use std::ffi::c_void;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, Write};
#[cfg(unix)]
use std::io::SeekFrom;
use std::path::Path;
#[cfg(unix)]
use std::sync::Arc;

pub const C8DB_MAGIC: &[u8; 4] = b"C8DB";
pub const C8DB_VERSION: u32 = 1;
pub const C8DB_FLAG_TBI_TAGS: u32 = 1 << 0;

/// Pack a Cuneiform 6D proposition's 3x8-bit radicals (`rc`, `rf`, `ra`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cuneiform6DRadicals {
    pub rc: u8,
    pub rf: u8,
    pub ra: u8,
    pub tag_byte: u8,
}

impl Cuneiform6DRadicals {
    pub fn from_axes(
        domain: u8,
        subdomain: u8,
        operation: u8,
        modality: u8,
        depth: u8,
        polarity: u8,
        expert_idx: u8,
    ) -> Self {
        let rc = ((domain & 0xF) << 4) | (subdomain & 0xF);
        let rf = ((operation & 0xF) << 4) | (modality & 0xF);
        let ra = ((depth & 0xF) << 4) | (polarity & 0xF);
        let tag_byte = ((domain & 0xF) << 4) | (expert_idx & 0xF);
        Self { rc, rf, ra, tag_byte }
    }

    pub fn opcode_24bit(self) -> u32 {
        (self.rc as u32) | ((self.rf as u32) << 8) | ((self.ra as u32) << 16)
    }
}

/// Pack an 8-bit metadata tag into ARMv8 TBI bits `[63:56]`.
#[inline]
pub fn tbi_tag_ptr(raw_addr: u64, tag: u8) -> u64 {
    ((tag as u64) << 56) | (raw_addr & 0x00FF_FFFF_FFFF_FFFF)
}

#[inline]
pub fn tbi_extract_tag(tagged_ptr: u64) -> u8 {
    (tagged_ptr >> 56) as u8
}

#[inline]
pub fn tbi_untag_ptr(tagged_ptr: u64) -> u64 {
    tagged_ptr & 0x00FF_FFFF_FFFF_FFFF
}

#[inline]
fn tbi_deref_slice(tagged_ptr: u64, byte_len: usize) -> &'static [u8] {
    let raw = if cfg!(all(target_arch = "aarch64", target_os = "linux")) {
        tagged_ptr as usize as *const u8
    } else {
        tbi_untag_ptr(tagged_ptr) as usize as *const u8
    };
    unsafe { std::slice::from_raw_parts(raw, byte_len) }
}

/// 8D normalized Sylvester-Hadamard transform (`H8 / sqrt(8)`).
#[inline]
pub fn fwht8_octonion(input: &[f32; 8]) -> [f32; 8] {
    let inv_sqrt8 = 0.35355339059327373f32;
    let a0 = input[0] + input[1];
    let a1 = input[0] - input[1];
    let a2 = input[2] + input[3];
    let a3 = input[2] - input[3];
    let a4 = input[4] + input[5];
    let a5 = input[4] - input[5];
    let a6 = input[6] + input[7];
    let a7 = input[6] - input[7];

    let b0 = a0 + a2;
    let b2 = a0 - a2;
    let b1 = a1 + a3;
    let b3 = a1 - a3;
    let b4 = a4 + a6;
    let b6 = a4 - a6;
    let b5 = a5 + a7;
    let b7 = a5 - a7;

    [
        (b0 + b4) * inv_sqrt8,
        (b1 + b5) * inv_sqrt8,
        (b2 + b6) * inv_sqrt8,
        (b3 + b7) * inv_sqrt8,
        (b0 - b4) * inv_sqrt8,
        (b1 - b5) * inv_sqrt8,
        (b2 - b6) * inv_sqrt8,
        (b3 - b7) * inv_sqrt8,
    ]
}

#[cfg(not(unix))]
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Align64Chunk([u8; 64]);

/// Owned 64-byte-aligned byte backing whose allocation layout matches `drop`.
#[cfg(not(unix))]
struct Aligned64Buf {
    chunks: Vec<Align64Chunk>,
}

#[cfg(not(unix))]
impl Aligned64Buf {
    fn from_slice(data: &[u8]) -> Self {
        let n_chunks = data.len().div_ceil(64).max(1);
        let mut chunks = vec![Align64Chunk([0u8; 64]); n_chunks];
        unsafe {
            let dst = std::slice::from_raw_parts_mut(chunks.as_mut_ptr() as *mut u8, data.len());
            dst.copy_from_slice(data);
        }
        Self { chunks }
    }

    fn as_ptr(&self) -> *const u8 {
        self.chunks.as_ptr() as *const u8
    }
}

#[cfg(unix)]
mod sys_unix {
    use std::ffi::c_void;
    pub const PROT_READ: i32 = 1;
    pub const MAP_SHARED: i32 = 1;
    pub const MADV_RANDOM: i32 = 1;
    pub const MADV_WILLNEED: i32 = 3;
    pub const MAP_FAILED: *mut c_void = !0usize as *mut c_void;

    extern "C" {
        pub fn mmap(addr: *mut c_void, length: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, length: usize) -> i32;
        pub fn madvise(addr: *mut c_void, length: usize, advice: i32) -> i32;
        pub fn dlopen(filename: *const std::ffi::c_char, flag: i32) -> *mut c_void;
        pub fn dlsym(handle: *mut c_void, symbol: *const std::ffi::c_char) -> *mut c_void;
        pub fn dlclose(handle: *mut c_void) -> i32;
    }
    pub const RTLD_NOW: i32 = 2;
}

#[cfg(windows)]
mod sys_win {
    use std::ffi::c_void;
    extern "system" {
        pub fn LoadLibraryW(lp_lib_file_name: *const u16) -> *mut c_void;
        pub fn GetProcAddress(h_module: *mut c_void, lp_proc_name: *const i8) -> *mut c_void;
        pub fn FreeLibrary(h_module: *mut c_void) -> i32;
    }
}

#[cfg(unix)]
struct BankMmap {
    ptr: *const u8,
    len: usize,
}

#[cfg(unix)]
unsafe impl Send for BankMmap {}
#[cfg(unix)]
unsafe impl Sync for BankMmap {}

#[cfg(unix)]
impl Drop for BankMmap {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.len > 0 {
            unsafe {
                sys_unix::munmap(self.ptr as *mut c_void, self.len);
            }
        }
    }
}

pub type TbiHardwareActiveFn = unsafe extern "C" fn() -> i32;
pub type CuneiformS7RouteBatchFn = unsafe extern "C" fn(
    x: *const f32,
    centroids: *const f32,
    k_tokens: i32,
    d_in: i32,
    n_experts: i32,
    top_k: i32,
    temperature: f32,
    out_idx: *mut i32,
    out_weights: *mut f32,
) -> i32;
pub type CuneiformBankApplyBatchFn = unsafe extern "C" fn(
    expert_ptrs_tagged: *const u64,
    centroids: *const f32,
    x_raw: *const f32,
    y: *mut f32,
    d_out: i32,
    d_in: i32,
    rank: i32,
    n_experts: i32,
    top_k: i32,
    temperature: f32,
    gate_scale: f32,
    k_tokens: i32,
    num_threads: i32,
    out_active_tags: *mut u8,
) -> i32;

pub struct NativeCuneiformKernel {
    handle: *mut c_void,
    pub path: String,
    pub tbi_hw_active: bool,
    pub route_fn: CuneiformS7RouteBatchFn,
    pub apply_fn: CuneiformBankApplyBatchFn,
}

unsafe impl Send for NativeCuneiformKernel {}
unsafe impl Sync for NativeCuneiformKernel {}

impl Drop for NativeCuneiformKernel {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            #[cfg(unix)]
            unsafe {
                sys_unix::dlclose(self.handle);
            }
            #[cfg(windows)]
            unsafe {
                sys_win::FreeLibrary(self.handle);
            }
        }
    }
}

impl NativeCuneiformKernel {
    pub fn try_load(path: &Path) -> Result<Self, String> {
        #[cfg(unix)]
        unsafe {
            use std::ffi::CString;
            let cpath = CString::new(path.to_string_lossy().as_bytes())
                .map_err(|_| format!("invalid path for dlopen: {}", path.display()))?;
            let h = sys_unix::dlopen(cpath.as_ptr(), sys_unix::RTLD_NOW);
            if h.is_null() {
                return Err(format!("dlopen {} failed", path.display()));
            }
            let sym_route = sys_unix::dlsym(h, b"cuneiform_s7_route_batch_c\0".as_ptr() as *const std::ffi::c_char);
            let sym_apply = sys_unix::dlsym(h, b"cuneiform_bank_apply_batch_c\0".as_ptr() as *const std::ffi::c_char);
            let sym_hw = sys_unix::dlsym(h, b"tbi_hardware_active_c\0".as_ptr() as *const std::ffi::c_char);
            if sym_route.is_null() || sym_apply.is_null() {
                sys_unix::dlclose(h);
                return Err(format!("missing cuneiform symbols in {}", path.display()));
            }
            let tbi_hw_active = if !sym_hw.is_null() {
                let f: TbiHardwareActiveFn = std::mem::transmute(sym_hw);
                f() == 1
            } else {
                false
            };
            Ok(Self {
                handle: h,
                path: path.display().to_string(),
                tbi_hw_active,
                route_fn: std::mem::transmute(sym_route),
                apply_fn: std::mem::transmute(sym_apply),
            })
        }
        #[cfg(windows)]
        unsafe {
            use std::os::windows::ffi::OsStrExt;
            let wpath: Vec<u16> = path.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
            let h = sys_win::LoadLibraryW(wpath.as_ptr());
            if h.is_null() {
                return Err(format!("LoadLibraryW {} failed", path.display()));
            }
            let sym_route = sys_win::GetProcAddress(h, b"cuneiform_s7_route_batch_c\0".as_ptr() as *const i8);
            let sym_apply = sys_win::GetProcAddress(h, b"cuneiform_bank_apply_batch_c\0".as_ptr() as *const i8);
            let sym_hw = sys_win::GetProcAddress(h, b"tbi_hardware_active_c\0".as_ptr() as *const i8);
            if sym_route.is_null() || sym_apply.is_null() {
                sys_win::FreeLibrary(h);
                return Err(format!("missing cuneiform symbols in {}", path.display()));
            }
            let tbi_hw_active = if !sym_hw.is_null() {
                let f: TbiHardwareActiveFn = std::mem::transmute(sym_hw);
                f() == 1
            } else {
                false
            };
            Ok(Self {
                handle: h,
                path: path.display().to_string(),
                tbi_hw_active,
                route_fn: std::mem::transmute(sym_route),
                apply_fn: std::mem::transmute(sym_apply),
            })
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = path;
            Err("dynamic FFI unsupported on this target".into())
        }
    }
}

#[allow(dead_code)]
enum ExpertBacking {
    #[cfg(not(unix))]
    Owned(Vec<Aligned64Buf>),
    #[cfg(unix)]
    Mmap(Arc<BankMmap>),
}

/// One routed sparse pattern bank attached to a transformer weight matrix.
pub struct CuneiformBankEntry {
    pub name: String,
    pub d_out: usize,
    pub d_in: usize,
    pub rank: usize,
    pub n_experts: usize,
    pub top_k: usize,
    pub c6d: Cuneiform6DRadicals,
    pub temperature: f32,
    pub gate_scale: f32,
    pub centroids: Vec<f32>,
    pub expert_radicals: Vec<Cuneiform6DRadicals>,
    pub expert_ptrs_tagged: Vec<u64>,
    expert_byte_len: usize,
    _backing: ExpertBacking,
}

unsafe impl Send for CuneiformBankEntry {}
unsafe impl Sync for CuneiformBankEntry {}

/// Builder specification for writing a `.c8d` bank entry.
pub struct CuneiformExpertInput<'a> {
    pub radicals: Cuneiform6DRadicals,
    pub su: &'a [f32],
    pub sv: &'a [f32],
    pub u_i8: &'a [i8],
    pub v_i8: &'a [i8],
}

pub struct CuneiformBankEntryInput<'a> {
    pub name: String,
    pub d_out: usize,
    pub d_in: usize,
    pub rank: usize,
    pub top_k: usize,
    pub c6d: Cuneiform6DRadicals,
    pub temperature: f32,
    pub gate_scale: f32,
    pub centroids: &'a [f32],
    pub experts: Vec<CuneiformExpertInput<'a>>,
}

#[inline]
fn round_up_64(v: usize) -> usize {
    (v + 63) & !63
}

#[inline]
fn expert_payload_byte_len(d_out: usize, d_in: usize, rank: usize) -> usize {
    d_out * 4 + rank * 4 + d_out * rank + rank * d_in
}

impl CuneiformBankEntry {
    /// Route each token via H8 -> S^7 cosine alignment and return `(top_k_indices, softmax_weights)`.
    pub fn route_tokens_rust(&self, x: &[f32], k_tokens: usize) -> Result<(Vec<usize>, Vec<f32>), String> {
        if x.len() != k_tokens * self.d_in {
            return Err(format!("route_tokens x len {} != {} * {}", x.len(), k_tokens, self.d_in));
        }
        if self.d_in < 8 || k_tokens == 0 {
            return Err("invalid k_tokens or d_in < 8".into());
        }
        let k_blocks = self.d_in / 8;
        let inv_blocks = 1.0f32 / k_blocks as f32;
        let inv_temp = 1.0f32 / self.temperature;

        let mut out_idx = Vec::with_capacity(k_tokens * self.top_k);
        let mut out_w = Vec::with_capacity(k_tokens * self.top_k);

        for t in 0..k_tokens {
            let xr = &x[t * self.d_in..(t + 1) * self.d_in];
            let mut sum8 = [0.0f32; 8];
            for b in 0..k_blocks {
                let bp = &xr[b * 8..(b + 1) * 8];
                for j in 0..8 {
                    sum8[j] += bp[j];
                }
            }
            for s in &mut sum8 {
                *s *= inv_blocks;
            }
            let u_bar = fwht8_octonion(&sum8);
            let sq: f32 = u_bar.iter().map(|v| v * v).sum();
            let nrm = sq.sqrt();
            let inv_nrm = 1.0f32 / if nrm > 1e-9 { nrm } else { 1e-9 };
            let mut g_hat = [0.0f32; 8];
            for j in 0..8 {
                g_hat[j] = u_bar[j] * inv_nrm;
            }

            let mut logits = Vec::with_capacity(self.n_experts);
            for e in 0..self.n_experts {
                let ce = &self.centroids[e * 8..(e + 1) * 8];
                let dot: f32 = g_hat.iter().zip(ce.iter()).map(|(a, b)| a * b).sum();
                logits.push(dot * inv_temp);
            }

            let mut used = vec![false; self.n_experts];
            let mut sel = Vec::with_capacity(self.top_k);
            for _ in 0..self.top_k {
                let mut best_e = 0usize;
                let mut best_v = f32::NEG_INFINITY;
                let mut found = false;
                for (e, &lv) in logits.iter().enumerate() {
                    if !used[e] && (!found || lv > best_v) {
                        best_e = e;
                        best_v = lv;
                        found = true;
                    }
                }
                used[best_e] = true;
                sel.push(best_e);
            }

            let max_l = sel.iter().map(|&e| logits[e]).fold(f32::NEG_INFINITY, f32::max);
            let mut ws = Vec::with_capacity(self.top_k);
            let mut w_sum = 0.0f32;
            for &e in &sel {
                let wv = (logits[e] - max_l).exp();
                ws.push(wv);
                w_sum += wv;
            }
            let inv_sum = if w_sum > 1e-12 { 1.0 / w_sum } else { 1.0 / self.top_k as f32 };
            for &e in &sel {
                out_idx.push(e);
            }
            for &wv in &ws {
                out_w.push(if w_sum > 1e-12 { wv * inv_sum } else { inv_sum });
            }
        }
        Ok((out_idx, out_w))
    }

    /// Add routed expert delta directly onto mutable `y[k_tokens * d_out]`.
    pub fn apply_batch_into(
        &self,
        x_raw: &[f32],
        k_tokens: usize,
        y: &mut [f32],
        native: Option<&NativeCuneiformKernel>,
        threads: usize,
        mut out_tags: Option<&mut Vec<u8>>,
    ) -> Result<(), String> {
        if x_raw.len() != k_tokens * self.d_in {
            return Err(format!("apply_batch x len {} != {} * {}", x_raw.len(), k_tokens, self.d_in));
        }
        if y.len() != k_tokens * self.d_out {
            return Err(format!("apply_batch y len {} != {} * {}", y.len(), k_tokens, self.d_out));
        }
        if let Some(nk) = native {
            let mut tag_buf = if out_tags.is_some() {
                vec![0u8; k_tokens * self.top_k]
            } else {
                Vec::new()
            };
            let tag_ptr = if out_tags.is_some() { tag_buf.as_mut_ptr() } else { std::ptr::null_mut() };
            let rc = unsafe {
                (nk.apply_fn)(
                    self.expert_ptrs_tagged.as_ptr(),
                    self.centroids.as_ptr(),
                    x_raw.as_ptr(),
                    y.as_mut_ptr(),
                    self.d_out as i32,
                    self.d_in as i32,
                    self.rank as i32,
                    self.n_experts as i32,
                    self.top_k as i32,
                    self.temperature,
                    self.gate_scale,
                    k_tokens as i32,
                    threads.max(1) as i32,
                    tag_ptr,
                )
            };
            if rc != 0 {
                return Err(format!("native cuneiform_bank_apply_batch_c failed (rc={rc}) for {}", self.name));
            }
            if let Some(dst) = out_tags.as_deref_mut() {
                dst.clear();
                dst.extend_from_slice(&tag_buf);
            }
            return Ok(());
        }

        // Pure-Rust reference path using identical TBI pointer slice resolution
        let (idx_buf, w_buf) = self.route_tokens_rust(x_raw, k_tokens)?;
        if let Some(dst) = out_tags.as_deref_mut() {
            dst.clear();
        }
        let su_bytes = self.d_out * 4;
        let sv_bytes = self.rank * 4;
        let u_bytes = self.d_out * self.rank;

        let read_f32_at = |buf: &[u8], i: usize| -> f32 {
            let o = i * 4;
            f32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]])
        };

        let mut z = vec![0.0f32; self.rank];
        for t in 0..k_tokens {
            let xt = &x_raw[t * self.d_in..(t + 1) * self.d_in];
            let yt = &mut y[t * self.d_out..(t + 1) * self.d_out];
            for k in 0..self.top_k {
                let e = idx_buf[t * self.top_k + k];
                let tp = self.expert_ptrs_tagged[e];
                if let Some(ref mut dst) = out_tags {
                    dst.push(tbi_extract_tag(tp));
                }
                let alpha = self.gate_scale * w_buf[t * self.top_k + k];
                if alpha.abs() < 1e-12 {
                    continue;
                }
                let payload = tbi_deref_slice(tp, self.expert_byte_len);
                let su_slice = &payload[0..su_bytes];
                let sv_slice = &payload[su_bytes..su_bytes + sv_bytes];
                let u_slice = &payload[su_bytes + sv_bytes..su_bytes + sv_bytes + u_bytes];
                let v_slice = &payload[su_bytes + sv_bytes + u_bytes..];

                for rk in 0..self.rank {
                    let sv = read_f32_at(sv_slice, rk);
                    let vrow = &v_slice[rk * self.d_in..(rk + 1) * self.d_in];
                    let mut dot = 0.0f32;
                    for c in 0..self.d_in {
                        dot += (vrow[c] as i8) as f32 * xt[c];
                    }
                    z[rk] = dot * sv * alpha;
                }
                for row in 0..self.d_out {
                    let su = read_f32_at(su_slice, row);
                    let urow = &u_slice[row * self.rank..(row + 1) * self.rank];
                    let mut acc = 0.0f32;
                    for rk in 0..self.rank {
                        acc += (urow[rk] as i8) as f32 * z[rk];
                    }
                    yt[row] += acc * su;
                }
            }
        }
        Ok(())
    }
}

/// Complete Cuneiform 6D/8D pattern bank container mapped from `.c8d`.
pub struct CuneiformBank {
    pub flags: u32,
    pub norms: HashMap<String, Vec<f32>>,
    pub entries: HashMap<String, CuneiformBankEntry>,
    pub total_payload_bytes: usize,
    pub native: Option<NativeCuneiformKernel>,
}

fn read_exact_u16<R: Read>(r: &mut R) -> Result<u16, String> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b).map_err(|e| format!("read u16: {e}"))?;
    Ok(u16::from_le_bytes(b))
}

fn read_exact_u32<R: Read>(r: &mut R) -> Result<u32, String> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|e| format!("read u32: {e}"))?;
    Ok(u32::from_le_bytes(b))
}

fn read_exact_u64<R: Read>(r: &mut R) -> Result<u64, String> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(|e| format!("read u64: {e}"))?;
    Ok(u64::from_le_bytes(b))
}

fn read_exact_f32<R: Read>(r: &mut R) -> Result<f32, String> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|e| format!("read f32: {e}"))?;
    Ok(f32::from_le_bytes(b))
}

impl CuneiformBank {
    pub fn load(
        path: &Path,
        native_lib: Option<&Path>,
        disable_native: bool,
        warm_touch: bool,
    ) -> Result<Self, String> {
        let f = File::open(path).map_err(|e| format!("open C8DB {}: {e}", path.display()))?;
        let native = if !disable_native {
            native_lib.and_then(|p| NativeCuneiformKernel::try_load(p).ok())
        } else {
            None
        };

        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let meta = f.metadata().map_err(|e| format!("stat {}: {e}", path.display()))?;
            let file_len = usize::try_from(meta.len())
                .map_err(|_| format!("C8DB too large for address space: {}", meta.len()))?;
            if file_len < 24 {
                return Err(format!("C8DB truncated ({file_len} bytes)"));
            }
            let raw_ptr = unsafe {
                sys_unix::mmap(
                    std::ptr::null_mut(),
                    file_len,
                    sys_unix::PROT_READ,
                    sys_unix::MAP_SHARED,
                    f.as_raw_fd(),
                    0,
                )
            };
            if raw_ptr == sys_unix::MAP_FAILED || raw_ptr.is_null() {
                return Err(format!("mmap C8DB {} failed", path.display()));
            }
            let map = Arc::new(BankMmap { ptr: raw_ptr as *const u8, len: file_len });
            unsafe {
                let advice = if warm_touch { sys_unix::MADV_WILLNEED } else { sys_unix::MADV_RANDOM };
                let _ = sys_unix::madvise(raw_ptr, file_len, advice);
                if warm_touch {
                    let mut off = 0usize;
                    while off < file_len {
                        let _ = std::ptr::read_volatile(map.ptr.add(off));
                        off += 4096;
                    }
                }
            }
            return Self::load_from_reader_and_mmap(&mut BufReader::new(f), Some(map), native);
        }

        #[cfg(not(unix))]
        {
            let _ = warm_touch;
            Self::load_from_reader_and_mmap(&mut BufReader::with_capacity(4 * 1024 * 1024, f), None, native)
        }
    }

    fn load_from_reader_and_mmap<R: Read + Seek>(
        r: &mut R,
        #[cfg(unix)] mmap_opt: Option<Arc<BankMmap>>,
        #[cfg(not(unix))] mmap_opt: Option<()>,
        native: Option<NativeCuneiformKernel>,
    ) -> Result<Self, String> {
        let mut hdr = [0u8; 24];
        r.read_exact(&mut hdr).map_err(|e| format!("read C8DB header: {e}"))?;
        if &hdr[0..4] != C8DB_MAGIC {
            return Err(format!("invalid C8DB magic {:?}", &hdr[0..4]));
        }
        let version = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]);
        if version != C8DB_VERSION {
            return Err(format!("unsupported C8DB version {version}"));
        }
        let n_norms = u32::from_le_bytes([hdr[8], hdr[9], hdr[10], hdr[11]]) as usize;
        let n_entries = u32::from_le_bytes([hdr[12], hdr[13], hdr[14], hdr[15]]) as usize;
        let flags = u32::from_le_bytes([hdr[16], hdr[17], hdr[18], hdr[19]]);
        let mut pos = 24usize;

        let mut norms = HashMap::with_capacity(n_norms);
        for _ in 0..n_norms {
            let nlen = read_exact_u16(r)? as usize;
            let mut nb = vec![0u8; nlen];
            r.read_exact(&mut nb).map_err(|e| format!("read norm name: {e}"))?;
            let name = String::from_utf8(nb).map_err(|e| format!("utf8 norm: {e}"))?;
            let cnt = read_exact_u32(r)? as usize;
            let mut vals = Vec::with_capacity(cnt);
            for _ in 0..cnt {
                vals.push(read_exact_f32(r)?);
            }
            pos += 2 + nlen + 4 + cnt * 4;
            norms.insert(name, vals);
        }

        let mut entries = HashMap::with_capacity(n_entries);
        let mut total_payload_bytes = 0usize;

        for _ in 0..n_entries {
            let nlen = read_exact_u16(r)? as usize;
            let d_out = read_exact_u32(r)? as usize;
            let d_in = read_exact_u32(r)? as usize;
            let rank = read_exact_u16(r)? as usize;
            let mut meta4 = [0u8; 4];
            r.read_exact(&mut meta4).map_err(|e| format!("read meta4: {e}"))?;
            let n_experts = meta4[0] as usize;
            let top_k = meta4[1] as usize;
            // meta4[2..4] unused alignment
            let mut c6d_b = [0u8; 4];
            r.read_exact(&mut c6d_b).map_err(|e| format!("read c6d: {e}"))?;
            let c6d = Cuneiform6DRadicals {
                rc: c6d_b[0],
                rf: c6d_b[1],
                ra: c6d_b[2],
                tag_byte: c6d_b[3],
            };
            let temperature = read_exact_f32(r)?;
            let gate_scale = read_exact_f32(r)?;
            let _reserved = read_exact_u32(r)?;
            let payload_bytes = read_exact_u64(r)? as usize;
            let mut nb = vec![0u8; nlen];
            r.read_exact(&mut nb).map_err(|e| format!("read entry name: {e}"))?;
            let name = String::from_utf8(nb).map_err(|e| format!("utf8 entry: {e}"))?;
            pos += 40 + nlen;

            if n_experts == 0 || n_experts > 64 || top_k == 0 || top_k > n_experts || top_k > 8 || rank == 0 || d_in < 8 || d_out == 0 {
                return Err(format!("invalid C8DB geometry for {name}: ({d_out}x{d_in}) r={rank} K={n_experts} top_k={top_k}"));
            }
            if !temperature.is_finite() || temperature <= 0.0 || !gate_scale.is_finite() {
                return Err(format!("invalid scalar params for {name}"));
            }

            let pad = (64 - (pos % 64)) % 64;
            if pad > 0 {
                let mut skip = [0u8; 64];
                r.read_exact(&mut skip[..pad]).map_err(|e| format!("read pad: {e}"))?;
                pos += pad;
            }
            let _payload_start = pos;

            let mut centroids = Vec::with_capacity(n_experts * 8);
            for _ in 0..n_experts * 8 {
                centroids.push(read_exact_f32(r)?);
            }
            let mut expert_radicals = Vec::with_capacity(n_experts);
            for _ in 0..n_experts {
                let mut rb = [0u8; 4];
                r.read_exact(&mut rb).map_err(|e| format!("read expert radical: {e}"))?;
                expert_radicals.push(Cuneiform6DRadicals {
                    rc: rb[0],
                    rf: rb[1],
                    ra: rb[2],
                    tag_byte: rb[3],
                });
            }
            let meta_bytes = n_experts * 8 * 4 + n_experts * 4;
            let meta_pad = (64 - (meta_bytes % 64)) % 64;
            if meta_pad > 0 {
                let mut skip = [0u8; 64];
                r.read_exact(&mut skip[..meta_pad]).map_err(|e| format!("read meta pad: {e}"))?;
            }

            let exp_bytes = expert_payload_byte_len(d_out, d_in, rank);
            let exp_stride = round_up_64(exp_bytes);
            let expected_payload = meta_bytes + meta_pad + n_experts * exp_stride;
            if payload_bytes != expected_payload {
                return Err(format!(
                    "C8DB payload_bytes mismatch for {name}: header={payload_bytes} expected={expected_payload}"
                ));
            }

            let mut expert_ptrs_tagged = Vec::with_capacity(n_experts);

            #[cfg(unix)]
            let backing = if let Some(ref map) = mmap_opt {
                let first_exp_offset = _payload_start + meta_bytes + meta_pad;
                if first_exp_offset + n_experts * exp_stride > map.len {
                    return Err(format!("C8DB {name} exceeds mapped file size"));
                }
                for e in 0..n_experts {
                    let abs_off = first_exp_offset + e * exp_stride;
                    let raw_va = unsafe { map.ptr.add(abs_off) } as usize as u64;
                    if raw_va % 64 != 0 {
                        return Err(format!("C8DB {name} expert {e} unaligned VA 0x{raw_va:x}"));
                    }
                    expert_ptrs_tagged.push(tbi_tag_ptr(raw_va, expert_radicals[e].tag_byte));
                }
                r.seek(SeekFrom::Start((_payload_start + payload_bytes) as u64))
                    .map_err(|e| format!("seek C8DB: {e}"))?;
                ExpertBacking::Mmap(Arc::clone(map))
            } else {
                unreachable!()
            };

            #[cfg(not(unix))]
            let backing = {
                let _ = mmap_opt;
                let mut owned_experts = Vec::with_capacity(n_experts);
                let mut raw = vec![0u8; exp_stride];
                for e in 0..n_experts {
                    r.read_exact(&mut raw).map_err(|er| format!("read expert {e} of {name}: {er}"))?;
                    let ab = Aligned64Buf::from_slice(&raw[..exp_bytes]);
                    let raw_va = ab.as_ptr() as usize as u64;
                    expert_ptrs_tagged.push(tbi_tag_ptr(raw_va, expert_radicals[e].tag_byte));
                    owned_experts.push(ab);
                }
                ExpertBacking::Owned(owned_experts)
            };

            pos += payload_bytes;
            total_payload_bytes += payload_bytes;

            entries.insert(
                name.clone(),
                CuneiformBankEntry {
                    name,
                    d_out,
                    d_in,
                    rank,
                    n_experts,
                    top_k,
                    c6d,
                    temperature,
                    gate_scale,
                    centroids,
                    expert_radicals,
                    expert_ptrs_tagged,
                    expert_byte_len: exp_bytes,
                    _backing: backing,
                },
            );
        }

        Ok(Self { flags, norms, entries, total_payload_bytes, native })
    }

    pub fn write_file(
        path: &Path,
        norms: &[(String, Vec<f32>)],
        entries: &[CuneiformBankEntryInput<'_>],
    ) -> Result<(), String> {
        let f = File::create(path).map_err(|e| format!("create C8DB {}: {e}", path.display()))?;
        let mut w = BufWriter::new(f);
        let mut hdr = [0u8; 24];
        hdr[0..4].copy_from_slice(C8DB_MAGIC);
        hdr[4..8].copy_from_slice(&C8DB_VERSION.to_le_bytes());
        hdr[8..12].copy_from_slice(&(norms.len() as u32).to_le_bytes());
        hdr[12..16].copy_from_slice(&(entries.len() as u32).to_le_bytes());
        hdr[16..20].copy_from_slice(&C8DB_FLAG_TBI_TAGS.to_le_bytes());
        w.write_all(&hdr).map_err(|e| format!("write C8DB hdr: {e}"))?;
        let mut pos = 24usize;

        for (name, vals) in norms {
            let nb = name.as_bytes();
            w.write_all(&(nb.len() as u16).to_le_bytes()).map_err(|e| format!("norm len: {e}"))?;
            w.write_all(nb).map_err(|e| format!("norm name: {e}"))?;
            w.write_all(&(vals.len() as u32).to_le_bytes()).map_err(|e| format!("norm cnt: {e}"))?;
            for &v in vals {
                w.write_all(&v.to_le_bytes()).map_err(|e| format!("norm f32: {e}"))?;
            }
            pos += 2 + nb.len() + 4 + vals.len() * 4;
        }

        let z64 = [0u8; 64];
        for e_in in entries {
            let n_exp = e_in.experts.len();
            if n_exp == 0 || n_exp > 64 || e_in.top_k == 0 || e_in.top_k > n_exp || e_in.top_k > 8 {
                return Err(format!("invalid expert count/top_k for {}", e_in.name));
            }
            if e_in.centroids.len() != n_exp * 8 {
                return Err(format!("centroids len mismatch for {}", e_in.name));
            }
            let nb = e_in.name.as_bytes();
            let meta_bytes = n_exp * 8 * 4 + n_exp * 4;
            let meta_pad = (64 - (meta_bytes % 64)) % 64;
            let exp_bytes = expert_payload_byte_len(e_in.d_out, e_in.d_in, e_in.rank);
            let exp_stride = round_up_64(exp_bytes);
            let payload_bytes = meta_bytes + meta_pad + n_exp * exp_stride;

            w.write_all(&(nb.len() as u16).to_le_bytes()).map_err(|e| format!("entry hdr: {e}"))?;
            w.write_all(&(e_in.d_out as u32).to_le_bytes()).map_err(|e| format!("entry hdr: {e}"))?;
            w.write_all(&(e_in.d_in as u32).to_le_bytes()).map_err(|e| format!("entry hdr: {e}"))?;
            w.write_all(&(e_in.rank as u16).to_le_bytes()).map_err(|e| format!("entry hdr: {e}"))?;
            w.write_all(&[n_exp as u8, e_in.top_k as u8, 0u8, 0u8]).map_err(|e| format!("entry hdr: {e}"))?;
            w.write_all(&[e_in.c6d.rc, e_in.c6d.rf, e_in.c6d.ra, e_in.c6d.tag_byte]).map_err(|e| format!("entry hdr: {e}"))?;
            w.write_all(&e_in.temperature.to_le_bytes()).map_err(|e| format!("entry hdr: {e}"))?;
            w.write_all(&e_in.gate_scale.to_le_bytes()).map_err(|e| format!("entry hdr: {e}"))?;
            w.write_all(&0u32.to_le_bytes()).map_err(|e| format!("entry hdr: {e}"))?;
            w.write_all(&(payload_bytes as u64).to_le_bytes()).map_err(|e| format!("entry hdr: {e}"))?;
            w.write_all(nb).map_err(|e| format!("entry name: {e}"))?;
            pos += 40 + nb.len();

            let pad = (64 - (pos % 64)) % 64;
            if pad > 0 {
                w.write_all(&z64[..pad]).map_err(|e| format!("entry pad: {e}"))?;
                pos += pad;
            }
            for &cv in e_in.centroids {
                w.write_all(&cv.to_le_bytes()).map_err(|e| format!("centroid f32: {e}"))?;
            }
            for ex in &e_in.experts {
                w.write_all(&[ex.radicals.rc, ex.radicals.rf, ex.radicals.ra, ex.radicals.tag_byte])
                    .map_err(|e| format!("expert rad: {e}"))?;
            }
            if meta_pad > 0 {
                w.write_all(&z64[..meta_pad]).map_err(|e| format!("meta pad: {e}"))?;
            }

            let exp_pad = exp_stride - exp_bytes;
            for ex in &e_in.experts {
                if ex.su.len() != e_in.d_out || ex.sv.len() != e_in.rank || ex.u_i8.len() != e_in.d_out * e_in.rank || ex.v_i8.len() != e_in.rank * e_in.d_in {
                    return Err(format!("expert tensor shape mismatch in {}", e_in.name));
                }
                for &v in ex.su {
                    w.write_all(&v.to_le_bytes()).map_err(|e| format!("su: {e}"))?;
                }
                for &v in ex.sv {
                    w.write_all(&v.to_le_bytes()).map_err(|e| format!("sv: {e}"))?;
                }
                let ub: Vec<u8> = ex.u_i8.iter().map(|&v| v as u8).collect();
                let vb: Vec<u8> = ex.v_i8.iter().map(|&v| v as u8).collect();
                w.write_all(&ub).map_err(|e| format!("ui8: {e}"))?;
                w.write_all(&vb).map_err(|e| format!("vi8: {e}"))?;
                if exp_pad > 0 {
                    w.write_all(&z64[..exp_pad]).map_err(|e| format!("exp pad: {e}"))?;
                }
            }
            pos += payload_bytes;
        }
        w.flush().map_err(|e| format!("flush C8DB: {e}"))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn repo_native_lib() -> Option<PathBuf> {
        let p = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("neon_runner")
            .join(if cfg!(windows) {
                "neon_stream_accelerator.dll"
            } else {
                "libneon_stream_accelerator.so"
            });
        p.is_file().then_some(p)
    }

    #[test]
    fn tbi_pointer_tags_and_c6d_radicals_roundtrip() {
        let r = Cuneiform6DRadicals::from_axes(10, 13, 0, 7, 10, 0, 5);
        assert_eq!(r.rc, 0xAD);
        assert_eq!(r.rf, 0x07);
        assert_eq!(r.ra, 0xA0);
        assert_eq!(r.tag_byte, 0xA5);
        assert_eq!(r.opcode_24bit(), 0x00A0_07AD);

        let raw = 0x0000_2345_6789_ABC0u64;
        let tp = tbi_tag_ptr(raw, r.tag_byte);
        assert_eq!(tbi_extract_tag(tp), 0xA5);
        assert_eq!(tbi_untag_ptr(tp), raw);
    }

    #[test]
    fn c8db_write_load_tbi_and_native_parity() {
        let (d_out, d_in, rk, n_exp, top_k, k_tok) = (48usize, 64usize, 6usize, 8usize, 2usize, 3usize);
        let mut centroids = vec![0.0f32; n_exp * 8];
        for e in 0..n_exp {
            centroids[e * 8 + e] = 1.0f32; // canonical 8D S^7 Clifford octants e0..e7
        }

        let mut su_all = Vec::with_capacity(n_exp);
        let mut sv_all = Vec::with_capacity(n_exp);
        let mut u_all = Vec::with_capacity(n_exp);
        let mut v_all = Vec::with_capacity(n_exp);
        let mut exp_rad = Vec::with_capacity(n_exp);
        let mut state: u64 = 0xCAFE_BABE_1234_5678;
        let mut next_r = || -> u64 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            state
        };

        for e in 0..n_exp {
            su_all.push((0..d_out).map(|_| 0.002 * (((next_r() >> 32) % 31) as i32 - 15) as f32).collect::<Vec<_>>());
            sv_all.push((0..rk).map(|_| 0.003 * (((next_r() >> 32) % 31) as i32 - 15) as f32).collect::<Vec<_>>());
            u_all.push((0..d_out * rk).map(|_| (((next_r() >> 32) % 201) as i32 - 100) as i8).collect::<Vec<_>>());
            v_all.push((0..rk * d_in).map(|_| (((next_r() >> 32) % 201) as i32 - 100) as i8).collect::<Vec<_>>());
            exp_rad.push(Cuneiform6DRadicals::from_axes(10, 14, 6, 7, 10, 0, e as u8));
        }

        let experts: Vec<CuneiformExpertInput<'_>> = (0..n_exp)
            .map(|e| CuneiformExpertInput {
                radicals: exp_rad[e],
                su: &su_all[e],
                sv: &sv_all[e],
                u_i8: &u_all[e],
                v_i8: &v_all[e],
            })
            .collect();

        let tmp = std::env::temp_dir().join("test_cuneiform_bank_v1.c8d");
        let norm_ref: Vec<f32> = (0..64).map(|i| 0.5 + 0.01 * i as f32).collect();
        let mat_name = "model.language_model.layers.39.mlp.down_proj.weight".to_string();
        let entry_in = CuneiformBankEntryInput {
            name: mat_name.clone(),
            d_out,
            d_in,
            rank: rk,
            top_k,
            c6d: exp_rad[0],
            temperature: 0.5,
            gate_scale: 0.85,
            centroids: &centroids,
            experts,
        };
        CuneiformBank::write_file(&tmp, &[("model.language_model.norm.weight".into(), norm_ref.clone())], &[entry_in]).unwrap();

        let lib_path = repo_native_lib();
        let bank = CuneiformBank::load(&tmp, lib_path.as_deref(), false, true).unwrap();
        let _ = std::fs::remove_file(&tmp);

        assert_eq!(bank.norms.get("model.language_model.norm.weight").unwrap(), &norm_ref);
        let entry = bank.entries.get(&mat_name).unwrap();

        let x: Vec<f32> = (0..k_tok * d_in).map(|i| ((i as f32) * 0.13).sin() + if i % 8 == (i / d_in) % 8 { 0.4 } else { 0.0 }).collect();
        let y_base: Vec<f32> = (0..k_tok * d_out).map(|i| 0.01 * (i as f32 % 7.0 - 3.0)).collect();

        let mut y_rust = y_base.clone();
        let mut tags_rust = Vec::new();
        entry.apply_batch_into(&x, k_tok, &mut y_rust, None, 2, Some(&mut tags_rust)).unwrap();
        assert_eq!(tags_rust.len(), k_tok * top_k);
        assert!(tags_rust.iter().all(|&t| (t >> 4) == 10));

        if let Some(ref nk) = bank.native {
            let mut y_nat = y_base;
            let mut tags_nat = Vec::new();
            entry.apply_batch_into(&x, k_tok, &mut y_nat, Some(nk), 2, Some(&mut tags_nat)).unwrap();
            assert_eq!(tags_nat, tags_rust);
            let max_diff = y_rust.iter().zip(y_nat.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(max_diff < 1e-4, "native vs rust C8DB max_diff={max_diff}");
        }
    }
}
