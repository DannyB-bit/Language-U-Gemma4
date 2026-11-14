//! T16C v1 codec: compact 64-byte-aligned Tile-16 sub-byte container (1..4-bit
//! symmetric/asymmetric) with optional Hadamard block rotation (`FWHT-64/128/256`),
//! input channel equalization (`inv_s`), and INT8 low-rank residual (`U_i8 @ V_i8`).
//!
//! Dispatches `gemm_bitserial_tile16_batch` through dynamically loaded native SIMD
//! shared libraries (`libneon_stream_accelerator.so` on ARMv8 Linux /
//! `neon_stream_accelerator.dll` on Windows) with a deterministic multi-threaded
//! pure-Rust fallback so `cargo test --release` passes with zero external deps.

use crate::codec::{fp16_to_f32, fp32_to_fp16};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::ptr;

pub const T16C_MAGIC: &[u8; 4] = b"T16C";
pub const T16C_VERSION: u32 = 1;
pub const ARCH_FLAG_GEMMA4: u32 = 1 << 0;
pub const ARCH_FLAG_TIED_HEAD: u32 = 1 << 1;

#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct Aligned64(pub [u8; 64]);

fn make_aligned_buf(byte_len: usize) -> Vec<Aligned64> {
    let n_blocks = byte_len.div_ceil(64).max(1);
    vec![Aligned64([0u8; 64]); n_blocks]
}

fn aligned_as_slice(buf: &[Aligned64], byte_len: usize) -> &[u8] {
    let raw = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len() * 64) };
    &raw[..byte_len]
}

fn aligned_as_mut_slice(buf: &mut [Aligned64], byte_len: usize) -> &mut [u8] {
    let raw = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, buf.len() * 64) };
    &mut raw[..byte_len]
}

#[derive(Clone)]
pub struct Int8LowRank {
    pub rank: usize,
    pub d_out: usize,
    pub d_in: usize,
    pub su: Vec<f32>,
    pub sv: Vec<f32>,
    pub u_i8: Vec<i8>,
    pub v_i8: Vec<i8>,
}

impl Int8LowRank {
    pub fn byte_len(&self) -> usize {
        self.su.len() * 4 + self.sv.len() * 4 + self.u_i8.len() + self.v_i8.len()
    }

    pub fn add_batch_into(&self, x_raw: &[f32], k_tokens: usize, y: &mut [f32], threads: usize) {
        let d_out = self.d_out;
        let d_in = self.d_in;
        let r = self.rank;
        if r == 0 || k_tokens == 0 {
            return;
        }
        debug_assert_eq!(x_raw.len(), k_tokens * d_in);
        debug_assert_eq!(y.len(), k_tokens * d_out);

        let mut z = vec![0f32; k_tokens * r];
        for tok in 0..k_tokens {
            let xk = &x_raw[tok * d_in..(tok + 1) * d_in];
            let zk = &mut z[tok * r..(tok + 1) * r];
            for k in 0..r {
                let vrow = &self.v_i8[k * d_in..(k + 1) * d_in];
                let mut acc0 = 0.0f32;
                let mut acc1 = 0.0f32;
                let mut acc2 = 0.0f32;
                let mut acc3 = 0.0f32;
                let mut j = 0usize;
                while j + 4 <= d_in {
                    acc0 += (vrow[j] as f32) * xk[j];
                    acc1 += (vrow[j + 1] as f32) * xk[j + 1];
                    acc2 += (vrow[j + 2] as f32) * xk[j + 2];
                    acc3 += (vrow[j + 3] as f32) * xk[j + 3];
                    j += 4;
                }
                let mut acc = (acc0 + acc1) + (acc2 + acc3);
                while j < d_in {
                    acc += (vrow[j] as f32) * xk[j];
                    j += 1;
                }
                zk[k] = acc * self.sv[k];
            }
        }

        let nthreads = threads.max(1);
        if nthreads == 1 || d_out < 512 {
            for tok in 0..k_tokens {
                let zk = &z[tok * r..(tok + 1) * r];
                let yk = &mut y[tok * d_out..(tok + 1) * d_out];
                for row in 0..d_out {
                    let urow = &self.u_i8[row * r..(row + 1) * r];
                    let mut acc = 0.0f32;
                    for k in 0..r {
                        acc += (urow[k] as f32) * zk[k];
                    }
                    yk[row] += acc * self.su[row];
                }
            }
        } else {
            let chunk = d_out.div_ceil(nthreads).max(1);
            let u_i8 = &self.u_i8;
            let su = &self.su;
            let z_ref = &z;
            for tok in 0..k_tokens {
                let zk = &z_ref[tok * r..(tok + 1) * r];
                let yk = &mut y[tok * d_out..(tok + 1) * d_out];
                std::thread::scope(|s| {
                    for (bi, yslice) in yk.chunks_mut(chunk).enumerate() {
                        let r0 = bi * chunk;
                        s.spawn(move || {
                            for (dr, yv) in yslice.iter_mut().enumerate() {
                                let row = r0 + dr;
                                let urow = &u_i8[row * r..(row + 1) * r];
                                let mut acc = 0.0f32;
                                for k in 0..r {
                                    acc += (urow[k] as f32) * zk[k];
                                }
                                *yv += acc * su[row];
                            }
                        });
                    }
                });
            }
        }
    }
}

pub fn fwht_block_inplace(x: &mut [f32], bsize: usize) -> Result<(), String> {
    if !bsize.is_power_of_two() || bsize == 0 || x.len() % bsize != 0 {
        return Err(format!("fwht_block_inplace invalid bsize={bsize} len={}", x.len()));
    }
    let inv_sqrt = 1.0f32 / (bsize as f32).sqrt();
    for chunk in x.chunks_exact_mut(bsize) {
        let mut len = 1usize;
        while len < bsize {
            let step = len * 2;
            let mut i = 0usize;
            while i < bsize {
                let (left, right) = chunk[i..i + step].split_at_mut(len);
                for j in 0..len {
                    let u = left[j];
                    let v = right[j];
                    left[j] = u + v;
                    right[j] = u - v;
                }
                i += step;
            }
            len = step;
        }
        for v in chunk.iter_mut() {
            *v *= inv_sqrt;
        }
    }
    Ok(())
}

pub fn expected_tile16_bytes(d_out: usize, d_in: usize, block_size: usize, n_bits: u8, is_asym: bool) -> Result<usize, String> {
    if d_out == 0 || d_in == 0 {
        return Err("d_out and d_in must be non-zero".into());
    }
    if !(block_size == 64 || block_size == 128 || block_size == 256) || d_in % block_size != 0 {
        return Err(format!("invalid tile16 block_size={block_size} for d_in={d_in}"));
    }
    if n_bits == 0 || n_bits > 4 {
        return Err(format!("invalid tile16 n_bits={n_bits}"));
    }
    let n_tiles = d_out.div_ceil(16);
    let n_blks = d_in / block_size;
    let hdr_bytes = if is_asym { 64 } else { 32 };
    let plane_bytes = (block_size / 8) * 16;
    Ok(n_tiles * n_blks * (hdr_bytes + (n_bits as usize) * plane_bytes))
}

pub struct T16CMat {
    pub name: String,
    pub d_out: usize,
    pub d_in: usize,
    pub block_size: usize,
    pub n_bits: u8,
    pub is_asym: bool,
    pub has_fwht: bool,
    pub packed_bytes_len: usize,
    packed: Vec<Aligned64>,
    pub inv_s: Option<Vec<f32>>,
    pub low_rank: Option<Int8LowRank>,
}

impl T16CMat {
    pub fn total_bytes(&self) -> usize {
        self.packed_bytes_len
            + self.inv_s.as_ref().map_or(0, |v| v.len() * 4)
            + self.low_rank.as_ref().map_or(0, |lr| lr.byte_len())
    }

    pub fn packed_slice(&self) -> &[u8] {
        aligned_as_slice(&self.packed, self.packed_bytes_len)
    }

    pub fn pack_from_components(
        name: String,
        d_out: usize,
        d_in: usize,
        block_size: usize,
        n_bits: u8,
        is_asym: bool,
        has_fwht: bool,
        sc_f32: &[f32],
        bi_f32: Option<&[f32]>,
        idx_u8: &[u8],
        inv_s: Option<Vec<f32>>,
        low_rank: Option<Int8LowRank>,
    ) -> Result<Self, String> {
        let pbytes = expected_tile16_bytes(d_out, d_in, block_size, n_bits, is_asym)?;
        let n_blks = d_in / block_size;
        if sc_f32.len() != d_out * n_blks {
            return Err(format!("sc_f32 len {} != {}", sc_f32.len(), d_out * n_blks));
        }
        if is_asym {
            let bi = bi_f32.ok_or("bi_f32 required when is_asym=true")?;
            if bi.len() != d_out * n_blks {
                return Err(format!("bi_f32 len {} != {}", bi.len(), d_out * n_blks));
            }
        }
        if idx_u8.len() != d_out * d_in {
            return Err(format!("idx_u8 len {} != {}", idx_u8.len(), d_out * d_in));
        }
        if let Some(ref isv) = inv_s {
            if isv.len() != d_in {
                return Err(format!("inv_s len {} != {}", isv.len(), d_in));
            }
        }
        if let Some(ref lr) = low_rank {
            if lr.d_out != d_out || lr.d_in != d_in {
                return Err(format!("low_rank dims ({}, {}) != ({d_out}, {d_in})", lr.d_out, lr.d_in));
            }
        }

        let mut packed = make_aligned_buf(pbytes);
        let dst = aligned_as_mut_slice(&mut packed, pbytes);
        let n_tiles = d_out.div_ceil(16);
        let hdr_bytes = if is_asym { 64 } else { 32 };
        let bytes_per_row_blk = block_size / 8;
        let plane_bytes = bytes_per_row_blk * 16;
        let tblk_bytes = hdr_bytes + (n_bits as usize) * plane_bytes;
        let max_level = (1u8 << n_bits) - 1;

        for t in 0..n_tiles {
            for blk in 0..n_blks {
                let off = (t * n_blks + blk) * tblk_bytes;
                for r in 0..16 {
                    let gr = t * 16 + r;
                    if gr >= d_out {
                        continue;
                    }
                    let sb_idx = gr * n_blks + blk;
                    let sc_h = fp32_to_fp16(sc_f32[sb_idx]).to_le_bytes();
                    dst[off + r * 2] = sc_h[0];
                    dst[off + r * 2 + 1] = sc_h[1];
                    if let Some(bi) = bi_f32 {
                        let bi_h = fp32_to_fp16(bi[sb_idx]).to_le_bytes();
                        dst[off + 32 + r * 2] = bi_h[0];
                        dst[off + 32 + r * 2 + 1] = bi_h[1];
                    }
                    let row_blk = &idx_u8[gr * d_in + blk * block_size..gr * d_in + (blk + 1) * block_size];
                    for &v in row_blk {
                        if v > max_level {
                            return Err(format!("idx value {v} exceeds max_level {max_level}"));
                        }
                    }
                    for p in 0..(n_bits as usize) {
                        let p_off = off + hdr_bytes + p * plane_bytes;
                        for bo in 0..bytes_per_row_blk {
                            let mut byte = 0u8;
                            for bit in 0..8 {
                                if ((row_blk[bo * 8 + bit] >> p) & 1) != 0 {
                                    byte |= 1u8 << (7 - bit);
                                }
                            }
                            dst[p_off + bo * 16 + r] = byte;
                        }
                    }
                }
            }
        }

        Ok(Self {
            name,
            d_out,
            d_in,
            block_size,
            n_bits,
            is_asym,
            has_fwht,
            packed_bytes_len: pbytes,
            packed,
            inv_s,
            low_rank,
        })
    }

    pub fn decode_row(&self, row: usize) -> Result<Vec<f32>, String> {
        if row >= self.d_out {
            return Err(format!("decode_row out of bounds: {row} >= {}", self.d_out));
        }
        let d_in = self.d_in;
        let bsize = self.block_size;
        let n_blks = d_in / bsize;
        let hdr_bytes = if self.is_asym { 64 } else { 32 };
        let bytes_per_row_blk = bsize / 8;
        let plane_bytes = bytes_per_row_blk * 16;
        let tblk_bytes = hdr_bytes + (self.n_bits as usize) * plane_bytes;
        let t = row / 16;
        let r = row % 16;
        let src = self.packed_slice();

        let mut w = vec![0f32; d_in];
        for blk in 0..n_blks {
            let off = (t * n_blks + blk) * tblk_bytes;
            let sc = fp16_to_f32(u16::from_le_bytes([src[off + r * 2], src[off + r * 2 + 1]]));
            let bi = if self.is_asym {
                fp16_to_f32(u16::from_le_bytes([src[off + 32 + r * 2], src[off + 32 + r * 2 + 1]]))
            } else {
                0.0f32
            };
            let w_blk = &mut w[blk * bsize..(blk + 1) * bsize];
            for v in w_blk.iter_mut() {
                *v = bi;
            }
            for p in 0..(self.n_bits as usize) {
                let wp = sc * ((1u32 << p) as f32);
                let p_off = off + hdr_bytes + p * plane_bytes;
                for bo in 0..bytes_per_row_blk {
                    let b = src[p_off + bo * 16 + r];
                    for bit in 0..8 {
                        let sgn = if ((b >> (7 - bit)) & 1) != 0 { 1.0f32 } else { -1.0f32 };
                        w_blk[bo * 8 + bit] += wp * sgn;
                    }
                }
            }
        }
        if self.has_fwht {
            fwht_block_inplace(&mut w, bsize)?;
        }
        if let Some(ref inv_s) = self.inv_s {
            for j in 0..d_in {
                w[j] *= inv_s[j];
            }
        }
        if let Some(ref lr) = self.low_rank {
            let rnk = lr.rank;
            let su_r = lr.su[row];
            let urow = &lr.u_i8[row * rnk..(row + 1) * rnk];
            for k in 0..rnk {
                let coef = su_r * (urow[k] as f32) * lr.sv[k];
                let vrow = &lr.v_i8[k * d_in..(k + 1) * d_in];
                for j in 0..d_in {
                    w[j] += coef * (vrow[j] as f32);
                }
            }
        }
        Ok(w)
    }

    pub fn project_batch(
        &self,
        x_raw: &[f32],
        k_tokens: usize,
        native: Option<&NativeTile16Kernel>,
        threads: usize,
    ) -> Result<Vec<f32>, String> {
        let d_out = self.d_out;
        let d_in = self.d_in;
        if x_raw.len() != k_tokens * d_in {
            return Err(format!(
                "{}: x len {} != k_tokens({}) * d_in({})",
                self.name,
                x_raw.len(),
                k_tokens,
                d_in
            ));
        }
        if k_tokens == 0 {
            return Ok(Vec::new());
        }

        let mut x_work: Option<Vec<f32>> = if self.inv_s.is_some() || self.has_fwht {
            let mut buf = x_raw.to_vec();
            if let Some(ref inv_s) = self.inv_s {
                for tok in 0..k_tokens {
                    let row = &mut buf[tok * d_in..(tok + 1) * d_in];
                    for j in 0..d_in {
                        row[j] *= inv_s[j];
                    }
                }
            }
            if self.has_fwht {
                fwht_block_inplace(&mut buf, self.block_size)?;
            }
            Some(buf)
        } else {
            None
        };
        let x_mat: &[f32] = match x_work.as_mut() {
            Some(v) => v.as_slice(),
            None => x_raw,
        };

        let mut y = vec![0f32; k_tokens * d_out];
        if let Some(kern) = native {
            let rc = unsafe {
                (kern.gemm_fn)(
                    self.packed_slice().as_ptr(),
                    ptr::null(),
                    x_mat.as_ptr(),
                    y.as_mut_ptr(),
                    d_out as i32,
                    d_in as i32,
                    self.block_size as i32,
                    self.n_bits as i32,
                    if self.is_asym { 1 } else { 0 },
                    k_tokens as i32,
                    0.0f32,
                    threads.max(1) as i32,
                    ptr::null_mut(),
                )
            };
            if rc != 0 {
                return Err(format!("native gemm_bitserial_tile16_batch failed rc={rc} on {}", self.name));
            }
        } else {
            self.matmul_rust(x_mat, k_tokens, &mut y, threads);
        }

        if let Some(ref lr) = self.low_rank {
            lr.add_batch_into(x_raw, k_tokens, &mut y, threads);
        }
        Ok(y)
    }

    fn matmul_rust(&self, x_mat: &[f32], k_tokens: usize, y: &mut [f32], threads: usize) {
        let d_out = self.d_out;
        let d_in = self.d_in;
        let bsize = self.block_size;
        let n_blks = d_in / bsize;
        let n_tiles = d_out.div_ceil(16);
        let hdr_bytes = if self.is_asym { 64 } else { 32 };
        let bytes_per_row_blk = bsize / 8;
        let plane_bytes = bytes_per_row_blk * 16;
        let tblk_bytes = hdr_bytes + (self.n_bits as usize) * plane_bytes;
        let packed = self.packed_slice();
        let is_asym = self.is_asym;
        let n_bits = self.n_bits as usize;

        let nthreads = threads.max(1).min(d_out);
        let chunk = d_out.div_ceil(nthreads).max(1);
        for tok in 0..k_tokens {
            let xk = &x_mat[tok * d_in..(tok + 1) * d_in];
            let yk = &mut y[tok * d_out..(tok + 1) * d_out];
            std::thread::scope(|s| {
                for (bi, yslice) in yk.chunks_mut(chunk).enumerate() {
                    let r_start = bi * chunk;
                    s.spawn(move || {
                        let _ = n_tiles;
                        for (dr, yv) in yslice.iter_mut().enumerate() {
                            let row = r_start + dr;
                            let t = row / 16;
                            let r = row % 16;
                            let mut acc = 0.0f64;
                            for blk in 0..n_blks {
                                let off = (t * n_blks + blk) * tblk_bytes;
                                let sc = fp16_to_f32(u16::from_le_bytes([packed[off + r * 2], packed[off + r * 2 + 1]])) as f64;
                                let bi_v = if is_asym {
                                    fp16_to_f32(u16::from_le_bytes([packed[off + 32 + r * 2], packed[off + 32 + r * 2 + 1]])) as f64
                                } else {
                                    0.0f64
                                };
                                let xb = &xk[blk * bsize..(blk + 1) * bsize];
                                if is_asym {
                                    let sx: f64 = xb.iter().map(|&v| v as f64).sum();
                                    acc += bi_v * sx;
                                }
                                for p in 0..n_bits {
                                    let wp = sc * ((1u32 << p) as f64);
                                    let p_off = off + hdr_bytes + p * plane_bytes;
                                    let mut isum = 0.0f64;
                                    for bo in 0..bytes_per_row_blk {
                                        let b = packed[p_off + bo * 16 + r];
                                        let base = bo * 8;
                                        for bit in 0..8 {
                                            let v = xb[base + bit] as f64;
                                            isum += if ((b >> (7 - bit)) & 1) != 0 { v } else { -v };
                                        }
                                    }
                                    acc += wp * isum;
                                }
                            }
                            *yv = acc as f32;
                        }
                    });
                }
            });
        }
    }
}

pub type GemmTile16BatchFn = unsafe extern "C" fn(
    packed: *const u8,
    tile_norm_fp16: *const u16,
    x: *const f32,
    y: *mut f32,
    d_out: i32,
    d_in: i32,
    block_size: i32,
    n_bits: i32,
    is_asym: i32,
    k_tokens: i32,
    cs_skip_thresh: f32,
    num_threads: i32,
    out_skipped: *mut i64,
) -> i32;

pub struct NativeTile16Kernel {
    handle: usize,
    pub path: String,
    pub gemm_fn: GemmTile16BatchFn,
}

unsafe impl Send for NativeTile16Kernel {}
unsafe impl Sync for NativeTile16Kernel {}

#[cfg(unix)]
mod sys {
    use std::ffi::c_void;
    pub const RTLD_NOW: i32 = 2;
    extern "C" {
        pub fn dlopen(filename: *const i8, flag: i32) -> *mut c_void;
        pub fn dlsym(handle: *mut c_void, symbol: *const i8) -> *mut c_void;
        pub fn dlclose(handle: *mut c_void) -> i32;
    }
}

#[cfg(windows)]
mod sys {
    use std::ffi::c_void;
    extern "system" {
        pub fn LoadLibraryW(lp_lib_file_name: *const u16) -> *mut c_void;
        pub fn GetProcAddress(h_module: *mut c_void, lp_proc_name: *const i8) -> *mut c_void;
        pub fn FreeLibrary(h_module: *mut c_void) -> i32;
    }
}

impl NativeTile16Kernel {
    pub fn try_load(path: &Path) -> Result<Self, String> {
        if !path.is_file() {
            return Err(format!("native library does not exist: {}", path.display()));
        }
        #[cfg(unix)]
        unsafe {
            let s = path.to_str().ok_or("non-utf8 path")?;
            let mut c_path = s.as_bytes().to_vec();
            c_path.push(0);
            let h = sys::dlopen(c_path.as_ptr() as *const i8, sys::RTLD_NOW);
            if h.is_null() {
                return Err(format!("dlopen failed for {}", path.display()));
            }
            let sym = sys::dlsym(h, b"gemm_bitserial_tile16_batch\0".as_ptr() as *const i8);
            if sym.is_null() {
                sys::dlclose(h);
                return Err(format!("missing gemm_bitserial_tile16_batch in {}", path.display()));
            }
            let gemm_fn: GemmTile16BatchFn = std::mem::transmute(sym);
            Ok(Self { handle: h as usize, path: s.to_string(), gemm_fn })
        }
        #[cfg(windows)]
        unsafe {
            use std::os::windows::ffi::OsStrExt;
            let mut wide: Vec<u16> = path.as_os_str().encode_wide().collect();
            wide.push(0);
            let h = sys::LoadLibraryW(wide.as_ptr());
            if h.is_null() {
                return Err(format!("LoadLibraryW failed for {}", path.display()));
            }
            let sym = sys::GetProcAddress(h, b"gemm_bitserial_tile16_batch\0".as_ptr() as *const i8);
            if sym.is_null() {
                sys::FreeLibrary(h);
                return Err(format!("missing gemm_bitserial_tile16_batch in {}", path.display()));
            }
            let gemm_fn: GemmTile16BatchFn = std::mem::transmute(sym);
            Ok(Self { handle: h as usize, path: path.display().to_string(), gemm_fn })
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = path;
            Err("unsupported OS for dynamic native loading".into())
        }
    }

    pub fn discover(explicit: Option<&Path>) -> Option<Self> {
        let mut cands: Vec<PathBuf> = Vec::new();
        if let Some(p) = explicit {
            cands.push(p.to_path_buf());
        }
        let lib_name = if cfg!(windows) {
            "neon_stream_accelerator.dll"
        } else {
            "libneon_stream_accelerator.so"
        };
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                cands.push(dir.join(lib_name));
                cands.push(dir.join("neon_runner").join(lib_name));
            }
        }
        if let Ok(cwd) = std::env::current_dir() {
            cands.push(cwd.join(lib_name));
            cands.push(cwd.join("neon_runner").join(lib_name));
            cands.push(cwd.join("..").join("..").join("neon_runner").join(lib_name));
        }
        for c in cands {
            if let Ok(k) = Self::try_load(&c) {
                return Some(k);
            }
        }
        None
    }
}

impl Drop for NativeTile16Kernel {
    fn drop(&mut self) {
        // Intentionally leak the loaded library handle for the lifetime of the
        // process on Windows/Unix: unloading an OpenMP-backed DLL/SO with
        // FreeLibrary/dlclose while OpenMP worker threads remain parked causes
        // 0xC0000005 access violations at test/process teardown.
        let _ = self.handle;
    }
}

pub struct T16CModel {
    pub arch_flags: u32,
    pub norms: HashMap<String, Vec<f32>>,
    pub mats: HashMap<String, T16CMat>,
    pub total_mat_bytes: usize,
    pub native: Option<NativeTile16Kernel>,
}

fn read_u8<R: Read>(r: &mut R, pos: &mut usize) -> Result<u8, String> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b).map_err(|e| format!("eof reading u8 at {pos}: {e}"))?;
    *pos += 1;
    Ok(b[0])
}
fn read_u16<R: Read>(r: &mut R, pos: &mut usize) -> Result<u16, String> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b).map_err(|e| format!("eof reading u16 at {pos}: {e}"))?;
    *pos += 2;
    Ok(u16::from_le_bytes(b))
}
fn read_u32<R: Read>(r: &mut R, pos: &mut usize) -> Result<u32, String> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|e| format!("eof reading u32 at {pos}: {e}"))?;
    *pos += 4;
    Ok(u32::from_le_bytes(b))
}
fn read_u64<R: Read>(r: &mut R, pos: &mut usize) -> Result<u64, String> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(|e| format!("eof reading u64 at {pos}: {e}"))?;
    *pos += 8;
    Ok(u64::from_le_bytes(b))
}
fn read_f32_vec<R: Read>(r: &mut R, pos: &mut usize, n: usize) -> Result<Vec<f32>, String> {
    let mut raw = vec![0u8; n * 4];
    r.read_exact(&mut raw).map_err(|e| format!("eof reading f32[{n}] at {pos}: {e}"))?;
    *pos += n * 4;
    Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

impl T16CModel {
    pub fn load(path: &Path, native_lib: Option<&Path>, disable_native: bool) -> Result<Self, String> {
        let f = File::open(path).map_err(|e| format!("open T16C {}: {e}", path.display()))?;
        let mut r = BufReader::with_capacity(4 * 1024 * 1024, f);
        let mut pos = 0usize;
        let mut hdr = [0u8; 24];
        r.read_exact(&mut hdr).map_err(|e| format!("read T16C header: {e}"))?;
        pos += 24;
        if &hdr[0..4] != T16C_MAGIC {
            return Err(format!("invalid T16C magic: {:?}", &hdr[0..4]));
        }
        let ver = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]);
        if ver != T16C_VERSION {
            return Err(format!("unsupported T16C version {ver}"));
        }
        let arch_flags = u32::from_le_bytes([hdr[8], hdr[9], hdr[10], hdr[11]]);
        let n_norms = u32::from_le_bytes([hdr[12], hdr[13], hdr[14], hdr[15]]) as usize;
        let n_mats = u32::from_le_bytes([hdr[16], hdr[17], hdr[18], hdr[19]]) as usize;

        let mut norms = HashMap::with_capacity(n_norms);
        for _ in 0..n_norms {
            let nlen = read_u16(&mut r, &mut pos)? as usize;
            let mut nbuf = vec![0u8; nlen];
            r.read_exact(&mut nbuf).map_err(|e| format!("read norm name: {e}"))?;
            pos += nlen;
            let name = String::from_utf8(nbuf).map_err(|e| format!("utf8 norm name: {e}"))?;
            let len = read_u32(&mut r, &mut pos)? as usize;
            let data = read_f32_vec(&mut r, &mut pos, len)?;
            norms.insert(name, data);
        }

        let mut mats = HashMap::with_capacity(n_mats);
        let mut total_mat_bytes = 0usize;
        for _ in 0..n_mats {
            let nlen = read_u16(&mut r, &mut pos)? as usize;
            let d_out = read_u32(&mut r, &mut pos)? as usize;
            let d_in = read_u32(&mut r, &mut pos)? as usize;
            let block_size = read_u16(&mut r, &mut pos)? as usize;
            let n_bits = read_u8(&mut r, &mut pos)?;
            let is_asym_u8 = read_u8(&mut r, &mut pos)?;
            let has_fwht_u8 = read_u8(&mut r, &mut pos)?;
            let inv_s_kind = read_u8(&mut r, &mut pos)?;
            let low_rank_u16 = read_u16(&mut r, &mut pos)? as usize;
            let packed_bytes = read_u64(&mut r, &mut pos)? as usize;
            let lr_bytes = read_u64(&mut r, &mut pos)? as usize;

            if is_asym_u8 > 1 || has_fwht_u8 > 1 || inv_s_kind > 2 {
                return Err(format!("invalid flags is_asym={is_asym_u8} has_fwht={has_fwht_u8} inv_s_kind={inv_s_kind}"));
            }
            let is_asym = is_asym_u8 == 1;
            let has_fwht = has_fwht_u8 == 1;

            let mut nbuf = vec![0u8; nlen];
            r.read_exact(&mut nbuf).map_err(|e| format!("read mat name: {e}"))?;
            pos += nlen;
            let name = String::from_utf8(nbuf).map_err(|e| format!("utf8 mat name: {e}"))?;

            let pad = (64 - (pos % 64)) % 64;
            if pad > 0 {
                let mut pbuf = [0u8; 64];
                r.read_exact(&mut pbuf[..pad]).map_err(|e| format!("read align pad: {e}"))?;
                pos += pad;
            }

            let exp_p = expected_tile16_bytes(d_out, d_in, block_size, n_bits, is_asym)?;
            if packed_bytes != exp_p {
                return Err(format!("{name}: packed_bytes {packed_bytes} != expected {exp_p}"));
            }
            let mut packed = make_aligned_buf(packed_bytes);
            r.read_exact(aligned_as_mut_slice(&mut packed, packed_bytes))
                .map_err(|e| format!("read {name} packed bytes ({packed_bytes}): {e}"))?;
            pos += packed_bytes;

            let inv_s = match inv_s_kind {
                0 => None,
                1 => Some(read_f32_vec(&mut r, &mut pos, d_in)?),
                2 => {
                    let mut raw = vec![0u8; d_in * 2];
                    r.read_exact(&mut raw).map_err(|e| format!("read {name} inv_s fp16: {e}"))?;
                    pos += d_in * 2;
                    Some(raw.chunks_exact(2).map(|c| fp16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect())
                }
                _ => return Err(format!("{name}: unsupported inv_s_kind={inv_s_kind}")),
            };

            let low_rank = if low_rank_u16 > 0 {
                let rk = low_rank_u16;
                let exp_lr = d_out * 4 + rk * 4 + d_out * rk + rk * d_in;
                if lr_bytes != exp_lr {
                    return Err(format!("{name}: lr_bytes {lr_bytes} != expected {exp_lr}"));
                }
                let su = read_f32_vec(&mut r, &mut pos, d_out)?;
                let sv = read_f32_vec(&mut r, &mut pos, rk)?;
                let mut uraw = vec![0u8; d_out * rk];
                r.read_exact(&mut uraw).map_err(|e| format!("read {name} u_i8: {e}"))?;
                pos += uraw.len();
                let u_i8 = uraw.into_iter().map(|b| b as i8).collect();
                let mut vraw = vec![0u8; rk * d_in];
                r.read_exact(&mut vraw).map_err(|e| format!("read {name} v_i8: {e}"))?;
                pos += vraw.len();
                let v_i8 = vraw.into_iter().map(|b| b as i8).collect();
                Some(Int8LowRank { rank: rk, d_out, d_in, su, sv, u_i8, v_i8 })
            } else {
                if lr_bytes != 0 {
                    return Err(format!("{name}: lr_bytes={lr_bytes} with low_rank=0"));
                }
                None
            };

            let mat = T16CMat {
                name: name.clone(),
                d_out,
                d_in,
                block_size,
                n_bits,
                is_asym,
                has_fwht,
                packed_bytes_len: packed_bytes,
                packed,
                inv_s,
                low_rank,
            };
            total_mat_bytes += mat.total_bytes();
            mats.insert(name, mat);
        }

        let native = if disable_native {
            None
        } else {
            NativeTile16Kernel::discover(native_lib)
        };
        Ok(Self { arch_flags, norms, mats, total_mat_bytes, native })
    }

    pub fn write_file(
        path: &Path,
        arch_flags: u32,
        norms: &[(String, Vec<f32>)],
        mats: &[T16CMat],
    ) -> Result<(), String> {
        let f = File::create(path).map_err(|e| format!("create T16C {}: {e}", path.display()))?;
        let mut w = BufWriter::new(f);
        let mut pos = 0usize;
        let mut hdr = [0u8; 24];
        hdr[0..4].copy_from_slice(T16C_MAGIC);
        hdr[4..8].copy_from_slice(&T16C_VERSION.to_le_bytes());
        hdr[8..12].copy_from_slice(&arch_flags.to_le_bytes());
        hdr[12..16].copy_from_slice(&(norms.len() as u32).to_le_bytes());
        hdr[16..20].copy_from_slice(&(mats.len() as u32).to_le_bytes());
        w.write_all(&hdr).map_err(|e| format!("write header: {e}"))?;
        pos += 24;

        for (name, data) in norms {
            let nb = name.as_bytes();
            w.write_all(&(nb.len() as u16).to_le_bytes()).map_err(|e| format!("write norm len: {e}"))?;
            w.write_all(nb).map_err(|e| format!("write norm name: {e}"))?;
            w.write_all(&(data.len() as u32).to_le_bytes()).map_err(|e| format!("write norm data len: {e}"))?;
            for &v in data {
                w.write_all(&v.to_le_bytes()).map_err(|e| format!("write norm f32: {e}"))?;
            }
            pos += 2 + nb.len() + 4 + data.len() * 4;
        }

        for m in mats {
            let nb = m.name.as_bytes();
            let inv_kind: u8 = if m.inv_s.is_some() { 1 } else { 0 };
            let lr_rank: u16 = m.low_rank.as_ref().map_or(0, |lr| lr.rank as u16);
            let lr_bytes: u64 = m.low_rank.as_ref().map_or(0, |lr| lr.byte_len() as u64);
            w.write_all(&(nb.len() as u16).to_le_bytes()).map_err(|e| format!("write mat hdr: {e}"))?;
            w.write_all(&(m.d_out as u32).to_le_bytes()).map_err(|e| format!("write mat hdr: {e}"))?;
            w.write_all(&(m.d_in as u32).to_le_bytes()).map_err(|e| format!("write mat hdr: {e}"))?;
            w.write_all(&(m.block_size as u16).to_le_bytes()).map_err(|e| format!("write mat hdr: {e}"))?;
            w.write_all(&[m.n_bits, if m.is_asym { 1 } else { 0 }, if m.has_fwht { 1 } else { 0 }, inv_kind]).map_err(|e| format!("write mat hdr: {e}"))?;
            w.write_all(&lr_rank.to_le_bytes()).map_err(|e| format!("write mat hdr: {e}"))?;
            w.write_all(&(m.packed_bytes_len as u64).to_le_bytes()).map_err(|e| format!("write mat hdr: {e}"))?;
            w.write_all(&lr_bytes.to_le_bytes()).map_err(|e| format!("write mat hdr: {e}"))?;
            w.write_all(nb).map_err(|e| format!("write mat name: {e}"))?;
            pos += 34 + nb.len();

            let pad = (64 - (pos % 64)) % 64;
            if pad > 0 {
                let z = [0u8; 64];
                w.write_all(&z[..pad]).map_err(|e| format!("write pad: {e}"))?;
                pos += pad;
            }
            w.write_all(m.packed_slice()).map_err(|e| format!("write packed: {e}"))?;
            pos += m.packed_bytes_len;

            if let Some(ref inv) = m.inv_s {
                for &v in inv {
                    w.write_all(&v.to_le_bytes()).map_err(|e| format!("write inv_s: {e}"))?;
                }
                pos += inv.len() * 4;
            }
            if let Some(ref lr) = m.low_rank {
                for &v in &lr.su {
                    w.write_all(&v.to_le_bytes()).map_err(|e| format!("write su: {e}"))?;
                }
                for &v in &lr.sv {
                    w.write_all(&v.to_le_bytes()).map_err(|e| format!("write sv: {e}"))?;
                }
                let ubytes: Vec<u8> = lr.u_i8.iter().map(|&v| v as u8).collect();
                let vbytes: Vec<u8> = lr.v_i8.iter().map(|&v| v as u8).collect();
                w.write_all(&ubytes).map_err(|e| format!("write u_i8: {e}"))?;
                w.write_all(&vbytes).map_err(|e| format!("write v_i8: {e}"))?;
                pos += lr.byte_len();
            }
        }
        w.flush().map_err(|e| format!("flush T16C: {e}"))?;
        Ok(())
    }

    pub fn mat(&self, name: &str) -> Result<&T16CMat, String> {
        self.mats.get(name).ok_or_else(|| format!("T16C missing matrix '{name}'"))
    }

    pub fn norm(&self, name: &str) -> Result<&[f32], String> {
        self.norms.get(name).map(|v| v.as_slice()).ok_or_else(|| format!("T16C missing norm '{name}'"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fwht_is_involutory() {
        let orig: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.37).sin()).collect();
        let mut x = orig.clone();
        fwht_block_inplace(&mut x, 128).unwrap();
        fwht_block_inplace(&mut x, 128).unwrap();
        for (a, b) in orig.iter().zip(x.iter()) {
            assert!((a - b).abs() < 1e-4, "orig={a} vs roundtrip={b}");
        }
    }

    #[test]
    fn t16c_roundtrip_and_decode_row_parity() {
        let (d_out, d_in, bsize, nb) = (19usize, 256usize, 128usize, 2u8);
        let n_blks = d_in / bsize;
        let sc: Vec<f32> = (0..d_out * n_blks).map(|i| 0.02 + 0.001 * (i as f32 % 7.0)).collect();
        let bi: Vec<f32> = (0..d_out * n_blks).map(|i| -0.005 + 0.002 * (i as f32 % 5.0)).collect();
        let idx: Vec<u8> = (0..d_out * d_in).map(|i| ((i * 13 + 5) % 4) as u8).collect();
        let inv_s: Vec<f32> = (0..d_in).map(|j| 0.8 + 0.002 * (j as f32)).collect();
        let rk = 4usize;
        let lr = Int8LowRank {
            rank: rk,
            d_out,
            d_in,
            su: vec![0.01f32; d_out],
            sv: vec![0.02f32; rk],
            u_i8: (0..d_out * rk).map(|i| ((i % 200) as i32 - 100) as i8).collect(),
            v_i8: (0..rk * d_in).map(|i| ((i % 200) as i32 - 100) as i8).collect(),
        };
        let mat = T16CMat::pack_from_components(
            "test.proj".into(), d_out, d_in, bsize, nb, true, true, &sc, Some(&bi), &idx, Some(inv_s), Some(lr)
        ).unwrap();

        let tmp = std::env::temp_dir().join("t16c_unit_test.t16c");
        T16CModel::write_file(&tmp, ARCH_FLAG_GEMMA4, &[("n0".into(), vec![1.0, 2.0])], &[mat]).unwrap();
        let loaded = T16CModel::load(&tmp, None, true).unwrap();
        let _ = std::fs::remove_file(&tmp);

        let lm = loaded.mat("test.proj").unwrap();
        let x: Vec<f32> = (0..2 * d_in).map(|i| ((i as f32) * 0.19).cos()).collect();
        let y_rust = lm.project_batch(&x, 2, None, 2).unwrap();
        for r in 0..d_out {
            let row = lm.decode_row(r).unwrap();
            let mut dot0 = 0.0f32;
            let mut dot1 = 0.0f32;
            for j in 0..d_in {
                dot0 += row[j] * x[j];
                dot1 += row[j] * x[d_in + j];
            }
            assert!((y_rust[r] - dot0).abs() < 2e-3, "row {r} tok0: {} vs {dot0}", y_rust[r]);
            assert!((y_rust[d_out + r] - dot1).abs() < 2e-3, "row {r} tok1: {} vs {dot1}", y_rust[d_out + r]);
        }

        if let Some(native_k) = NativeTile16Kernel::discover(None) {
            let y_nat = lm.project_batch(&x, 2, Some(&native_k), 2).unwrap();
            let mut dot = 0.0f64;
            let mut n1 = 0.0f64;
            let mut n2 = 0.0f64;
            for (a, b) in y_rust.iter().zip(y_nat.iter()) {
                dot += (*a as f64) * (*b as f64);
                n1 += (*a as f64) * (*a as f64);
                n2 += (*b as f64) * (*b as f64);
            }
            let cos = dot / (n1.sqrt() * n2.sqrt() + 1e-12);
            assert!(cos > 0.999, "native vs rust tile16 cos={cos}");
        }
    }
}
