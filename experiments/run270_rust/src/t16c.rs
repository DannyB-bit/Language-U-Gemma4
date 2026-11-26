//! T16C v1 codec: compact 64-byte-aligned Tile-16 sub-byte container (1..4-bit
//! symmetric/asymmetric) with optional Hadamard block rotation (`FWHT-64/128/256`),
//! input channel equalization (`inv_s`), and INT8 low-rank residual (`U_i8 @ V_i8`).
//!
//! Dispatches `gemm_bitserial_tile16_batch` through dynamically loaded native SIMD
//! shared libraries (`libneon_stream_accelerator.so` on ARMv8 Linux /
//! `neon_stream_accelerator.dll` on Windows) with a deterministic multi-threaded
//! pure-Rust fallback so `cargo test --release` passes with zero external deps.

use crate::codec::{fp16_to_f32, fp32_to_fp16};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::Arc;

pub const T16C_MAGIC: &[u8; 4] = b"T16C";
pub const T16C_VERSION: u32 = 1;
pub const ARCH_FLAG_GEMMA4: u32 = 1 << 0;
pub const ARCH_FLAG_TIED_HEAD: u32 = 1 << 1;
pub const T16C_GAUSS_BIT: u8 = 0x80;
pub const T16C_ROWMAJOR_BIT: u8 = 0x40;
pub const T16C_COLBLOCK_BIT: u8 = 0x20;

/// Reconstruct additive bit-plane scaling factor (`1<<p` uniform, or non-uniform Gaussian weights when `T16C_GAUSS_BIT` is set).
#[inline]
pub fn plane_weight(n_bits_raw: u8, p: usize) -> f32 {
    let gauss = (n_bits_raw & T16C_GAUSS_BIT) != 0;
    let eff = n_bits_raw & 0x1F;
    if gauss && eff == 2 {
        match p {
            0 => 1.087,
            1 => 2.087,
            _ => (1u32 << p) as f32,
        }
    } else if gauss && eff == 3 {
        match p {
            0 => 1.000,
            1 => 2.080,
            2 => 4.350,
            _ => (1u32 << p) as f32,
        }
    } else {
        (1u32 << p) as f32
    }
}

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

const fn build_tern5_lut() -> [[i8; 5]; 243] {
    let mut lut = [[0i8; 5]; 243];
    let mut c = 0usize;
    while c < 243 {
        let mut v = c;
        let mut k = 0usize;
        while k < 5 {
            lut[c][k] = (v % 3) as i8 - 1;
            v /= 3;
            k += 1;
        }
        c += 1;
    }
    lut
}

const TERN5_LUT: [[i8; 5]; 243] = build_tern5_lut();

/// Packed byte size of a single `16 x block_size` tile-block.
pub fn tile16_blk_bytes(block_size: usize, n_bits: u8, is_asym: bool) -> usize {
    let eff_bits = n_bits & 0x1F;
    if eff_bits == 5 {
        32 + block_size.div_ceil(5) * 16
    } else if eff_bits == 8 {
        32 + block_size * 16
    } else {
        let hdr_bytes = if is_asym { 64 } else { 32 };
        let plane_bytes = (block_size / 8) * 16;
        hdr_bytes + (eff_bits as usize) * plane_bytes
    }
}

pub fn expected_tile16_bytes(d_out: usize, d_in: usize, block_size: usize, n_bits: u8, is_asym: bool) -> Result<usize, String> {
    if d_out == 0 || d_in == 0 {
        return Err("d_out and d_in must be non-zero".into());
    }
    if !(block_size == 64 || block_size == 128 || block_size == 256) || d_in % block_size != 0 {
        return Err(format!("invalid tile16 block_size={block_size} for d_in={d_in}"));
    }
    let gauss = (n_bits & T16C_GAUSS_BIT) != 0;
    let rowmajor = (n_bits & T16C_ROWMAJOR_BIT) != 0;
    let eff_bits = n_bits & 0x1F;
    if eff_bits == 0 || (eff_bits > 5 && eff_bits != 8) {
        return Err(format!("invalid tile16 n_bits={n_bits}"));
    }
    if gauss && eff_bits != 2 && eff_bits != 3 {
        return Err(format!("invalid tile16 T16C_GAUSS_BIT with eff_bits={eff_bits}"));
    }
    if rowmajor && eff_bits > 4 {
        return Err(format!("invalid tile16 T16C_ROWMAJOR_BIT with eff_bits={eff_bits}"));
    }
    if (eff_bits == 5 || eff_bits == 8) && is_asym {
        return Err(format!("invalid tile16 n_bits={n_bits} with is_asym=true"));
    }
    let n_tiles = d_out.div_ceil(16);
    let n_blks = d_in / block_size;
    Ok(n_tiles * n_blks * tile16_blk_bytes(block_size, n_bits, is_asym))
}

#[cfg(unix)]
struct MappedFile {
    ptr: *const u8,
    len: usize,
    file: Arc<File>,
}

#[cfg(unix)]
unsafe impl Send for MappedFile {}
#[cfg(unix)]
unsafe impl Sync for MappedFile {}

#[cfg(unix)]
impl Drop for MappedFile {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.len > 0 {
            unsafe {
                sys::munmap(self.ptr as *mut std::ffi::c_void, self.len);
            }
        }
    }
}

fn pread_exact_at(file: &File, offset: u64, dst: &mut [u8]) -> Result<(), String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(dst, offset)
            .map_err(|e| format!("pread_exact_at(off={offset}, len={}): {e}", dst.len()))
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut cur_off = offset;
        let mut rem = dst;
        while !rem.is_empty() {
            let n = file
                .seek_read(rem, cur_off)
                .map_err(|e| format!("seek_read(off={cur_off}, len={}): {e}", rem.len()))?;
            if n == 0 {
                return Err(format!("seek_read EOF at off={cur_off} rem={}", rem.len()));
            }
            cur_off += n as u64;
            let tmp = rem;
            rem = &mut tmp[n..];
        }
        Ok(())
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (file, offset, dst);
        Err("pread_exact_at unsupported on this OS".into())
    }
}

fn fadvise_dontneed(file: &File, offset: u64, len: usize) {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::AsRawFd;
        if len > 0 {
            unsafe {
                let _ = sys::posix_fadvise(
                    file.as_raw_fd(),
                    offset as i64,
                    len as i64,
                    sys::POSIX_FADV_DONTNEED,
                );
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (file, offset, len);
    }
}

enum PackedStorage {
    Owned(Vec<Aligned64>),
    #[cfg(unix)]
    Mmap {
        map: Arc<MappedFile>,
        offset: usize,
        len: usize,
    },
    StreamedFile {
        file: Arc<File>,
        file_offset: u64,
        len: usize,
    },
}

/// Extracted coalesced/compacted `.t16c` tile-major slab buffered from SSD (or resident memory)
/// for CRTR/Gate-guided sparse execution at high NEON efficiency.
pub struct StreamedSparseSlab {
    pub packed: Vec<Aligned64>,
    pub packed_len: usize,
    pub active_tiles: Option<Vec<usize>>,
    pub active_blks: Option<Vec<usize>>,
    pub eff_d_out: usize,
    pub eff_d_in: usize,
    pub is_full_fallback: bool,
    pub bytes_read: u64,
}

impl StreamedSparseSlab {
    pub fn slice(&self) -> &[u8] {
        aligned_as_slice(&self.packed, self.packed_len)
    }
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
    packed: PackedStorage,
    pub inv_s: Option<Vec<f32>>,
    pub low_rank: Option<Int8LowRank>,
}

impl T16CMat {
    pub fn is_colblock(&self) -> bool {
        (self.n_bits & T16C_COLBLOCK_BIT) != 0
    }

    pub fn is_streamed(&self) -> bool {
        matches!(self.packed, PackedStorage::StreamedFile { .. })
    }

    pub fn total_bytes(&self) -> usize {
        self.packed_bytes_len
            + self.inv_s.as_ref().map_or(0, |v| v.len() * 4)
            + self.low_rank.as_ref().map_or(0, |lr| lr.byte_len())
    }

    pub fn resident_bytes(&self) -> usize {
        let pk = if self.is_streamed() { 0 } else { self.packed_bytes_len };
        pk + self.inv_s.as_ref().map_or(0, |v| v.len() * 4)
            + self.low_rank.as_ref().map_or(0, |lr| lr.byte_len())
    }

    pub fn packed_slice(&self) -> &[u8] {
        match &self.packed {
            PackedStorage::Owned(buf) => aligned_as_slice(buf, self.packed_bytes_len),
            #[cfg(unix)]
            PackedStorage::Mmap { map, offset, len } => unsafe {
                std::slice::from_raw_parts(map.ptr.add(*offset), *len)
            },
            PackedStorage::StreamedFile { .. } => {
                panic!("packed_slice() called directly on StreamedFile matrix {}", self.name)
            }
        }
    }

    fn packed_cow(&self) -> Result<Cow<'_, [u8]>, String> {
        match &self.packed {
            PackedStorage::Owned(buf) => Ok(Cow::Borrowed(aligned_as_slice(buf, self.packed_bytes_len))),
            #[cfg(unix)]
            PackedStorage::Mmap { map, offset, len } => Ok(Cow::Borrowed(unsafe {
                std::slice::from_raw_parts(map.ptr.add(*offset), *len)
            })),
            PackedStorage::StreamedFile { file, file_offset, len } => {
                let mut buf = vec![0u8; *len];
                pread_exact_at(file, *file_offset, &mut buf)?;
                fadvise_dontneed(file, *file_offset, *len);
                Ok(Cow::Owned(buf))
            }
        }
    }

    fn read_packed_range(&self, rel_off: usize, dst: &mut [u8]) -> Result<(), String> {
        let end = rel_off
            .checked_add(dst.len())
            .ok_or_else(|| format!("{}: read_packed_range overflow", self.name))?;
        if end > self.packed_bytes_len {
            return Err(format!(
                "{}: read_packed_range out of bounds ({end} > {})",
                self.name, self.packed_bytes_len
            ));
        }
        match &self.packed {
            PackedStorage::Owned(buf) => {
                dst.copy_from_slice(&aligned_as_slice(buf, self.packed_bytes_len)[rel_off..end]);
                Ok(())
            }
            #[cfg(unix)]
            PackedStorage::Mmap { map, offset, .. } => {
                let src = unsafe { std::slice::from_raw_parts(map.ptr.add(*offset + rel_off), dst.len()) };
                dst.copy_from_slice(src);
                Ok(())
            }
            PackedStorage::StreamedFile { file, file_offset, .. } => {
                pread_exact_at(file, *file_offset + rel_off as u64, dst)
            }
        }
    }

    /// Touch & `MADV_WILLNEED` resident mmap pages for this matrix into physical RAM.
    pub fn advise_willneed_and_touch(&self) -> usize {
        #[cfg(unix)]
        {
            if let PackedStorage::Mmap { map, offset, len } = &self.packed {
                if *len > 0 {
                    return advise_willneed_span(map, *offset, offset.saturating_add(*len));
                }
            }
        }
        0
    }

    /// Evict strictly interior 4096-byte OS pages via `MADV_DONTNEED` + `POSIX_FADV_DONTNEED`,
    /// preserving boundary pages that may be shared with adjacent tensors.
    pub fn advise_dontneed(&self) -> usize {
        #[cfg(unix)]
        {
            if let PackedStorage::Mmap { map, offset, len } = &self.packed {
                let base_addr = (map.ptr as usize).saturating_add(*offset);
                let start_page = (base_addr + 4095) & !4095usize;
                let end_page = (base_addr + *len) & !4095usize;
                if end_page > start_page {
                    let ev_len = end_page - start_page;
                    let rel_page_off = start_page - (map.ptr as usize);
                    unsafe {
                        let _ = sys::madvise(start_page as *mut std::ffi::c_void, ev_len, sys::MADV_DONTNEED);
                    }
                    fadvise_dontneed(&map.file, rel_page_off as u64, ev_len);
                    return ev_len;
                }
            } else if let PackedStorage::StreamedFile { file, file_offset, len } = &self.packed {
                fadvise_dontneed(file, *file_offset, *len);
            }
        }
        0
    }

    /// Repack a resident/loaded matrix into Column-Block-Major (`T16C_COLBLOCK_BIT`) layout
    /// so contiguous `[n_tiles, tblk_bytes]` slices per input block can be streamed cleanly.
    pub fn to_colblock(&self) -> Result<Self, String> {
        let raw_cow = self.packed_cow()?;
        let src: &[u8] = raw_cow.as_ref();
        let n_tiles = self.d_out.div_ceil(16);
        let n_blks = self.d_in / self.block_size;
        let tblk_bytes = tile16_blk_bytes(self.block_size, self.n_bits, self.is_asym);
        let mut out = make_aligned_buf(self.packed_bytes_len);
        let dst = aligned_as_mut_slice(&mut out, self.packed_bytes_len);
        if self.is_colblock() {
            dst.copy_from_slice(src);
        } else {
            for t in 0..n_tiles {
                for blk in 0..n_blks {
                    let s_off = (t * n_blks + blk) * tblk_bytes;
                    let d_off = (blk * n_tiles + t) * tblk_bytes;
                    dst[d_off..d_off + tblk_bytes].copy_from_slice(&src[s_off..s_off + tblk_bytes]);
                }
            }
        }
        Ok(Self {
            name: self.name.clone(),
            d_out: self.d_out,
            d_in: self.d_in,
            block_size: self.block_size,
            n_bits: self.n_bits | T16C_COLBLOCK_BIT,
            is_asym: self.is_asym,
            has_fwht: self.has_fwht,
            packed_bytes_len: self.packed_bytes_len,
            packed: PackedStorage::Owned(out),
            inv_s: self.inv_s.clone(),
            low_rank: self.low_rank.clone(),
        })
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
        let tblk_bytes = tile16_blk_bytes(block_size, n_bits, is_asym);
        let quintets = block_size.div_ceil(5);
        let is_colblock = (n_bits & T16C_COLBLOCK_BIT) != 0;
        let rowmajor = (n_bits & T16C_ROWMAJOR_BIT) != 0;
        let eff_bits = n_bits & 0x1F;
        let max_level = if eff_bits == 5 {
            2
        } else if eff_bits == 8 {
            255
        } else {
            (1u8 << eff_bits) - 1
        };

        for t in 0..n_tiles {
            for blk in 0..n_blks {
                let off = if is_colblock {
                    (blk * n_tiles + t) * tblk_bytes
                } else {
                    (t * n_blks + blk) * tblk_bytes
                };
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
                    if eff_bits == 5 {
                        for g in 0..quintets {
                            let base = g * 5;
                            let mut code = 0u16;
                            let mut p3 = 1u16;
                            for k in 0..5 {
                                let col = base + k;
                                let d = if col < block_size { row_blk[col] } else { 1u8 };
                                code += (d as u16) * p3;
                                p3 *= 3;
                            }
                            dst[off + 32 + g * 16 + r] = code as u8;
                        }
                    } else if eff_bits == 8 {
                        for c in 0..block_size {
                            dst[off + 32 + c * 16 + r] = row_blk[c];
                        }
                    } else {
                        for p in 0..(eff_bits as usize) {
                            let p_off = off + hdr_bytes + p * plane_bytes;
                            for bo in 0..bytes_per_row_blk {
                                let mut byte = 0u8;
                                for bit in 0..8 {
                                    if ((row_blk[bo * 8 + bit] >> p) & 1) != 0 {
                                        byte |= 1u8 << (7 - bit);
                                    }
                                }
                                let dst_i = if rowmajor {
                                    p_off + r * bytes_per_row_blk + bo
                                } else {
                                    p_off + bo * 16 + r
                                };
                                dst[dst_i] = byte;
                            }
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
            packed: PackedStorage::Owned(packed),
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
        let n_tiles = self.d_out.div_ceil(16);
        let n_blks = d_in / bsize;
        let hdr_bytes = if self.is_asym { 64 } else { 32 };
        let bytes_per_row_blk = bsize / 8;
        let plane_bytes = bytes_per_row_blk * 16;
        let tblk_bytes = tile16_blk_bytes(bsize, self.n_bits, self.is_asym);
        let quintets = bsize.div_ceil(5);
        let t = row / 16;
        let r = row % 16;
        let src_cow = self.packed_cow()?;
        let src: &[u8] = src_cow.as_ref();
        let is_colblock = self.is_colblock();

        let mut w = vec![0f32; d_in];
        for blk in 0..n_blks {
            let off = if is_colblock {
                (blk * n_tiles + t) * tblk_bytes
            } else {
                (t * n_blks + blk) * tblk_bytes
            };
            let sc = fp16_to_f32(u16::from_le_bytes([src[off + r * 2], src[off + r * 2 + 1]]));
            let bi = if self.is_asym {
                fp16_to_f32(u16::from_le_bytes([src[off + 32 + r * 2], src[off + 32 + r * 2 + 1]]))
            } else {
                0.0f32
            };
            let w_blk = &mut w[blk * bsize..(blk + 1) * bsize];
            let rowmajor = (self.n_bits & T16C_ROWMAJOR_BIT) != 0;
            let eff_bits = self.n_bits & 0x1F;
            if eff_bits == 5 {
                for g in 0..quintets {
                    let code = src[off + 32 + g * 16 + r] as usize;
                    if code >= 243 {
                        return Err(format!("{}: invalid ternary5 code {code} at row={row} blk={blk} g={g}", self.name));
                    }
                    let t5 = TERN5_LUT[code];
                    let base = g * 5;
                    for k in 0..5 {
                        let col = base + k;
                        if col < bsize {
                            w_blk[col] = sc * (t5[k] as f32);
                        }
                    }
                }
            } else if eff_bits == 8 {
                for c in 0..bsize {
                    let q = src[off + 32 + c * 16 + r] as i8;
                    w_blk[c] = sc * (q as f32);
                }
            } else {
                for v in w_blk.iter_mut() {
                    *v = bi;
                }
                for p in 0..(eff_bits as usize) {
                    let wp = sc * plane_weight(self.n_bits, p);
                    let p_off = off + hdr_bytes + p * plane_bytes;
                    for bo in 0..bytes_per_row_blk {
                        let b = if rowmajor {
                            src[p_off + r * bytes_per_row_blk + bo]
                        } else {
                            src[p_off + bo * 16 + r]
                        };
                        for bit in 0..8 {
                            let sgn = if ((b >> (7 - bit)) & 1) != 0 { 1.0f32 } else { -1.0f32 };
                            w_blk[bo * 8 + bit] += wp * sgn;
                        }
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
        self.project_batch_cs(x_raw, k_tokens, native, threads, None, 0.0)
            .map(|(y, _)| y)
    }

    /// Precompute FP16 max-row Cauchy-Schwarz L2 block norms `[n_tiles, n_blks]` from packed tiles.
    pub fn compute_tile_norms_fp16(&self) -> Vec<u16> {
        let d_out = self.d_out;
        let d_in = self.d_in;
        let bsize = self.block_size;
        let n_blks = d_in / bsize;
        let n_tiles = d_out.div_ceil(16);
        let hdr_bytes = if self.is_asym { 64 } else { 32 };
        let bytes_per_row_blk = bsize / 8;
        let plane_bytes = bytes_per_row_blk * 16;
        let tblk_bytes = tile16_blk_bytes(bsize, self.n_bits, self.is_asym);
        let quintets = bsize.div_ceil(5);
        let is_colblock = self.is_colblock();
        let src_cow = self.packed_cow().expect("compute_tile_norms_fp16 packed read");
        let src: &[u8] = src_cow.as_ref();

        let mut out = vec![0u16; n_tiles * n_blks];
        let mut w_blk = [0.0f32; 256];
        for t in 0..n_tiles {
            for blk in 0..n_blks {
                let off = if is_colblock {
                    (blk * n_tiles + t) * tblk_bytes
                } else {
                    (t * n_blks + blk) * tblk_bytes
                };
                let mut max_n2 = 0.0f32;
                for r in 0..16 {
                    let row = t * 16 + r;
                    if row >= d_out {
                        break;
                    }
                    let sc = fp16_to_f32(u16::from_le_bytes([src[off + r * 2], src[off + r * 2 + 1]]));
                    let rowmajor = (self.n_bits & T16C_ROWMAJOR_BIT) != 0;
                    let eff_bits = self.n_bits & 0x1F;
                    if eff_bits == 5 {
                        w_blk[..bsize].fill(0.0);
                        for g in 0..quintets {
                            let code = src[off + 32 + g * 16 + r] as usize;
                            if code < 243 {
                                let t5 = TERN5_LUT[code];
                                let base = g * 5;
                                for k in 0..5 {
                                    let col = base + k;
                                    if col < bsize {
                                        w_blk[col] = sc * (t5[k] as f32);
                                    }
                                }
                            }
                        }
                    } else if eff_bits == 8 {
                        for c in 0..bsize {
                            let q = src[off + 32 + c * 16 + r] as i8;
                            w_blk[c] = sc * (q as f32);
                        }
                    } else {
                        let bi = if self.is_asym {
                            fp16_to_f32(u16::from_le_bytes([src[off + 32 + r * 2], src[off + 32 + r * 2 + 1]]))
                        } else {
                            0.0f32
                        };
                        w_blk[..bsize].fill(bi);
                        for p in 0..(eff_bits as usize) {
                            let wp = sc * plane_weight(self.n_bits, p);
                            let p_off = off + hdr_bytes + p * plane_bytes;
                            for bo in 0..bytes_per_row_blk {
                                let b = if rowmajor {
                                    src[p_off + r * bytes_per_row_blk + bo]
                                } else {
                                    src[p_off + bo * 16 + r]
                                };
                                for bit in 0..8 {
                                    let sgn = if ((b >> (7 - bit)) & 1) != 0 { 1.0f32 } else { -1.0f32 };
                                    w_blk[bo * 8 + bit] += wp * sgn;
                                }
                            }
                        }
                    }
                    let mut n2 = 0.0f32;
                    for &v in &w_blk[..bsize] {
                        n2 += v * v;
                    }
                    if n2 > max_n2 {
                        max_n2 = n2;
                    }
                }
                out[t * n_blks + blk] = fp32_to_fp16(max_n2.sqrt());
            }
        }
        out
    }

    fn prepare_x_fwht(
        &self,
        x_raw: &[f32],
        k_tokens: usize,
        native: Option<&NativeTile16Kernel>,
    ) -> Result<Option<Vec<f32>>, String> {
        prepare_x_fwht_with(
            x_raw,
            k_tokens,
            self.d_in,
            self.block_size,
            self.has_fwht,
            self.inv_s.as_deref(),
            native,
            &self.name,
        )
    }

    fn apply_low_rank(
        &self,
        x_raw: &[f32],
        k_tokens: usize,
        y: &mut [f32],
        native: Option<&NativeTile16Kernel>,
        threads: usize,
    ) -> Result<(), String> {
        if let Some(ref lr) = self.low_rank {
            if let Some(lr_fn) = native.and_then(|k| k.lowrank_add_fn) {
                let rc = unsafe {
                    lr_fn(
                        lr.u_i8.as_ptr(),
                        lr.su.as_ptr(),
                        lr.v_i8.as_ptr(),
                        lr.sv.as_ptr(),
                        x_raw.as_ptr(),
                        y.as_mut_ptr(),
                        lr.d_out as i32,
                        lr.d_in as i32,
                        lr.rank as i32,
                        k_tokens as i32,
                        threads.max(1) as i32,
                    )
                };
                if rc != 0 {
                    return Err(format!("native int8_lowrank_add_batch_c failed rc={rc} on {}", self.name));
                }
            } else {
                lr.add_batch_into(x_raw, k_tokens, y, threads);
            }
        }
        Ok(())
    }

    /// Execute tile-16 GEMV/GEMM with optional Cauchy-Schwarz block-tile pruning, returning `(y, skipped_blocks)`.
    pub fn project_batch_cs(
        &self,
        x_raw: &[f32],
        k_tokens: usize,
        native: Option<&NativeTile16Kernel>,
        threads: usize,
        tile_norms_fp16: Option<&[u16]>,
        cs_skip_thresh: f32,
    ) -> Result<(Vec<f32>, i64), String> {
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
            return Ok((Vec::new(), 0));
        }
        let can_native_colblock = !self.is_streamed()
            && self.is_colblock()
            && native.and_then(|k| k.gemm_sparse_fn).is_some()
            && (self.n_bits & 0x1F) != 8;
        if self.is_streamed() || (self.is_colblock() && !can_native_colblock) {
            let slab = self.read_sparse_slab(None, None)?;
            if tile_norms_fp16.is_some() && cs_skip_thresh > 0.0 {
                let expected_norms = d_out.div_ceil(16) * (d_in / self.block_size);
                let norms = tile_norms_fp16.unwrap();
                if norms.len() != expected_norms {
                    return Err(format!("{}: tile_norms_fp16 len {} != expected {expected_norms}", self.name, norms.len()));
                }
                let x_work = self.prepare_x_fwht(x_raw, k_tokens, native)?;
                let x_mat = x_work.as_deref().unwrap_or(x_raw);
                let nb_tile = self.n_bits & !T16C_COLBLOCK_BIT;
                let (mut y, skipped) = matmul_tilemajor_slice(
                    slab.slice(),
                    d_out,
                    d_in,
                    self.block_size,
                    nb_tile,
                    self.is_asym,
                    k_tokens,
                    x_mat,
                    native,
                    threads,
                    Some(norms),
                    cs_skip_thresh,
                    &self.name,
                )?;
                self.apply_low_rank(x_raw, k_tokens, &mut y, native, threads)?;
                return Ok((y, skipped));
            }
            return self.eval_sparse_slab(&slab, x_raw, k_tokens, native, threads, None, None, true, false);
        }

        let expected_norms = d_out.div_ceil(16) * (d_in / self.block_size);
        let active_norms = match tile_norms_fp16 {
            Some(norms) => {
                if norms.len() != expected_norms {
                    return Err(format!(
                        "{}: tile_norms_fp16 len {} != expected {expected_norms}",
                        self.name,
                        norms.len()
                    ));
                }
                if cs_skip_thresh > 0.0 {
                    Some(norms)
                } else {
                    None
                }
            }
            None => None,
        };

        let x_work = self.prepare_x_fwht(x_raw, k_tokens, native)?;
        let x_mat = x_work.as_deref().unwrap_or(x_raw);
        let (mut y, skipped) = matmul_tilemajor_slice(
            self.packed_slice(),
            d_out,
            d_in,
            self.block_size,
            self.n_bits,
            self.is_asym,
            k_tokens,
            x_mat,
            native,
            threads,
            active_norms,
            cs_skip_thresh,
            &self.name,
        )?;
        self.apply_low_rank(x_raw, k_tokens, &mut y, native, threads)?;
        Ok((y, skipped))
    }

    /// Execute tile-16 GEMV/GEMM with explicit Gate-Guided Block-Sparse row-tile and/or column-block masks.
    /// Skips inactive DRAM tiles/blocks cleanly in native NEON/AVX2 and falls back to exact masked Rust.
    #[allow(clippy::too_many_arguments)]
    pub fn project_batch_sparse(
        &self,
        x_raw: &[f32],
        k_tokens: usize,
        native: Option<&NativeTile16Kernel>,
        threads: usize,
        row_tile_mask: Option<&[u8]>,
        col_blk_mask: Option<&[u8]>,
    ) -> Result<(Vec<f32>, i64), String> {
        self.project_batch_sparse_ext(x_raw, k_tokens, native, threads, row_tile_mask, col_blk_mask, true, false)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn project_batch_sparse_ext(
        &self,
        x_raw: &[f32],
        k_tokens: usize,
        native: Option<&NativeTile16Kernel>,
        threads: usize,
        row_tile_mask: Option<&[u8]>,
        col_blk_mask: Option<&[u8]>,
        include_low_rank: bool,
        unmasked_low_rank: bool,
    ) -> Result<(Vec<f32>, i64), String> {
        if row_tile_mask.is_none() && col_blk_mask.is_none() && include_low_rank {
            return self.project_batch_cs(x_raw, k_tokens, native, threads, None, 0.0);
        }
        let can_native_colblock = !self.is_streamed()
            && self.is_colblock()
            && native.and_then(|k| k.gemm_sparse_fn).is_some()
            && (self.n_bits & 0x1F) != 8;
        if self.is_streamed() || (self.is_colblock() && !can_native_colblock) {
            let slab = self.read_sparse_slab(row_tile_mask, col_blk_mask)?;
            return self.eval_sparse_slab(
                &slab,
                x_raw,
                k_tokens,
                native,
                threads,
                row_tile_mask,
                col_blk_mask,
                include_low_rank,
                unmasked_low_rank,
            );
        }
        let d_out = self.d_out;
        let d_in = self.d_in;
        let bsize = self.block_size;
        let n_tiles = d_out.div_ceil(16);
        let n_blks = d_in / bsize;
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
            return Ok((Vec::new(), 0));
        }
        if let Some(rm) = row_tile_mask {
            if rm.len() != n_tiles {
                return Err(format!("{}: row_tile_mask len {} != n_tiles({n_tiles})", self.name, rm.len()));
            }
        }
        if let Some(cm) = col_blk_mask {
            if cm.len() != n_blks {
                return Err(format!("{}: col_blk_mask len {} != n_blks({n_blks})", self.name, cm.len()));
            }
        }

        // Mask inactive input blocks prior to low_rank and block-local inv_s + FWHT so output matches exact masked dense.
        let x_masked_owned: Option<Vec<f32>> = if let Some(cm) = col_blk_mask {
            if cm.iter().any(|&v| v == 0) {
                let mut xm = x_raw.to_vec();
                for tok in 0..k_tokens {
                    let row = &mut xm[tok * d_in..(tok + 1) * d_in];
                    for (blk, &active) in cm.iter().enumerate() {
                        if active == 0 {
                            row[blk * bsize..(blk + 1) * bsize].fill(0.0);
                        }
                    }
                }
                Some(xm)
            } else {
                None
            }
        } else {
            None
        };
        let x_eff: &[f32] = x_masked_owned.as_deref().unwrap_or(x_raw);

        let x_work = self.prepare_x_fwht(x_eff, k_tokens, native)?;
        let x_mat: &[f32] = x_work.as_deref().unwrap_or(x_eff);

        let eff_bits = self.n_bits & 0x1F;
        let mut y = vec![0f32; k_tokens * d_out];
        let mut skipped: i64 = 0;
        if let Some(sp_fn) = native.and_then(|k| k.gemm_sparse_fn) {
            if eff_bits != 8 {
                let rptr = row_tile_mask.map_or(ptr::null(), |m| m.as_ptr());
                let cptr = col_blk_mask.map_or(ptr::null(), |m| m.as_ptr());
                let rc = unsafe {
                    sp_fn(
                        self.packed_slice().as_ptr(),
                        ptr::null(),
                        x_mat.as_ptr(),
                        y.as_mut_ptr(),
                        d_out as i32,
                        d_in as i32,
                        bsize as i32,
                        self.n_bits as i32,
                        if self.is_asym { 1 } else { 0 },
                        k_tokens as i32,
                        0.0f32,
                        rptr,
                        cptr,
                        threads.max(1) as i32,
                        &mut skipped as *mut i64,
                    )
                };
                if rc != 0 {
                    return Err(format!("native gemm_bitserial_tile16_sparse_batch failed rc={rc} on {}", self.name));
                }
            } else {
                skipped = matmul_rust_sparse_slice(
                    self.packed_slice(),
                    d_out,
                    d_in,
                    bsize,
                    self.n_bits,
                    self.is_asym,
                    x_mat,
                    k_tokens,
                    &mut y,
                    threads,
                    row_tile_mask,
                    col_blk_mask,
                );
            }
        } else {
            skipped = matmul_rust_sparse_slice(
                self.packed_slice(),
                d_out,
                d_in,
                bsize,
                self.n_bits,
                self.is_asym,
                x_mat,
                k_tokens,
                &mut y,
                threads,
                row_tile_mask,
                col_blk_mask,
            );
        }

        if include_low_rank {
            let lr_x = if unmasked_low_rank { x_raw } else { x_eff };
            self.apply_low_rank(lr_x, k_tokens, &mut y, native, threads)?;
            if let Some(rm) = row_tile_mask {
                if self.low_rank.is_some() {
                    for tok in 0..k_tokens {
                        let yk = &mut y[tok * d_out..(tok + 1) * d_out];
                        for (t, &active) in rm.iter().enumerate() {
                            if active == 0 {
                                let r0 = t * 16;
                                let r1 = (r0 + 16).min(d_out);
                                yk[r0..r1].fill(0.0);
                            }
                        }
                    }
                }
            }
        }
        Ok((y, skipped))
    }

    /// Read ONLY active row tiles or column-blocks from SSD/resident storage into an aligned tile-major slab,
    /// using coalesced range reads and `POSIX_FADV_DONTNEED` to keep page cache bounded.
    pub fn read_sparse_slab(
        &self,
        row_tile_mask: Option<&[u8]>,
        col_blk_mask: Option<&[u8]>,
    ) -> Result<StreamedSparseSlab, String> {
        let d_out = self.d_out;
        let d_in = self.d_in;
        let bsize = self.block_size;
        let n_tiles = d_out.div_ceil(16);
        let n_blks = d_in / bsize;
        let tblk_bytes = tile16_blk_bytes(bsize, self.n_bits, self.is_asym);

        if let Some(rm) = row_tile_mask {
            if rm.len() != n_tiles {
                return Err(format!("{}: row_tile_mask len {} != n_tiles({n_tiles})", self.name, rm.len()));
            }
        }
        if let Some(cm) = col_blk_mask {
            if cm.len() != n_blks {
                return Err(format!("{}: col_blk_mask len {} != n_blks({n_blks})", self.name, cm.len()));
            }
        }

        let want_row_compact = !self.is_colblock() && row_tile_mask.is_some() && col_blk_mask.is_none();
        if want_row_compact {
            let rm = row_tile_mask.unwrap();
            let active_tiles: Vec<usize> = (0..n_tiles).filter(|&t| rm[t] != 0).collect();
            if active_tiles.is_empty() {
                return Ok(StreamedSparseSlab {
                    packed: Vec::new(),
                    packed_len: 0,
                    active_tiles: Some(active_tiles),
                    active_blks: None,
                    eff_d_out: 0,
                    eff_d_in: d_in,
                    is_full_fallback: false,
                    bytes_read: 0,
                });
            }
            if active_tiles.len() < n_tiles {
                let tile_row_bytes = n_blks * tblk_bytes;
                let out_len = active_tiles.len() * tile_row_bytes;
                let mut out_buf = make_aligned_buf(out_len);
                let dst = aligned_as_mut_slice(&mut out_buf, out_len);
                let mut dst_pos = 0usize;
                let mut idx_i = 0usize;
                while idx_i < active_tiles.len() {
                    let t0 = active_tiles[idx_i];
                    let mut t_end = t0 + 1;
                    while idx_i + 1 < active_tiles.len() && active_tiles[idx_i + 1] == t_end {
                        t_end += 1;
                        idx_i += 1;
                    }
                    let run_bytes = (t_end - t0) * tile_row_bytes;
                    self.read_packed_range(t0 * tile_row_bytes, &mut dst[dst_pos..dst_pos + run_bytes])?;
                    dst_pos += run_bytes;
                    idx_i += 1;
                }
                if let PackedStorage::StreamedFile { file, file_offset, len } = &self.packed {
                    fadvise_dontneed(file, *file_offset, *len);
                }
                let eff_d_out = active_tiles.len() * 16;
                return Ok(StreamedSparseSlab {
                    packed: out_buf,
                    packed_len: out_len,
                    active_tiles: Some(active_tiles),
                    active_blks: None,
                    eff_d_out,
                    eff_d_in: d_in,
                    is_full_fallback: false,
                    bytes_read: out_len as u64,
                });
            }
        }

        let want_col_compact = self.is_colblock() && row_tile_mask.is_none() && col_blk_mask.is_some();
        if want_col_compact {
            let cm = col_blk_mask.unwrap();
            let active_blks: Vec<usize> = (0..n_blks).filter(|&b| cm[b] != 0).collect();
            let k_act = active_blks.len();
            if k_act == 0 {
                return Ok(StreamedSparseSlab {
                    packed: Vec::new(),
                    packed_len: 0,
                    active_tiles: None,
                    active_blks: Some(active_blks),
                    eff_d_out: d_out,
                    eff_d_in: 0,
                    is_full_fallback: false,
                    bytes_read: 0,
                });
            }
            let blk_col_bytes = n_tiles * tblk_bytes;
            let raw_len = k_act * blk_col_bytes;
            let mut tmp_col = vec![0u8; raw_len];
            let mut tmp_pos = 0usize;
            let mut idx_i = 0usize;
            while idx_i < k_act {
                let b0 = active_blks[idx_i];
                let mut b_end = b0 + 1;
                while idx_i + 1 < k_act && active_blks[idx_i + 1] == b_end {
                    b_end += 1;
                    idx_i += 1;
                }
                let run_bytes = (b_end - b0) * blk_col_bytes;
                self.read_packed_range(b0 * blk_col_bytes, &mut tmp_col[tmp_pos..tmp_pos + run_bytes])?;
                tmp_pos += run_bytes;
                idx_i += 1;
            }
            if let PackedStorage::StreamedFile { file, file_offset, len } = &self.packed {
                fadvise_dontneed(file, *file_offset, *len);
            }
            let mut out_buf = make_aligned_buf(raw_len);
            let dst = aligned_as_mut_slice(&mut out_buf, raw_len);
            for (bi, _) in active_blks.iter().enumerate() {
                for t in 0..n_tiles {
                    let s_off = (bi * n_tiles + t) * tblk_bytes;
                    let d_off = (t * k_act + bi) * tblk_bytes;
                    dst[d_off..d_off + tblk_bytes].copy_from_slice(&tmp_col[s_off..s_off + tblk_bytes]);
                }
            }
            return Ok(StreamedSparseSlab {
                packed: out_buf,
                packed_len: raw_len,
                active_tiles: None,
                active_blks: Some(active_blks),
                eff_d_out: d_out,
                eff_d_in: k_act * bsize,
                is_full_fallback: false,
                bytes_read: raw_len as u64,
            });
        }

        // Full matrix read (dense or general mask fallback); transpose COLBLOCK to tile-major if needed.
        let full_len = self.packed_bytes_len;
        let mut out_buf = make_aligned_buf(full_len);
        {
            let dst = aligned_as_mut_slice(&mut out_buf, full_len);
            if self.is_colblock() {
                let mut tmp = vec![0u8; full_len];
                self.read_packed_range(0, &mut tmp)?;
                for blk in 0..n_blks {
                    for t in 0..n_tiles {
                        let s_off = (blk * n_tiles + t) * tblk_bytes;
                        let d_off = (t * n_blks + blk) * tblk_bytes;
                        dst[d_off..d_off + tblk_bytes].copy_from_slice(&tmp[s_off..s_off + tblk_bytes]);
                    }
                }
            } else {
                self.read_packed_range(0, dst)?;
            }
        }
        if let PackedStorage::StreamedFile { file, file_offset, len } = &self.packed {
            fadvise_dontneed(file, *file_offset, *len);
        }
        let is_dense = row_tile_mask.is_none() && col_blk_mask.is_none();
        Ok(StreamedSparseSlab {
            packed: out_buf,
            packed_len: full_len,
            active_tiles: None,
            active_blks: None,
            eff_d_out: d_out,
            eff_d_in: d_in,
            is_full_fallback: !is_dense,
            bytes_read: full_len as u64,
        })
    }

    /// Evaluate a compacted `StreamedSparseSlab` (produced by `read_sparse_slab`) in NEON or Rust.
    #[allow(clippy::too_many_arguments)]
    pub fn eval_sparse_slab(
        &self,
        slab: &StreamedSparseSlab,
        x_raw: &[f32],
        k_tokens: usize,
        native: Option<&NativeTile16Kernel>,
        threads: usize,
        row_tile_mask: Option<&[u8]>,
        col_blk_mask: Option<&[u8]>,
        include_low_rank: bool,
        unmasked_low_rank: bool,
    ) -> Result<(Vec<f32>, i64), String> {
        let d_out = self.d_out;
        let d_in = self.d_in;
        let bsize = self.block_size;
        let n_tiles = d_out.div_ceil(16);
        let n_blks = d_in / bsize;
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
            return Ok((Vec::new(), 0));
        }
        let nb_tile = self.n_bits & !T16C_COLBLOCK_BIT;

        if let Some(ref at) = slab.active_tiles {
            let skipped = ((n_tiles - at.len()) * n_blks * k_tokens.div_ceil(4).max(1)) as i64;
            let mut y = vec![0.0f32; k_tokens * d_out];
            if !at.is_empty() {
                let x_work = self.prepare_x_fwht(x_raw, k_tokens, native)?;
                let x_mat = x_work.as_deref().unwrap_or(x_raw);
                let (y_comp, _) = matmul_tilemajor_slice(
                    slab.slice(),
                    slab.eff_d_out,
                    d_in,
                    bsize,
                    nb_tile,
                    self.is_asym,
                    k_tokens,
                    x_mat,
                    native,
                    threads,
                    None,
                    0.0,
                    &self.name,
                )?;
                for tok in 0..k_tokens {
                    let yc = &y_comp[tok * slab.eff_d_out..(tok + 1) * slab.eff_d_out];
                    let yd = &mut y[tok * d_out..(tok + 1) * d_out];
                    for (ci, &tile_idx) in at.iter().enumerate() {
                        let r0 = tile_idx * 16;
                        let vr = (d_out - r0).min(16);
                        yd[r0..r0 + vr].copy_from_slice(&yc[ci * 16..ci * 16 + vr]);
                    }
                }
            }
            if include_low_rank && self.low_rank.is_some() {
                self.apply_low_rank(x_raw, k_tokens, &mut y, native, threads)?;
                if let Some(rm) = row_tile_mask {
                    for tok in 0..k_tokens {
                        let yk = &mut y[tok * d_out..(tok + 1) * d_out];
                        for (t, &active) in rm.iter().enumerate() {
                            if active == 0 {
                                let r0 = t * 16;
                                let r1 = (r0 + 16).min(d_out);
                                yk[r0..r1].fill(0.0);
                            }
                        }
                    }
                }
            }
            return Ok((y, skipped));
        }

        if let Some(ref ab) = slab.active_blks {
            let skipped = (n_tiles * (n_blks - ab.len()) * k_tokens.div_ceil(4).max(1)) as i64;
            let mut y = if ab.is_empty() {
                vec![0.0f32; k_tokens * d_out]
            } else {
                let eff_d_in = slab.eff_d_in;
                let mut x_comp = vec![0.0f32; k_tokens * eff_d_in];
                for tok in 0..k_tokens {
                    let xs = &x_raw[tok * d_in..(tok + 1) * d_in];
                    let xd = &mut x_comp[tok * eff_d_in..(tok + 1) * eff_d_in];
                    for (ci, &blk) in ab.iter().enumerate() {
                        xd[ci * bsize..(ci + 1) * bsize].copy_from_slice(&xs[blk * bsize..(blk + 1) * bsize]);
                    }
                }
                let inv_comp = self.inv_s.as_ref().map(|inv| {
                    let mut out = vec![0.0f32; eff_d_in];
                    for (ci, &blk) in ab.iter().enumerate() {
                        out[ci * bsize..(ci + 1) * bsize].copy_from_slice(&inv[blk * bsize..(blk + 1) * bsize]);
                    }
                    out
                });
                let x_work = prepare_x_fwht_with(
                    &x_comp,
                    k_tokens,
                    eff_d_in,
                    bsize,
                    self.has_fwht,
                    inv_comp.as_deref(),
                    native,
                    &self.name,
                )?;
                let x_mat = x_work.as_deref().unwrap_or(&x_comp);
                let (y_out, _) = matmul_tilemajor_slice(
                    slab.slice(),
                    d_out,
                    eff_d_in,
                    bsize,
                    nb_tile,
                    self.is_asym,
                    k_tokens,
                    x_mat,
                    native,
                    threads,
                    None,
                    0.0,
                    &self.name,
                )?;
                y_out
            };
            if include_low_rank && self.low_rank.is_some() {
                if unmasked_low_rank || ab.len() == n_blks {
                    self.apply_low_rank(x_raw, k_tokens, &mut y, native, threads)?;
                } else {
                    let mut x_z = vec![0.0f32; k_tokens * d_in];
                    for tok in 0..k_tokens {
                        let xs = &x_raw[tok * d_in..(tok + 1) * d_in];
                        let xd = &mut x_z[tok * d_in..(tok + 1) * d_in];
                        for &blk in ab {
                            xd[blk * bsize..(blk + 1) * bsize].copy_from_slice(&xs[blk * bsize..(blk + 1) * bsize]);
                        }
                    }
                    self.apply_low_rank(&x_z, k_tokens, &mut y, native, threads)?;
                }
            }
            return Ok((y, skipped));
        }

        if !slab.is_full_fallback {
            let x_work = self.prepare_x_fwht(x_raw, k_tokens, native)?;
            let x_mat = x_work.as_deref().unwrap_or(x_raw);
            let (mut y, skipped) = matmul_tilemajor_slice(
                slab.slice(),
                d_out,
                d_in,
                bsize,
                nb_tile,
                self.is_asym,
                k_tokens,
                x_mat,
                native,
                threads,
                None,
                0.0,
                &self.name,
            )?;
            if include_low_rank {
                self.apply_low_rank(x_raw, k_tokens, &mut y, native, threads)?;
            }
            return Ok((y, skipped));
        }

        // General masked fallback on full tile-major slab buffer
        let x_masked_owned: Option<Vec<f32>> = if let Some(cm) = col_blk_mask {
            if cm.iter().any(|&v| v == 0) {
                let mut xm = x_raw.to_vec();
                for tok in 0..k_tokens {
                    let row = &mut xm[tok * d_in..(tok + 1) * d_in];
                    for (blk, &active) in cm.iter().enumerate() {
                        if active == 0 {
                            row[blk * bsize..(blk + 1) * bsize].fill(0.0);
                        }
                    }
                }
                Some(xm)
            } else {
                None
            }
        } else {
            None
        };
        let x_eff = x_masked_owned.as_deref().unwrap_or(x_raw);
        let x_work = self.prepare_x_fwht(x_eff, k_tokens, native)?;
        let x_mat = x_work.as_deref().unwrap_or(x_eff);
        let eff_bits = nb_tile & 0x1F;
        let mut y = vec![0.0f32; k_tokens * d_out];
        let mut skipped: i64 = 0;
        if let Some(sp_fn) = native.and_then(|k| k.gemm_sparse_fn) {
            if eff_bits != 8 {
                let rptr = row_tile_mask.map_or(ptr::null(), |m| m.as_ptr());
                let cptr = col_blk_mask.map_or(ptr::null(), |m| m.as_ptr());
                let rc = unsafe {
                    sp_fn(
                        slab.slice().as_ptr(),
                        ptr::null(),
                        x_mat.as_ptr(),
                        y.as_mut_ptr(),
                        d_out as i32,
                        d_in as i32,
                        bsize as i32,
                        nb_tile as i32,
                        if self.is_asym { 1 } else { 0 },
                        k_tokens as i32,
                        0.0f32,
                        rptr,
                        cptr,
                        threads.max(1) as i32,
                        &mut skipped as *mut i64,
                    )
                };
                if rc != 0 {
                    return Err(format!("native gemm_bitserial_tile16_sparse_batch failed rc={rc} on {}", self.name));
                }
            } else {
                skipped = matmul_rust_sparse_slice(
                    slab.slice(),
                    d_out,
                    d_in,
                    bsize,
                    nb_tile,
                    self.is_asym,
                    x_mat,
                    k_tokens,
                    &mut y,
                    threads,
                    row_tile_mask,
                    col_blk_mask,
                );
            }
        } else {
            skipped = matmul_rust_sparse_slice(
                slab.slice(),
                d_out,
                d_in,
                bsize,
                nb_tile,
                self.is_asym,
                x_mat,
                k_tokens,
                &mut y,
                threads,
                row_tile_mask,
                col_blk_mask,
            );
        }
        if include_low_rank {
            let lr_x = if unmasked_low_rank { x_raw } else { x_eff };
            self.apply_low_rank(lr_x, k_tokens, &mut y, native, threads)?;
            if let Some(rm) = row_tile_mask {
                if self.low_rank.is_some() {
                    for tok in 0..k_tokens {
                        let yk = &mut y[tok * d_out..(tok + 1) * d_out];
                        for (t, &active) in rm.iter().enumerate() {
                            if active == 0 {
                                let r0 = t * 16;
                                let r1 = (r0 + 16).min(d_out);
                                yk[r0..r1].fill(0.0);
                            }
                        }
                    }
                }
            }
        }
        Ok((y, skipped))
    }

    /// Execute two-stage block-ordered Cauchy-Schwarz / sieve LM-head projection on resident `.t16c` tiles.
    #[allow(clippy::too_many_arguments)]
    pub fn project_lmhead_cs(
        &self,
        x_raw: &[f32],
        k_tokens: usize,
        native: Option<&NativeTile16Kernel>,
        threads: usize,
        tile_norms_fp16: Option<&[u16]>,
        top_k_keep: usize,
        sieve_frac: f32,
    ) -> Result<(Vec<f32>, i64), String> {
        self.project_lmhead_cs_seeded(
            x_raw,
            k_tokens,
            native,
            threads,
            tile_norms_fp16,
            top_k_keep,
            sieve_frac,
            None,
        )
    }

    /// Execute Draft-Seeded two-stage block-ordered Cauchy-Schwarz LM-head projection (`seed_row_ids` locks exact draft threshold).
    #[allow(clippy::too_many_arguments)]
    pub fn project_lmhead_cs_seeded(
        &self,
        x_raw: &[f32],
        k_tokens: usize,
        native: Option<&NativeTile16Kernel>,
        threads: usize,
        tile_norms_fp16: Option<&[u16]>,
        top_k_keep: usize,
        sieve_frac: f32,
        seed_row_ids: Option<&[i32]>,
    ) -> Result<(Vec<f32>, i64), String> {
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
        if let Some(sr) = seed_row_ids {
            if sr.len() != k_tokens {
                return Err(format!("{}: seed_row_ids len {} != k_tokens({k_tokens})", self.name, sr.len()));
            }
        }
        if k_tokens == 0 {
            return Ok((Vec::new(), 0));
        }
        let expected_norms = d_out.div_ceil(16) * (d_in / self.block_size);
        let seeded_fn = native.and_then(|k| k.lmhead_cs_seeded_fn);
        let fallback_fn = native.and_then(|k| k.lmhead_cs_fn);
        if let (Some(norms), true) = (tile_norms_fp16, seeded_fn.is_some() || fallback_fn.is_some()) {
            if norms.len() != expected_norms {
                return Err(format!(
                    "{}: tile_norms_fp16 len {} != expected {expected_norms}",
                    self.name,
                    norms.len()
                ));
            }
            let mut x_work = self.prepare_x_fwht(x_raw, k_tokens, native)?;
            let x_mat: &[f32] = match x_work.as_mut() {
                Some(v) => v.as_slice(),
                None => x_raw,
            };
            let mut y = vec![0f32; k_tokens * d_out];
            let mut skipped: i64 = 0;
            let rc = if let Some(s_fn) = seeded_fn {
                let s_ptr = seed_row_ids.map_or(ptr::null(), |s| s.as_ptr());
                unsafe {
                    s_fn(
                        self.packed_slice().as_ptr(),
                        norms.as_ptr(),
                        x_mat.as_ptr(),
                        y.as_mut_ptr(),
                        d_out as i32,
                        d_in as i32,
                        self.block_size as i32,
                        self.n_bits as i32,
                        if self.is_asym { 1 } else { 0 },
                        k_tokens as i32,
                        top_k_keep.clamp(1, 32) as i32,
                        sieve_frac.max(0.0),
                        s_ptr,
                        threads.max(1) as i32,
                        &mut skipped as *mut i64,
                    )
                }
            } else {
                let cs_fn = fallback_fn.unwrap();
                unsafe {
                    cs_fn(
                        self.packed_slice().as_ptr(),
                        norms.as_ptr(),
                        x_mat.as_ptr(),
                        y.as_mut_ptr(),
                        d_out as i32,
                        d_in as i32,
                        self.block_size as i32,
                        self.n_bits as i32,
                        if self.is_asym { 1 } else { 0 },
                        k_tokens as i32,
                        top_k_keep.clamp(1, 32) as i32,
                        sieve_frac.max(0.0),
                        threads.max(1) as i32,
                        &mut skipped as *mut i64,
                    )
                }
            };
            if rc != 0 {
                return Err(format!("native lmhead_cs_seeded_tile16_batch failed rc={rc} on {}", self.name));
            }
            self.apply_low_rank(x_raw, k_tokens, &mut y, native, threads)?;
            Ok((y, skipped))
        } else {
            self.project_batch(x_raw, k_tokens, native, threads).map(|y| (y, 0))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_x_fwht_with(
    x_raw: &[f32],
    k_tokens: usize,
    d_in: usize,
    block_size: usize,
    has_fwht: bool,
    inv_s: Option<&[f32]>,
    native: Option<&NativeTile16Kernel>,
    name: &str,
) -> Result<Option<Vec<f32>>, String> {
    if inv_s.is_none() && !has_fwht {
        return Ok(None);
    }
    let mut buf = x_raw.to_vec();
    let used_native_prep = if let Some(inv_fn) = native.and_then(|k| k.inv_s_fwht_fn) {
        let inv_ptr = inv_s.map_or(ptr::null(), |v| v.as_ptr());
        let bsz = if has_fwht { block_size as i32 } else { 1i32 };
        let rc = unsafe { inv_fn(buf.as_mut_ptr(), inv_ptr, k_tokens as i32, d_in as i32, bsz) };
        if rc != 0 {
            return Err(format!("native inv_s_and_fwht_f32_c failed rc={rc} on {name}"));
        }
        true
    } else {
        false
    };
    if !used_native_prep {
        if let Some(inv) = inv_s {
            for tok in 0..k_tokens {
                let row = &mut buf[tok * d_in..(tok + 1) * d_in];
                for j in 0..d_in {
                    row[j] *= inv[j];
                }
            }
        }
        if has_fwht {
            fwht_block_inplace(&mut buf, block_size)?;
        }
    }
    Ok(Some(buf))
}

#[allow(clippy::too_many_arguments)]
fn matmul_tilemajor_slice(
    packed_slice: &[u8],
    d_out: usize,
    d_in: usize,
    block_size: usize,
    n_bits_tile: u8,
    is_asym: bool,
    k_tokens: usize,
    x_mat: &[f32],
    native: Option<&NativeTile16Kernel>,
    threads: usize,
    tile_norms_fp16: Option<&[u16]>,
    cs_skip_thresh: f32,
    name: &str,
) -> Result<(Vec<f32>, i64), String> {
    let mut y = vec![0f32; k_tokens * d_out];
    let mut skipped: i64 = 0;
    if let Some(kern) = native {
        let (norm_ptr, eff_thresh) = match tile_norms_fp16 {
            Some(n) if cs_skip_thresh > 0.0 => (n.as_ptr(), cs_skip_thresh),
            _ => (ptr::null(), 0.0f32),
        };
        let rc = unsafe {
            (kern.gemm_fn)(
                packed_slice.as_ptr(),
                norm_ptr,
                x_mat.as_ptr(),
                y.as_mut_ptr(),
                d_out as i32,
                d_in as i32,
                block_size as i32,
                n_bits_tile as i32,
                if is_asym { 1 } else { 0 },
                k_tokens as i32,
                eff_thresh,
                threads.max(1) as i32,
                &mut skipped as *mut i64,
            )
        };
        if rc != 0 {
            return Err(format!("native gemm_bitserial_tile16_batch failed rc={rc} on {name}"));
        }
    } else {
        skipped = matmul_rust_slice(
            packed_slice,
            d_out,
            d_in,
            block_size,
            n_bits_tile,
            is_asym,
            x_mat,
            k_tokens,
            &mut y,
            threads,
            tile_norms_fp16,
            cs_skip_thresh,
        );
    }
    Ok((y, skipped))
}

#[allow(clippy::too_many_arguments)]
fn matmul_rust_slice(
    packed: &[u8],
    d_out: usize,
    d_in: usize,
    bsize: usize,
    n_bits_raw: u8,
    is_asym: bool,
    x_mat: &[f32],
    k_tokens: usize,
    y: &mut [f32],
    threads: usize,
    tile_norms_fp16: Option<&[u16]>,
    cs_skip_thresh: f32,
) -> i64 {
    let n_blks = d_in / bsize;
    let n_tiles = d_out.div_ceil(16);
    let hdr_bytes = if is_asym { 64 } else { 32 };
    let bytes_per_row_blk = bsize / 8;
    let plane_bytes = bytes_per_row_blk * 16;
    let tblk_bytes = tile16_blk_bytes(bsize, n_bits_raw, is_asym);
    let quintets = bsize.div_ceil(5);
    let is_colblock = (n_bits_raw & T16C_COLBLOCK_BIT) != 0;
    let rowmajor = (n_bits_raw & T16C_ROWMAJOR_BIT) != 0;
    let eff_bits = (n_bits_raw & 0x1F) as usize;
    let total_skipped = std::sync::atomic::AtomicI64::new(0);

    let nthreads = threads.max(1).min(d_out);
    let chunk = d_out.div_ceil(nthreads).max(1);
    for tok in 0..k_tokens {
        let xk = &x_mat[tok * d_in..(tok + 1) * d_in];
        let x_blk_norms: Vec<f32> = if tile_norms_fp16.is_some() && cs_skip_thresh > 0.0 {
            (0..n_blks)
                .map(|blk| {
                    let xb = &xk[blk * bsize..(blk + 1) * bsize];
                    xb.iter().map(|&v| v * v).sum::<f32>().sqrt()
                })
                .collect()
        } else {
            Vec::new()
        };
        let yk = &mut y[tok * d_out..(tok + 1) * d_out];
        std::thread::scope(|s| {
            for (bi, yslice) in yk.chunks_mut(chunk).enumerate() {
                let r_start = bi * chunk;
                let xbn = &x_blk_norms;
                let skip_ref = &total_skipped;
                s.spawn(move || {
                    for (dr, yv) in yslice.iter_mut().enumerate() {
                        let row = r_start + dr;
                        let t = row / 16;
                        let r = row % 16;
                        let mut acc = 0.0f64;
                        for blk in 0..n_blks {
                            if let Some(norms) = tile_norms_fp16 {
                                let wn = fp16_to_f32(norms[t * n_blks + blk]);
                                if wn * xbn[blk] <= cs_skip_thresh {
                                    if r == 0 {
                                        skip_ref.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    }
                                    continue;
                                }
                            }
                            let off = if is_colblock {
                                (blk * n_tiles + t) * tblk_bytes
                            } else {
                                (t * n_blks + blk) * tblk_bytes
                            };
                            let sc = fp16_to_f32(u16::from_le_bytes([packed[off + r * 2], packed[off + r * 2 + 1]])) as f64;
                            let xb = &xk[blk * bsize..(blk + 1) * bsize];
                            if eff_bits == 5 {
                                let mut bsum = 0.0f64;
                                for g in 0..quintets {
                                    let code = packed[off + 32 + g * 16 + r] as usize;
                                    debug_assert!(code < 243);
                                    let t5 = TERN5_LUT.get(code).copied().unwrap_or([0; 5]);
                                    let base = g * 5;
                                    for k in 0..5 {
                                        let col = base + k;
                                        if col < bsize && t5[k] != 0 {
                                            bsum += (t5[k] as f64) * (xb[col] as f64);
                                        }
                                    }
                                }
                                acc += sc * bsum;
                            } else if eff_bits == 8 {
                                let mut wsum = 0.0f64;
                                for c in 0..bsize {
                                    let q = packed[off + 32 + c * 16 + r] as i8;
                                    wsum += (q as f64) * (xb[c] as f64);
                                }
                                acc += sc * wsum;
                            } else {
                                let bi_v = if is_asym {
                                    fp16_to_f32(u16::from_le_bytes([packed[off + 32 + r * 2], packed[off + 32 + r * 2 + 1]])) as f64
                                } else {
                                    0.0f64
                                };
                                if is_asym {
                                    let sx: f64 = xb.iter().map(|&v| v as f64).sum();
                                    acc += bi_v * sx;
                                }
                                for p in 0..eff_bits {
                                    let wp = sc * (plane_weight(n_bits_raw, p) as f64);
                                    let p_off = off + hdr_bytes + p * plane_bytes;
                                    let mut isum = 0.0f64;
                                    for bo in 0..bytes_per_row_blk {
                                        let b = if rowmajor {
                                            packed[p_off + r * bytes_per_row_blk + bo]
                                        } else {
                                            packed[p_off + bo * 16 + r]
                                        };
                                        let base = bo * 8;
                                        for bit in 0..8 {
                                            let v = xb[base + bit] as f64;
                                            isum += if ((b >> (7 - bit)) & 1) != 0 { v } else { -v };
                                        }
                                    }
                                    acc += wp * isum;
                                }
                            }
                        }
                        *yv = acc as f32;
                    }
                });
            }
        });
    }
    total_skipped.load(std::sync::atomic::Ordering::Relaxed)
}

#[allow(clippy::too_many_arguments)]
fn matmul_rust_sparse_slice(
    packed: &[u8],
    d_out: usize,
    d_in: usize,
    bsize: usize,
    n_bits_raw: u8,
    is_asym: bool,
    x_mat: &[f32],
    k_tokens: usize,
    y: &mut [f32],
    threads: usize,
    row_tile_mask: Option<&[u8]>,
    col_blk_mask: Option<&[u8]>,
) -> i64 {
    let n_blks = d_in / bsize;
    let n_tiles = d_out.div_ceil(16);
    let hdr_bytes = if is_asym { 64 } else { 32 };
    let bytes_per_row_blk = bsize / 8;
    let plane_bytes = bytes_per_row_blk * 16;
    let tblk_bytes = tile16_blk_bytes(bsize, n_bits_raw, is_asym);
    let quintets = bsize.div_ceil(5);
    let is_colblock = (n_bits_raw & T16C_COLBLOCK_BIT) != 0;
    let rowmajor = (n_bits_raw & T16C_ROWMAJOR_BIT) != 0;
    let eff_bits = (n_bits_raw & 0x1F) as usize;
    let active_blks: Vec<usize> = match col_blk_mask {
        Some(cm) => (0..n_blks).filter(|&b| cm[b] != 0).collect(),
        None => (0..n_blks).collect(),
    };
    let active_tiles = match row_tile_mask {
        Some(rm) => rm.iter().filter(|&&v| v != 0).count(),
        None => n_tiles,
    };
    let skipped_per_tok = ((n_tiles - active_tiles) * n_blks + active_tiles * (n_blks - active_blks.len())) as i64;

    let nthreads = threads.max(1).min(d_out);
    let chunk = d_out.div_ceil(nthreads).max(1);
    for tok in 0..k_tokens {
        let xk = &x_mat[tok * d_in..(tok + 1) * d_in];
        let yk = &mut y[tok * d_out..(tok + 1) * d_out];
        std::thread::scope(|s| {
            for (bi, yslice) in yk.chunks_mut(chunk).enumerate() {
                let r_start = bi * chunk;
                let ablks = &active_blks;
                s.spawn(move || {
                    for (dr, yv) in yslice.iter_mut().enumerate() {
                        let row = r_start + dr;
                        let t = row / 16;
                        if let Some(rm) = row_tile_mask {
                            if rm[t] == 0 {
                                *yv = 0.0;
                                continue;
                            }
                        }
                        let r = row % 16;
                        let mut acc = 0.0f64;
                        for &blk in ablks {
                            let off = if is_colblock {
                                (blk * n_tiles + t) * tblk_bytes
                            } else {
                                (t * n_blks + blk) * tblk_bytes
                            };
                            let sc = fp16_to_f32(u16::from_le_bytes([packed[off + r * 2], packed[off + r * 2 + 1]])) as f64;
                            let xb = &xk[blk * bsize..(blk + 1) * bsize];
                            if eff_bits == 5 {
                                let mut bsum = 0.0f64;
                                for g in 0..quintets {
                                    let code = packed[off + 32 + g * 16 + r] as usize;
                                    let t5 = TERN5_LUT.get(code).copied().unwrap_or([0; 5]);
                                    let base = g * 5;
                                    for k in 0..5 {
                                        let col = base + k;
                                        if col < bsize && t5[k] != 0 {
                                            bsum += (t5[k] as f64) * (xb[col] as f64);
                                        }
                                    }
                                }
                                acc += sc * bsum;
                            } else if eff_bits == 8 {
                                let mut wsum = 0.0f64;
                                for c in 0..bsize {
                                    let q = packed[off + 32 + c * 16 + r] as i8;
                                    wsum += (q as f64) * (xb[c] as f64);
                                }
                                acc += sc * wsum;
                            } else {
                                let bi_v = if is_asym {
                                    fp16_to_f32(u16::from_le_bytes([packed[off + 32 + r * 2], packed[off + 32 + r * 2 + 1]])) as f64
                                } else {
                                    0.0f64
                                };
                                if is_asym {
                                    let sx: f64 = xb.iter().map(|&v| v as f64).sum();
                                    acc += bi_v * sx;
                                }
                                for p in 0..eff_bits {
                                    let wp = sc * (plane_weight(n_bits_raw, p) as f64);
                                    let p_off = off + hdr_bytes + p * plane_bytes;
                                    let mut isum = 0.0f64;
                                    for bo in 0..bytes_per_row_blk {
                                        let b = if rowmajor {
                                            packed[p_off + r * bytes_per_row_blk + bo]
                                        } else {
                                            packed[p_off + bo * 16 + r]
                                        };
                                        let base = bo * 8;
                                        for bit in 0..8 {
                                            let v = xb[base + bit] as f64;
                                            isum += if ((b >> (7 - bit)) & 1) != 0 { v } else { -v };
                                        }
                                    }
                                    acc += wp * isum;
                                }
                            }
                        }
                        *yv = acc as f32;
                    }
                });
            }
        });
    }
    skipped_per_tok * (k_tokens.div_ceil(4).max(1) as i64)
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

pub type GemmTile16SparseBatchFn = unsafe extern "C" fn(
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
    row_tile_mask: *const u8,
    col_blk_mask: *const u8,
    num_threads: i32,
    out_skipped: *mut i64,
) -> i32;

pub type InvSFwhtF32Fn = unsafe extern "C" fn(
    x: *mut f32,
    inv_s: *const f32,
    k_tokens: i32,
    d_in: i32,
    bsize: i32,
) -> i32;

pub type Int8LowRankAddBatchFn = unsafe extern "C" fn(
    u_i8: *const i8,
    su: *const f32,
    v_i8: *const i8,
    sv: *const f32,
    x_raw: *const f32,
    y: *mut f32,
    d_out: i32,
    d_in: i32,
    rank: i32,
    k_tokens: i32,
    num_threads: i32,
) -> i32;

pub type LmHeadCsTile16BatchFn = unsafe extern "C" fn(
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
    top_k_keep: i32,
    sieve_frac: f32,
    num_threads: i32,
    out_skipped_blks: *mut i64,
) -> i32;

pub type LmHeadCsSeededTile16BatchFn = unsafe extern "C" fn(
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
    top_k_keep: i32,
    sieve_frac: f32,
    seed_row_ids: *const i32,
    num_threads: i32,
    out_skipped_blks: *mut i64,
) -> i32;

pub type EncodeKvGlyphFastFn = unsafe extern "C" fn(
    kv_in: *const f32,
    cb: *const f32,
    lut_6561: *const u16,
    glyphs_out: *mut u8,
    n_glyphs: i32,
    codec_id: i32,
    bytes_per_glyph: i32,
    apply_fwht8: i32,
    num_threads: i32,
);

pub type FlashGlyphAttentionGqaFn = unsafe extern "C" fn(
    q: *const f32,
    k_glyphs: *const u8,
    v_glyphs: *const u8,
    codebook_s7: *const f32,
    out_context: *mut f32,
    out_max_score: *mut f32,
    out_sum_exp: *mut f32,
    n_q_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    seq_len: i32,
    kv_seq_stride_bytes: usize,
    kv_head_stride_bytes: usize,
    attn_scale: f32,
    codec_id: i32,
    bytes_per_glyph: i32,
    apply_fwht8: i32,
    k_eq_v: i32,
    num_threads: i32,
) -> i32;

pub type SetAndVcntAbitsFn = unsafe extern "C" fn(a_bits: i32) -> i32;

pub type NeonGemvI8ScaleBiasFn = unsafe extern "C" fn(
    w_i8: *const i8,
    scales: *const f32,
    biases: *const f32,
    x_f32: *const f32,
    y_out: *mut f32,
    d_out: i32,
    d_in: i32,
    do_tanh: i32,
    num_threads: i32,
) -> i32;

pub struct NativeTile16Kernel {
    handle: usize,
    pub path: String,
    pub gemm_fn: GemmTile16BatchFn,
    pub gemm_sparse_fn: Option<GemmTile16SparseBatchFn>,
    pub inv_s_fwht_fn: Option<InvSFwhtF32Fn>,
    pub lowrank_add_fn: Option<Int8LowRankAddBatchFn>,
    pub lmhead_cs_fn: Option<LmHeadCsTile16BatchFn>,
    pub lmhead_cs_seeded_fn: Option<LmHeadCsSeededTile16BatchFn>,
    pub encode_kv_glyph_fn: Option<EncodeKvGlyphFastFn>,
    pub flash_glyph_attn_fn: Option<FlashGlyphAttentionGqaFn>,
    pub set_andvcnt_abits_fn: Option<SetAndVcntAbitsFn>,
    pub gemv_i8_scale_bias_fn: Option<NeonGemvI8ScaleBiasFn>,
}

unsafe impl Send for NativeTile16Kernel {}
unsafe impl Sync for NativeTile16Kernel {}

#[cfg(unix)]
mod sys {
    use std::ffi::c_void;
    pub const RTLD_NOW: i32 = 2;
    pub const PROT_READ: i32 = 1;
    pub const MAP_SHARED: i32 = 1;
    pub const MADV_WILLNEED: i32 = 3;
    pub const MADV_DONTNEED: i32 = 4;
    #[cfg(target_os = "linux")]
    pub const POSIX_FADV_DONTNEED: i32 = 4;
    pub const MAP_FAILED: *mut c_void = !0usize as *mut c_void;
    extern "C" {
        pub fn dlopen(filename: *const std::ffi::c_char, flag: i32) -> *mut c_void;
        pub fn dlsym(handle: *mut c_void, symbol: *const std::ffi::c_char) -> *mut c_void;
        pub fn dlclose(handle: *mut c_void) -> i32;
        pub fn mmap(
            addr: *mut c_void,
            length: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, length: usize) -> i32;
        pub fn madvise(addr: *mut c_void, length: usize, advice: i32) -> i32;
        #[cfg(target_os = "linux")]
        pub fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
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
            let h = sys::dlopen(c_path.as_ptr() as *const std::ffi::c_char, sys::RTLD_NOW);
            if h.is_null() {
                return Err(format!("dlopen failed for {}", path.display()));
            }
            let sym = sys::dlsym(h, b"gemm_bitserial_tile16_batch\0".as_ptr() as *const std::ffi::c_char);
            if sym.is_null() {
                sys::dlclose(h);
                return Err(format!("missing gemm_bitserial_tile16_batch in {}", path.display()));
            }
            let gemm_fn: GemmTile16BatchFn = std::mem::transmute(sym);
            let sp_sym = sys::dlsym(h, b"gemm_bitserial_tile16_sparse_batch\0".as_ptr() as *const std::ffi::c_char);
            let gemm_sparse_fn: Option<GemmTile16SparseBatchFn> = if sp_sym.is_null() {
                None
            } else {
                Some(std::mem::transmute::<*mut std::ffi::c_void, GemmTile16SparseBatchFn>(sp_sym))
            };
            let inv_sym = sys::dlsym(h, b"inv_s_and_fwht_f32_c\0".as_ptr() as *const std::ffi::c_char);
            let inv_s_fwht_fn: Option<InvSFwhtF32Fn> = if inv_sym.is_null() {
                None
            } else {
                Some(std::mem::transmute::<*mut std::ffi::c_void, InvSFwhtF32Fn>(inv_sym))
            };
            let lr_sym = sys::dlsym(h, b"int8_lowrank_add_batch_c\0".as_ptr() as *const std::ffi::c_char);
            let lowrank_add_fn: Option<Int8LowRankAddBatchFn> = if lr_sym.is_null() {
                None
            } else {
                Some(std::mem::transmute::<*mut std::ffi::c_void, Int8LowRankAddBatchFn>(lr_sym))
            };
            let lm_sym = sys::dlsym(h, b"lmhead_cs_top1_tile16_batch\0".as_ptr() as *const std::ffi::c_char);
            let lmhead_cs_fn: Option<LmHeadCsTile16BatchFn> = if lm_sym.is_null() {
                None
            } else {
                Some(std::mem::transmute::<*mut std::ffi::c_void, LmHeadCsTile16BatchFn>(lm_sym))
            };
            let lms_sym = sys::dlsym(h, b"lmhead_cs_seeded_tile16_batch\0".as_ptr() as *const std::ffi::c_char);
            let lmhead_cs_seeded_fn: Option<LmHeadCsSeededTile16BatchFn> = if lms_sym.is_null() {
                None
            } else {
                Some(std::mem::transmute::<*mut std::ffi::c_void, LmHeadCsSeededTile16BatchFn>(lms_sym))
            };
            let ekv_sym = sys::dlsym(h, b"encode_kv_glyph_fast_c\0".as_ptr() as *const std::ffi::c_char);
            let encode_kv_glyph_fn: Option<EncodeKvGlyphFastFn> = if ekv_sym.is_null() {
                None
            } else {
                Some(std::mem::transmute::<*mut std::ffi::c_void, EncodeKvGlyphFastFn>(ekv_sym))
            };
            let fg_sym = sys::dlsym(h, b"flash_glyph_attention_gqa_c\0".as_ptr() as *const std::ffi::c_char);
            let flash_glyph_attn_fn: Option<FlashGlyphAttentionGqaFn> = if fg_sym.is_null() {
                None
            } else {
                Some(std::mem::transmute::<*mut std::ffi::c_void, FlashGlyphAttentionGqaFn>(fg_sym))
            };
            let av_sym = sys::dlsym(h, b"neon_set_andvcnt_abits\0".as_ptr() as *const std::ffi::c_char);
            let set_andvcnt_abits_fn: Option<SetAndVcntAbitsFn> = if av_sym.is_null() {
                None
            } else {
                Some(std::mem::transmute::<*mut std::ffi::c_void, SetAndVcntAbitsFn>(av_sym))
            };
            let gb_sym = sys::dlsym(h, b"neon_gemv_i8_scale_bias_c\0".as_ptr() as *const std::ffi::c_char);
            let gemv_i8_scale_bias_fn: Option<NeonGemvI8ScaleBiasFn> = if gb_sym.is_null() {
                None
            } else {
                Some(std::mem::transmute::<*mut std::ffi::c_void, NeonGemvI8ScaleBiasFn>(gb_sym))
            };
            Ok(Self {
                handle: h as usize,
                path: s.to_string(),
                gemm_fn,
                gemm_sparse_fn,
                inv_s_fwht_fn,
                lowrank_add_fn,
                lmhead_cs_fn,
                lmhead_cs_seeded_fn,
                encode_kv_glyph_fn,
                flash_glyph_attn_fn,
                set_andvcnt_abits_fn,
                gemv_i8_scale_bias_fn,
            })
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
            let sp_sym = sys::GetProcAddress(h, b"gemm_bitserial_tile16_sparse_batch\0".as_ptr() as *const i8);
            let gemm_sparse_fn: Option<GemmTile16SparseBatchFn> = if sp_sym.is_null() {
                None
            } else {
                Some(std::mem::transmute::<*mut std::ffi::c_void, GemmTile16SparseBatchFn>(sp_sym))
            };
            let inv_sym = sys::GetProcAddress(h, b"inv_s_and_fwht_f32_c\0".as_ptr() as *const i8);
            let inv_s_fwht_fn: Option<InvSFwhtF32Fn> = if inv_sym.is_null() {
                None
            } else {
                Some(std::mem::transmute::<*mut std::ffi::c_void, InvSFwhtF32Fn>(inv_sym))
            };
            let lr_sym = sys::GetProcAddress(h, b"int8_lowrank_add_batch_c\0".as_ptr() as *const i8);
            let lowrank_add_fn: Option<Int8LowRankAddBatchFn> = if lr_sym.is_null() {
                None
            } else {
                Some(std::mem::transmute::<*mut std::ffi::c_void, Int8LowRankAddBatchFn>(lr_sym))
            };
            let lm_sym = sys::GetProcAddress(h, b"lmhead_cs_top1_tile16_batch\0".as_ptr() as *const i8);
            let lmhead_cs_fn: Option<LmHeadCsTile16BatchFn> = if lm_sym.is_null() {
                None
            } else {
                Some(std::mem::transmute::<*mut std::ffi::c_void, LmHeadCsTile16BatchFn>(lm_sym))
            };
            let lms_sym = sys::GetProcAddress(h, b"lmhead_cs_seeded_tile16_batch\0".as_ptr() as *const i8);
            let lmhead_cs_seeded_fn: Option<LmHeadCsSeededTile16BatchFn> = if lms_sym.is_null() {
                None
            } else {
                Some(std::mem::transmute::<*mut std::ffi::c_void, LmHeadCsSeededTile16BatchFn>(lms_sym))
            };
            let ekv_sym = sys::GetProcAddress(h, b"encode_kv_glyph_fast_c\0".as_ptr() as *const i8);
            let encode_kv_glyph_fn: Option<EncodeKvGlyphFastFn> = if ekv_sym.is_null() {
                None
            } else {
                Some(std::mem::transmute::<*mut std::ffi::c_void, EncodeKvGlyphFastFn>(ekv_sym))
            };
            let fg_sym = sys::GetProcAddress(h, b"flash_glyph_attention_gqa_c\0".as_ptr() as *const i8);
            let flash_glyph_attn_fn: Option<FlashGlyphAttentionGqaFn> = if fg_sym.is_null() {
                None
            } else {
                Some(std::mem::transmute::<*mut std::ffi::c_void, FlashGlyphAttentionGqaFn>(fg_sym))
            };
            let av_sym = sys::GetProcAddress(h, b"neon_set_andvcnt_abits\0".as_ptr() as *const i8);
            let set_andvcnt_abits_fn: Option<SetAndVcntAbitsFn> = if av_sym.is_null() {
                None
            } else {
                Some(std::mem::transmute::<*mut std::ffi::c_void, SetAndVcntAbitsFn>(av_sym))
            };
            let gb_sym = sys::GetProcAddress(h, b"neon_gemv_i8_scale_bias_c\0".as_ptr() as *const i8);
            let gemv_i8_scale_bias_fn: Option<NeonGemvI8ScaleBiasFn> = if gb_sym.is_null() {
                None
            } else {
                Some(std::mem::transmute::<*mut std::ffi::c_void, NeonGemvI8ScaleBiasFn>(gb_sym))
            };
            Ok(Self {
                handle: h as usize,
                path: path.display().to_string(),
                gemm_fn,
                gemm_sparse_fn,
                inv_s_fwht_fn,
                lowrank_add_fn,
                lmhead_cs_fn,
                lmhead_cs_seeded_fn,
                encode_kv_glyph_fn,
                flash_glyph_attn_fn,
                set_andvcnt_abits_fn,
                gemv_i8_scale_bias_fn,
            })
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
                cands.push(dir.join("..").join("..").join("..").join("..").join("neon_runner").join(lib_name));
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

    pub fn set_andvcnt_abits(&self, a_bits: i32) -> Option<i32> {
        self.set_andvcnt_abits_fn.map(|f| unsafe { f(a_bits) })
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

#[derive(Clone, Debug, Default)]
pub struct T16CLoadPolicy {
    pub defer_vision_fault: bool,
    pub cold_mlp_layers: HashSet<usize>,
}

pub fn parse_mlp_layer_idx(name: &str) -> Option<usize> {
    let rest = name
        .strip_prefix("model.language_model.layers.")
        .or_else(|| name.strip_prefix("model.layers."))?;
    let (idx_s, suf) = rest.split_once('.')?;
    if matches!(suf, "mlp.gate_proj.weight" | "mlp.up_proj.weight" | "mlp.down_proj.weight") {
        idx_s.parse::<usize>().ok()
    } else {
        None
    }
}

pub fn is_vision_mat(name: &str) -> bool {
    name.starts_with("model.vision_tower.") || name.starts_with("model.embed_vision.")
}

pub struct T16CModel {
    pub arch_flags: u32,
    pub norms: HashMap<String, Vec<f32>>,
    pub mats: HashMap<String, T16CMat>,
    pub total_mat_bytes: usize,
    pub native: Option<NativeTile16Kernel>,
}

trait T16CSource: Read {
    fn read_packed_with_policy(
        &mut self,
        len: usize,
        pos: &mut usize,
        name: &str,
        is_cold: bool,
    ) -> Result<PackedStorage, String>;
}

struct FileSource {
    reader: BufReader<File>,
    file: Option<Arc<File>>,
}

impl Read for FileSource {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.reader.read(buf)
    }
}

impl T16CSource for FileSource {
    fn read_packed_with_policy(
        &mut self,
        len: usize,
        pos: &mut usize,
        name: &str,
        is_cold: bool,
    ) -> Result<PackedStorage, String> {
        if is_cold {
            if let Some(ref fa) = self.file {
                let off = *pos as u64;
                self.reader
                    .seek_relative(len as i64)
                    .map_err(|e| format!("seek past cold {name} ({len} B): {e}"))?;
                *pos += len;
                return Ok(PackedStorage::StreamedFile {
                    file: Arc::clone(fa),
                    file_offset: off,
                    len,
                });
            }
        }
        let mut packed = make_aligned_buf(len);
        self.reader
            .read_exact(aligned_as_mut_slice(&mut packed, len))
            .map_err(|e| format!("read {name} packed bytes ({len}): {e}"))?;
        *pos += len;
        Ok(PackedStorage::Owned(packed))
    }
}

#[cfg(unix)]
struct MmapCursor {
    map: Arc<MappedFile>,
    offset: usize,
}

#[cfg(unix)]
impl Read for MmapCursor {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let rem = self.map.len.saturating_sub(self.offset);
        let n = rem.min(buf.len());
        if n > 0 {
            unsafe {
                ptr::copy_nonoverlapping(self.map.ptr.add(self.offset), buf.as_mut_ptr(), n);
            }
            self.offset += n;
        }
        Ok(n)
    }
}

#[cfg(unix)]
impl T16CSource for MmapCursor {
    fn read_packed_with_policy(
        &mut self,
        len: usize,
        pos: &mut usize,
        name: &str,
        is_cold: bool,
    ) -> Result<PackedStorage, String> {
        let end = self.offset.checked_add(len).ok_or_else(|| format!("{name}: packed length overflow"))?;
        if end > self.map.len {
            return Err(format!("{name}: eof in mmap packed bytes ({end} > {})", self.map.len));
        }
        let ptr_addr = (self.map.ptr as usize) + self.offset;
        if ptr_addr % 64 != 0 {
            return Err(format!(
                "{name}: mmap packed offset {} (ptr {ptr_addr:#x}) is not 64-byte aligned",
                self.offset
            ));
        }
        let off = self.offset;
        self.offset = end;
        *pos += len;
        if is_cold {
            Ok(PackedStorage::StreamedFile {
                file: Arc::clone(&self.map.file),
                file_offset: off as u64,
                len,
            })
        } else {
            Ok(PackedStorage::Mmap {
                map: Arc::clone(&self.map),
                offset: off,
                len,
            })
        }
    }
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

#[cfg(unix)]
fn advise_willneed_span(map: &MappedFile, rel_start: usize, rel_end: usize) -> usize {
    let p_s = rel_start & !4095usize;
    let p_e = (rel_end.saturating_add(4095) & !4095usize).min(map.len);
    if p_e <= p_s {
        return 0;
    }
    let span = p_e - p_s;
    unsafe {
        let p = map.ptr.add(p_s);
        let _ = sys::madvise(p as *mut std::ffi::c_void, span, sys::MADV_WILLNEED);
        let mut off = 0usize;
        while off < span {
            let _ = std::ptr::read_volatile(p.add(off));
            off += 4096;
        }
    }
    span
}

#[cfg(unix)]
fn advise_selective_mmap_policy(map: &Arc<MappedFile>, model: &T16CModel, policy: &T16CLoadPolicy) {
    let mut pinned: Vec<(usize, usize)> = Vec::new();
    let mut deferred: Vec<(usize, usize)> = Vec::new();
    for m in model.mats.values() {
        let is_deferred = m.is_streamed() || (policy.defer_vision_fault && is_vision_mat(&m.name));
        match &m.packed {
            PackedStorage::Mmap { map: mm, offset, len } if Arc::ptr_eq(mm, map) && *len > 0 => {
                if is_deferred {
                    deferred.push((*offset, offset.saturating_add(*len)));
                } else {
                    pinned.push((*offset, offset.saturating_add(*len)));
                }
            }
            PackedStorage::StreamedFile { file_offset, len, .. } if *len > 0 => {
                if let Ok(off) = usize::try_from(*file_offset) {
                    deferred.push((off, off.saturating_add(*len)));
                }
            }
            _ => {}
        }
    }
    pinned.sort_unstable_by_key(|&(s, _)| s);
    deferred.sort_unstable_by_key(|&(s, _)| s);

    // Merge consecutive pinned intervals across small (< 1 MiB) metadata/inv_s/low_rank gaps
    // so USB3/NVMe readahead stays sequential, while never bridging across deferred tensors.
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(pinned.len());
    for (s, e) in pinned {
        if let Some(last) = merged.last_mut() {
            let crosses_deferred = deferred.iter().any(|&(ds, de)| ds < e && de > last.1);
            if !crosses_deferred && s <= last.1.saturating_add(1_048_576) {
                last.1 = last.1.max(e);
                continue;
            }
        }
        merged.push((s, e));
    }

    for (r_s, r_e) in merged {
        let _ = advise_willneed_span(map, r_s, r_e);
    }

    // Evict strictly-interior pages of deferred/cold matrices after the sequential pinned sweep so
    // any kernel readahead tail spilling into Vision/cold ranges is dropped cleanly.
    for m in model.mats.values() {
        if m.is_streamed() || (policy.defer_vision_fault && is_vision_mat(&m.name)) {
            m.advise_dontneed();
        }
    }
}

impl T16CModel {
    fn load_from_source_with_policy<S: T16CSource>(
        r: &mut S,
        native_lib: Option<&Path>,
        disable_native: bool,
        policy: &T16CLoadPolicy,
    ) -> Result<Self, String> {
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
            let nlen = read_u16(r, &mut pos)? as usize;
            let mut nbuf = vec![0u8; nlen];
            r.read_exact(&mut nbuf).map_err(|e| format!("read norm name: {e}"))?;
            pos += nlen;
            let name = String::from_utf8(nbuf).map_err(|e| format!("utf8 norm name: {e}"))?;
            let len = read_u32(r, &mut pos)? as usize;
            let data = read_f32_vec(r, &mut pos, len)?;
            norms.insert(name, data);
        }

        let mut mats = HashMap::with_capacity(n_mats);
        let mut total_mat_bytes = 0usize;
        for _ in 0..n_mats {
            let nlen = read_u16(r, &mut pos)? as usize;
            let d_out = read_u32(r, &mut pos)? as usize;
            let d_in = read_u32(r, &mut pos)? as usize;
            let block_size = read_u16(r, &mut pos)? as usize;
            let n_bits = read_u8(r, &mut pos)?;
            let is_asym_u8 = read_u8(r, &mut pos)?;
            let has_fwht_u8 = read_u8(r, &mut pos)?;
            let inv_s_kind = read_u8(r, &mut pos)?;
            let low_rank_u16 = read_u16(r, &mut pos)? as usize;
            let packed_bytes = read_u64(r, &mut pos)? as usize;
            let lr_bytes = read_u64(r, &mut pos)? as usize;

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
            let is_cold = parse_mlp_layer_idx(&name).is_some_and(|l| policy.cold_mlp_layers.contains(&l));
            let packed = r.read_packed_with_policy(packed_bytes, &mut pos, &name, is_cold)?;

            let inv_s = match inv_s_kind {
                0 => None,
                1 => Some(read_f32_vec(r, &mut pos, d_in)?),
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
                let su = read_f32_vec(r, &mut pos, d_out)?;
                let sv = read_f32_vec(r, &mut pos, rk)?;
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

    pub fn load(path: &Path, native_lib: Option<&Path>, disable_native: bool) -> Result<Self, String> {
        Self::load_with_policy(path, native_lib, disable_native, &T16CLoadPolicy::default())
    }

    pub fn load_with_policy(
        path: &Path,
        native_lib: Option<&Path>,
        disable_native: bool,
        policy: &T16CLoadPolicy,
    ) -> Result<Self, String> {
        if let Some(s) = path.to_str() {
            if s.contains('+') {
                let parts: Vec<&str> = s.split('+').filter(|p| !p.is_empty()).collect();
                if parts.is_empty() {
                    return Err("empty multi-shard T16C path".into());
                }
                let mut base = Self::load_single_with_policy(Path::new(parts[0]), native_lib, disable_native, policy)?;
                for &part in &parts[1..] {
                    let sub = Self::load_single_with_policy(Path::new(part), None, true, policy)?;
                    for (k, v) in sub.norms {
                        base.norms.insert(k, v);
                    }
                    for (k, m) in sub.mats {
                        base.mats.insert(k, m);
                    }
                }
                base.total_mat_bytes = base.mats.values().map(|m| m.total_bytes()).sum();
                return Ok(base);
            }
        }
        Self::load_single_with_policy(path, native_lib, disable_native, policy)
    }

    fn load_single_with_policy(
        path: &Path,
        native_lib: Option<&Path>,
        disable_native: bool,
        policy: &T16CLoadPolicy,
    ) -> Result<Self, String> {
        let f = File::open(path).map_err(|e| format!("open T16C {}: {e}", path.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let meta = f.metadata().map_err(|e| format!("stat T16C {}: {e}", path.display()))?;
            if !meta.is_file() {
                let mut src = FileSource {
                    reader: BufReader::with_capacity(4 * 1024 * 1024, f),
                    file: None,
                };
                return Self::load_from_source_with_policy(&mut src, native_lib, disable_native, policy);
            }
            let file_len = usize::try_from(meta.len())
                .map_err(|_| format!("T16C too large for address space: {}", meta.len()))?;
            if file_len < 24 {
                return Err(format!("T16C file truncated ({file_len} bytes): {}", path.display()));
            }
            let file_arc = Arc::new(f);
            let raw_ptr = unsafe {
                sys::mmap(ptr::null_mut(), file_len, sys::PROT_READ, sys::MAP_SHARED, file_arc.as_raw_fd(), 0)
            };
            if raw_ptr == sys::MAP_FAILED || raw_ptr.is_null() {
                return Err(format!("mmap T16C {} ({file_len} B) failed", path.display()));
            }
            let map = Arc::new(MappedFile {
                ptr: raw_ptr as *const u8,
                len: file_len,
                file: Arc::clone(&file_arc),
            });
            let map_ref = Arc::clone(&map);
            let selective = policy.defer_vision_fault || !policy.cold_mlp_layers.is_empty();
            if !selective {
                unsafe {
                    let _ = sys::madvise(raw_ptr, file_len, sys::MADV_WILLNEED);
                    let mut off = 0usize;
                    while off < file_len {
                        let _ = std::ptr::read_volatile(map.ptr.add(off));
                        off += 4096;
                    }
                }
            }
            let mut cursor = MmapCursor { map, offset: 0 };
            let model = Self::load_from_source_with_policy(&mut cursor, native_lib, disable_native, policy)?;
            if selective {
                advise_selective_mmap_policy(&map_ref, &model, policy);
            }
            Ok(model)
        }
        #[cfg(not(unix))]
        {
            let file_seek = File::open(path).map_err(|e| format!("open T16C seek {}: {e}", path.display()))?;
            let meta = f.metadata().map_err(|e| format!("stat T16C {}: {e}", path.display()))?;
            let file_opt = if meta.is_file() { Some(Arc::new(file_seek)) } else { None };
            let mut src = FileSource {
                reader: BufReader::with_capacity(4 * 1024 * 1024, f),
                file: file_opt,
            };
            Self::load_from_source_with_policy(&mut src, native_lib, disable_native, policy)
        }
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
            let cow = m.packed_cow()?;
            w.write_all(cow.as_ref()).map_err(|e| format!("write packed: {e}"))?;
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

    #[test]
    fn tern5_expected_bytes_and_validation() {
        assert_eq!(expected_tile16_bytes(19, 64, 64, 5, false).unwrap(), 2 * 240);
        assert_eq!(expected_tile16_bytes(19, 128, 128, 5, false).unwrap(), 2 * 448);
        assert_eq!(expected_tile16_bytes(19, 256, 256, 5, false).unwrap(), 2 * 864);
        assert!(expected_tile16_bytes(19, 256, 256, 5, true).is_err());
        assert!(expected_tile16_bytes(19, 256, 256, 6, false).is_err());
    }

    #[test]
    fn tern5_roundtrip_decode_row_and_batch_parity() {
        let native_k = NativeTile16Kernel::discover(None);
        for &(bsize, d_in) in &[(64usize, 128usize), (128, 256), (256, 512)] {
            let d_out = 19usize;
            let n_blks = d_in / bsize;
            let sc: Vec<f32> = (0..d_out * n_blks).map(|i| 0.02 + 0.003 * (i as f32 % 7.0)).collect();
            let idx: Vec<u8> = (0..d_out * d_in).map(|i| ((i * 17 + 3) % 3) as u8).collect();
            let inv_s: Vec<f32> = (0..d_in).map(|j| 0.75 + 0.0015 * (j as f32)).collect();
            let rk = 4usize;
            let lr = Int8LowRank {
                rank: rk,
                d_out,
                d_in,
                su: vec![0.012f32; d_out],
                sv: vec![0.018f32; rk],
                u_i8: (0..d_out * rk).map(|i| ((i * 11 % 199) as i32 - 99) as i8).collect(),
                v_i8: (0..rk * d_in).map(|i| ((i * 7 % 199) as i32 - 99) as i8).collect(),
            };
            let mat = T16CMat::pack_from_components(
                "test.tern5".into(), d_out, d_in, bsize, 5, false, true, &sc, None, &idx, Some(inv_s), Some(lr),
            ).unwrap();

            let tmp = std::env::temp_dir().join(format!("t16c_tern5_{bsize}.t16c"));
            T16CModel::write_file(&tmp, ARCH_FLAG_GEMMA4, &[("n0".into(), vec![1.0, 2.0])], &[mat]).unwrap();
            let loaded = T16CModel::load(&tmp, None, native_k.is_none()).unwrap();
            let _ = std::fs::remove_file(&tmp);

            let lm = loaded.mat("test.tern5").unwrap();
            for &k_tokens in &[1usize, 3usize] {
                let x: Vec<f32> = (0..k_tokens * d_in).map(|i| ((i as f32) * 0.23).sin()).collect();
                let y_rust = lm.project_batch(&x, k_tokens, None, 2).unwrap();
                for tok in 0..k_tokens {
                    for r in 0..d_out {
                        let row = lm.decode_row(r).unwrap();
                        let dot: f32 = row.iter().zip(&x[tok * d_in..(tok + 1) * d_in]).map(|(w, v)| w * v).sum();
                        let got = y_rust[tok * d_out + r];
                        assert!((got - dot).abs() < 3e-3, "bsize={bsize} k={k_tokens} tok={tok} r={r}: {got} vs {dot}");
                    }
                }
                if let Some(ref nk) = native_k {
                    let y_nat = lm.project_batch(&x, k_tokens, Some(nk), 2).unwrap();
                    let (mut dot, mut n1, mut n2) = (0.0f64, 0.0f64, 0.0f64);
                    for (a, b) in y_rust.iter().zip(y_nat.iter()) {
                        dot += (*a as f64) * (*b as f64);
                        n1 += (*a as f64) * (*a as f64);
                        n2 += (*b as f64) * (*b as f64);
                    }
                    let y_rms = (n1 / (y_rust.len().max(1) as f64)).sqrt() as f32;
                    let cos = dot / (n1.sqrt() * n2.sqrt() + 1e-12);
                    assert!(cos > 0.9999, "native tern5 cos={cos} (bsize={bsize} k={k_tokens})");
                    // Unified single-plane INT8 (7-bit) Q8 precision across all smooth k_tokens
                    let tol = 2e-3f32 + 2e-4f32 * y_rms;
                    for (idx_y, (a, b)) in y_rust.iter().zip(y_nat.iter()).enumerate() {
                        assert!((a - b).abs() < tol, "native mismatch bsize={bsize} k={k_tokens} idx={idx_y}: rust={a} nat={b} tol={tol}");
                    }
                }
            }
        }
    }

    #[test]
    fn native_ffi_inv_s_fwht_lowrank_and_spiky_dual_plane_parity() {
        let native_k = NativeTile16Kernel::discover(None);
        if let Some(ref nk) = native_k {
            assert!(nk.inv_s_fwht_fn.is_some(), "expected inv_s_and_fwht_f32_c in {}", nk.path);
            assert!(nk.lowrank_add_fn.is_some(), "expected int8_lowrank_add_batch_c in {}", nk.path);
        }

        for &(n_bits, is_asym) in &[(5u8, false), (1u8, true)] {
            let (d_out, d_in, bsize, kt) = (32usize, 128usize, 64usize, 2usize);
            let n_blks = d_in / bsize;
            let sc = vec![0.25f32; d_out * n_blks];
            let bi = if is_asym { Some(vec![-0.02f32; d_out * n_blks]) } else { None };
            let mut idx = vec![0u8; d_out * d_in];
            for r in 0..d_out {
                for blk in 0..n_blks {
                    for c in 0..bsize {
                        let col = blk * bsize + c;
                        let val = if n_bits == 5 {
                            if c < 2 { 1u8 } else { ((r * 7 + col * 13) % 3) as u8 }
                        } else if c == 0 {
                            1u8
                        } else if c == 1 {
                            0u8
                        } else {
                            ((r * 11 + col * 5) % 2) as u8
                        };
                        idx[r * d_in + col] = val;
                    }
                }
            }
            let inv_s: Vec<f32> = (0..d_in).map(|j| 0.7 + 0.002 * (j as f32)).collect();
            let rk = 16usize;
            let lr = Int8LowRank {
                rank: rk,
                d_out,
                d_in,
                su: (0..d_out).map(|r| 0.01 + 0.0005 * (r as f32)).collect(),
                sv: (0..rk).map(|k| 0.015 - 0.0003 * (k as f32)).collect(),
                u_i8: (0..d_out * rk).map(|i| ((i * 19 % 211) as i32 - 105) as i8).collect(),
                v_i8: (0..rk * d_in).map(|i| ((i * 23 % 211) as i32 - 105) as i8).collect(),
            };
            let mat = T16CMat::pack_from_components(
                format!("spiky.b{n_bits}"),
                d_out,
                d_in,
                bsize,
                n_bits,
                is_asym,
                false,
                &sc,
                bi.as_deref(),
                &idx,
                Some(inv_s),
                Some(lr),
            )
            .unwrap();

            let tmp = std::env::temp_dir().join(format!("t16c_spiky_{n_bits}.t16c"));
            T16CModel::write_file(&tmp, ARCH_FLAG_GEMMA4, &[("n0".into(), vec![1.0])], &[mat]).unwrap();
            let loaded = T16CModel::load(&tmp, None, true).unwrap();
            let _ = std::fs::remove_file(&tmp);
            let lm = loaded.mat(&format!("spiky.b{n_bits}")).unwrap();

            for &k_check in &[1usize, kt] {
                let mut x = vec![0.0f32; k_check * d_in];
                for tok in 0..k_check {
                    for j in 0..d_in {
                        let loc = j % 64;
                        x[tok * d_in + j] = if loc < 2 {
                            if tok == 0 { 800.0f32 } else { -800.0f32 }
                        } else {
                            0.04f32 * (((tok * 31 + j * 17) % 19) as f32 - 9.0) / 9.0
                        };
                    }
                }
                let y_rust = lm.project_batch(&x, k_check, None, 2).unwrap();
                if let Some(ref nk) = native_k {
                    let y_nat = lm.project_batch(&x, k_check, Some(nk), 2).unwrap();
                    for tok in 0..k_check {
                        let s = tok * d_out;
                        let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
                        for r in 0..d_out {
                            let a = y_rust[s + r] as f64;
                            let b = y_nat[s + r] as f64;
                            dot += a * b;
                            na += a * a;
                            nb += b * b;
                        }
                        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
                        assert!(cos > 0.9999, "spiky n_bits={n_bits} k={k_check} tok={tok} cos={cos}");
                    }
                }
            }
        }
    }

    #[test]
    fn int8_roundtrip_decode_row_and_spiky_batch_parity() {
        assert_eq!(expected_tile16_bytes(19, 64, 64, 8, false).unwrap(), 2 * (32 + 64 * 16));
        assert!(expected_tile16_bytes(19, 256, 256, 8, true).is_err());

        let native_k = NativeTile16Kernel::discover(None);
        for &(bsize, d_in) in &[(64usize, 128usize), (128, 256)] {
            let d_out = 19usize;
            let n_blks = d_in / bsize;
            let sc: Vec<f32> = (0..d_out * n_blks).map(|i| 0.002 + 0.0004 * (i as f32 % 11.0)).collect();
            let mut idx: Vec<u8> = Vec::with_capacity(d_out * d_in);
            for r in 0..d_out {
                for c in 0..d_in {
                    let q_i8 = (((r * 19 + c * 37) % 253) as i32 - 126) as i8;
                    idx.push(q_i8 as u8);
                }
            }
            let mat = T16CMat::pack_from_components(
                "test.int8".into(), d_out, d_in, bsize, 8, false, false, &sc, None, &idx, None, None,
            ).unwrap();
            let tmp = std::env::temp_dir().join(format!("t16c_int8_{bsize}.t16c"));
            T16CModel::write_file(&tmp, ARCH_FLAG_GEMMA4, &[("n0".into(), vec![1.0])], &[mat]).unwrap();
            let loaded = T16CModel::load(&tmp, None, native_k.is_none()).unwrap();
            let _ = std::fs::remove_file(&tmp);

            let lm = loaded.mat("test.int8").unwrap();
            for &k_tokens in &[1usize, 2usize, 4usize] {
                let mut x = vec![0.0f32; k_tokens * d_in];
                for tok in 0..k_tokens {
                    for j in 0..d_in {
                        x[tok * d_in + j] = if j % 64 == 0 {
                            if tok % 2 == 0 { 500.0 } else { -500.0 }
                        } else {
                            ((tok * 13 + j * 29) as f32 * 0.17).sin() * 0.25
                        };
                    }
                }
                let y_rust = lm.project_batch(&x, k_tokens, None, 2).unwrap();
                for tok in 0..k_tokens {
                    for r in 0..d_out {
                        let row = lm.decode_row(r).unwrap();
                        let dot: f32 = row.iter().zip(&x[tok * d_in..(tok + 1) * d_in]).map(|(w, v)| w * v).sum();
                        let got = y_rust[tok * d_out + r];
                        assert!((got - dot).abs() < 2e-3, "int8 bsize={bsize} k={k_tokens} tok={tok} r={r}: {got} vs {dot}");
                    }
                }
                if let Some(ref nk) = native_k {
                    let y_nat = lm.project_batch(&x, k_tokens, Some(nk), 2).unwrap();
                    for tok in 0..k_tokens {
                        let s = tok * d_out;
                        let (mut dot, mut n1, mut n2) = (0.0f64, 0.0f64, 0.0f64);
                        for r in 0..d_out {
                            let a = y_rust[s + r] as f64;
                            let b = y_nat[s + r] as f64;
                            dot += a * b;
                            n1 += a * a;
                            n2 += b * b;
                        }
                        let cos = dot / (n1.sqrt() * n2.sqrt() + 1e-12);
                        assert!(cos > 0.999, "native int8 cos={cos} (bsize={bsize} k={k_tokens} tok={tok})");
                    }
                }
            }
        }
    }

    #[test]
    fn cs_tile_skip_skips_orthogonal_zero_blocks_consistently() {
        let d_out = 32usize;
        let d_in = 128usize;
        let bsize = 64usize;
        let mut sc = vec![0.25f32; d_out * 2];
        // Zero weight scale in block 1 for tile 1 (rows 16..32)
        for r in 16..32 {
            sc[r * 2 + 1] = 0.0;
        }
        let idx = vec![15u8; d_out * d_in];
        let mat = T16CMat::pack_from_components("cs.test".into(), d_out, d_in, bsize, 4, false, false, &sc, None, &idx, None, None).unwrap();
        let norms = mat.compute_tile_norms_fp16();
        let mut x = vec![0.0f32; d_in];
        for j in 0..bsize {
            x[j] = 1.0f32; // block 0 active, block 1 zero
        }
        let (y_full, sk0) = mat.project_batch_cs(&x, 1, None, 2, Some(&norms), 0.0).unwrap();
        let (y_cs, sk1) = mat.project_batch_cs(&x, 1, None, 2, Some(&norms), 1e-4).unwrap();
        assert_eq!(sk0, 0);
        assert!(sk1 >= 2, "expected both tile zero-activation blocks skipped, got {sk1}");
        for (a, b) in y_full.iter().zip(y_cs.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        if let Some(nk) = NativeTile16Kernel::discover(None) {
            let (y_nat_full, sk_nat0) = mat.project_batch_cs(&x, 1, Some(&nk), 2, Some(&norms), 0.0).unwrap();
            let (y_nat, sk_nat) = mat.project_batch_cs(&x, 1, Some(&nk), 2, Some(&norms), 1e-4).unwrap();
            assert_eq!(sk_nat0, 0);
            assert!(sk_nat >= 2);
            for (r_idx, ((a, b0), b)) in y_full.iter().zip(y_nat_full.iter()).zip(y_nat.iter()).enumerate() {
                assert!((b0 - b).abs() < 1e-6, "CS skip altered native output row={r_idx}: {b0} vs {b}");
                let tol = 1e-3f32 + 2e-4f32 * a.abs();
                assert!(
                    (a - b).abs() < tol,
                    "cs_tile_skip mismatch row={r_idx}: y_full={a} vs y_nat={b} tol={tol} (path={})",
                    nk.path
                );
            }
        }
    }

    #[test]
    fn gauss_nonuniform_2bit_improves_rotated_gaussian_snr_and_matches_native() {
        let d_out = 32usize;
        let d_in = 128usize;
        let bsize = 64usize;
        let n_blks = d_in / bsize;

        // Generate Gaussian weights via deterministic Box-Muller
        let mut s: u64 = 0x517cc1b727220a95;
        let mut next_f = || -> f32 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (((s >> 40) as u32).max(1) as f32) / (1u32 << 24) as f32
        };
        let mut w = vec![0.0f32; d_out * d_in];
        for r in 0..d_out {
            let row_std = 0.04 + 0.002 * (r as f32);
            let mut j = 0;
            while j < d_in {
                let u1 = next_f().max(1e-6);
                let u2 = next_f();
                let rad = (-2.0f32 * u1.ln()).sqrt();
                let th = std::f32::consts::TAU * u2;
                w[r * d_in + j] = rad * th.cos() * row_std;
                if j + 1 < d_in {
                    w[r * d_in + j + 1] = rad * th.sin() * row_std;
                }
                j += 2;
            }
        }

        let quantize_pack = |n_bits: u8| -> (T16CMat, f64) {
            let levels: [f32; 4] = [
                -(plane_weight(n_bits, 1) + plane_weight(n_bits, 0)),
                -(plane_weight(n_bits, 1) - plane_weight(n_bits, 0)),
                plane_weight(n_bits, 1) - plane_weight(n_bits, 0),
                plane_weight(n_bits, 1) + plane_weight(n_bits, 0),
            ];
            let mut sc = vec![0.0f32; d_out * n_blks];
            let mut idx = vec![0u8; d_out * d_in];
            let mut sse = 0.0f64;
            for r in 0..d_out {
                for blk in 0..n_blks {
                    let s0 = r * d_in + blk * bsize;
                    let sl = &w[s0..s0 + bsize];
                    let var = sl.iter().map(|&v| v * v).sum::<f32>() / (bsize as f32);
                    let std_b = var.sqrt().max(1e-6);
                    // Lloyd-Max 2-bit optimal inner threshold factor ~0.498 on N(0,1)
                    let mut best_step = std_b * 0.50;
                    let mut best_err = f64::INFINITY;
                    for c in 35..=70 {
                        let step = std_b * (c as f32 * 0.01);
                        let mut err = 0.0f64;
                        for &wv in sl {
                            let mut min_d = f32::INFINITY;
                            for &lv in &levels {
                                let d = (wv - step * lv).abs();
                                if d < min_d {
                                    min_d = d;
                                }
                            }
                            err += (min_d as f64) * (min_d as f64);
                        }
                        if err < best_err {
                            best_err = err;
                            best_step = step;
                        }
                    }
                    sc[r * n_blks + blk] = best_step;
                    for (j, &wv) in sl.iter().enumerate() {
                        let mut best_q = 0usize;
                        let mut min_d = f32::INFINITY;
                        for (qi, &lv) in levels.iter().enumerate() {
                            let d = (wv - best_step * lv).abs();
                            if d < min_d {
                                min_d = d;
                                best_q = qi;
                            }
                        }
                        idx[s0 + j] = best_q as u8;
                        let diff = (wv - best_step * levels[best_q]) as f64;
                        sse += diff * diff;
                    }
                }
            }
            let mat = T16CMat::pack_from_components(
                format!("gauss.nb{n_bits}"),
                d_out,
                d_in,
                bsize,
                n_bits,
                false,
                false,
                &sc,
                None,
                &idx,
                None,
                None,
            )
            .unwrap();
            (mat, sse / (w.len() as f64))
        };

        let (_, mse_u) = quantize_pack(2);
        let (mat_g, mse_g) = quantize_pack(2 | T16C_GAUSS_BIT);
        assert!(mse_g < mse_u, "expected Gaussian levels MSE ({mse_g}) < uniform ({mse_u})");

        let tmp = std::env::temp_dir().join("t16c_gauss_test.t16c");
        T16CModel::write_file(&tmp, ARCH_FLAG_GEMMA4, &[("n0".into(), vec![1.0])], &[mat_g]).unwrap();
        let loaded = T16CModel::load(&tmp, None, true).unwrap();
        let _ = std::fs::remove_file(&tmp);
        let lm = loaded.mat("gauss.nb130").unwrap();
        assert_eq!(lm.n_bits, 2 | T16C_GAUSS_BIT);

        let kt = 4usize;
        let x: Vec<f32> = (0..kt * d_in).map(|i| ((i as f32) * 0.13).sin()).collect();
        let y_rust = lm.project_batch(&x, kt, None, 2).unwrap();
        for tok in 0..kt {
            for r in 0..d_out {
                let row = lm.decode_row(r).unwrap();
                let dot: f32 = row.iter().zip(&x[tok * d_in..(tok + 1) * d_in]).map(|(a, b)| a * b).sum();
                assert!((y_rust[tok * d_out + r] - dot).abs() < 1e-4);
            }
        }
        if let Some(nk) = NativeTile16Kernel::discover(None) {
            let y_b = lm.project_batch(&x, kt, Some(&nk), 2).unwrap();
            for tok in 0..kt {
                let y_s = lm.project_batch(&x[tok * d_in..(tok + 1) * d_in], 1, Some(&nk), 2).unwrap();
                let (mut dot, mut n1, mut n2) = (0.0f64, 0.0f64, 0.0f64);
                for r in 0..d_out {
                    let a = y_b[tok * d_out + r];
                    let b = y_s[r];
                    let yr = y_rust[tok * d_out + r];
                    assert_eq!(a.to_bits(), b.to_bits(), "gauss native k=4 vs k=1 bit mismatch");
                    assert!((a - yr).abs() < 0.05, "gauss native vs rust diff={} ({a} vs {yr})", (a - yr).abs());
                    dot += (a as f64) * (yr as f64);
                    n1 += (a as f64) * (a as f64);
                    n2 += (yr as f64) * (yr as f64);
                }
                let cos = dot / (n1.sqrt() * n2.sqrt() + 1e-12);
                assert!(cos > 0.9999, "gauss native vs rust cos={cos} at tok={tok}");
            }
        }
    }

    #[test]
    fn lmhead_cs_prunes_and_matches_exact_top1_batch() {
        let d_out = 128usize;
        let d_in = 512usize;
        let bsize = 64usize;
        let n_blks = d_in / bsize;
        let mut sc = vec![0.02f32; d_out * n_blks];
        let mut idx = vec![0u8; d_out * d_in];
        for r in 0..d_out {
            for b in 0..n_blks {
                sc[r * n_blks + b] = if r == 37 {
                    0.50
                } else if r < 48 {
                    0.08
                } else {
                    0.01
                };
            }
            for j in 0..d_in {
                idx[r * d_in + j] = if r == 37 && j < 128 {
                    7
                } else {
                    ((r * 13 + j * 7) % 8) as u8
                };
            }
        }
        let mat = T16CMat::pack_from_components(
            "lmhead.cs.test".into(),
            d_out,
            d_in,
            bsize,
            3,
            false,
            true,
            &sc,
            None,
            &idx,
            None,
            None,
        )
        .unwrap();
        let norms = mat.compute_tile_norms_fp16();
        let mut x = vec![0.0f32; 2 * d_in];
        for tok in 0..2 {
            for j in 0..d_in {
                x[tok * d_in + j] = if j < 128 {
                    if tok == 0 { 1.4 } else { -1.4 }
                } else {
                    (((j % 11) as f32) - 5.0) * 0.01
                };
            }
        }

        if let Some(nk) = NativeTile16Kernel::discover(None) {
            assert!(nk.lmhead_cs_fn.is_some(), "expected lmhead_cs_top1_tile16_batch in {}", nk.path);
            assert!(nk.lmhead_cs_seeded_fn.is_some(), "expected lmhead_cs_seeded_tile16_batch in {}", nk.path);
            let y_full = mat.project_batch(&x, 2, Some(&nk), 2).unwrap();
            let (y_cs, sk_cs) = mat.project_lmhead_cs(&x, 2, Some(&nk), 2, Some(&norms), 8, 0.0).unwrap();
            let (y_sv, sk_sv) = mat.project_lmhead_cs(&x, 2, Some(&nk), 2, Some(&norms), 8, 0.25).unwrap();
            assert!(sk_cs > 0, "expected exact CS to prune low-norm tiles, got sk_cs={sk_cs}");
            assert!(sk_sv > 0, "expected sieve CS to prune blocks, got sk_sv={sk_sv}");
            for tok in 0..2 {
                let s0 = tok * d_out;
                let top_f = y_full[s0..s0 + d_out]
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap();
                let top_c = y_cs[s0..s0 + d_out]
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap();
                let top_s = y_sv[s0..s0 + d_out]
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap();
                assert_eq!(top_f.0, top_c.0, "tok {tok}: exact CS top1 mismatch");
                assert_eq!(top_f.0, top_s.0, "tok {tok}: sieve CS top1 mismatch");
                assert!((top_f.1 - top_c.1).abs() < 1e-3);
            }

            // 16-slot speculative tree seeded exact CS parity
            let mut x16 = Vec::with_capacity(16 * d_in);
            let mut seed16 = vec![-1i32; 16];
            for k in 0..16usize {
                x16.extend_from_slice(&x[(k % 2) * d_in..((k % 2) + 1) * d_in]);
                seed16[k] = if k % 2 == 0 { 35 } else { 19 };
            }
            seed16[15] = -1;
            let y16_full = mat.project_batch(&x16, 16, Some(&nk), 2).unwrap();
            let (y16_seed, sk16) = mat
                .project_lmhead_cs_seeded(&x16, 16, Some(&nk), 2, Some(&norms), 1, 0.0, Some(&seed16))
                .unwrap();
            assert!(sk16 > 0, "expected k=16 seeded CS to prune blocks, got sk16={sk16}");
            for tok in 0..16usize {
                let s0 = tok * d_out;
                let top_f = y16_full[s0..s0 + d_out]
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0;
                let top_s = y16_seed[s0..s0 + d_out]
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0;
                assert_eq!(top_f, top_s, "tok {tok}: k=16 seeded exact CS argmax mismatch");
            }
        }
    }

    #[test]
    fn t16c_sparse_block_tile_matches_dense_masked() {
        let d_out = 64usize;
        let d_in = 256usize;
        let bsize = 64usize;
        let n_blks = d_in / bsize;
        let rmask = [1u8, 0u8, 1u8, 0u8];
        let cmask = [0u8, 1u8, 0u8, 1u8];
        for &(n_bits, is_asym) in &[(1u8, true), (2u8 | T16C_GAUSS_BIT, false)] {
            let max_q = if (n_bits & 0x7F) == 1 { 1u8 } else { 3u8 };
            let mut sc = vec![0.0f32; d_out * n_blks];
            let mut bi = vec![0.0f32; d_out * n_blks];
            let mut idx = vec![0u8; d_out * d_in];
            for r in 0..d_out {
                for b in 0..n_blks {
                    sc[r * n_blks + b] = 0.04 + 0.01 * ((r + b) % 5) as f32;
                    bi[r * n_blks + b] = -0.02 + 0.005 * ((r + 2 * b) % 4) as f32;
                }
                for j in 0..d_in {
                    idx[r * d_in + j] = ((r * 7 + j * 13) % ((max_q as usize) + 1)) as u8;
                }
            }
            let inv_s: Vec<f32> = (0..d_in).map(|j| 0.8 + 0.05 * ((j % 7) as f32)).collect();
            let lr = Int8LowRank {
                rank: 4,
                d_out,
                d_in,
                su: vec![0.01f32; d_out],
                sv: vec![0.02f32; 4],
                u_i8: (0..d_out * 4).map(|i| ((i % 11) as i8) - 5).collect(),
                v_i8: (0..4 * d_in).map(|i| ((i % 13) as i8) - 6).collect(),
            };
            let mat = T16CMat::pack_from_components(
                "sparse.test".into(),
                d_out,
                d_in,
                bsize,
                n_bits,
                is_asym,
                true,
                &sc,
                if is_asym { Some(&bi) } else { None },
                &idx,
                Some(inv_s),
                Some(lr),
            )
            .unwrap();
            let k_tok = 4usize;
            let mut x = vec![0.0f32; k_tok * d_in];
            let mut x_z = vec![0.0f32; k_tok * d_in];
            for t in 0..k_tok {
                for j in 0..d_in {
                    let v = (((t * 17 + j * 5) % 19) as f32 - 9.0) * 0.1;
                    x[t * d_in + j] = v;
                    x_z[t * d_in + j] = if cmask[j / bsize] != 0 { v } else { 0.0 };
                }
            }
            let y_ref = mat.project_batch(&x_z, k_tok, None, 2).unwrap();
            let (y_rust, sk_rust) = mat
                .project_batch_sparse(&x, k_tok, None, 2, Some(&rmask), Some(&cmask))
                .unwrap();
            assert_eq!(sk_rust, 12);
            for t in 0..k_tok {
                for r in 0..d_out {
                    let want = if rmask[r / 16] != 0 { y_ref[t * d_out + r] } else { 0.0 };
                    assert!(
                        (y_rust[t * d_out + r] - want).abs() < 1e-4,
                        "rust sparse mismatch mode={n_bits} t={t} r={r}: got={} want={want}",
                        y_rust[t * d_out + r]
                    );
                }
            }
            if let Some(nk) = NativeTile16Kernel::discover(None) {
                if nk.gemm_sparse_fn.is_some() {
                    let y_nat_ref = mat.project_batch(&x_z, k_tok, Some(&nk), 2).unwrap();
                    let (y_nat, sk_nat) = mat
                        .project_batch_sparse(&x, k_tok, Some(&nk), 2, Some(&rmask), Some(&cmask))
                        .unwrap();
                    assert_eq!(sk_nat, 12);
                    for t in 0..k_tok {
                        for r in 0..d_out {
                            let idx_o = t * d_out + r;
                            let want_nat = if rmask[r / 16] != 0 { y_nat_ref[idx_o] } else { 0.0 };
                            assert!(
                                (y_nat[idx_o] - want_nat).abs() < 1e-6,
                                "native sparse vs native dense mismatch mode={n_bits} t={t} r={r}: nat={} want={want_nat}",
                                y_nat[idx_o]
                            );
                            assert!(
                                (y_nat[idx_o] - y_rust[idx_o]).abs() < 2.5e-2,
                                "native sparse vs fp64 rust mismatch mode={n_bits} t={t} r={r}: nat={} rust={}",
                                y_nat[idx_o],
                                y_rust[idx_o]
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn t16c_rowmajor_andvcnt_roundtrip_and_sparse_parity() {
        let d_out = 64usize;
        let d_in = 512usize;
        let bsize = 256usize;
        let n_blks = d_in / bsize;
        let rmask = [1u8, 0u8, 1u8, 0u8];
        let mut sc = vec![0.0f32; d_out * n_blks];
        let mut bi = vec![0.0f32; d_out * n_blks];
        let mut idx = vec![0u8; d_out * d_in];
        for r in 0..d_out {
            for b in 0..n_blks {
                sc[r * n_blks + b] = 0.04 + 0.01 * ((r + b) % 5) as f32;
                bi[r * n_blks + b] = -0.015 + 0.005 * ((r + 2 * b) % 3) as f32;
            }
            for j in 0..d_in {
                idx[r * d_in + j] = ((r * 7 + j * 13) & 1) as u8;
            }
        }
        let mat = T16CMat::pack_from_components(
            "rowmajor.test".into(),
            d_out,
            d_in,
            bsize,
            1 | T16C_ROWMAJOR_BIT,
            true,
            true,
            &sc,
            Some(&bi),
            &idx,
            None,
            None,
        )
        .unwrap();
        let mat_col = T16CMat::pack_from_components(
            "colmajor.test".into(),
            d_out,
            d_in,
            bsize,
            1,
            true,
            true,
            &sc,
            Some(&bi),
            &idx,
            None,
            None,
        )
        .unwrap();
        for r in 0..d_out {
            let a = mat.decode_row(r).unwrap();
            let b = mat_col.decode_row(r).unwrap();
            for (va, vb) in a.iter().zip(b.iter()) {
                assert!((va - vb).abs() < 1e-6, "rowmajor decode_row mismatch r={r}: {va} vs {vb}");
            }
        }
        let kt = 4usize;
        let x: Vec<f32> = (0..kt * d_in)
            .map(|i| (((i * 19 + 7) % 29) as f32 - 14.0) * 0.08)
            .collect();
        let y_rust = mat.project_batch(&x, kt, None, 2).unwrap();
        let y_col_rust = mat_col.project_batch(&x, kt, None, 2).unwrap();
        for (a, b) in y_rust.iter().zip(y_col_rust.iter()) {
            assert!((a - b).abs() < 1e-5, "rowmajor vs colmajor rust mismatch: {a} vs {b}");
        }
        if let Some(nk) = NativeTile16Kernel::discover(None) {
            if nk.set_andvcnt_abits_fn.is_some() {
                nk.set_andvcnt_abits(5);
                let y_nat4 = mat.project_batch(&x, kt, Some(&nk), 2).unwrap();
                for tok in 0..kt {
                    let y_nat1 = mat.project_batch(&x[tok * d_in..(tok + 1) * d_in], 1, Some(&nk), 2).unwrap();
                    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
                    for r in 0..d_out {
                        let a = y_nat4[tok * d_out + r];
                        let b = y_nat1[r];
                        let yr = y_rust[tok * d_out + r];
                        assert_eq!(a.to_bits(), b.to_bits(), "rowmajor native k=4 vs k=1 bit mismatch");
                        dot += (a as f64) * (yr as f64);
                        na += (a as f64) * (a as f64);
                        nb += (yr as f64) * (yr as f64);
                    }
                    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
                    assert!(cos > 0.995, "rowmajor native 5-bit cos={cos} tok={tok}");
                }
                let (y_sp, sk) = mat.project_batch_sparse(&x, kt, Some(&nk), 2, Some(&rmask), None).unwrap();
                assert_eq!(sk, (2 * n_blks) as i64);
                for tok in 0..kt {
                    for r in 0..d_out {
                        let want = if rmask[r / 16] != 0 { y_nat4[tok * d_out + r] } else { 0.0 };
                        assert!((y_sp[tok * d_out + r] - want).abs() < 1e-6);
                    }
                }
            }
        }
    }

    #[test]
    fn colblock_streamed_sparse_matches_resident_dense_for_all_precisions_and_batch_k() {
        let native_k = NativeTile16Kernel::discover(None);
        let d_out = 64usize;
        let d_in = 512usize;
        let bsize = 128usize;
        let n_blks = d_in / bsize; // 4 blocks
        let n_tiles = d_out / 16;  // 4 tiles
        let rmask = [1u8, 0u8, 1u8, 0u8];
        let cmask = [1u8, 0u8, 0u8, 1u8]; // 50% active blocks
        let inv_s: Vec<f32> = (0..d_in).map(|j| 0.75 + 0.03 * ((j % 7) as f32)).collect();
        let specs: &[(u8, bool, &str)] = &[
            (1 | T16C_ROWMAJOR_BIT, true, "1b_rm"),
            (2 | T16C_GAUSS_BIT | T16C_ROWMAJOR_BIT, false, "2b_gauss_rm"),
            (3, false, "3b"),
            (5, false, "tern5"),
        ];
        for &(nb, is_asym, tag) in specs {
            let eff = nb & 0x1F;
            let max_q: usize = if eff == 5 { 3 } else { 1usize << eff };
            let mut sc = vec![0.0f32; d_out * n_blks];
            let mut bi = vec![0.0f32; d_out * n_blks];
            let mut idx = vec![0u8; d_out * d_in];
            for r in 0..d_out {
                for b in 0..n_blks {
                    sc[r * n_blks + b] = 0.03 + 0.008 * ((r + 3 * b) % 5) as f32;
                    bi[r * n_blks + b] = -0.015 + 0.004 * ((r + b) % 4) as f32;
                }
                for c in 0..d_in {
                    idx[r * d_in + c] = ((r * 13 + c * 7 + (eff as usize)) % max_q) as u8;
                }
            }
            let lr_g = Int8LowRank {
                rank: 4,
                d_out,
                d_in,
                su: vec![0.012f32; d_out],
                sv: vec![0.015f32; 4],
                u_i8: (0..d_out * 4).map(|i| ((i * 7 % 21) as i8) - 10).collect(),
                v_i8: (0..4 * d_in).map(|i| ((i * 11 % 23) as i8) - 11).collect(),
            };
            let gate_res = T16CMat::pack_from_components(
                "model.language_model.layers.0.mlp.gate_proj.weight".into(),
                d_out,
                d_in,
                bsize,
                nb,
                is_asym,
                true,
                &sc,
                if is_asym { Some(&bi) } else { None },
                &idx,
                Some(inv_s.clone()),
                Some(lr_g.clone()),
            )
            .unwrap();
            let down_res = T16CMat::pack_from_components(
                "model.language_model.layers.0.mlp.down_proj.weight".into(),
                d_out,
                d_in,
                bsize,
                nb,
                is_asym,
                true,
                &sc,
                if is_asym { Some(&bi) } else { None },
                &idx,
                Some(inv_s.clone()),
                Some(lr_g.clone()),
            )
            .unwrap();
            let down_cb = down_res.to_colblock().unwrap();
            assert!(down_cb.is_colblock());
            assert_eq!(down_cb.packed_bytes_len, down_res.packed_bytes_len);

            let tmp = std::env::temp_dir().join(format!("t16c_colblock_{tag}_{}.t16c", std::process::id()));
            T16CModel::write_file(&tmp, ARCH_FLAG_GEMMA4, &[("n0".into(), vec![1.0])], &[gate_res, down_cb]).unwrap();
            let loaded_res = T16CModel::load(&tmp, None, true).unwrap();
            // Keep a tile-major resident `down_proj` copy (`down_res`) for exact native NEON parity checks.
            let down_tile_ref = T16CMat::pack_from_components(
                "model.language_model.layers.0.mlp.down_proj.weight".into(),
                d_out,
                d_in,
                bsize,
                nb,
                is_asym,
                true,
                &sc,
                if is_asym { Some(&bi) } else { None },
                &idx,
                Some(inv_s.clone()),
                Some(lr_g.clone()),
            )
            .unwrap();

            let mut policy = T16CLoadPolicy::default();
            policy.cold_mlp_layers.insert(0);
            let loaded_cold = T16CModel::load_with_policy(&tmp, None, true, &policy).unwrap();
            let _ = std::fs::remove_file(&tmp);

            let gate_cold = loaded_cold.mat("model.language_model.layers.0.mlp.gate_proj.weight").unwrap();
            let down_cold = loaded_cold.mat("model.language_model.layers.0.mlp.down_proj.weight").unwrap();
            let gate_ref = loaded_res.mat("model.language_model.layers.0.mlp.gate_proj.weight").unwrap();
            let down_ref = loaded_res.mat("model.language_model.layers.0.mlp.down_proj.weight").unwrap();
            assert!(gate_cold.is_streamed() && down_cold.is_streamed());
            assert!(gate_cold.resident_bytes() < gate_cold.total_bytes());
            assert!(down_cold.resident_bytes() < down_cold.total_bytes());

            let assert_close = |a: &[f32], b: &[f32], tol: f32, min_cos: f64, label: &str| {
                let (mut dot, mut na, mut nb_acc, mut max_abs) = (0.0f64, 0.0f64, 0.0f64, 0.0f32);
                for (&x, &y) in a.iter().zip(b) {
                    dot += (x as f64) * (y as f64);
                    na += (x as f64) * (x as f64);
                    nb_acc += (y as f64) * (y as f64);
                    max_abs = max_abs.max((x - y).abs());
                }
                let cos = dot / (na.sqrt() * nb_acc.sqrt() + 1e-12);
                assert!(cos > min_cos && max_abs < tol, "{label}: cos={cos} max_abs={max_abs}");
            };

            for &kt in &[1usize, 4usize] {
                let x: Vec<f32> = (0..kt * d_in)
                    .map(|i| (((i * 19 + 5) % 31) as f32 - 15.0) * 0.06)
                    .collect();

                // Row-masked gate_proj sparse slab vs resident (Rust exact + NEON exact)
                let slab_g = gate_cold.read_sparse_slab(Some(&rmask), None).unwrap();
                assert!(
                    slab_g.bytes_read <= (gate_cold.packed_bytes_len as u64 * 55) / 100,
                    "{tag}: expected gate sparse bytes <= 55%, got {} / {}",
                    slab_g.bytes_read,
                    gate_cold.packed_bytes_len
                );
                let (y_g_ref_rust, _) = gate_ref
                    .project_batch_sparse_ext(&x, kt, None, 2, Some(&rmask), None, true, true)
                    .unwrap();
                let (y_g_cold_rust, _) = gate_cold
                    .eval_sparse_slab(&slab_g, &x, kt, None, 2, Some(&rmask), None, true, true)
                    .unwrap();
                assert_close(&y_g_ref_rust, &y_g_cold_rust, 1e-4, 0.99999, &format!("{tag} gate rust k={kt}"));
                if let Some(ref nk) = native_k {
                    // For eff_bits==5, project_batch_sparse falls back to FP64 Rust while eval_sparse_slab
                    // upgrades to native NEON Q8 `gemm_ternary5_tile16_batch` (~0.9998+ vs FP64).
                    let (nat_tol, nat_cos) = if eff == 5 { (2.5e-2, 0.9995) } else { (1e-4, 0.99999) };
                    let (y_g_ref_nat, _) = gate_ref
                        .project_batch_sparse_ext(&x, kt, Some(nk), 2, Some(&rmask), None, true, true)
                        .unwrap();
                    let (y_g_cold_nat, _) = gate_cold
                        .eval_sparse_slab(&slab_g, &x, kt, Some(nk), 2, Some(&rmask), None, true, true)
                        .unwrap();
                    assert_close(&y_g_ref_nat, &y_g_cold_nat, nat_tol, nat_cos, &format!("{tag} gate neon k={kt}"));
                }

                // Col-masked COLBLOCK down_proj sparse slab vs resident (Rust exact + NEON exact)
                let slab_d = down_cold.read_sparse_slab(None, Some(&cmask)).unwrap();
                let exp_half = ((2 * n_tiles) as u64) * (tile16_blk_bytes(bsize, nb, is_asym) as u64);
                assert_eq!(slab_d.bytes_read, exp_half, "{tag}: colblock sparse bytes mismatch");
                let (y_d_ref_rust, _) = down_ref
                    .project_batch_sparse(&x, kt, None, 2, None, Some(&cmask))
                    .unwrap();
                let (y_d_cold_rust, _) = down_cold
                    .eval_sparse_slab(&slab_d, &x, kt, None, 2, None, Some(&cmask), true, false)
                    .unwrap();
                assert_close(&y_d_ref_rust, &y_d_cold_rust, 1e-4, 0.99999, &format!("{tag} down colblock rust k={kt}"));
                if let Some(ref nk) = native_k {
                    let (nat_tol, nat_cos) = if eff == 5 { (2.5e-2, 0.9995) } else { (1e-4, 0.99999) };
                    let (y_d_ref_nat, _) = down_tile_ref
                        .project_batch_sparse(&x, kt, Some(nk), 2, None, Some(&cmask))
                        .unwrap();
                    let (y_d_cold_nat, _) = down_cold
                        .eval_sparse_slab(&slab_d, &x, kt, Some(nk), 2, None, Some(&cmask), true, false)
                        .unwrap();
                    assert_close(&y_d_ref_nat, &y_d_cold_nat, nat_tol, nat_cos, &format!("{tag} down colblock neon k={kt}"));
                }
            }
        }
    }

    #[test]
    fn selective_load_policy_preserves_pinned_vision_and_cold_mlp_parity() {
        let d_out = 32usize;
        let d_in = 128usize;
        let bsize = 64usize;
        let n_blks = d_in / bsize;
        let sc = vec![0.05f32; d_out * n_blks];
        let bi = vec![-0.01f32; d_out * n_blks];
        let idx: Vec<u8> = (0..d_out * d_in).map(|i| (i % 4) as u8).collect();
        let make_m = |name: &str| {
            T16CMat::pack_from_components(
                name.into(),
                d_out,
                d_in,
                bsize,
                2 | T16C_GAUSS_BIT,
                true,
                true,
                &sc,
                Some(&bi),
                &idx,
                None,
                None,
            )
            .unwrap()
        };
        let m_pinned = make_m("model.language_model.layers.0.mlp.gate_proj.weight");
        let m_cold = make_m("model.language_model.layers.6.mlp.gate_proj.weight");
        let m_vis = make_m("model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight");

        let tmp = std::env::temp_dir().join(format!("t16c_selective_{}.t16c", std::process::id()));
        T16CModel::write_file(&tmp, ARCH_FLAG_GEMMA4, &[("norm0".into(), vec![1.0])], &[m_pinned, m_cold, m_vis]).unwrap();

        let ref_m = T16CModel::load(&tmp, None, true).unwrap();
        let mut pol = T16CLoadPolicy {
            defer_vision_fault: true,
            cold_mlp_layers: HashSet::new(),
        };
        pol.cold_mlp_layers.insert(6);
        let sel_m = T16CModel::load_with_policy(&tmp, None, true, &pol).unwrap();
        let _ = std::fs::remove_file(&tmp);

        let pm = sel_m.mat("model.language_model.layers.0.mlp.gate_proj.weight").unwrap();
        let cm = sel_m.mat("model.language_model.layers.6.mlp.gate_proj.weight").unwrap();
        let vm = sel_m.mat("model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight").unwrap();
        assert!(!pm.is_streamed());
        assert!(cm.is_streamed());
        assert!(!vm.is_streamed());

        let x: Vec<f32> = (0..2 * d_in).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
        for (nm, m_sel) in [
            ("model.language_model.layers.0.mlp.gate_proj.weight", pm),
            ("model.language_model.layers.6.mlp.gate_proj.weight", cm),
            ("model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight", vm),
        ] {
            let m_ref = ref_m.mat(nm).unwrap();
            let _ = m_sel.advise_willneed_and_touch();
            let y_ref = m_ref.project_batch(&x, 2, None, 2).unwrap();
            let y_sel = m_sel.project_batch(&x, 2, None, 2).unwrap();
            assert_eq!(y_ref.len(), y_sel.len());
            for (a, b) in y_ref.iter().zip(y_sel.iter()) {
                assert!((a - b).abs() < 1e-6, "{nm}: mismatch {a} vs {b}");
            }
            let _ = m_sel.advise_dontneed();
        }
    }
}
