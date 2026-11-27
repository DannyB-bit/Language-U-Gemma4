//! Unified Input-to-Output Meaning-Pointer Mirror (`va_mirror`) over Sparse Linux VA Space
//! & ARMv8 Top-Byte-Ignore (`TBI [63:56]`) Hardware Pointers.
//!
//! Translates BOTH incoming token/causal input states (Layer `0`) AND intermediate hidden states
//! (`Layers 8, 20, 32`) AND target output tokens into a shared `FWHT-64` orthogonal hypercube:
//!   - Address in VA space = multi-probe pigeonhole LSH bands over `code64` inside a `32 GiB`
//!     `mmap(MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE)` hardware page-table arena (costing
//!     `0 B` of physical DRAM for untouched slots, faulting `4 KiB` only on populated pages).
//!   - Top byte `[63:56]` of every VA lookup/entry pointer = `TbiCognitiveTag` (`tier | c6d_quad |
//!     support_confidence`), ignored by ARM64 EL0 hardware MMU during pointer dereference.
//!   - Bidirectional mirror cell (`16 B`): packs predicted full 60-layer next-token ID, output
//!     mirror signature, confidence support counter, and full `64-bit` structural verification key.

use crate::cuneiform_tbi::{tbi_extract_tag, tbi_tag_ptr, tbi_untag_ptr};
use crate::t16c::fwht_block_inplace;
#[cfg(not(unix))]
use std::collections::HashMap;
use std::collections::HashSet;
#[cfg(unix)]
use std::ffi::c_void;

pub const MIRROR_DEPTHS: [usize; 4] = [0, 8, 20, 32];
const BUCKET_SLOTS: usize = 8;
const BUCKET_BYTES: usize = BUCKET_SLOTS * std::mem::size_of::<VaMirrorSlot>(); // 128 B (2 cache lines)
const NUM_LSH_LANES: usize = 4; // 4 x 16-bit bands => 100% Pigeonhole recall for Hamming <= 3!
const VALID_FLAG_BIT: u64 = 1u64 << 47;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaMirrorSlot {
    /// `[63:56]` TBI tag | `[55:48]` support count | `[47]` valid bit | `[46:24]` out_tok | `[23:0]` out_mirror_sig24
    pub word0: u64,
    /// Full 64-bit orthogonal `FWHT-64` SimHash verification key.
    pub key64: u64,
}

impl VaMirrorSlot {
    #[inline]
    pub fn pack(tbi_tag: u8, support: u8, out_tok: u32, out_sig24: u32, key64: u64) -> Self {
        let word0 = ((tbi_tag as u64) << 56)
            | ((support.max(1) as u64) << 48)
            | VALID_FLAG_BIT
            | (((out_tok as u64) & 0x7F_FFFF) << 24)
            | ((out_sig24 as u64) & 0xFF_FFFF);
        Self { word0, key64 }
    }

    #[inline]
    pub fn is_valid(self) -> bool {
        (self.word0 & VALID_FLAG_BIT) != 0
    }

    #[inline]
    pub fn tbi_tag(self) -> u8 {
        (self.word0 >> 56) as u8
    }

    #[inline]
    pub fn support(self) -> u8 {
        ((self.word0 >> 48) & 0xFF) as u8
    }

    #[inline]
    pub fn out_tok(self) -> u32 {
        ((self.word0 >> 24) & 0x7F_FFFF) as u32
    }

    #[inline]
    pub fn out_sig24(self) -> u32 {
        (self.word0 & 0xFF_FFFF) as u32
    }

    #[inline]
    pub fn with_support(mut self, support: u8) -> Self {
        self.word0 = (self.word0 & !(0xFFu64 << 48)) | ((support as u64) << 48);
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VaMirrorCode {
    pub code64: u64,
    pub sig24: u32,
    pub cuneiform_u8: u8,
}

impl VaMirrorCode {
    /// Pack a top-byte TBI descriptor: `[7:6]` depth_tier (0..3) | `[5:2]` support/conf | `[1:0]` cuneiform_quad.
    #[inline]
    pub fn tbi_tag(self, depth_tier: u8, support_4b: u8) -> u8 {
        ((depth_tier & 0x3) << 6) | ((support_4b & 0xF) << 2) | (self.cuneiform_u8 & 0x3)
    }
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic orthogonal `FWHT-64` + 64-axis Rademacher SimHash mirror encoder shared across
/// Input (`L=0`), Hidden (`L=8,20,32`), and Target Output spaces.
pub struct MirrorProjector {
    pub d_model: usize,
    /// `[d_model]` XOR sign masks per dimension packing bits `0..64`: stored column-major `[d_model]` `u64`
    /// where bit `b` of `col_masks[j]` gives the sign `{-1, +1}` of projection axis `b` on dimension `j`.
    col_masks: Vec<u64>,
}

impl MirrorProjector {
    pub fn new(d_model: usize) -> Self {
        let mut col_masks = Vec::with_capacity(d_model);
        for j in 0..d_model {
            let w = splitmix64(0x4D49_5252_4F52_3131u64 ^ ((j as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)));
            col_masks.push(w);
        }
        Self { d_model, col_masks }
    }

    /// Encode any `d_model` state vector via normalized `FWHT-64` rotation + 64 orthogonal sign hyperplanes.
    pub fn encode_state(&self, vec_d: &[f32]) -> Result<VaMirrorCode, String> {
        if vec_d.len() != self.d_model {
            return Err(format!("MirrorProjector len {} != d_model {}", vec_d.len(), self.d_model));
        }
        let mut rot = vec_d.to_vec();
        // L2-normalize first so extreme outliers stay numerically bounded before FWHT.
        let mut sq = 0.0f64;
        for &v in &rot {
            sq += (v as f64) * (v as f64);
        }
        let inv_n = if sq > 1e-24 { (1.0 / sq.sqrt()) as f32 } else { 0.0 };
        if inv_n > 0.0 {
            for v in &mut rot {
                *v *= inv_n;
            }
        }
        let bsz = if self.d_model >= 64 && self.d_model % 64 == 0 {
            64
        } else if self.d_model >= 8 && self.d_model % 8 == 0 {
            8
        } else {
            0
        };
        if bsz > 0 {
            fwht_block_inplace(&mut rot, bsz)?;
        }

        let mut acc = [0.0f32; 64];
        for j in 0..self.d_model {
            let v = rot[j];
            if v == 0.0 {
                continue;
            }
            let m = self.col_masks[j];
            for k in 0..64 {
                if ((m >> k) & 1) != 0 {
                    acc[k] += v;
                } else {
                    acc[k] -= v;
                }
            }
        }

        let mut code64 = 0u64;
        for k in 0..64 {
            if acc[k] >= 0.0 {
                code64 |= 1u64 << k;
            }
        }
        // Derive Cuneiform-6D radical byte from the first 6 orthogonal sign octants (`[-/+]^6`).
        let mut c6d = 0u8;
        for q in 0..6 {
            if acc[q] >= 0.0 {
                c6d |= 1u8 << q;
            }
        }
        let fold = (code64 ^ (code64 >> 24) ^ (code64 >> 48)) as u32;
        Ok(VaMirrorCode {
            code64,
            sig24: fold & 0xFF_FFFF,
            cuneiform_u8: c6d,
        })
    }

    /// Convert raw input embedding sequence `[seq_len, d_model]` into causal multi-scale mirror vectors
    /// (`0.65 * e[t] + 0.25 * e[t-1] + 0.10 * e[t-2]`) + their corresponding Layer-0 `VaMirrorCode`s
    /// so inputs enter the runtime already translated into the VA pointer mirror space.
    pub fn convert_input_sequence(
        &self,
        embeds: &[f32],
        seq_len: usize,
    ) -> Result<(Vec<f32>, Vec<VaMirrorCode>), String> {
        let d = self.d_model;
        if embeds.len() != seq_len * d {
            return Err(format!("convert_input_sequence len {} != {}*{d}", embeds.len(), seq_len));
        }
        let mut mirrored_vecs = vec![0.0f32; seq_len * d];
        let mut codes = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let dst = &mut mirrored_vecs[t * d..(t + 1) * d];
            let e0 = &embeds[t * d..(t + 1) * d];
            if t == 0 {
                dst.copy_from_slice(e0);
            } else if t == 1 {
                let e1 = &embeds[0..d];
                for j in 0..d {
                    dst[j] = 0.72 * e0[j] + 0.28 * e1[j];
                }
            } else {
                let e1 = &embeds[(t - 1) * d..t * d];
                let e2 = &embeds[(t - 2) * d..(t - 1) * d];
                for j in 0..d {
                    dst[j] = 0.65 * e0[j] + 0.25 * e1[j] + 0.10 * e2[j];
                }
            }
            codes.push(self.encode_state(dst)?);
        }
        Ok((mirrored_vecs, codes))
    }
}

#[cfg(unix)]
mod sys_va {
    use std::ffi::c_void;
    pub const PROT_READ: i32 = 0x1;
    pub const PROT_WRITE: i32 = 0x2;
    pub const MAP_PRIVATE: i32 = 0x2;
    pub const MAP_ANONYMOUS: i32 = 0x20;
    #[cfg(target_os = "linux")]
    pub const MAP_NORESERVE: i32 = 0x4000;
    #[cfg(not(target_os = "linux"))]
    pub const MAP_NORESERVE: i32 = 0;
    pub const MAP_FAILED: *mut c_void = !0usize as *mut c_void;

    extern "C" {
        pub fn mmap(addr: *mut c_void, length: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, length: usize) -> i32;
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VaMirrorHit {
    pub out_tok: u32,
    pub out_sig24: u32,
    pub hamming_dist: u32,
    pub support: u8,
    pub tbi_tag_read: u8,
    pub tbi_va_ptr: u64,
    pub lane: usize,
}

/// Hardware-backed Sparse Virtual Address Associative Memory (`mmap(MAP_NORESERVE)` on Linux).
pub struct VaMirrorArena {
    #[cfg(unix)]
    base_ptr: *mut c_void,
    #[cfg(not(unix))]
    emu_buckets: HashMap<usize, [VaMirrorSlot; BUCKET_SLOTS]>,
    pub arena_bytes: usize,
    lane_bytes: usize,
    buckets_per_lane: usize,
    bucket_mask: usize,
    pub hardware_mmap: bool,
    pub hardware_tbi: bool,
    touched_pages_4k: HashSet<usize>,
    pub total_writes: usize,
}

#[cfg(unix)]
unsafe impl Send for VaMirrorArena {}
#[cfg(unix)]
unsafe impl Sync for VaMirrorArena {}

impl VaMirrorArena {
    /// Reserve up to `32 GiB` (configurable candidates down to `256 MiB`) of zero-RAM sparse VA space.
    pub fn reserve_default() -> Result<Self, String> {
        #[cfg(target_pointer_width = "64")]
        let candidates: &[usize] = &[
            1usize << 35, // 32.0 GiB VA (4 x 8 GiB lanes)
            1usize << 34, // 16.0 GiB VA
            1usize << 32, // 4.0 GiB VA
            1usize << 28, // 256.0 MiB VA
        ];
        #[cfg(not(target_pointer_width = "64"))]
        let candidates: &[usize] = &[1usize << 28];

        #[cfg(unix)]
        {
            for &sz in candidates {
                let flags = sys_va::MAP_PRIVATE | sys_va::MAP_ANONYMOUS | sys_va::MAP_NORESERVE;
                let ptr = unsafe {
                    sys_va::mmap(
                        std::ptr::null_mut(),
                        sz,
                        sys_va::PROT_READ | sys_va::PROT_WRITE,
                        flags,
                        -1,
                        0,
                    )
                };
                if ptr != sys_va::MAP_FAILED && !ptr.is_null() {
                    let lane_bytes = sz / NUM_LSH_LANES;
                    let buckets_per_lane = lane_bytes / BUCKET_BYTES;
                    debug_assert!(buckets_per_lane.is_power_of_two());
                    return Ok(Self {
                        base_ptr: ptr,
                        arena_bytes: sz,
                        lane_bytes,
                        buckets_per_lane,
                        bucket_mask: buckets_per_lane - 1,
                        hardware_mmap: true,
                        hardware_tbi: cfg!(all(target_arch = "aarch64", target_os = "linux")),
                        touched_pages_4k: HashSet::new(),
                        total_writes: 0,
                    });
                }
            }
            Err("VaMirrorArena: mmap(MAP_ANONYMOUS | MAP_NORESERVE) failed across all VA sizes".into())
        }
        #[cfg(not(unix))]
        {
            let sz = candidates[0];
            let lane_bytes = sz / NUM_LSH_LANES;
            let buckets_per_lane = lane_bytes / BUCKET_BYTES;
            Ok(Self {
                emu_buckets: HashMap::new(),
                arena_bytes: sz,
                lane_bytes,
                buckets_per_lane,
                bucket_mask: buckets_per_lane - 1,
                hardware_mmap: false,
                hardware_tbi: false,
                touched_pages_4k: HashSet::new(),
                total_writes: 0,
            })
        }
    }

    pub fn va_reserved_gib(&self) -> f64 {
        (self.arena_bytes as f64) / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn faulted_pages_4k(&self) -> usize {
        self.touched_pages_4k.len()
    }

    pub fn faulted_ram_kb(&self) -> usize {
        self.touched_pages_4k.len() * 4
    }

    #[inline]
    fn band16(code64: u64, lane: usize) -> usize {
        ((code64 >> (lane * 16)) & 0xFFFF) as usize
    }

    /// Map `(lane, depth_tier, band16)` bijectively across the sparse VA lane so every 16-bit
    /// pigeonhole bucket gets its own cache-line pair dispersed across the `32 GiB` address space.
    #[inline]
    fn bucket_byte_offset(&self, lane: usize, depth_tier: u8, code64: u64) -> usize {
        let raw_idx = (((depth_tier as usize) & 0x3) << 16) | Self::band16(code64, lane);
        let sparse_b = raw_idx.wrapping_mul(0x9E37_79B9) & self.bucket_mask;
        debug_assert!(sparse_b < self.buckets_per_lane);
        let off = lane * self.lane_bytes + sparse_b * BUCKET_BYTES;
        debug_assert!(off + BUCKET_BYTES <= self.arena_bytes);
        off
    }

    #[inline]
    fn read_bucket(&self, byte_off: usize, tbi_tag: u8) -> ([VaMirrorSlot; BUCKET_SLOTS], u64) {
        #[cfg(unix)]
        unsafe {
            let raw_va = (self.base_ptr as *const u8).add(byte_off) as usize as u64;
            let tagged_va = tbi_tag_ptr(raw_va, tbi_tag);
            // On ARM64 Linux (`TBI`), hardware MMU ignores bits `[63:56]` directly on dereference!
            let eff_ptr = if cfg!(all(target_arch = "aarch64", target_os = "linux")) {
                tagged_va as usize as *const [VaMirrorSlot; BUCKET_SLOTS]
            } else {
                tbi_untag_ptr(tagged_va) as usize as *const [VaMirrorSlot; BUCKET_SLOTS]
            };
            (std::ptr::read(eff_ptr), tagged_va)
        }
        #[cfg(not(unix))]
        {
            let tagged_va = tbi_tag_ptr(byte_off as u64, tbi_tag);
            let b = self
                .emu_buckets
                .get(&byte_off)
                .copied()
                .unwrap_or([VaMirrorSlot::default(); BUCKET_SLOTS]);
            (b, tagged_va)
        }
    }

    #[inline]
    fn write_bucket(&mut self, byte_off: usize, tbi_tag: u8, slots: &[VaMirrorSlot; BUCKET_SLOTS]) -> u64 {
        self.touched_pages_4k.insert(byte_off / 4096);
        self.touched_pages_4k.insert((byte_off + BUCKET_BYTES - 1) / 4096);
        #[cfg(unix)]
        unsafe {
            let raw_va = (self.base_ptr as *mut u8).add(byte_off) as usize as u64;
            let tagged_va = tbi_tag_ptr(raw_va, tbi_tag);
            let eff_ptr = if cfg!(all(target_arch = "aarch64", target_os = "linux")) {
                tagged_va as usize as *mut [VaMirrorSlot; BUCKET_SLOTS]
            } else {
                tbi_untag_ptr(tagged_va) as usize as *mut [VaMirrorSlot; BUCKET_SLOTS]
            };
            std::ptr::write(eff_ptr, *slots);
            tagged_va
        }
        #[cfg(not(unix))]
        {
            self.emu_buckets.insert(byte_off, *slots);
            tbi_tag_ptr(byte_off as u64, tbi_tag)
        }
    }

    /// Write an Input/Hidden -> Target Token + Output Mirror association across all 4 LSH probe lanes.
    pub fn insert(&mut self, depth_tier: u8, in_code: VaMirrorCode, out_tok: u32, out_sig24: u32) -> u64 {
        let mut last_va = 0u64;
        for lane in 0..NUM_LSH_LANES {
            let off = self.bucket_byte_offset(lane, depth_tier, in_code.code64);
            let tag = in_code.tbi_tag(depth_tier, 1);
            let (mut bkt, _) = self.read_bucket(off, tag);
            let mut placed = false;
            for sl in &mut bkt {
                if sl.is_valid() && sl.key64 == in_code.code64 && sl.out_tok() == out_tok {
                    let sup = sl.support().saturating_add(1);
                    *sl = sl.with_support(sup);
                    placed = true;
                    break;
                }
            }
            if !placed {
                let mut target_idx = None;
                let mut min_sup = u8::MAX;
                for (si, sl) in bkt.iter().enumerate() {
                    if !sl.is_valid() {
                        target_idx = Some(si);
                        break;
                    }
                    if sl.support() < min_sup {
                        min_sup = sl.support();
                        target_idx = Some(si);
                    }
                }
                if let Some(ti) = target_idx {
                    bkt[ti] = VaMirrorSlot::pack(tag, 1, out_tok, out_sig24, in_code.code64);
                }
            }
            last_va = self.write_bucket(off, tag, &bkt);
        }
        self.total_writes += 1;
        last_va
    }

    /// Probe all 4 LSH lanes in the `32 GiB` VA arena, dereferencing each candidate via its TBI pointer
    /// and returning the closest semantic neighbor within `max_hamming` bits (breaking ties by support).
    pub fn query(&self, depth_tier: u8, query_code: VaMirrorCode, max_hamming: u32) -> Option<VaMirrorHit> {
        let mut best: Option<VaMirrorHit> = None;
        let query_tag = query_code.tbi_tag(depth_tier, 1);
        for lane in 0..NUM_LSH_LANES {
            let off = self.bucket_byte_offset(lane, depth_tier, query_code.code64);
            // Skip clean zero pages quickly on Unix? Wait: on Linux `mmap(MAP_NORESERVE)` reading an
            // untouched page returns zeros without allocating physical write pages, or we can check our
            // `touched_pages_4k` guard FIRST so querying an empty bucket doesn't even read a zero page!
            if !self.touched_pages_4k.contains(&(off / 4096)) {
                continue;
            }
            let (bkt, tagged_va) = self.read_bucket(off, query_tag);
            for (si, &sl) in bkt.iter().enumerate() {
                if !sl.is_valid() {
                    continue;
                }
                let dist = (sl.key64 ^ query_code.code64).count_ones();
                if dist <= max_hamming {
                    let slot_va = tbi_tag_ptr(
                        tbi_untag_ptr(tagged_va) + (si * std::mem::size_of::<VaMirrorSlot>()) as u64,
                        tbi_extract_tag(tagged_va),
                    );
                    let replace = match best {
                        None => true,
                        Some(ref cur) => dist < cur.hamming_dist || (dist == cur.hamming_dist && sl.support() > cur.support),
                    };
                    if replace {
                        best = Some(VaMirrorHit {
                            out_tok: sl.out_tok(),
                            out_sig24: sl.out_sig24(),
                            hamming_dist: dist,
                            support: sl.support(),
                            tbi_tag_read: sl.tbi_tag(),
                            tbi_va_ptr: slot_va,
                            lane,
                        });
                    }
                }
            }
        }
        best
    }
}

impl Drop for VaMirrorArena {
    fn drop(&mut self) {
        #[cfg(unix)]
        if self.hardware_mmap && !self.base_ptr.is_null() && self.arena_bytes > 0 {
            unsafe {
                let _ = sys_va::munmap(self.base_ptr, self.arena_bytes);
            }
            self.base_ptr = std::ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn va_mirror_reserves_32gb_zero_ram_converts_inputs_and_preserves_lsh_locality() {
        let mut arena = VaMirrorArena::reserve_default().expect("reserve VA arena");
        assert!(arena.arena_bytes >= (1usize << 28));
        assert_eq!(arena.faulted_pages_4k(), 0);

        let proj = MirrorProjector::new(256);
        let tok0: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.13).sin()).collect();
        let tok1: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.29).cos()).collect();
        let mut embeds = Vec::with_capacity(512);
        embeds.extend_from_slice(&tok0);
        embeds.extend_from_slice(&tok1);
        let (_mvecs, mcodes) = proj.convert_input_sequence(&embeds, 2).expect("convert inputs");
        assert_ne!(mcodes[0].code64, mcodes[1].code64);

        let out_code = proj.encode_state(&tok1).expect("out mirror");
        let ptr_written = arena.insert(0, mcodes[1], 12345, out_code.sig24);
        assert_eq!(tbi_extract_tag(ptr_written), mcodes[1].tbi_tag(0, 1));
        assert!(arena.faulted_pages_4k() > 0 && arena.faulted_pages_4k() <= 8);

        // Exact query hits with dist == 0
        let hit0 = arena.query(0, mcodes[1], 0).expect("exact VA mirror hit");
        assert_eq!(hit0.out_tok, 12345);
        assert_eq!(hit0.out_sig24, out_code.sig24);
        assert_eq!(hit0.hamming_dist, 0);

        // 2-bit perturbed query must STILL hit with 100% guarantee via 4x16-bit pigeonhole LSH!
        let mut noisy = mcodes[1];
        noisy.code64 ^= (1u64 << 3) | (1u64 << 39);
        let hit2 = arena.query(0, noisy, 4).expect("2-bit noisy pigeonhole LSH hit");
        assert_eq!(hit2.out_tok, 12345);
        assert_eq!(hit2.hamming_dist, 2);

        // Orthogonal unrelated query must fail closed
        let mut ortho = mcodes[1];
        ortho.code64 ^= 0x5555_AAAA_3333_CCCC;
        assert!(arena.query(0, ortho, 6).is_none());
    }
}
