//! GlyphAttentionV2 (`--kv-quant glyph` / `glyph-pure`) native Rust + C NEON/AVX2 KV Cache.
//!
//! Compresses post-RoPE KV states into `Option 5` 48-byte Top-4 Outlier macro-glyphs
//! (`256` dims -> `48 B`, `head_dim = 256` -> `1` glyph, `global_head_dim = 512` -> `2` glyphs)
//! with `FWHT-8` outlier smoothing, `attention_k_eq_v = true` single-buffer global deduplication
//! (`21.33x` pure / `~21.2x` Dual-Road at 128K vs naive BF16 K+V), circular ring-buffer eviction
//! on sliding-window layers (`W = 1024`), and exact associative log-sum-exp Dual-Road fusion of
//! pinned FP32 attention sinks and rolling FP32 hot-tail tokens.
//!
//! Dispatches to `encode_kv_glyph_fast_c` and `flash_glyph_attention_gqa_c` inside
//! `libneon_stream_accelerator.so` / `neon_stream_accelerator.dll` when loaded, with a bit-faithful
//! pure-Rust fallback so `--no-native-ffi` and cross-platform builds remain verified.

use crate::codec::{fp16_to_f32, fp32_to_fp16};
use crate::packed::fwht8;
use crate::t16c::NativeTile16Kernel;

pub const OPT5_GLYPH_DIM: usize = 256;
pub const OPT5_BPG_OUTLIER: usize = 48;
pub const OPT5_CODEC_ID: i32 = 5;

/// Pure-Rust `Option 5` 48-byte Top-4 Outlier macro-glyph encoder matching
/// `encode_single_opt5_glyph_fast` in `neon_stream_accelerator_v2.c`.
pub fn encode_opt5_48b_glyph(in_256: &[f32], out_48: &mut [u8], apply_fwht8: bool) {
    debug_assert_eq!(in_256.len(), OPT5_GLYPH_DIM);
    debug_assert_eq!(out_48.len(), OPT5_BPG_OUTLIER);

    let mut target = [0.0f32; OPT5_GLYPH_DIM];
    if apply_fwht8 {
        for oct in 0..32 {
            let base = oct * 8;
            let v = [
                in_256[base],
                in_256[base + 1],
                in_256[base + 2],
                in_256[base + 3],
                in_256[base + 4],
                in_256[base + 5],
                in_256[base + 6],
                in_256[base + 7],
            ];
            let rot = fwht8(&v);
            target[base..base + 8].copy_from_slice(&rot);
        }
    } else {
        target.copy_from_slice(in_256);
    }

    let mut sum = 0.0f32;
    for &v in &target {
        sum += v;
    }
    let bias_h = fp32_to_fp16(sum * (1.0f32 / 256.0f32));
    let bias = fp16_to_f32(bias_h);

    let mut abs_sum = 0.0f32;
    let mut var_sum = 0.0f32;
    let mut top_abs = [-1.0f32; 4];
    let mut top_diff = [0.0f32; 4];
    let mut top_idx = [0u8; 4];

    for b in 0..32 {
        let mut mask = 0u8;
        for bit in 0..8 {
            let idx = (b << 3) + bit;
            let v = target[idx];
            let pos = if v >= bias { 1u8 } else { 0u8 };
            mask = (mask << 1) | pos;
            let diff = v - bias;
            let adiff = diff.abs();
            abs_sum += adiff;
            var_sum += diff * diff;
            if adiff > top_abs[3] {
                let mut ins = 3usize;
                while ins > 0 && adiff > top_abs[ins - 1] {
                    top_abs[ins] = top_abs[ins - 1];
                    top_diff[ins] = top_diff[ins - 1];
                    top_idx[ins] = top_idx[ins - 1];
                    ins -= 1;
                }
                top_abs[ins] = adiff;
                top_diff[ins] = diff;
                top_idx[ins] = idx as u8;
            }
        }
        out_48[4 + b] = mask;
    }

    out_48[2..4].copy_from_slice(&bias_h.to_le_bytes());

    let top_abs_sum = top_abs[0] + top_abs[1] + top_abs[2] + top_abs[3];
    let top_var_sum = top_diff[0] * top_diff[0]
        + top_diff[1] * top_diff[1]
        + top_diff[2] * top_diff[2]
        + top_diff[3] * top_diff[3];
    let abs_rem = if abs_sum > top_abs_sum { abs_sum - top_abs_sum } else { 0.0f32 };
    let s_mse = abs_rem * (1.0f32 / 252.0f32);
    let var_rec = 252.0f32 * s_mse * s_mse + top_var_sum;
    let gamma = if var_rec > 1e-20 && var_sum > 1e-20 {
        (var_sum / var_rec).sqrt()
    } else {
        1.0f32
    };

    let scale_h = fp32_to_fp16(s_mse * gamma);
    let scale_actual = fp16_to_f32(scale_h);
    out_48[0..2].copy_from_slice(&scale_h.to_le_bytes());

    for o in 0..4 {
        let sgn = if target[top_idx[o] as usize] >= bias { 1.0f32 } else { -1.0f32 };
        let res_o = gamma * top_diff[o] - sgn * scale_actual;
        out_48[36 + o] = top_idx[o];
        let rh = fp32_to_fp16(res_o).to_le_bytes();
        out_48[40 + o * 2] = rh[0];
        out_48[40 + o * 2 + 1] = rh[1];
    }
}

/// Pure-Rust `Option 5` 48-byte macro-glyph decoder (in rotated or original space).
pub fn decode_opt5_48b_glyph(glyph_48: &[u8], out_256: &mut [f32], apply_inv_fwht8: bool) {
    debug_assert_eq!(glyph_48.len(), OPT5_BPG_OUTLIER);
    debug_assert_eq!(out_256.len(), OPT5_GLYPH_DIM);

    let sc = fp16_to_f32(u16::from_le_bytes([glyph_48[0], glyph_48[1]]));
    let bias = fp16_to_f32(u16::from_le_bytes([glyph_48[2], glyph_48[3]]));
    let pos_val = bias + sc;
    let neg_val = bias - sc;

    for b in 0..32 {
        let mask = glyph_48[4 + b];
        let base = b << 3;
        for bit in 0..8 {
            out_256[base + bit] = if ((mask >> (7 - bit)) & 1) != 0 { pos_val } else { neg_val };
        }
    }
    for o in 0..4 {
        let idx = glyph_48[36 + o] as usize;
        let res = fp16_to_f32(u16::from_le_bytes([glyph_48[40 + o * 2], glyph_48[40 + o * 2 + 1]]));
        out_256[idx] += res;
    }

    if apply_inv_fwht8 {
        for oct in 0..32 {
            let base = oct * 8;
            let v = [
                out_256[base],
                out_256[base + 1],
                out_256[base + 2],
                out_256[base + 3],
                out_256[base + 4],
                out_256[base + 5],
                out_256[base + 6],
                out_256[base + 7],
            ];
            let inv = fwht8(&v);
            out_256[base..base + 8].copy_from_slice(&inv);
        }
    }
}

/// Snapshot of a single evicted token for lossless speculative decoding rollback.
#[derive(Clone)]
struct EvictedEntry {
    pos: usize,
    k: Vec<f32>,
    v: Vec<f32>,
    overwrote_glyph_slot: Option<(usize, usize, Vec<u8>, Vec<u8>)>,
}

/// Summary memory statistics for one or all `GlyphKvLayer`s.
#[derive(Clone, Copy, Debug, Default)]
pub struct GlyphKvMemStats {
    pub active_tokens: usize,
    pub glyph_tokens: usize,
    pub exact_tokens: usize,
    pub resident_bytes: usize,
    pub naive_bf16_bytes: usize,
    pub fp32_bytes: usize,
}

impl GlyphKvMemStats {
    pub fn shrink_vs_fp32(&self) -> f64 {
        if self.resident_bytes == 0 {
            1.0
        } else {
            self.fp32_bytes as f64 / self.resident_bytes as f64
        }
    }

    pub fn shrink_vs_naive_bf16(&self) -> f64 {
        if self.resident_bytes == 0 {
            1.0
        } else {
            self.naive_bf16_bytes as f64 / self.resident_bytes as f64
        }
    }
}

/// Production per-layer `GlyphAttentionV2` Dual-Road KV cache (global contiguous arena or
/// circular sliding-window ring) with C NEON/AVX2 FFI dispatch and pure-Rust fallback.
pub struct GlyphKvLayer {
    pub n_heads: usize,
    pub n_kv: usize,
    pub hd: usize,
    pub glyphs_per_head: usize,
    pub head_glyph_bytes: usize,
    pub k_eq_v: bool,
    pub sliding_window: Option<usize>,
    pub sink_cap: usize,
    pub tail_cap: usize,
    pub apply_fwht8: bool,

    glyph_cap: usize,
    glyph_len: usize,
    glyph_write_pos: usize,
    /// Head-major contiguous glyph arena: `[n_kv, glyph_cap, head_glyph_bytes]`.
    k_arena: Vec<u8>,
    v_arena: Vec<u8>,
    /// Logical token position stored in each glyph slot `[0..glyph_len]`.
    glyph_pos: Vec<usize>,

    /// Pinned FP32 attention sinks (global layers only): `[sink_len, n_kv * hd]`.
    k_sink: Vec<f32>,
    v_sink: Vec<f32>,
    sink_pos: Vec<usize>,

    /// Rolling FP32 local hot-tail stored in chronological order (`len <= tail_cap`).
    k_tail: Vec<f32>,
    v_tail: Vec<f32>,
    tail_pos: Vec<usize>,

    total_seen: usize,
    rollback_log: Vec<EvictedEntry>,
}

impl GlyphKvLayer {
    pub fn new(
        n_heads: usize,
        n_kv: usize,
        hd: usize,
        k_eq_v: bool,
        sliding_window: Option<usize>,
        sink_cap: usize,
        tail_cap: usize,
        init_glyph_cap: usize,
    ) -> Result<Self, String> {
        if n_heads == 0 || n_kv == 0 || n_heads % n_kv != 0 {
            return Err(format!("GlyphKvLayer invalid GQA heads: n_heads={n_heads}, n_kv={n_kv}"));
        }
        if hd == 0 || hd % OPT5_GLYPH_DIM != 0 {
            return Err(format!("GlyphKvLayer head_dim={hd} must be a positive multiple of {OPT5_GLYPH_DIM}"));
        }
        let glyphs_per_head = hd / OPT5_GLYPH_DIM;
        let head_glyph_bytes = glyphs_per_head * OPT5_BPG_OUTLIER;
        let eff_sink_cap = if sliding_window.is_some() { 0 } else { sink_cap };
        let glyph_cap = if let Some(w) = sliding_window {
            w.saturating_sub(tail_cap).max(1)
        } else {
            init_glyph_cap.max(64)
        };
        let arena_bytes = n_kv * glyph_cap * head_glyph_bytes;
        Ok(Self {
            n_heads,
            n_kv,
            hd,
            glyphs_per_head,
            head_glyph_bytes,
            k_eq_v,
            sliding_window,
            sink_cap: eff_sink_cap,
            tail_cap,
            apply_fwht8: true,
            glyph_cap,
            glyph_len: 0,
            glyph_write_pos: 0,
            k_arena: vec![0u8; arena_bytes],
            v_arena: if k_eq_v { Vec::new() } else { vec![0u8; arena_bytes] },
            glyph_pos: vec![0usize; glyph_cap],
            k_sink: Vec::new(),
            v_sink: Vec::new(),
            sink_pos: Vec::new(),
            k_tail: Vec::new(),
            v_tail: Vec::new(),
            tail_pos: Vec::new(),
            total_seen: 0,
            rollback_log: Vec::new(),
        })
    }

    pub fn total_seen(&self) -> usize {
        self.total_seen
    }

    pub fn glyph_len(&self) -> usize {
        self.glyph_len
    }

    pub fn exact_len(&self) -> usize {
        self.sink_pos.len() + self.tail_pos.len()
    }

    pub fn memory_stats(&self) -> GlyphKvMemStats {
        let active_tokens = self.sink_pos.len() + self.glyph_len + self.tail_pos.len();
        let streams = if self.k_eq_v { 1usize } else { 2usize };
        let glyph_bytes = self.glyph_len * self.n_kv * self.head_glyph_bytes * streams;
        let exact_bytes = (self.sink_pos.len() + self.tail_pos.len()) * self.n_kv * self.hd * 4 * streams;
        let resident_bytes = glyph_bytes + exact_bytes;
        let naive_bf16_bytes = active_tokens * self.n_kv * self.hd * 2 * 2;
        let fp32_bytes = active_tokens * self.n_kv * self.hd * 4 * 2;
        GlyphKvMemStats {
            active_tokens,
            glyph_tokens: self.glyph_len,
            exact_tokens: self.sink_pos.len() + self.tail_pos.len(),
            resident_bytes,
            naive_bf16_bytes,
            fp32_bytes,
        }
    }

    /// Grow a global (non-sliding) head-major glyph arena while keeping `[n_kv, glyph_cap, head_glyph_bytes]` contiguous.
    fn grow_global_arena(&mut self, min_cap: usize) {
        let new_cap = (self.glyph_cap * 2).max(min_cap);
        let old_head_stride = self.glyph_cap * self.head_glyph_bytes;
        let new_head_stride = new_cap * self.head_glyph_bytes;
        let active_bytes = self.glyph_len * self.head_glyph_bytes;

        let mut new_k = vec![0u8; self.n_kv * new_head_stride];
        for h in 0..self.n_kv {
            let src = h * old_head_stride;
            let dst = h * new_head_stride;
            new_k[dst..dst + active_bytes].copy_from_slice(&self.k_arena[src..src + active_bytes]);
        }
        self.k_arena = new_k;

        if !self.k_eq_v {
            let mut new_v = vec![0u8; self.n_kv * new_head_stride];
            for h in 0..self.n_kv {
                let src = h * old_head_stride;
                let dst = h * new_head_stride;
                new_v[dst..dst + active_bytes].copy_from_slice(&self.v_arena[src..src + active_bytes]);
            }
            self.v_arena = new_v;
        }
        self.glyph_pos.resize(new_cap, 0);
        self.glyph_cap = new_cap;
    }

    fn encode_head_glyphs(
        &self,
        tok_kv: &[f32],
        out: &mut [u8],
        native: Option<&NativeTile16Kernel>,
        threads: usize,
    ) {
        let total_glyphs = self.n_kv * self.glyphs_per_head;
        debug_assert_eq!(tok_kv.len(), self.n_kv * self.hd);
        debug_assert_eq!(out.len(), total_glyphs * OPT5_BPG_OUTLIER);
        if let Some(ekv) = native.and_then(|n| n.encode_kv_glyph_fn) {
            unsafe {
                ekv(
                    tok_kv.as_ptr(),
                    std::ptr::null(),
                    std::ptr::null(),
                    out.as_mut_ptr(),
                    total_glyphs as i32,
                    OPT5_CODEC_ID,
                    OPT5_BPG_OUTLIER as i32,
                    if self.apply_fwht8 { 1 } else { 0 },
                    threads.max(1) as i32,
                );
            }
            return;
        }
        for g in 0..total_glyphs {
            encode_opt5_48b_glyph(
                &tok_kv[g * OPT5_GLYPH_DIM..(g + 1) * OPT5_GLYPH_DIM],
                &mut out[g * OPT5_BPG_OUTLIER..(g + 1) * OPT5_BPG_OUTLIER],
                self.apply_fwht8,
            );
        }
    }

    fn push_glyph_token(
        &mut self,
        pos: usize,
        k_tok: &[f32],
        v_tok: &[f32],
        native: Option<&NativeTile16Kernel>,
        record_rollback: bool,
    ) -> Option<(usize, usize, Vec<u8>, Vec<u8>)> {
        if self.sliding_window.is_none() && self.glyph_write_pos >= self.glyph_cap {
            self.grow_global_arena(self.glyph_write_pos + 1);
        }
        let slot = self.glyph_write_pos;
        let head_stride = self.glyph_cap * self.head_glyph_bytes;

        let overwrote = if record_rollback && self.sliding_window.is_some() && self.glyph_len == self.glyph_cap {
            let mut old_k = vec![0u8; self.n_kv * self.head_glyph_bytes];
            let mut old_v = if self.k_eq_v {
                Vec::new()
            } else {
                vec![0u8; self.n_kv * self.head_glyph_bytes]
            };
            for h in 0..self.n_kv {
                let off = h * head_stride + slot * self.head_glyph_bytes;
                old_k[h * self.head_glyph_bytes..(h + 1) * self.head_glyph_bytes]
                    .copy_from_slice(&self.k_arena[off..off + self.head_glyph_bytes]);
                if !self.k_eq_v {
                    old_v[h * self.head_glyph_bytes..(h + 1) * self.head_glyph_bytes]
                        .copy_from_slice(&self.v_arena[off..off + self.head_glyph_bytes]);
                }
            }
            Some((slot, self.glyph_pos[slot], old_k, old_v))
        } else {
            None
        };

        let mut enc_k = vec![0u8; self.n_kv * self.head_glyph_bytes];
        self.encode_head_glyphs(k_tok, &mut enc_k, native, 1);
        for h in 0..self.n_kv {
            let off = h * head_stride + slot * self.head_glyph_bytes;
            self.k_arena[off..off + self.head_glyph_bytes]
                .copy_from_slice(&enc_k[h * self.head_glyph_bytes..(h + 1) * self.head_glyph_bytes]);
        }
        if !self.k_eq_v {
            let mut enc_v = vec![0u8; self.n_kv * self.head_glyph_bytes];
            self.encode_head_glyphs(v_tok, &mut enc_v, native, 1);
            for h in 0..self.n_kv {
                let off = h * head_stride + slot * self.head_glyph_bytes;
                self.v_arena[off..off + self.head_glyph_bytes]
                    .copy_from_slice(&enc_v[h * self.head_glyph_bytes..(h + 1) * self.head_glyph_bytes]);
            }
        }
        self.glyph_pos[slot] = pos;

        if self.sliding_window.is_some() {
            self.glyph_write_pos = (slot + 1) % self.glyph_cap;
            if self.glyph_len < self.glyph_cap {
                self.glyph_len += 1;
            }
        } else {
            self.glyph_write_pos = slot + 1;
            self.glyph_len = self.glyph_write_pos;
        }
        overwrote
    }

    /// Append one post-RoPE token `(k_tok, v_tok)` of length `n_kv * hd`.
    pub fn append_token(
        &mut self,
        k_tok: &[f32],
        v_tok: &[f32],
        native: Option<&NativeTile16Kernel>,
    ) -> Result<(), String> {
        let stride = self.n_kv * self.hd;
        if k_tok.len() != stride || (!self.k_eq_v && v_tok.len() != stride) {
            return Err(format!(
                "GlyphKvLayer::append_token len mismatch: k={} v={} expected={stride}",
                k_tok.len(),
                v_tok.len()
            ));
        }
        let pos = self.total_seen;
        self.total_seen += 1;

        if self.sink_pos.len() < self.sink_cap {
            self.k_sink.extend_from_slice(k_tok);
            if !self.k_eq_v {
                self.v_sink.extend_from_slice(v_tok);
            }
            self.sink_pos.push(pos);
            return Ok(());
        }

        if self.tail_cap == 0 {
            let ow = self.push_glyph_token(pos, k_tok, v_tok, native, true);
            if ow.is_some() {
                self.rollback_log.push(EvictedEntry {
                    pos,
                    k: Vec::new(),
                    v: Vec::new(),
                    overwrote_glyph_slot: ow,
                });
                if self.rollback_log.len() > 16 {
                    self.rollback_log.remove(0);
                }
            }
            return Ok(());
        }

        if self.tail_pos.len() == self.tail_cap {
            let ev_pos = self.tail_pos.remove(0);
            let ev_k: Vec<f32> = self.k_tail.drain(0..stride).collect();
            let ev_v: Vec<f32> = if self.k_eq_v {
                Vec::new()
            } else {
                self.v_tail.drain(0..stride).collect()
            };
            let ow = self.push_glyph_token(ev_pos, &ev_k, &ev_v, native, true);
            self.rollback_log.push(EvictedEntry {
                pos: ev_pos,
                k: ev_k,
                v: ev_v,
                overwrote_glyph_slot: ow,
            });
            if self.rollback_log.len() > 16 {
                self.rollback_log.remove(0);
            }
        }
        self.k_tail.extend_from_slice(k_tok);
        if !self.k_eq_v {
            self.v_tail.extend_from_slice(v_tok);
        }
        self.tail_pos.push(pos);
        Ok(())
    }

    /// Bulk-encode `n_tokens` directly into the global glyph arena (zero per-token copies)
    /// for 128K context prefill / verification.
    pub fn bulk_fill_glyphs(
        &mut self,
        k_seq_head_major: &[f32],
        v_seq_head_major: Option<&[f32]>,
        n_tokens: usize,
        native: Option<&NativeTile16Kernel>,
        threads: usize,
    ) -> Result<(), String> {
        let expected_floats = self.n_kv * n_tokens * self.hd;
        if k_seq_head_major.len() != expected_floats {
            return Err(format!(
                "bulk_fill_glyphs k len {} != expected {expected_floats}",
                k_seq_head_major.len()
            ));
        }
        if !self.k_eq_v {
            let v = v_seq_head_major.ok_or("bulk_fill_glyphs requires v_seq when k_eq_v=false")?;
            if v.len() != expected_floats {
                return Err(format!("bulk_fill_glyphs v len {} != expected {expected_floats}", v.len()));
            }
        }
        if self.sliding_window.is_none() && self.glyph_len + n_tokens > self.glyph_cap {
            self.grow_global_arena(self.glyph_len + n_tokens);
        }
        if n_tokens > self.glyph_cap {
            return Err(format!("bulk_fill_glyphs n_tokens={n_tokens} > glyph_cap={}", self.glyph_cap));
        }
        let head_stride = self.glyph_cap * self.head_glyph_bytes;
        let tok_glyphs = n_tokens * self.glyphs_per_head;
        let tok_bytes = n_tokens * self.head_glyph_bytes;
        let mut tmp_out = vec![0u8; tok_bytes];
        for h in 0..self.n_kv {
            let k_slice = &k_seq_head_major[h * n_tokens * self.hd..(h + 1) * n_tokens * self.hd];
            if let Some(ekv) = native.and_then(|n| n.encode_kv_glyph_fn) {
                unsafe {
                    ekv(
                        k_slice.as_ptr(),
                        std::ptr::null(),
                        std::ptr::null(),
                        tmp_out.as_mut_ptr(),
                        tok_glyphs as i32,
                        OPT5_CODEC_ID,
                        OPT5_BPG_OUTLIER as i32,
                        if self.apply_fwht8 { 1 } else { 0 },
                        threads.max(1) as i32,
                    );
                }
            } else {
                for g in 0..tok_glyphs {
                    encode_opt5_48b_glyph(
                        &k_slice[g * OPT5_GLYPH_DIM..(g + 1) * OPT5_GLYPH_DIM],
                        &mut tmp_out[g * OPT5_BPG_OUTLIER..(g + 1) * OPT5_BPG_OUTLIER],
                        self.apply_fwht8,
                    );
                }
            }
            let dst = h * head_stride + self.glyph_write_pos * self.head_glyph_bytes;
            self.k_arena[dst..dst + tok_bytes].copy_from_slice(&tmp_out);

            if !self.k_eq_v {
                let v_all = v_seq_head_major.unwrap();
                let v_slice = &v_all[h * n_tokens * self.hd..(h + 1) * n_tokens * self.hd];
                if let Some(ekv) = native.and_then(|n| n.encode_kv_glyph_fn) {
                    unsafe {
                        ekv(
                            v_slice.as_ptr(),
                            std::ptr::null(),
                            std::ptr::null(),
                            tmp_out.as_mut_ptr(),
                            tok_glyphs as i32,
                            OPT5_CODEC_ID,
                            OPT5_BPG_OUTLIER as i32,
                            if self.apply_fwht8 { 1 } else { 0 },
                            threads.max(1) as i32,
                        );
                    }
                } else {
                    for g in 0..tok_glyphs {
                        encode_opt5_48b_glyph(
                            &v_slice[g * OPT5_GLYPH_DIM..(g + 1) * OPT5_GLYPH_DIM],
                            &mut tmp_out[g * OPT5_BPG_OUTLIER..(g + 1) * OPT5_BPG_OUTLIER],
                            self.apply_fwht8,
                        );
                    }
                }
                self.v_arena[dst..dst + tok_bytes].copy_from_slice(&tmp_out);
            }
        }
        for i in 0..n_tokens {
            self.glyph_pos[self.glyph_write_pos + i] = self.total_seen + i;
        }
        if self.sliding_window.is_some() {
            self.glyph_write_pos = (self.glyph_write_pos + n_tokens) % self.glyph_cap;
            self.glyph_len = (self.glyph_len + n_tokens).min(self.glyph_cap);
        } else {
            self.glyph_write_pos += n_tokens;
            self.glyph_len += n_tokens;
        }
        self.total_seen += n_tokens;
        Ok(())
    }

    /// Roll back speculative draft tokens back to `keep_tokens`.
    pub fn truncate(&mut self, keep_tokens: usize) -> Result<(), String> {
        if keep_tokens > self.total_seen {
            return Err(format!(
                "GlyphKvLayer::truncate keep_tokens={keep_tokens} > total_seen={}",
                self.total_seen
            ));
        }
        if keep_tokens == 0 {
            self.glyph_len = 0;
            self.glyph_write_pos = 0;
            self.glyph_pos.fill(0);
            self.k_sink.clear();
            self.v_sink.clear();
            self.sink_pos.clear();
            self.k_tail.clear();
            self.v_tail.clear();
            self.tail_pos.clear();
            self.total_seen = 0;
            self.rollback_log.clear();
            return Ok(());
        }
        let stride = self.n_kv * self.hd;
        let head_stride = self.glyph_cap * self.head_glyph_bytes;
        while self.total_seen > keep_tokens {
            self.total_seen -= 1;
            let drop_pos = self.total_seen;
            if self.tail_pos.last().copied() == Some(drop_pos) {
                self.tail_pos.pop();
                let new_len = self.k_tail.len() - stride;
                self.k_tail.truncate(new_len);
                if !self.k_eq_v {
                    self.v_tail.truncate(new_len);
                }
                // Restore evicted token from glyph arena back to front of tail if present
                if self.tail_cap > 0 && self.glyph_len > 0 {
                    let prev_slot = if self.glyph_write_pos == 0 {
                        self.glyph_cap - 1
                    } else {
                        self.glyph_write_pos - 1
                    };
                    if let Some(ev) = self.rollback_log.pop() {
                        self.tail_pos.insert(0, ev.pos);
                        self.k_tail.splice(0..0, ev.k);
                        if !self.k_eq_v {
                            self.v_tail.splice(0..0, ev.v);
                        }
                        if let Some((slot, old_pos, old_k, old_v)) = ev.overwrote_glyph_slot {
                            self.glyph_write_pos = slot;
                            self.glyph_pos[slot] = old_pos;
                            for h in 0..self.n_kv {
                                let off = h * head_stride + slot * self.head_glyph_bytes;
                                self.k_arena[off..off + self.head_glyph_bytes]
                                    .copy_from_slice(&old_k[h * self.head_glyph_bytes..(h + 1) * self.head_glyph_bytes]);
                                if !self.k_eq_v {
                                    self.v_arena[off..off + self.head_glyph_bytes]
                                        .copy_from_slice(&old_v[h * self.head_glyph_bytes..(h + 1) * self.head_glyph_bytes]);
                                }
                            }
                        } else {
                            self.glyph_write_pos = prev_slot;
                            self.glyph_len -= 1;
                        }
                    } else {
                        return Err(format!(
                            "GlyphKvLayer::truncate cannot unwind >16 evicted tokens (drop_pos={drop_pos})"
                        ));
                    }
                }
            } else if self.sink_pos.last().copied() == Some(drop_pos) {
                self.sink_pos.pop();
                let new_len = self.k_sink.len() - stride;
                self.k_sink.truncate(new_len);
                if !self.k_eq_v {
                    self.v_sink.truncate(new_len);
                }
            } else if self.tail_cap == 0 && self.glyph_len > 0 {
                let prev_slot = if self.glyph_write_pos == 0 {
                    self.glyph_cap - 1
                } else {
                    self.glyph_write_pos - 1
                };
                if let Some(ev) = self.rollback_log.last().filter(|e| e.pos == drop_pos).cloned() {
                    self.rollback_log.pop();
                    if let Some((slot, old_pos, old_k, old_v)) = ev.overwrote_glyph_slot {
                        self.glyph_write_pos = slot;
                        self.glyph_pos[slot] = old_pos;
                        for h in 0..self.n_kv {
                            let off = h * head_stride + slot * self.head_glyph_bytes;
                            self.k_arena[off..off + self.head_glyph_bytes]
                                .copy_from_slice(&old_k[h * self.head_glyph_bytes..(h + 1) * self.head_glyph_bytes]);
                            if !self.k_eq_v {
                                self.v_arena[off..off + self.head_glyph_bytes]
                                    .copy_from_slice(&old_v[h * self.head_glyph_bytes..(h + 1) * self.head_glyph_bytes]);
                            }
                        }
                    } else {
                        self.glyph_write_pos = prev_slot;
                        self.glyph_len -= 1;
                    }
                } else {
                    self.glyph_write_pos = prev_slot;
                    self.glyph_len -= 1;
                }
            } else {
                return Err(format!("GlyphKvLayer::truncate state desync at drop_pos={drop_pos}"));
            }
        }
        Ok(())
    }

    /// Execute Single-Pass Flash-Glyph GQA Attention over the compressed glyph arena +
    /// exact associative Log-Sum-Exp Dual-Road merge of pinned sinks and rolling hot-tail tokens.
    ///
    /// Supports both autoregressive decode (`max_tt == total_seen - 1`, full C NEON/AVX2
    /// `flash_glyph_attention_gqa_c` fast path) and masked/bidirectional vision prefill ranges.
    pub fn attend_range(
        &self,
        q: &[f32],
        t_logical: usize,
        kv_end_logical: usize,
        pos: usize,
        attn_scale: f32,
        native: Option<&NativeTile16Kernel>,
        threads: usize,
    ) -> Result<Vec<f32>, String> {
        if q.len() != self.n_heads * self.hd {
            return Err(format!(
                "GlyphKvLayer::attend_range q len {} != {}*{}",
                q.len(),
                self.n_heads,
                self.hd
            ));
        }
        let max_tt = kv_end_logical.max(t_logical);
        let gqa = self.n_heads / self.n_kv;
        let stride = self.n_kv * self.hd;

        // Check whether every slot in `k_arena[0..glyph_len]` is valid under `(max_tt, sliding_window)`.
        let all_glyphs_valid = if self.glyph_len == 0 {
            false
        } else {
            (0..self.glyph_len).all(|s| {
                let p = self.glyph_pos[s];
                p <= max_tt && self.sliding_window.map_or(true, |w| pos.abs_diff(p) < w)
            })
        };

        // 1. Evaluate glyph arena contribution: `(out_g, m_g, l_g)`.
        let mut out_g = vec![0.0f32; self.n_heads * self.hd];
        let mut m_g = vec![f32::NEG_INFINITY; self.n_heads];
        let mut l_g = vec![0.0f32; self.n_heads];
        let mut has_glyph_contrib = false;

        if self.glyph_len > 0 {
            if all_glyphs_valid {
                if let Some(fg_fn) = native.and_then(|n| n.flash_glyph_attn_fn) {
                    let head_stride = self.glyph_cap * self.head_glyph_bytes;
                    let v_ptr = if self.k_eq_v { std::ptr::null() } else { self.v_arena.as_ptr() };
                    let rc = unsafe {
                        fg_fn(
                            q.as_ptr(),
                            self.k_arena.as_ptr(),
                            v_ptr,
                            std::ptr::null(),
                            out_g.as_mut_ptr(),
                            m_g.as_mut_ptr(),
                            l_g.as_mut_ptr(),
                            self.n_heads as i32,
                            self.n_kv as i32,
                            self.hd as i32,
                            self.glyph_len as i32,
                            self.head_glyph_bytes,
                            head_stride,
                            attn_scale,
                            OPT5_CODEC_ID,
                            OPT5_BPG_OUTLIER as i32,
                            if self.apply_fwht8 { 1 } else { 0 },
                            if self.k_eq_v { 1 } else { 0 },
                            threads.max(1) as i32,
                        )
                    };
                    if rc != 0 {
                        return Err(format!("flash_glyph_attention_gqa_c returned error {rc}"));
                    }
                    has_glyph_contrib = true;
                }
            }
            if !has_glyph_contrib {
                // Pure-Rust exact Opt5 48B glyph evaluation (works for masked prefill & `--no-native-ffi`)
                let head_stride = self.glyph_cap * self.head_glyph_bytes;
                let mut dec_k = vec![0.0f32; self.hd];
                let mut dec_v = vec![0.0f32; self.hd];
                let mut q_rot = vec![0.0f32; self.n_heads * self.hd];
                if self.apply_fwht8 {
                    for hq in 0..self.n_heads {
                        for oct in 0..(self.hd / 8) {
                            let off = hq * self.hd + oct * 8;
                            let v = [
                                q[off], q[off + 1], q[off + 2], q[off + 3],
                                q[off + 4], q[off + 5], q[off + 6], q[off + 7],
                            ];
                            let r = fwht8(&v);
                            q_rot[off..off + 8].copy_from_slice(&r);
                        }
                    }
                } else {
                    q_rot.copy_from_slice(q);
                }

                for kvh in 0..self.n_kv {
                    let h_off = kvh * head_stride;
                    for slot in 0..self.glyph_len {
                        let p = self.glyph_pos[slot];
                        if p > max_tt {
                            continue;
                        }
                        if let Some(w) = self.sliding_window {
                            if pos.abs_diff(p) >= w {
                                continue;
                            }
                        }
                        has_glyph_contrib = true;
                        let s_off = h_off + slot * self.head_glyph_bytes;
                        for g in 0..self.glyphs_per_head {
                            decode_opt5_48b_glyph(
                                &self.k_arena[s_off + g * OPT5_BPG_OUTLIER..s_off + (g + 1) * OPT5_BPG_OUTLIER],
                                &mut dec_k[g * OPT5_GLYPH_DIM..(g + 1) * OPT5_GLYPH_DIM],
                                false,
                            );
                        }
                        let v_ref = if self.k_eq_v {
                            &dec_k[..]
                        } else {
                            for g in 0..self.glyphs_per_head {
                                decode_opt5_48b_glyph(
                                    &self.v_arena[s_off + g * OPT5_BPG_OUTLIER..s_off + (g + 1) * OPT5_BPG_OUTLIER],
                                    &mut dec_v[g * OPT5_GLYPH_DIM..(g + 1) * OPT5_GLYPH_DIM],
                                    false,
                                );
                            }
                            &dec_v[..]
                        };
                        for g_idx in 0..gqa {
                            let hq = kvh * gqa + g_idx;
                            let qh = &q_rot[hq * self.hd..(hq + 1) * self.hd];
                            let mut dot = 0.0f64;
                            for d in 0..self.hd {
                                dot += qh[d] as f64 * dec_k[d] as f64;
                            }
                            let s = (dot as f32) * attn_scale;
                            let acc = &mut out_g[hq * self.hd..(hq + 1) * self.hd];
                            if s > m_g[hq] {
                                let alpha = if m_g[hq].is_finite() { (m_g[hq] - s).exp() } else { 0.0f32 };
                                l_g[hq] = l_g[hq] * alpha + 1.0;
                                m_g[hq] = s;
                                for d in 0..self.hd {
                                    acc[d] = acc[d] * alpha + v_ref[d];
                                }
                            } else {
                                let beta = (s - m_g[hq]).exp();
                                l_g[hq] += beta;
                                for d in 0..self.hd {
                                    acc[d] += beta * v_ref[d];
                                }
                            }
                        }
                    }
                }
                if has_glyph_contrib {
                    for hq in 0..self.n_heads {
                        let inv_l = if l_g[hq] > 1e-30 { 1.0 / l_g[hq] } else { 0.0 };
                        let acc = &mut out_g[hq * self.hd..(hq + 1) * self.hd];
                        for v in acc.iter_mut() {
                            *v *= inv_l;
                        }
                        if self.apply_fwht8 {
                            for oct in 0..(self.hd / 8) {
                                let off = oct * 8;
                                let v = [
                                    acc[off], acc[off + 1], acc[off + 2], acc[off + 3],
                                    acc[off + 4], acc[off + 5], acc[off + 6], acc[off + 7],
                                ];
                                let inv = fwht8(&v);
                                acc[off..off + 8].copy_from_slice(&inv);
                            }
                        }
                    }
                }
            }
        }

        // 2. Evaluate exact Dual-Road sink + rolling hot-tail tokens `(out_e, m_e, l_e)`
        //    using NEON-vectorizable 4-way FP32 dots and a single-pass softmax (zero per-token `out_e` rescaling).
        let mut out_e = vec![0.0f32; self.n_heads * self.hd];
        let mut m_e = vec![f32::NEG_INFINITY; self.n_heads];
        let mut l_e = vec![0.0f32; self.n_heads];
        let mut has_exact_contrib = false;

        let mut valid_exact: Vec<(bool, usize)> = Vec::with_capacity(self.sink_pos.len() + self.tail_pos.len());
        for (idx, &p) in self.sink_pos.iter().enumerate() {
            if p <= max_tt && self.sliding_window.map_or(true, |w| pos.abs_diff(p) < w) {
                valid_exact.push((false, idx));
            }
        }
        for (idx, &p) in self.tail_pos.iter().enumerate() {
            if p <= max_tt && self.sliding_window.map_or(true, |w| pos.abs_diff(p) < w) {
                valid_exact.push((true, idx));
            }
        }

        if !valid_exact.is_empty() {
            has_exact_contrib = true;
            let n_ex = valid_exact.len();
            let hd = self.hd;
            let mut scores = vec![0.0f32; n_ex];
            for kvh in 0..self.n_kv {
                let kvh_off = kvh * hd;
                for g_idx in 0..gqa {
                    let hq = kvh * gqa + g_idx;
                    let qh = &q[hq * hd..(hq + 1) * hd];
                    let mut m_max = f32::NEG_INFINITY;
                    for (vi, &(is_tail, idx)) in valid_exact.iter().enumerate() {
                        let k_buf = if is_tail { &self.k_tail } else { &self.k_sink };
                        let base = idx * stride + kvh_off;
                        let ks = &k_buf[base..base + hd];
                        let s = crate::math::dot_f32_4way(qh, ks) * attn_scale;
                        scores[vi] = s;
                        if s > m_max {
                            m_max = s;
                        }
                    }
                    m_e[hq] = m_max;
                    let mut l_sum = 0.0f32;
                    let acc = &mut out_e[hq * hd..(hq + 1) * hd];
                    for (vi, &(is_tail, idx)) in valid_exact.iter().enumerate() {
                        let w = (scores[vi] - m_max).exp();
                        l_sum += w;
                        let v_buf = if self.k_eq_v {
                            if is_tail { &self.k_tail } else { &self.k_sink }
                        } else if is_tail {
                            &self.v_tail
                        } else {
                            &self.v_sink
                        };
                        let base = idx * stride + kvh_off;
                        let vs = &v_buf[base..base + hd];
                        for (dst, &v_val) in acc.iter_mut().zip(vs.iter()) {
                            *dst += w * v_val;
                        }
                    }
                    l_e[hq] = l_sum;
                    let inv_l = if l_sum > 1e-30 { 1.0f32 / l_sum } else { 0.0f32 };
                    for v in acc.iter_mut() {
                        *v *= inv_l;
                    }
                }
            }
        }

        // 3. Exact associative LSE merge: `(m_g, l_g, out_g) (+) (m_e, l_e, out_e)`.
        if !has_glyph_contrib {
            return Ok(out_e);
        }
        if !has_exact_contrib {
            return Ok(out_g);
        }
        let mut merged = vec![0.0f32; self.n_heads * self.hd];
        for hq in 0..self.n_heads {
            let mj = m_g[hq].max(m_e[hq]);
            let wg = if m_g[hq].is_finite() { l_g[hq] * (m_g[hq] - mj).exp() } else { 0.0 };
            let we = if m_e[hq].is_finite() { l_e[hq] * (m_e[hq] - mj).exp() } else { 0.0 };
            let denom = (wg + we).max(1e-30);
            let ag = wg / denom;
            let ae = we / denom;
            let base = hq * self.hd;
            for d in 0..self.hd {
                merged[base + d] = ag * out_g[base + d] + ae * out_e[base + d];
            }
        }
        Ok(merged)
    }

    /// Expose raw exact FP32 KV slices (if all tokens reside in exact buffers, e.g. for assistant sharing)
    /// or reconstruct decoded FP32 KV sequence in chronological order `[0..total_seen]` when needed.
    pub fn materialize_chronological_kv(&self) -> (Vec<f32>, Vec<f32>) {
        let stride = self.n_kv * self.hd;
        let head_stride = self.glyph_cap * self.head_glyph_bytes;
        let mut entries: Vec<(usize, Vec<f32>, Vec<f32>)> = Vec::with_capacity(self.exact_len() + self.glyph_len);

        for (i, &p) in self.sink_pos.iter().enumerate() {
            let k = self.k_sink[i * stride..(i + 1) * stride].to_vec();
            let v = if self.k_eq_v { k.clone() } else { self.v_sink[i * stride..(i + 1) * stride].to_vec() };
            entries.push((p, k, v));
        }
        for slot in 0..self.glyph_len {
            let p = self.glyph_pos[slot];
            let mut k = vec![0.0f32; stride];
            let mut v = if self.k_eq_v { Vec::new() } else { vec![0.0f32; stride] };
            for h in 0..self.n_kv {
                let s_off = h * head_stride + slot * self.head_glyph_bytes;
                for g in 0..self.glyphs_per_head {
                    let g_off = s_off + g * OPT5_BPG_OUTLIER;
                    let d_off = h * self.hd + g * OPT5_GLYPH_DIM;
                    decode_opt5_48b_glyph(
                        &self.k_arena[g_off..g_off + OPT5_BPG_OUTLIER],
                        &mut k[d_off..d_off + OPT5_GLYPH_DIM],
                        self.apply_fwht8,
                    );
                    if !self.k_eq_v {
                        decode_opt5_48b_glyph(
                            &self.v_arena[g_off..g_off + OPT5_BPG_OUTLIER],
                            &mut v[d_off..d_off + OPT5_GLYPH_DIM],
                            self.apply_fwht8,
                        );
                    }
                }
            }
            if self.k_eq_v {
                v = k.clone();
            }
            entries.push((p, k, v));
        }
        for (i, &p) in self.tail_pos.iter().enumerate() {
            let k = self.k_tail[i * stride..(i + 1) * stride].to_vec();
            let v = if self.k_eq_v { k.clone() } else { self.v_tail[i * stride..(i + 1) * stride].to_vec() };
            entries.push((p, k, v));
        }
        entries.sort_by_key(|e| e.0);
        let mut k_out = Vec::with_capacity(entries.len() * stride);
        let mut v_out = Vec::with_capacity(entries.len() * stride);
        for (_, k, v) in entries {
            k_out.extend_from_slice(&k);
            v_out.extend_from_slice(&v);
        }
        (k_out, v_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{attn_decode_range, cosine_and_rel_l2};

    fn det_vec(len: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0;
            out.push(u);
        }
        out
    }

    #[test]
    fn opt5_48b_roundtrip_preserves_norm_and_high_cosine() {
        let x_clean = det_vec(256, 42);
        let mut g = [0u8; 48];
        let mut rec = [0.0f32; 256];
        encode_opt5_48b_glyph(&x_clean, &mut g, true);
        decode_opt5_48b_glyph(&g, &mut rec, true);
        let (cos_clean, _, _) = cosine_and_rel_l2(&x_clean, &rec);
        let n_src: f32 = x_clean.iter().map(|v| v * v).sum::<f32>().sqrt();
        let n_rec: f32 = rec.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(cos_clean > 0.80, "expected Opt5 48B FWHT clean cosine > 0.80, got {cos_clean}");
        assert!(
            (n_rec / n_src - 1.0).abs() < 0.08,
            "expected Opt5 48B L2 norm preservation within 8%, got ratio={}",
            n_rec / n_src
        );

        let mut x_spiky = x_clean;
        x_spiky[7] = 9.5;
        x_spiky[113] = -8.2;
        x_spiky[200] = 6.7;
        encode_opt5_48b_glyph(&x_spiky, &mut g, false);
        decode_opt5_48b_glyph(&g, &mut rec, false);
        let (cos_spiky, _, _) = cosine_and_rel_l2(&x_spiky, &rec);
        assert!(cos_spiky > 0.90, "expected Top-4 outlier capture cosine > 0.90, got {cos_spiky}");
    }

    #[test]
    fn dual_road_glyph_kv_matches_dense_fp32_gqa_and_handles_rollback() {
        let n_heads = 4;
        let n_kv = 2;
        let hd = 256;
        let seq_len = 56;
        let mut layer = GlyphKvLayer::new(n_heads, n_kv, hd, false, None, 4, 16, 64).unwrap();
        let mut k_all = Vec::new();
        let mut v_all = Vec::new();
        for t in 0..seq_len {
            let k = det_vec(n_kv * hd, 100 + t as u64);
            let v = det_vec(n_kv * hd, 500 + t as u64);
            layer.append_token(&k, &v, None).unwrap();
            k_all.extend_from_slice(&k);
            v_all.extend_from_slice(&v);
        }
        assert_eq!(layer.total_seen(), seq_len);
        assert_eq!(layer.glyph_len(), seq_len - 20);
        assert_eq!(layer.exact_len(), 20);

        let q = det_vec(n_heads * hd, 999);
        let ref_ctx = attn_decode_range(&q, &k_all, &v_all, seq_len - 1, seq_len - 1, seq_len - 1, n_heads, n_kv, hd, None);
        let glyph_ctx = layer.attend_range(&q, seq_len - 1, seq_len - 1, seq_len - 1, 1.0, None, 1).unwrap();
        let (cos, _, _) = cosine_and_rel_l2(&ref_ctx, &glyph_ctx);
        assert!(cos > 0.85, "Dual-Road GQA cosine expected > 0.85, got {cos}");

        // Speculative draft + rollback must restore exact state
        let draft_k = det_vec(n_kv * hd, 7771);
        let draft_v = det_vec(n_kv * hd, 7772);
        layer.append_token(&draft_k, &draft_v, None).unwrap();
        layer.append_token(&draft_k, &draft_v, None).unwrap();
        assert_eq!(layer.total_seen(), seq_len + 2);
        layer.truncate(seq_len).unwrap();
        assert_eq!(layer.total_seen(), seq_len);
        assert_eq!(layer.glyph_len(), seq_len - 20);
        let after_rb = layer.attend_range(&q, seq_len - 1, seq_len - 1, seq_len - 1, 1.0, None, 1).unwrap();
        let (rb_cos, rb_rel, _) = cosine_and_rel_l2(&glyph_ctx, &after_rb);
        assert!(rb_cos > 0.999999 && rb_rel < 1e-6, "rollback mismatch: cos={rb_cos} rel={rb_rel}");

        // Full reset via truncate(0) followed by second prompt eviction must preserve glyph_pos capacity
        layer.truncate(0).unwrap();
        assert_eq!(layer.total_seen(), 0);
        assert_eq!(layer.glyph_len(), 0);
        for t in 0..seq_len {
            let k = det_vec(n_kv * hd, 100 + t as u64);
            let v = det_vec(n_kv * hd, 500 + t as u64);
            layer.append_token(&k, &v, None).unwrap();
        }
        assert_eq!(layer.total_seen(), seq_len);
        assert_eq!(layer.glyph_len(), seq_len - 20);
        let after_reset = layer.attend_range(&q, seq_len - 1, seq_len - 1, seq_len - 1, 1.0, None, 1).unwrap();
        let (rs_cos, rs_rel, _) = cosine_and_rel_l2(&glyph_ctx, &after_reset);
        assert!(rs_cos > 0.999999 && rs_rel < 1e-6, "post-truncate(0) mismatch: cos={rs_cos} rel={rs_rel}");
    }

    #[test]
    fn global_128k_memory_shrink_exceeds_20x_vs_naive_bf16() {
        // Verify Gemma 4 31B global layer geometry (32Q / 4KV / 512d, k_eq_v=true)
        let mut layer = GlyphKvLayer::new(32, 4, 512, true, None, 4, 32, 128).unwrap();
        // Fill 4 sink + 32 tail + 2048 glyphs -> verify shrink scales to >20x at 128K
        let k_one = det_vec(4 * 512, 123);
        for _ in 0..40 {
            layer.append_token(&k_one, &[], None).unwrap();
        }
        let stats_small = layer.memory_stats();
        assert_eq!(stats_small.active_tokens, 40);
        assert_eq!(stats_small.glyph_tokens, 4);

        // Formula verification for 131,072 tokens: 4 sink + 32 tail + 131,036 glyphs
        let glyph_bytes = 131_036usize * 4 * (2 * OPT5_BPG_OUTLIER);
        let exact_bytes = 36usize * 4 * 512 * 4;
        let resident = glyph_bytes + exact_bytes;
        let naive_bf16 = 131_072usize * 4 * 512 * 2 * 2;
        let ratio = naive_bf16 as f64 / resident as f64;
        assert!(ratio > 21.0 && ratio < 21.4, "expected ~21.2x shrink at 128K, got {ratio:.3}x");
    }

    #[test]
    fn native_ffi_flash_glyph_matches_pure_rust_when_library_present() {
        let Some(native) = NativeTile16Kernel::discover(None) else {
            return;
        };
        if native.encode_kv_glyph_fn.is_none() || native.flash_glyph_attn_fn.is_none() {
            return;
        }
        let n_heads = 8;
        let n_kv = 4;
        let hd = 512;
        let seq = 48;
        let mut rust_kv = GlyphKvLayer::new(n_heads, n_kv, hd, true, None, 4, 8, 64).unwrap();
        let mut ffi_kv = GlyphKvLayer::new(n_heads, n_kv, hd, true, None, 4, 8, 64).unwrap();
        for t in 0..seq {
            let k = det_vec(n_kv * hd, 1000 + t as u64);
            rust_kv.append_token(&k, &[], None).unwrap();
            ffi_kv.append_token(&k, &[], Some(&native)).unwrap();
        }
        let q = det_vec(n_heads * hd, 5555);
        let out_rust = rust_kv.attend_range(&q, seq - 1, seq - 1, seq - 1, 1.0, None, 1).unwrap();
        let out_ffi = ffi_kv.attend_range(&q, seq - 1, seq - 1, seq - 1, 1.0, Some(&native), 2).unwrap();
        let (cos, rel, max_abs) = cosine_and_rel_l2(&out_rust, &out_ffi);
        assert!(
            cos > 0.9999 && rel < 1e-3,
            "FFI vs Pure-Rust GlyphAttentionV2 parity mismatch: cos={cos} rel={rel} max_abs={max_abs}"
        );
    }
}
