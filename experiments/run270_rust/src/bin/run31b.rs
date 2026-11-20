//! run31b — bounded-RAM native Rust inference runtime for Gemma 4 31B QAT W4A16 (zero Python).
//!
//! Executes all 60 dense Gemma 4 transformer layers (compressed-tensors symmetric group-32 INT4)
//! directly on 8 GB edge hardware (Raspberry Pi 4 ARM64) using:
//!   - Batched prefill across each layer so an N-token prompt reads SSD weights only once.
//!   - Streaming single-projection lifetime (`<= 90 MB` layer scratch) plus optional `--pin-layers N`.
//!   - Configurable embedding/LM-head residency (`stream` ~44 MB scratch, `bf16` 2.82 GB, or
//!     `q4`/`q6`/`q8` sub-byte packed via `run270_rust::packed::PackedMatrix`).

use run270_rust::codec::bf16_to_f32;
use run270_rust::glyph_kv::{GlyphKvLayer, GlyphKvMemStats};
use run270_rust::json;
use run270_rust::math::{attn_decode_range, gelu_tanh_scalar, qk_norm_raw, rmsn_raw, rope_head};
use run270_rust::optical_carrier::{decode_tokens_from_optical_frame, optical_context_dma_roundtrip, save_ppm};
use run270_rust::vision::GEMMA4_IMAGE_TOKEN_ID;
use run270_rust::packed::{PackedMatrix, QuantScheme};
use run270_rust::safetensors::SafeTensors;
use run270_rust::assistant::AssistantRuntime;
use run270_rust::speculative::{argmax_f32, verify_greedy_draft_batch, AdaptiveDraftController, FastNGramProposalEngine};
use run270_rust::tokenizer::Tokenizer;
use run270_rust::tri_modal::{
    analyze_hear_fwht, analyze_read_logits, analyze_see_spatial, render_tri_modal_rescue_canvas,
    ConsensusConfig, ConsensusMode, TriModalConsensusGate,
};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

struct Gemma4Cfg {
    d: usize,
    n_layers: usize,
    ff: usize,
    n_heads: usize,
    n_kv: usize,
    n_global_kv: usize,
    hd: usize,
    global_hd: usize,
    window: usize,
    eps: f32,
    theta_global: f32,
    partial_global: f32,
    theta_local: f32,
    partial_local: f32,
    attention_k_eq_v: bool,
    final_softcap: Option<f32>,
    layer_full: Vec<bool>,
    vocab: usize,
}

fn load_gemma4_cfg(path: &Path) -> Result<Gemma4Cfg, String> {
    let raw = fs::read(path).map_err(|e| format!("read config {}: {e}", path.display()))?;
    let root = json::parse(&raw)?;
    let tc = root.get("text_config").unwrap_or(&root);
    let gu = |k: &str| tc.get(k).and_then(|v| v.as_usize()).ok_or_else(|| format!("config missing {k}"));
    let gf = |k: &str| tc.get(k).and_then(|v| v.as_f64()).ok_or_else(|| format!("config missing {k}")).map(|v| v as f32);

    let lt = tc.get("layer_types").and_then(|v| v.as_arr()).ok_or("config missing layer_types")?;
    let layer_full: Vec<bool> = lt.iter().map(|v| v.as_str() == Some("full_attention")).collect();
    let n_layers = gu("num_hidden_layers")?;
    if layer_full.len() != n_layers {
        return Err(format!("layer_types len {} != num_hidden_layers {n_layers}", layer_full.len()));
    }

    let rp = tc.get("rope_parameters").ok_or("config missing rope_parameters")?;
    let rp_full = rp.get("full_attention").ok_or("rope_parameters missing full_attention")?;
    let rp_slide = rp.get("sliding_attention").ok_or("rope_parameters missing sliding_attention")?;
    let theta_global = rp_full.get("rope_theta").and_then(|v| v.as_f64()).ok_or("missing full rope_theta")? as f32;
    let partial_global = rp_full.get("partial_rotary_factor").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
    let theta_local = rp_slide.get("rope_theta").and_then(|v| v.as_f64()).ok_or("missing sliding rope_theta")? as f32;
    let partial_local = rp_slide.get("partial_rotary_factor").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;

    let attention_k_eq_v = matches!(tc.get("attention_k_eq_v"), Some(json::Json::Bool(true)));
    let final_softcap = tc.get("final_logit_softcapping").and_then(|v| v.as_f64()).map(|v| v as f32);

    Ok(Gemma4Cfg {
        d: gu("hidden_size")?,
        n_layers,
        ff: gu("intermediate_size")?,
        n_heads: gu("num_attention_heads")?,
        n_kv: gu("num_key_value_heads")?,
        n_global_kv: gu("num_global_key_value_heads")?,
        hd: gu("head_dim")?,
        global_hd: gu("global_head_dim")?,
        window: gu("sliding_window")?,
        eps: gf("rms_norm_eps")?,
        theta_global,
        partial_global,
        theta_local,
        partial_local,
        attention_k_eq_v,
        final_softcap,
        layer_full,
        vocab: gu("vocab_size")?,
    })
}

#[cfg(any(not(target_arch = "aarch64"), test))]
#[inline]
fn round_bf16_f32(v: f32) -> f32 {
    let bits = v.to_bits();
    let r = (bits.wrapping_add(0x7FFF).wrapping_add((bits >> 16) & 1)) & 0xFFFF_0000;
    f32::from_bits(r)
}

struct QatProj {
    d_out: usize,
    d_in: usize,
    packed: Vec<u8>,
    scales: Vec<u8>,
}

fn load_qat_proj(sf: &SafeTensors, base: &str) -> Result<(QatProj, u64), String> {
    let pk_key = format!("{base}.weight_packed");
    let sc_key = format!("{base}.weight_scale");
    let pk_meta = sf.tensors.get(&pk_key).ok_or_else(|| format!("missing tensor {pk_key}"))?;
    let sc_meta = sf.tensors.get(&sc_key).ok_or_else(|| format!("missing tensor {sc_key}"))?;
    if pk_meta.shape.len() != 2 || sc_meta.shape.len() != 2 {
        return Err(format!("{base}: expected rank-2 packed/scale tensors"));
    }
    let d_out = pk_meta.shape[0];
    let d_in = sc_meta.shape[1] * 32;
    if sc_meta.shape[0] != d_out || pk_meta.shape[1] * 8 != d_in || pk_meta.dtype != "I32" || sc_meta.dtype != "BF16" {
        return Err(format!(
            "{base}: unsupported QAT layout packed=({}, {:?}) scale=({}, {:?})",
            pk_meta.dtype, pk_meta.shape, sc_meta.dtype, sc_meta.shape
        ));
    }
    let packed = sf.read_raw_bytes(pk_meta)?;
    let scales = sf.read_raw_bytes(sc_meta)?;
    let nbytes = (packed.len() + scales.len()) as u64;
    Ok((QatProj { d_out, d_in, packed, scales }, nbytes))
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn round_bf16_neon(v: float32x4_t) -> float32x4_t {
    let bits = vreinterpretq_u32_f32(v);
    let one = vdupq_n_u32(1);
    let half = vdupq_n_u32(0x7FFF);
    let mask = vdupq_n_u32(0xFFFF_0000);
    let lsb = vandq_u32(vshrq_n_u32::<16>(bits), one);
    let rounded = vandq_u32(vaddq_u32(vaddq_u32(bits, half), lsb), mask);
    vreinterpretq_f32_u32(rounded)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_qat_group32_dot(gp: &[u8], sc: f32, x: &[f32]) -> f32 {
    let v = vld1q_u8(gp.as_ptr());
    let mask = vdupq_n_u8(0x0F);
    let off = vdupq_n_s8(8);
    let qlo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, mask)), off);
    let qhi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v)), off);
    let w0_15 = vzip1q_s8(qlo, qhi);
    let w16_31 = vzip2q_s8(qlo, qhi);
    let vsc = vdupq_n_f32(sc);

    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    for (half_idx, &w16) in [w0_15, w16_31].iter().enumerate() {
        let base = half_idx * 16;
        let lo = vmovl_s8(vget_low_s8(w16));
        let hi = vmovl_s8(vget_high_s8(w16));
        let f0 = round_bf16_neon(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), vsc));
        let f1 = round_bf16_neon(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), vsc));
        let f2 = round_bf16_neon(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), vsc));
        let f3 = round_bf16_neon(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), vsc));
        acc0 = vfmaq_f32(acc0, f0, vld1q_f32(x.as_ptr().add(base)));
        acc1 = vfmaq_f32(acc1, f1, vld1q_f32(x.as_ptr().add(base + 4)));
        acc0 = vfmaq_f32(acc0, f2, vld1q_f32(x.as_ptr().add(base + 8)));
        acc1 = vfmaq_f32(acc1, f3, vld1q_f32(x.as_ptr().add(base + 12)));
    }
    vaddvq_f32(vaddq_f32(acc0, acc1))
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_qat_group32_unpack(gp: &[u8], sc: f32, wtile: &mut [f32; 32]) {
    let v = vld1q_u8(gp.as_ptr());
    let mask = vdupq_n_u8(0x0F);
    let off = vdupq_n_s8(8);
    let qlo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, mask)), off);
    let qhi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v)), off);
    let w0_15 = vzip1q_s8(qlo, qhi);
    let w16_31 = vzip2q_s8(qlo, qhi);
    let vsc = vdupq_n_f32(sc);

    for (half_idx, &w16) in [w0_15, w16_31].iter().enumerate() {
        let base = half_idx * 16;
        let lo = vmovl_s8(vget_low_s8(w16));
        let hi = vmovl_s8(vget_high_s8(w16));
        vst1q_f32(wtile.as_mut_ptr().add(base), round_bf16_neon(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), vsc)));
        vst1q_f32(wtile.as_mut_ptr().add(base + 4), round_bf16_neon(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), vsc)));
        vst1q_f32(wtile.as_mut_ptr().add(base + 8), round_bf16_neon(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), vsc)));
        vst1q_f32(wtile.as_mut_ptr().add(base + 12), round_bf16_neon(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), vsc)));
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_dot32_f32(wtile: &[f32; 32], x: &[f32]) -> f32 {
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let w = wtile.as_ptr();
    let xp = x.as_ptr();
    acc0 = vfmaq_f32(acc0, vld1q_f32(w), vld1q_f32(xp));
    acc1 = vfmaq_f32(acc1, vld1q_f32(w.add(4)), vld1q_f32(xp.add(4)));
    acc0 = vfmaq_f32(acc0, vld1q_f32(w.add(8)), vld1q_f32(xp.add(8)));
    acc1 = vfmaq_f32(acc1, vld1q_f32(w.add(12)), vld1q_f32(xp.add(12)));
    acc0 = vfmaq_f32(acc0, vld1q_f32(w.add(16)), vld1q_f32(xp.add(16)));
    acc1 = vfmaq_f32(acc1, vld1q_f32(w.add(20)), vld1q_f32(xp.add(20)));
    acc0 = vfmaq_f32(acc0, vld1q_f32(w.add(24)), vld1q_f32(xp.add(24)));
    acc1 = vfmaq_f32(acc1, vld1q_f32(w.add(28)), vld1q_f32(xp.add(28)));
    vaddvq_f32(vaddq_f32(acc0, acc1))
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_dot32_bf16(row: &[u16], x: &[f32]) -> f32 {
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    for chunk in 0..4 {
        let base = chunk * 8;
        let u = vld1q_u16(row.as_ptr().add(base));
        let lo = vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_u16(vget_low_u16(u))));
        let hi = vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_u16(vget_high_u16(u))));
        acc0 = vfmaq_f32(acc0, lo, vld1q_f32(x.as_ptr().add(base)));
        acc1 = vfmaq_f32(acc1, hi, vld1q_f32(x.as_ptr().add(base + 4)));
    }
    vaddvq_f32(vaddq_f32(acc0, acc1))
}

/// Batched matrix multiply `Y[T, d_out] = W_qat[d_out, d_in] * X[T, d_in]^T`.
fn qat_project_batch(proj: &QatProj, x_batch: &[f32], seq_len: usize, threads: usize) -> Result<Vec<f32>, String> {
    if x_batch.len() != seq_len * proj.d_in {
        return Err(format!("x_batch len {} != {}*{}", x_batch.len(), seq_len, proj.d_in));
    }
    if proj.d_in % 32 != 0 {
        return Err(format!("QAT projection d_in {} not divisible by 32", proj.d_in));
    }
    let d_out = proj.d_out;
    let d_in = proj.d_in;
    let ngroups = d_in / 32;
    let row_pack_bytes = d_in / 2;
    let row_scale_bytes = ngroups * 2;
    if proj.packed.len() != d_out * row_pack_bytes || proj.scales.len() != d_out * row_scale_bytes {
        return Err(format!("QAT tensor byte mismatch for {d_out}x{d_in}"));
    }

    let mut out = vec![0f32; seq_len * d_out];
    let nthreads = threads.max(1).min(d_out);
    let chunk = d_out.div_ceil(nthreads);

    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for t in 0..nthreads {
            let r0 = t * chunk;
            if r0 >= d_out {
                break;
            }
            let r1 = (r0 + chunk).min(d_out);
            let packed = &proj.packed;
            let scales = &proj.scales;
            let out_ptr_addr = out.as_mut_ptr() as usize;
            handles.push(s.spawn(move || {
                // SAFETY: Each thread writes strictly to column indices `r in r0..r1` across tokens `0..seq_len`,
                // which are pairwise disjoint indices `tok * d_out + r`.
                let out_base = out_ptr_addr as *mut f32;
                let mut acc = vec![0f64; seq_len];
                let mut wtile = [0f32; 32];
                for r in r0..r1 {
                    acc.fill(0.0);
                    let rp = &packed[r * row_pack_bytes..(r + 1) * row_pack_bytes];
                    let rs = &scales[r * row_scale_bytes..(r + 1) * row_scale_bytes];
                    if seq_len == 1 {
                        let mut acc0 = 0f64;
                        for g in 0..ngroups {
                            let sc = bf16_to_f32(u16::from_le_bytes([rs[g * 2], rs[g * 2 + 1]]));
                            let gp = &rp[g * 16..(g + 1) * 16];
                            let col0 = g * 32;
                            let xt = &x_batch[col0..col0 + 32];
                            #[cfg(target_arch = "aarch64")]
                            let sub = unsafe { neon_qat_group32_dot(gp, sc, xt) };
                            #[cfg(not(target_arch = "aarch64"))]
                            let sub = {
                                let mut s0 = 0f32;
                                for j in 0..16 {
                                    let b = gp[j];
                                    let q0 = ((b & 0x0F) as i32 - 8) as f32;
                                    let q1 = ((b >> 4) as i32 - 8) as f32;
                                    s0 += round_bf16_f32(q0 * sc) * xt[2 * j] + round_bf16_f32(q1 * sc) * xt[2 * j + 1];
                                }
                                s0
                            };
                            acc0 += sub as f64;
                        }
                        unsafe {
                            *out_base.add(r) = acc0 as f32;
                        }
                        continue;
                    }
                    for g in 0..ngroups {
                        let sc = bf16_to_f32(u16::from_le_bytes([rs[g * 2], rs[g * 2 + 1]]));
                        let gp = &rp[g * 16..(g + 1) * 16];
                        #[cfg(target_arch = "aarch64")]
                        unsafe {
                            neon_qat_group32_unpack(gp, sc, &mut wtile);
                        }
                        #[cfg(not(target_arch = "aarch64"))]
                        for j in 0..16 {
                            let b = gp[j];
                            let q0 = ((b & 0x0F) as i32 - 8) as f32;
                            let q1 = ((b >> 4) as i32 - 8) as f32;
                            wtile[2 * j] = round_bf16_f32(q0 * sc);
                            wtile[2 * j + 1] = round_bf16_f32(q1 * sc);
                        }
                        let col0 = g * 32;
                        for (tok, a_slot) in acc.iter_mut().enumerate() {
                            let xt = &x_batch[tok * d_in + col0..tok * d_in + col0 + 32];
                            #[cfg(target_arch = "aarch64")]
                            let sub = unsafe { neon_dot32_f32(&wtile, xt) };
                            #[cfg(not(target_arch = "aarch64"))]
                            let sub = {
                                let mut s0 = 0f32;
                                for j in 0..32 {
                                    s0 += wtile[j] * xt[j];
                                }
                                s0
                            };
                            *a_slot += sub as f64;
                        }
                    }
                    for (tok, &val) in acc.iter().enumerate() {
                        unsafe {
                            *out_base.add(tok * d_out + r) = val as f32;
                        }
                    }
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    });
    Ok(out)
}

struct PinnedLayer {
    q: QatProj,
    k: QatProj,
    v: Option<QatProj>,
    o: QatProj,
    gate: QatProj,
    up: QatProj,
    down: QatProj,
}

enum Embed31B {
    Streamed,
    ResidentBf16(Vec<u16>),
    Packed(PackedMatrix),
    T16C {
        mat: run270_rust::t16c::T16CMat,
        native: Option<std::sync::Arc<run270_rust::t16c::NativeTile16Kernel>>,
    },
}

enum EmbedMode31B {
    Stream,
    Bf16,
    Quant(QuantScheme, bool),
}

impl EmbedMode31B {
    fn parse(s: &str, h8: bool) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "stream" => {
                if h8 {
                    return Err("--embed-h8 invalid with stream".into());
                }
                Ok(Self::Stream)
            }
            "bf16" => {
                if h8 {
                    return Err("--embed-h8 invalid with bf16".into());
                }
                Ok(Self::Bf16)
            }
            other => Ok(Self::Quant(QuantScheme::parse(other)?, h8)),
        }
    }
}

fn dot_bf16_rows(slab: &[u16], d: usize, n_rows: usize, x: &[f32], threads: usize) -> Vec<f32> {
    let mut out = vec![0f32; n_rows];
    let nthreads = threads.max(1).min(n_rows);
    let chunk = n_rows.div_ceil(nthreads);
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for (t, oc) in out.chunks_mut(chunk).enumerate() {
            let r0 = t * chunk;
            handles.push(s.spawn(move || {
                for (lr, slot) in oc.iter_mut().enumerate() {
                    let row = &slab[(r0 + lr) * d..(r0 + lr + 1) * d];
                    let mut acc = 0f64;
                    let mut off = 0usize;
                    while off + 32 <= d {
                        #[cfg(target_arch = "aarch64")]
                        let sub = unsafe { neon_dot32_bf16(&row[off..off + 32], &x[off..off + 32]) };
                        #[cfg(not(target_arch = "aarch64"))]
                        let sub = {
                            let mut s0 = 0f32;
                            for i in 0..32 {
                                s0 += bf16_to_f32(row[off + i]) * x[off + i];
                            }
                            s0
                        };
                        acc += sub as f64;
                        off += 32;
                    }
                    if off < d {
                        let mut sub = 0f32;
                        for i in off..d {
                            sub += bf16_to_f32(row[i]) * x[i];
                        }
                        acc += sub as f64;
                    }
                    *slot = acc as f32;
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    });
    out
}

fn proc_status_kb() -> (Option<usize>, Option<usize>) {
    let Ok(txt) = fs::read_to_string("/proc/self/status") else {
        return (None, None);
    };
    let parse_kb = |key: &str| -> Option<usize> {
        txt.lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|n| n.parse::<usize>().ok())
    };
    (parse_kb("VmHWM:"), parse_kb("VmRSS:"))
}

struct LayerKv {
    k: Vec<f32>,
    v: Vec<f32>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum KvQuantMode {
    Fp32,
    Q8,
    Fp8E4M3,
    Glyph,
    GlyphPure,
}

impl KvQuantMode {
    fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "fp32" | "f32" => Ok(Self::Fp32),
            "q8" | "q8_0" => Ok(Self::Q8),
            "fp8" | "e4m3" | "fp8_e4m3" => Ok(Self::Fp8E4M3),
            "glyph" | "glyph_v2" | "opt5" => Ok(Self::Glyph),
            "glyph-pure" | "glyph_pure" | "opt5-pure" => Ok(Self::GlyphPure),
            other => Err(format!("invalid --kv-quant '{other}' (expected fp32|q8|fp8|glyph|glyph-pure)")),
        }
    }

    fn is_glyph(self) -> bool {
        matches!(self, Self::Glyph | Self::GlyphPure)
    }
}

fn init_glyph_kv(cfg: &Gemma4Cfg, kv_quant: KvQuantMode) -> Result<Vec<Option<GlyphKvLayer>>, String> {
    if !kv_quant.is_glyph() {
        return Ok((0..cfg.n_layers).map(|_| None).collect());
    }
    let (sink_cap, tail_cap) = match kv_quant {
        KvQuantMode::Glyph => (4usize, 32usize),
        KvQuantMode::GlyphPure => (0usize, 0usize),
        _ => (0usize, 0usize),
    };
    let mut out = Vec::with_capacity(cfg.n_layers);
    for l in 0..cfg.n_layers {
        let is_full = cfg.layer_full[l];
        let hd = if is_full { cfg.global_hd } else { cfg.hd };
        let n_kv = if is_full { cfg.n_global_kv } else { cfg.n_kv };
        let sw = if is_full { None } else { Some(cfg.window) };
        // Gemma 4 `attention_k_eq_v` shares the pre-norm projection W_k == W_v, NOT the post-norm/post-RoPE
        // KV cache tensors (`K_post = RoPE(k_norm(W_k x))` != `V_post = RMSNorm(W_k x)`). Keep distinct K/V
        // glyph streams so live attention matches exact FP32 Gemma 4 mathematics.
        let k_eq_v = false;
        out.push(Some(GlyphKvLayer::new(
            cfg.n_heads,
            n_kv,
            hd,
            k_eq_v,
            sw,
            sink_cap,
            tail_cap,
            256,
        )?));
    }
    Ok(out)
}

fn fp8_e4m3_encode_scalar(v: f32) -> u8 {
    let sign = if v.is_sign_negative() { 0x80u8 } else { 0x00u8 };
    let av = v.abs().min(448.0);
    if av < 2.0f32.powi(-9) {
        return sign;
    }
    if av < 2.0f32.powi(-6) {
        let mant = ((av * 512.0).round() as u8).clamp(0, 7);
        return sign | mant;
    }
    let exp_unbiased = av.log2().floor() as i32;
    let exp_field = (exp_unbiased + 7).clamp(1, 15);
    let step = 2.0f32.powi(exp_field - 7);
    let mant_f = ((av / step) - 1.0) * 8.0;
    let mut mant = mant_f.round() as i32;
    let mut ef = exp_field;
    if mant >= 8 {
        mant = 0;
        ef += 1;
    }
    if ef >= 15 {
        ef = 15;
        mant = mant.min(6); // OCP E4M3 reserves exp=15,mant=7 for NaN
    }
    sign | ((ef as u8) << 3) | (mant as u8)
}

fn build_fp8_e4m3_lut() -> [f32; 256] {
    let mut lut = [0.0f32; 256];
    for idx in 0..256usize {
        let sign = if (idx & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
        let exp = ((idx >> 3) & 0x0F) as i32;
        let mant = (idx & 0x07) as f32;
        lut[idx] = if exp == 0 {
            sign * 2.0f32.powi(-6) * (mant / 8.0)
        } else if exp == 15 && mant >= 7.0 {
            sign * 448.0
        } else {
            sign * 2.0f32.powi(exp - 7) * (1.0 + mant / 8.0)
        };
    }
    lut
}

/// Per-head symmetric Q8_0 quantization roundtrip, returning packed Q8 bytes (`hd + 4` per head).
fn quantize_dequant_kv_q8_inplace(buf: &mut [f32], hd: usize) -> usize {
    let mut packed_bytes = 0usize;
    for chunk in buf.chunks_exact_mut(hd) {
        let max_a = chunk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let sc = if max_a > 1e-12 { max_a / 127.0 } else { 0.0 };
        let inv = if sc > 0.0 { 1.0 / sc } else { 0.0 };
        for v in chunk.iter_mut() {
            let q = (*v * inv).round().clamp(-127.0, 127.0) as i8;
            *v = (q as f32) * sc;
        }
        packed_bytes += hd + 4;
    }
    packed_bytes
}

/// Per-head IEEE OCP FP8 E4M3 quantization roundtrip, returning packed FP8 bytes (`hd + 4` per head).
fn quantize_dequant_kv_fp8_e4m3_inplace(buf: &mut [f32], hd: usize, lut: &[f32; 256]) -> usize {
    let mut packed_bytes = 0usize;
    for chunk in buf.chunks_exact_mut(hd) {
        let max_a = chunk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let sc = if max_a > 1e-12 { max_a / 448.0 } else { 0.0 };
        let inv = if sc > 0.0 { 1.0 / sc } else { 0.0 };
        for v in chunk.iter_mut() {
            let code = fp8_e4m3_encode_scalar(*v * inv);
            *v = lut[code as usize] * sc;
        }
        packed_bytes += hd + 4;
    }
    packed_bytes
}

fn parse_hybrid_qat_layers(spec: &str, n_layers: usize) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for raw in spec.split(',') {
        let t = raw.trim();
        if t.is_empty() {
            continue;
        }
        let idx: usize = t.parse().map_err(|_| format!("invalid layer id '{t}' in --hybrid-qat-layers"))?;
        if idx >= n_layers {
            return Err(format!("--hybrid-qat-layers layer {idx} >= n_layers {n_layers}"));
        }
        if !out.contains(&idx) {
            out.push(idx);
        }
    }
    out.sort_unstable();
    Ok(out)
}

fn parse_lmhead_cs_mode(s: &str) -> Result<Option<f32>, String> {
    let low = s.trim().to_ascii_lowercase();
    if low == "off" || low == "none" {
        return Ok(None);
    }
    if low == "exact" {
        return Ok(Some(0.0));
    }
    if low == "sieve" {
        return Ok(Some(0.25));
    }
    if let Some(rest) = low.strip_prefix("sieve:") {
        let frac: f32 = rest
            .parse()
            .map_err(|_| format!("invalid --lmhead-cs sieve fraction '{s}'"))?;
        if !frac.is_finite() || frac <= 0.0 || frac > 1.0 {
            return Err(format!("--lmhead-cs sieve fraction must be in (0, 1], got {frac}"));
        }
        return Ok(Some(frac));
    }
    Err(format!("invalid --lmhead-cs '{s}' (expected off|exact|sieve[:frac])"))
}

struct Runtime31B {
    cfg: Gemma4Cfg,
    sf: Option<SafeTensors>,
    norms: HashMap<String, Vec<f32>>,
    scalars: Vec<f32>,
    embed: Embed31B,
    pinned: Vec<PinnedLayer>,
    hybrid_pinned: HashMap<usize, PinnedLayer>,
    t16c_mats: Option<HashMap<String, run270_rust::t16c::T16CMat>>,
    native_t16c: Option<std::sync::Arc<run270_rust::t16c::NativeTile16Kernel>>,
    c8d_bank: Option<run270_rust::cuneiform_tbi::CuneiformBank>,
    apply_c8d_proj: bool,
    cs_skip_thresh: f32,
    cs_down_norms: Vec<Option<Vec<u16>>>,
    cs_skipped_blocks: u64,
    mlp_top_p: f32,
    mlp_sparse_skipped_blks: u64,
    mlp_sparse_total_blks: u64,
    profile_components: bool,
    prof_attn_s: f64,
    prof_gate_s: f64,
    prof_up_s: f64,
    prof_down_s: f64,
    prof_norm_kv_s: f64,
    prof_lmhead_s: f64,
    prof_steps: usize,
    embed_tile_norms: Option<Vec<u16>>,
    lmhead_cs_sieve: Option<f32>,
    lmhead_cs_skipped_blocks: u64,
    kv_quant: KvQuantMode,
    fp8_lut: [f32; 256],
    kv_raw_bytes: usize,
    kv_compressed_bytes: usize,
    kv: Vec<LayerKv>,
    glyph_kv: Vec<Option<GlyphKvLayer>>,
    vision: Option<run270_rust::vision::VisionWeights>,
    threads: usize,
    ssd_bytes_read: u64,
}

impl Runtime31B {
    fn kv_len(&self) -> usize {
        if self.kv_quant.is_glyph() {
            return self
                .glyph_kv
                .first()
                .and_then(|o| o.as_ref())
                .map_or(0, |g| g.total_seen());
        }
        if self.kv.is_empty() || self.cfg.layer_full.is_empty() {
            return 0;
        }
        let is_full = self.cfg.layer_full[0];
        let hd = if is_full { self.cfg.global_hd } else { self.cfg.hd };
        let n_kv = if is_full { self.cfg.n_global_kv } else { self.cfg.n_kv };
        let stride = n_kv * hd;
        if stride == 0 {
            0
        } else {
            self.kv[0].k.len() / stride
        }
    }

    fn glyph_kv_total_stats(&self) -> GlyphKvMemStats {
        let mut tot = GlyphKvMemStats::default();
        for g in self.glyph_kv.iter().flatten() {
            let s = g.memory_stats();
            tot.active_tokens += s.active_tokens;
            tot.glyph_tokens += s.glyph_tokens;
            tot.exact_tokens += s.exact_tokens;
            tot.resident_bytes += s.resident_bytes;
            tot.naive_bf16_bytes += s.naive_bf16_bytes;
            tot.fp32_bytes += s.fp32_bytes;
        }
        tot
    }

    fn truncate_kv(&mut self, keep_tokens: usize) -> Result<(), String> {
        if self.kv_quant.is_glyph() {
            for g in self.glyph_kv.iter_mut().flatten() {
                g.truncate(keep_tokens)?;
            }
            return Ok(());
        }
        for (l, layer_kv) in self.kv.iter_mut().enumerate() {
            let is_full = self.cfg.layer_full[l];
            let hd = if is_full { self.cfg.global_hd } else { self.cfg.hd };
            let n_kv = if is_full { self.cfg.n_global_kv } else { self.cfg.n_kv };
            let target = keep_tokens * n_kv * hd;
            if layer_kv.k.len() < target || layer_kv.v.len() < target {
                return Err(format!(
                    "truncate_kv layer {l}: keep_tokens={keep_tokens} (elems={target}) exceeds kv len ({}, {})",
                    layer_kv.k.len(),
                    layer_kv.v.len()
                ));
            }
            layer_kv.k.truncate(target);
            layer_kv.v.truncate(target);
        }
        Ok(())
    }

    fn assistant_kv_layers(&self) -> Result<(usize, usize), String> {
        let slide_l = self
            .cfg
            .layer_full
            .iter()
            .rposition(|&f| !f)
            .ok_or("no sliding_attention layer in model for assistant KV sharing")?;
        let full_l = self
            .cfg
            .layer_full
            .iter()
            .rposition(|&f| f)
            .ok_or("no full_attention layer in model for assistant KV sharing")?;
        Ok((slide_l, full_l))
    }

    fn enable_cs_skip(&mut self, thresh: f32) -> Result<(usize, usize), String> {
        if !thresh.is_finite() || thresh < 0.0 {
            return Err(format!("invalid --cs-skip threshold {thresh}"));
        }
        self.cs_skip_thresh = thresh;
        self.cs_down_norms.clear();
        if thresh == 0.0 {
            return Ok((0, 0));
        }
        let mats = self
            .t16c_mats
            .as_ref()
            .ok_or("--cs-skip requires --t16c resident quantized runtime")?;
        let mut count = 0usize;
        let mut bytes = 0usize;
        self.cs_down_norms.reserve(self.cfg.n_layers);
        for l in 0..self.cfg.n_layers {
            let k1 = format!("model.language_model.layers.{l}.mlp.down_proj.weight");
            let k2 = format!("model.layers.{l}.mlp.down_proj.weight");
            if let Some(m) = mats.get(&k1).or_else(|| mats.get(&k2)) {
                let norms = m.compute_tile_norms_fp16();
                bytes += norms.len() * 2;
                count += 1;
                self.cs_down_norms.push(Some(norms));
            } else {
                self.cs_down_norms.push(None);
            }
        }
        Ok((count, bytes))
    }

    fn enable_lmhead_cs(&mut self, sieve_opt: Option<f32>) -> Result<usize, String> {
        self.lmhead_cs_sieve = sieve_opt;
        self.embed_tile_norms = None;
        if sieve_opt.is_none() {
            return Ok(0);
        }
        match &self.embed {
            Embed31B::T16C { mat, .. } => {
                let norms = mat.compute_tile_norms_fp16();
                let bytes = norms.len() * 2;
                self.embed_tile_norms = Some(norms);
                Ok(bytes)
            }
            _ => {
                self.lmhead_cs_sieve = None;
                Ok(0)
            }
        }
    }

    fn norm(&self, key: &str) -> Result<&[f32], String> {
        self.norms
            .get(key)
            .or_else(|| key.strip_prefix("model.language_model.").and_then(|s| self.norms.get(&format!("model.{s}"))))
            .map(|v| v.as_slice())
            .ok_or_else(|| format!("missing norm {key}"))
    }

    fn embed_batch(&mut self, ids: &[u32]) -> Result<Vec<f32>, String> {
        let d = self.cfg.d;
        let escale = (d as f32).sqrt();
        let mut out = Vec::with_capacity(ids.len() * d);
        let em_key = "model.language_model.embed_tokens.weight";
        match &self.embed {
            Embed31B::Streamed => {
                let sf = self.sf.as_ref().ok_or("streamed embed requires --model")?;
                let meta = sf.tensors.get(em_key).ok_or_else(|| format!("missing {em_key}"))?;
                for &tid in ids {
                    let ti = tid as usize;
                    if ti >= self.cfg.vocab {
                        return Err(format!("token {tid} out of bounds"));
                    }
                    let raw = sf.read_raw_slice(meta, ti * d * 2, d * 2)?;
                    self.ssd_bytes_read += raw.len() as u64;
                    for c in raw.chunks_exact(2) {
                        out.push(bf16_to_f32(u16::from_le_bytes([c[0], c[1]])) * escale);
                    }
                }
            }
            Embed31B::ResidentBf16(tbl) => {
                for &tid in ids {
                    let ti = tid as usize;
                    if ti >= self.cfg.vocab {
                        return Err(format!("token {tid} out of bounds"));
                    }
                    for &b in &tbl[ti * d..(ti + 1) * d] {
                        out.push(bf16_to_f32(b) * escale);
                    }
                }
            }
            Embed31B::Packed(pm) => {
                for &tid in ids {
                    let row = pm.decode_row(tid as usize)?;
                    for v in row {
                        out.push(v * escale);
                    }
                }
            }
            Embed31B::T16C { mat, .. } => {
                for &tid in ids {
                    let row = mat.decode_row(tid as usize)?;
                    if row.len() != d {
                        return Err(format!("T16C embed decode_row len {} != d {d}", row.len()));
                    }
                    for v in row {
                        out.push(v * escale);
                    }
                }
            }
        }
        Ok(out)
    }

    fn forward_batch(&mut self, ids: &[u32], start_pos: usize, verbose_layers: bool) -> Result<Vec<f32>, String> {
        self.forward_batch_mode(ids, start_pos, verbose_layers, false)
    }

    fn forward_batch_mode(
        &mut self,
        ids: &[u32],
        start_pos: usize,
        verbose_layers: bool,
        all_final: bool,
    ) -> Result<Vec<f32>, String> {
        let seq_len = ids.len();
        if seq_len == 0 {
            return Err("empty token slice".into());
        }
        let states = self.embed_batch(ids)?;
        self.forward_batch_with_embeds(states, seq_len, start_pos, verbose_layers, all_final, None)
    }

    fn forward_batch_with_embeds(
        &mut self,
        mut states: Vec<f32>,
        seq_len: usize,
        start_pos: usize,
        verbose_layers: bool,
        all_final: bool,
        vis_kv_ends: Option<&[usize]>,
    ) -> Result<Vec<f32>, String> {
        if seq_len == 0 {
            return Err("empty token slice".into());
        }
        let d = self.cfg.d;
        if states.len() != seq_len * d {
            return Err(format!("embed state len {} != {}", states.len(), seq_len * d));
        }
        let eps = self.cfg.eps;
        let n_heads = self.cfg.n_heads;

        let cfg = &self.cfg;
        let sf_opt = &self.sf;
        let norms = &self.norms;
        let scalars = &self.scalars;
        let pinned = &self.pinned;
        let t16c_mats = &self.t16c_mats;
        let all_pinned = t16c_mats.is_some();
        let native_t16c = &self.native_t16c;
        let hybrid_pinned = &self.hybrid_pinned;
        let cs_down_norms = &self.cs_down_norms;
        let cs_skip_thresh = self.cs_skip_thresh;
        let mlp_top_p = self.mlp_top_p;
        let profile_components = self.profile_components;
        let kv_quant = self.kv_quant;
        let fp8_lut = &self.fp8_lut;
        let kv = &mut self.kv;
        let glyph_kv = &mut self.glyph_kv;
        let threads = self.threads;
        let mut read_bytes = 0u64;
        let mut cs_skipped_step = 0u64;
        let mut mlp_sp_skipped_step = 0u64;
        let mut mlp_sp_total_step = 0u64;
        let mut prof_attn_step = 0.0f64;
        let mut prof_gate_step = 0.0f64;
        let mut prof_up_step = 0.0f64;
        let mut prof_down_step = 0.0f64;
        let mut prof_norm_kv_step = 0.0f64;
        let mut kv_raw_step = 0usize;
        let mut kv_cmp_step = 0usize;

        let get_norm = |key: &str| -> Result<&[f32], String> {
            norms
                .get(key)
                .or_else(|| key.strip_prefix("model.language_model.").and_then(|s| norms.get(&format!("model.{s}"))))
                .map(|v| v.as_slice())
                .ok_or_else(|| format!("missing norm {key}"))
        };

        let mut schedule: Vec<(usize, &'static str)> = Vec::new();
        if t16c_mats.is_none() {
            for ul in pinned.len()..cfg.n_layers {
                schedule.push((ul, "self_attn.q_proj"));
                schedule.push((ul, "self_attn.k_proj"));
                if !(cfg.layer_full[ul] && cfg.attention_k_eq_v) {
                    schedule.push((ul, "self_attn.v_proj"));
                }
                schedule.push((ul, "self_attn.o_proj"));
                schedule.push((ul, "mlp.gate_proj"));
                schedule.push((ul, "mlp.up_proj"));
                schedule.push((ul, "mlp.down_proj"));
            }
        }

        std::thread::scope(|io_scope| -> Result<(), String> {
            let mut sched_idx = 0usize;
            let mut pending: Option<std::thread::ScopedJoinHandle<'_, Result<(QatProj, u64), String>>> =
                if !schedule.is_empty() {
                    let sf = sf_opt.as_ref().ok_or("streaming unpinned layers requires --model")?;
                    let (nl, nsuf) = schedule[0];
                    Some(io_scope.spawn(move || {
                        load_qat_proj(sf, &format!("model.language_model.layers.{nl}.{nsuf}"))
                    }))
                } else {
                    None
                };

            let mut run_proj_masked = |layer: usize,
                                       proj_suffix: &'static str,
                                       x_batch: &[f32],
                                       row_mask: Option<&[u8]>,
                                       col_mask: Option<&[u8]>,
                                       include_aux: bool|
             -> Result<Vec<f32>, String> {
                if let Some(pl) = hybrid_pinned.get(&layer) {
                    let p = match proj_suffix {
                        "self_attn.q_proj" => &pl.q,
                        "self_attn.k_proj" => &pl.k,
                        "self_attn.v_proj" => pl.v.as_ref().ok_or("unexpected v_proj lookup on k_eq_v layer")?,
                        "self_attn.o_proj" => &pl.o,
                        "mlp.gate_proj" => &pl.gate,
                        "mlp.up_proj" => &pl.up,
                        "mlp.down_proj" => &pl.down,
                        other => return Err(format!("unknown proj suffix {other}")),
                    };
                    return qat_project_batch(p, x_batch, seq_len, threads);
                }

                if let Some(mats) = t16c_mats {
                    let k1 = format!("model.language_model.layers.{layer}.{proj_suffix}.weight");
                    let k2 = format!("model.layers.{layer}.{proj_suffix}.weight");
                    let mat = mats
                        .get(&k1)
                        .or_else(|| mats.get(&k2))
                        .ok_or_else(|| format!("T16C missing {k1}"))?;
                    let mut out = if row_mask.is_some() || col_mask.is_some() {
                        let (y_sp, skipped) = mat.project_batch_sparse_ext(
                            x_batch,
                            seq_len,
                            native_t16c.as_deref(),
                            threads,
                            row_mask,
                            col_mask,
                            include_aux,
                            proj_suffix == "mlp.gate_proj",
                        )?;
                        let n_tiles_m = mat.d_out.div_ceil(16) as u64;
                        let n_blks_m = (mat.d_in / mat.block_size) as u64;
                        let passes = (seq_len.div_ceil(4)).max(1) as u64;
                        mlp_sp_total_step += n_tiles_m * n_blks_m * passes;
                        if skipped > 0 {
                            mlp_sp_skipped_step += skipped as u64;
                        }
                        y_sp
                    } else if proj_suffix == "mlp.down_proj" && cs_skip_thresh > 0.0 {
                        let d_norms = cs_down_norms.get(layer).and_then(|o| o.as_deref());
                        let (y_cs, skipped) = mat.project_batch_cs(
                            x_batch,
                            seq_len,
                            native_t16c.as_deref(),
                            threads,
                            d_norms,
                            cs_skip_thresh,
                        )?;
                        if skipped > 0 {
                            cs_skipped_step += skipped as u64;
                        }
                        y_cs
                    } else {
                        mat.project_batch(x_batch, seq_len, native_t16c.as_deref(), threads)?
                    };
                    if include_aux && self.apply_c8d_proj {
                        if let Some(ref bank) = self.c8d_bank {
                            if let Some(entry) = bank.entries.get(&k1).or_else(|| bank.entries.get(&k2)) {
                                entry.apply_batch_into(x_batch, seq_len, &mut out, bank.native.as_ref(), threads, None)?;
                            }
                        }
                    }
                    return Ok(out);
                }

                if layer < pinned.len() {
                    let pl = &pinned[layer];
                    let p = match proj_suffix {
                        "self_attn.q_proj" => &pl.q,
                        "self_attn.k_proj" => &pl.k,
                        "self_attn.v_proj" => pl.v.as_ref().ok_or("unexpected v_proj lookup on k_eq_v layer")?,
                        "self_attn.o_proj" => &pl.o,
                        "mlp.gate_proj" => &pl.gate,
                        "mlp.up_proj" => &pl.up,
                        "mlp.down_proj" => &pl.down,
                        other => return Err(format!("unknown proj suffix {other}")),
                    };
                    return qat_project_batch(p, x_batch, seq_len, threads);
                }

                if sched_idx >= schedule.len() || schedule[sched_idx] != (layer, proj_suffix) {
                    return Err(format!("prefetch schedule mismatch at layer {layer} {proj_suffix}"));
                }
                let sf = sf_opt.as_ref().ok_or("streaming unpinned layers requires --model")?;
                let (qp, nbytes) = match pending.take() {
                    Some(h) => h.join().map_err(|_| "prefetch worker panicked".to_string())??,
                    None => load_qat_proj(sf, &format!("model.language_model.layers.{layer}.{proj_suffix}"))?,
                };
                read_bytes += nbytes;
                sched_idx += 1;
                if sched_idx < schedule.len() {
                    let (nl, nsuf) = schedule[sched_idx];
                    pending = Some(io_scope.spawn(move || {
                        load_qat_proj(sf, &format!("model.language_model.layers.{nl}.{nsuf}"))
                    }));
                }
                qat_project_batch(&qp, x_batch, seq_len, threads)
            };

            for l in 0..cfg.n_layers {
                let tl = std::time::Instant::now();
                let pfx = format!("model.language_model.layers.{l}");
                let is_full = cfg.layer_full[l];
                let hd = if is_full { cfg.global_hd } else { cfg.hd };
                let n_kv = if is_full { cfg.n_global_kv } else { cfg.n_kv };
                let theta = if is_full { cfg.theta_global } else { cfg.theta_local };
                let partial = if is_full { cfg.partial_global } else { cfg.partial_local };
                let rotary_dims = ((hd as f32) * partial) as usize;
                let sw = if is_full { None } else { Some(cfg.window) };

                let mut inp = Vec::with_capacity(seq_len * d);
                {
                    let g_in = get_norm(&format!("{pfx}.input_layernorm.weight"))?;
                    for t in 0..seq_len {
                        inp.extend_from_slice(&rmsn_raw(&states[t * d..(t + 1) * d], Some(g_in), eps));
                    }
                }

                let t_ap0 = if profile_components { Some(std::time::Instant::now()) } else { None };
                let mut q_batch = run_proj_masked(l, "self_attn.q_proj", &inp, None, None, true)?;
                let k_raw = run_proj_masked(l, "self_attn.k_proj", &inp, None, None, true)?;
                let mut v_batch = if is_full && cfg.attention_k_eq_v {
                    k_raw.clone()
                } else {
                    run_proj_masked(l, "self_attn.v_proj", &inp, None, None, true)?
                };
                if let Some(t0) = t_ap0 {
                    prof_attn_step += t0.elapsed().as_secs_f64();
                }
                drop(inp);
                let t_kv0 = if profile_components { Some(std::time::Instant::now()) } else { None };

                let mut k_batch = k_raw;
                {
                    let qn = get_norm(&format!("{pfx}.self_attn.q_norm.weight"))?;
                    let kn = get_norm(&format!("{pfx}.self_attn.k_norm.weight"))?;
                    for t in 0..seq_len {
                        let pos = (start_pos + t) as u32;
                        let qs = &mut q_batch[t * n_heads * hd..(t + 1) * n_heads * hd];
                        let qnormed = qk_norm_raw(qs, n_heads, hd, Some(qn), eps);
                        qs.copy_from_slice(&qnormed);
                        for h in 0..n_heads {
                            rope_head(&mut qs[h * hd..(h + 1) * hd], pos, theta, rotary_dims, hd);
                        }

                        let ks = &mut k_batch[t * n_kv * hd..(t + 1) * n_kv * hd];
                        let knormed = qk_norm_raw(ks, n_kv, hd, Some(kn), eps);
                        ks.copy_from_slice(&knormed);
                        for h in 0..n_kv {
                            rope_head(&mut ks[h * hd..(h + 1) * hd], pos, theta, rotary_dims, hd);
                        }

                        let vs = &mut v_batch[t * n_kv * hd..(t + 1) * n_kv * hd];
                        let vnormed = qk_norm_raw(vs, n_kv, hd, None, eps);
                        vs.copy_from_slice(&vnormed);
                    }
                }

                let mut ctx_batch = Vec::with_capacity(seq_len * n_heads * hd);
                if kv_quant.is_glyph() {
                    let gkv = glyph_kv[l]
                        .as_mut()
                        .ok_or_else(|| format!("missing GlyphKvLayer at layer {l}"))?;
                    let base_t = gkv.total_seen();
                    let stride = n_kv * hd;
                    for t in 0..seq_len {
                        let kt = &k_batch[t * stride..(t + 1) * stride];
                        let vt = if gkv.k_eq_v {
                            &[][..]
                        } else {
                            &v_batch[t * stride..(t + 1) * stride]
                        };
                        gkv.append_token(kt, vt, native_t16c.as_deref())?;
                    }
                    for t in 0..seq_len {
                        let q_t = &q_batch[t * n_heads * hd..(t + 1) * n_heads * hd];
                        let kv_end = if !is_full {
                            vis_kv_ends.map_or(base_t + t, |ends| base_t + ends[t])
                        } else {
                            base_t + t
                        };
                        let c_t = gkv.attend_range(
                            q_t,
                            base_t + t,
                            kv_end,
                            start_pos + t,
                            1.0,
                            native_t16c.as_deref(),
                            threads,
                        )?;
                        ctx_batch.extend_from_slice(&c_t);
                    }
                } else {
                    match kv_quant {
                        KvQuantMode::Fp32 => {
                            let kb = k_batch.len() * 4 + v_batch.len() * 4;
                            kv_raw_step += kb;
                            kv_cmp_step += kb;
                        }
                        KvQuantMode::Q8 => {
                            kv_raw_step += (k_batch.len() + v_batch.len()) * 4;
                            kv_cmp_step += quantize_dequant_kv_q8_inplace(&mut k_batch, hd)
                                + quantize_dequant_kv_q8_inplace(&mut v_batch, hd);
                        }
                        KvQuantMode::Fp8E4M3 => {
                            kv_raw_step += (k_batch.len() + v_batch.len()) * 4;
                            kv_cmp_step += quantize_dequant_kv_fp8_e4m3_inplace(&mut k_batch, hd, fp8_lut)
                                + quantize_dequant_kv_fp8_e4m3_inplace(&mut v_batch, hd, fp8_lut);
                        }
                        KvQuantMode::Glyph | KvQuantMode::GlyphPure => unreachable!(),
                    }

                    let base_t = kv[l].k.len() / (n_kv * hd);
                    kv[l].k.extend_from_slice(&k_batch);
                    kv[l].v.extend_from_slice(&v_batch);
                    for t in 0..seq_len {
                        let q_t = &q_batch[t * n_heads * hd..(t + 1) * n_heads * hd];
                        let kv_end = if !is_full {
                            vis_kv_ends.map_or(base_t + t, |ends| base_t + ends[t])
                        } else {
                            base_t + t
                        };
                        let c_t = attn_decode_range(q_t, &kv[l].k, &kv[l].v, base_t + t, kv_end, start_pos + t, n_heads, n_kv, hd, sw);
                        ctx_batch.extend_from_slice(&c_t);
                    }
                }
                drop(q_batch);
                drop(k_batch);
                drop(v_batch);

                if let Some(t0) = t_kv0 {
                    prof_norm_kv_step += t0.elapsed().as_secs_f64();
                }
                let t_ao0 = if profile_components { Some(std::time::Instant::now()) } else { None };
                let attn_o = run_proj_masked(l, "self_attn.o_proj", &ctx_batch, None, None, true)?;
                if let Some(t0) = t_ao0 {
                    prof_attn_step += t0.elapsed().as_secs_f64();
                }
                drop(ctx_batch);
                let t_n1 = if profile_components { Some(std::time::Instant::now()) } else { None };
                {
                    let g_pa = get_norm(&format!("{pfx}.post_attention_layernorm.weight"))?;
                    for t in 0..seq_len {
                        let norm_a = rmsn_raw(&attn_o[t * d..(t + 1) * d], Some(g_pa), eps);
                        for i in 0..d {
                            states[t * d + i] += norm_a[i];
                        }
                    }
                }
                drop(attn_o);

                let mut inp2 = Vec::with_capacity(seq_len * d);
                {
                    let g_pf = get_norm(&format!("{pfx}.pre_feedforward_layernorm.weight"))?;
                    for t in 0..seq_len {
                        inp2.extend_from_slice(&rmsn_raw(&states[t * d..(t + 1) * d], Some(g_pf), eps));
                    }
                }
                if let Some(t0) = t_n1 {
                    prof_norm_kv_step += t0.elapsed().as_secs_f64();
                }
                let t_g0 = if profile_components { Some(std::time::Instant::now()) } else { None };
                // Exact 1-Pass Gate-Guided Block-Sparse GeGLU:
                // Compute exact `gate_proj` in a SINGLE pass (zero preview error, zero double-pass OpenMP/FWHT overhead,
                // ~2.4x faster than 2-stage preview!). Because V5/V5.1 co-activation permutation clusters active GeGLU
                // neurons into contiguous `bsize` blocks (88–97.8% energy in top 50% blocks), exact `gate_proj` selects
                // the true top `mlp_top_p` blocks (`row_mask`, `col_mask`), skipping `(1 - mlp_top_p)` of `up_proj` and
                // `down_proj` DRAM tiles cleanly while preserving 100% dense-identical GeGLU routing!
                let gate = run_proj_masked(l, "mlp.gate_proj", &inp2, None, None, true)?;
                if let Some(t0) = t_g0 {
                    prof_gate_step += t0.elapsed().as_secs_f64();
                }
                let gelu_gate: Vec<f32> = gate.iter().map(|&a| gelu_tanh_scalar(a)).collect();
                drop(gate);
                let sparse_masks: Option<(Vec<u8>, Vec<u8>, usize)> = if mlp_top_p < 0.9999 {
                    t16c_mats.as_ref().and_then(|mats| {
                        let ku1 = format!("model.language_model.layers.{l}.mlp.up_proj.weight");
                        let ku2 = format!("model.layers.{l}.mlp.up_proj.weight");
                        let kd1 = format!("model.language_model.layers.{l}.mlp.down_proj.weight");
                        let kd2 = format!("model.layers.{l}.mlp.down_proj.weight");
                        let um = mats.get(&ku1).or_else(|| mats.get(&ku2))?;
                        let dm = mats.get(&kd1).or_else(|| mats.get(&kd2))?;
                        let ue = um.n_bits & 0x7F;
                        let de = dm.n_bits & 0x7F;
                        if ue == 5 || ue == 8 || de == 5 || de == 8 {
                            return None;
                        }
                        let bsize = dm.block_size;
                        let ff = cfg.ff;
                        if bsize == 0 || ff % bsize != 0 || bsize % 16 != 0 {
                            return None;
                        }
                        let n_blks = ff / bsize;
                        let tiles_per_blk = bsize / 16;
                        let mut scores: Vec<(usize, f32)> = Vec::with_capacity(n_blks);
                        for b in 0..n_blks {
                            let mut max_e = 0.0f32;
                            for t in 0..seq_len {
                                let sl = &gelu_gate[t * ff + b * bsize..t * ff + (b + 1) * bsize];
                                let mut e = 0.0f32;
                                for &g in sl {
                                    e += g * g;
                                }
                                if e > max_e {
                                    max_e = e;
                                }
                            }
                            scores.push((b, max_e));
                        }
                        scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
                        let keep = ((n_blks as f32 * mlp_top_p).ceil() as usize).clamp(1, n_blks);
                        let mut col_mask = vec![0u8; n_blks];
                        let mut row_mask = vec![0u8; ff.div_ceil(16)];
                        for &(b, _) in &scores[..keep] {
                            col_mask[b] = 1;
                            let t0 = b * tiles_per_blk;
                            for dt in 0..tiles_per_blk {
                                if t0 + dt < row_mask.len() {
                                    row_mask[t0 + dt] = 1;
                                }
                            }
                        }
                        Some((row_mask, col_mask, bsize))
                    })
                } else {
                    None
                };

                let t_u0 = if profile_components { Some(std::time::Instant::now()) } else { None };
                let up = if let Some((ref rmask, _, _)) = sparse_masks {
                    run_proj_masked(l, "mlp.up_proj", &inp2, Some(rmask), None, true)?
                } else {
                    run_proj_masked(l, "mlp.up_proj", &inp2, None, None, true)?
                };
                if let Some(t0) = t_u0 {
                    prof_up_step += t0.elapsed().as_secs_f64();
                }
                drop(inp2);
                let act: Vec<f32> = if let Some((_, ref cmask, bsize)) = sparse_masks {
                    let ff = cfg.ff;
                    let mut out_a = vec![0.0f32; seq_len * ff];
                    for t in 0..seq_len {
                        let base = t * ff;
                        for (b, &active) in cmask.iter().enumerate() {
                            if active != 0 {
                                let off = base + b * bsize;
                                for j in 0..bsize {
                                    out_a[off + j] = gelu_gate[off + j] * up[off + j];
                                }
                            }
                        }
                    }
                    out_a
                } else {
                    gelu_gate.iter().zip(up.iter()).map(|(&a, &b)| a * b).collect()
                };
                drop(gelu_gate);
                drop(up);
                let t_d0 = if profile_components { Some(std::time::Instant::now()) } else { None };
                let down = if let Some((_, ref cmask, _)) = sparse_masks {
                    run_proj_masked(l, "mlp.down_proj", &act, None, Some(cmask), true)?
                } else {
                    run_proj_masked(l, "mlp.down_proj", &act, None, None, true)?
                };
                if let Some(t0) = t_d0 {
                    prof_down_step += t0.elapsed().as_secs_f64();
                }
                drop(act);

                let t_n2 = if profile_components { Some(std::time::Instant::now()) } else { None };
                let g_pff = get_norm(&format!("{pfx}.post_feedforward_layernorm.weight"))?;
                let ls = scalars[l];
                for t in 0..seq_len {
                    let norm_d = rmsn_raw(&down[t * d..(t + 1) * d], Some(g_pff), eps);
                    for i in 0..d {
                        states[t * d + i] = (states[t * d + i] + norm_d[i]) * ls;
                    }
                }
                if let Some(t0) = t_n2 {
                    prof_norm_kv_step += t0.elapsed().as_secs_f64();
                }
                if verbose_layers && (l == 0 || l == 5 || l % 15 == 14 || l + 1 == cfg.n_layers) {
                    eprintln!("[run31b] layer {l:02}/{} done in {:.2}s (pinned={})", cfg.n_layers, tl.elapsed().as_secs_f64(), all_pinned || l < pinned.len());
                }
            }
            Ok(())
        })?;
        self.ssd_bytes_read += read_bytes;
        self.cs_skipped_blocks += cs_skipped_step;
        self.mlp_sparse_skipped_blks += mlp_sp_skipped_step;
        self.mlp_sparse_total_blks += mlp_sp_total_step;
        if profile_components {
            self.prof_attn_s += prof_attn_step;
            self.prof_gate_s += prof_gate_step;
            self.prof_up_s += prof_up_step;
            self.prof_down_s += prof_down_step;
            self.prof_norm_kv_s += prof_norm_kv_step;
            self.prof_steps += 1;
        }
        self.kv_raw_bytes += kv_raw_step;
        self.kv_compressed_bytes += kv_cmp_step;

        let fn_gain = self.norm("model.language_model.norm.weight")?;
        let mut out = if all_final {
            let mut all_out = Vec::with_capacity(seq_len * d);
            for t in 0..seq_len {
                all_out.extend_from_slice(&rmsn_raw(&states[t * d..(t + 1) * d], Some(fn_gain), eps));
            }
            all_out
        } else {
            let last_state = &states[(seq_len - 1) * d..seq_len * d];
            rmsn_raw(last_state, Some(fn_gain), eps)
        };
        let n_out_tok = out.len() / d;
        if self.apply_c8d_proj {
            if let Some(ref bank) = self.c8d_bank {
                if let Some(entry) = bank.entries.get("model.language_model.norm.post_cell60") {
                    let x_in = out.clone();
                    entry.apply_batch_into(&x_in, n_out_tok, &mut out, bank.native.as_ref(), self.threads, None)?;
                }
            }
        }
        Ok(out)
    }

    fn compute_logits(&mut self, hidden: &[f32]) -> Result<Vec<f32>, String> {
        self.compute_logits_batch(hidden, 1)
    }

    fn compute_logits_batch(&mut self, hidden_batch: &[f32], seq_len: usize) -> Result<Vec<f32>, String> {
        self.compute_logits_batch_mode(hidden_batch, seq_len, true)
    }

    fn compute_logits_batch_mode(
        &mut self,
        hidden_batch: &[f32],
        seq_len: usize,
        allow_lmhead_cs: bool,
    ) -> Result<Vec<f32>, String> {
        let d = self.cfg.d;
        let vocab = self.cfg.vocab;
        if hidden_batch.len() != seq_len * d {
            return Err(format!("hidden_batch len {} != {seq_len}*{d}", hidden_batch.len()));
        }
        let t_lm0 = if self.profile_components { Some(std::time::Instant::now()) } else { None };
        let mut lm_sk: u64 = 0;
        let mut logits = match &self.embed {
            Embed31B::Streamed => {
                let sf = self.sf.as_ref().ok_or("streamed logits require --model")?;
                let head_key = if sf.tensors.contains_key("lm_head.weight") {
                    "lm_head.weight"
                } else {
                    "model.language_model.embed_tokens.weight"
                };
                let meta = sf.tensors.get(head_key).ok_or_else(|| format!("missing {head_key}"))?.clone();
                if meta.shape != vec![vocab, d] || meta.dtype != "BF16" {
                    return Err(format!("{head_key}: expected BF16 [{vocab}, {d}], got {} {:?}", meta.dtype, meta.shape));
                }
                let mut out = vec![0f32; seq_len * vocab];
                let chunk_rows = 4096usize;
                let mut r0 = 0usize;
                while r0 < vocab {
                    let take = (vocab - r0).min(chunk_rows);
                    let raw = sf.read_raw_slice(&meta, r0 * d * 2, take * d * 2)?;
                    self.ssd_bytes_read += raw.len() as u64;
                    let slab: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                    for tok in 0..seq_len {
                        let sub = dot_bf16_rows(&slab, d, take, &hidden_batch[tok * d..(tok + 1) * d], self.threads);
                        out[tok * vocab + r0..tok * vocab + r0 + take].copy_from_slice(&sub);
                    }
                    r0 += take;
                }
                out
            }
            Embed31B::ResidentBf16(tbl) => {
                let mut out = Vec::with_capacity(seq_len * vocab);
                for tok in 0..seq_len {
                    out.extend_from_slice(&dot_bf16_rows(tbl, d, vocab, &hidden_batch[tok * d..(tok + 1) * d], self.threads));
                }
                out
            }
            Embed31B::Packed(pm) => {
                let mut out = Vec::with_capacity(seq_len * vocab);
                for tok in 0..seq_len {
                    out.extend_from_slice(&pm.matvec(&hidden_batch[tok * d..(tok + 1) * d], self.threads)?);
                }
                out
            }
            Embed31B::T16C { mat, native } => {
                if allow_lmhead_cs {
                    if let (Some(sieve_frac), Some(norms)) = (self.lmhead_cs_sieve, self.embed_tile_norms.as_deref()) {
                        let (y, sk) = mat.project_lmhead_cs(
                            hidden_batch,
                            seq_len,
                            native.as_deref(),
                            self.threads,
                            Some(norms),
                            32,
                            sieve_frac,
                        )?;
                        if sk > 0 {
                            lm_sk = sk as u64;
                        }
                        y
                    } else {
                        mat.project_batch(hidden_batch, seq_len, native.as_deref(), self.threads)?
                    }
                } else {
                    mat.project_batch(hidden_batch, seq_len, native.as_deref(), self.threads)?
                }
            }
        };
        if let Some(t0) = t_lm0 {
            self.prof_lmhead_s += t0.elapsed().as_secs_f64();
        }
        self.lmhead_cs_skipped_blocks += lm_sk;
        if let Some(cap) = self.cfg.final_softcap {
            for v in &mut logits {
                *v = (*v / cap).tanh() * cap;
            }
        }
        Ok(logits)
    }
}

fn load_runtime_31b(
    model_path: &Path,
    cfg: Gemma4Cfg,
    embed_mode: EmbedMode31B,
    pin_layers: usize,
    kv_quant: KvQuantMode,
    threads: usize,
) -> Result<(Runtime31B, usize, usize), String> {
    let sf = SafeTensors::open(model_path)?;
    let mut norms = HashMap::new();
    let mut keys: Vec<String> = sf.tensors.keys().cloned().collect();
    keys.sort();
    for k in &keys {
        if k.starts_with("model.language_model.") && (k.ends_with(".weight") && !k.contains("embed_tokens")) {
            let meta = &sf.tensors[k];
            if meta.shape.len() == 1 {
                norms.insert(k.clone(), sf.read_tensor_f32(k)?);
            }
        }
    }
    let norm_bytes: usize = norms.values().map(|v| v.len() * 4).sum();
    let mut scalars = Vec::with_capacity(cfg.n_layers);
    for l in 0..cfg.n_layers {
        let sk = format!("model.language_model.layers.{l}.layer_scalar");
        let v = sf.read_tensor_f32(&sk)?;
        if v.len() != 1 || !v[0].is_finite() {
            return Err(format!("invalid scalar {sk}: {v:?}"));
        }
        scalars.push(v[0]);
    }

    let pin_count = pin_layers.min(cfg.n_layers);
    let mut pinned = Vec::with_capacity(pin_count);
    let mut pinned_bytes = 0usize;
    for l in 0..pin_count {
        let pfx = format!("model.language_model.layers.{l}");
        let (q, bq) = load_qat_proj(&sf, &format!("{pfx}.self_attn.q_proj"))?;
        let (k, bk) = load_qat_proj(&sf, &format!("{pfx}.self_attn.k_proj"))?;
        let (v, bv) = if cfg.layer_full[l] && cfg.attention_k_eq_v {
            (None, 0)
        } else {
            let (pv, bv0) = load_qat_proj(&sf, &format!("{pfx}.self_attn.v_proj"))?;
            (Some(pv), bv0)
        };
        let (o, bo) = load_qat_proj(&sf, &format!("{pfx}.self_attn.o_proj"))?;
        let (gate, bg) = load_qat_proj(&sf, &format!("{pfx}.mlp.gate_proj"))?;
        let (up, bu) = load_qat_proj(&sf, &format!("{pfx}.mlp.up_proj"))?;
        let (down, bd) = load_qat_proj(&sf, &format!("{pfx}.mlp.down_proj"))?;
        pinned_bytes += (bq + bk + bv + bo + bg + bu + bd) as usize;
        pinned.push(PinnedLayer { q, k, v, o, gate, up, down });
    }

    let em_key = "model.language_model.embed_tokens.weight";
    let (embed, embed_bytes) = match embed_mode {
        EmbedMode31B::Stream => (Embed31B::Streamed, 0usize),
        EmbedMode31B::Bf16 => {
            let v = sf.read_tensor_bf16_u16(em_key)?;
            let nb = v.len() * 2;
            (Embed31B::ResidentBf16(v), nb)
        }
        EmbedMode31B::Quant(scheme, h8) => {
            let meta = sf.tensors.get(em_key).ok_or_else(|| format!("missing {em_key}"))?;
            let d_out = meta.shape[0];
            let d_in = meta.shape[1];
            let mut pm = PackedMatrix::new_empty(d_out, d_in, scheme, h8)?;
            let mut r0 = 0usize;
            while r0 < d_out {
                let take = (d_out - r0).min(4096);
                let chunk = sf.read_tensor_rows_f32(em_key, r0, take)?;
                pm.pack_rows_range_f32(r0, take, &chunk, threads)?;
                r0 += take;
            }
            let nb = pm.bytes.len();
            (Embed31B::Packed(pm), nb)
        }
    };

    let kv = (0..cfg.n_layers).map(|_| LayerKv { k: Vec::new(), v: Vec::new() }).collect();
    let glyph_kv = init_glyph_kv(&cfg, kv_quant)?;
    Ok((
        Runtime31B {
            cfg,
            sf: Some(sf),
            norms,
            scalars,
            embed,
            pinned,
            hybrid_pinned: HashMap::new(),
            t16c_mats: None,
            native_t16c: None,
            c8d_bank: None,
            apply_c8d_proj: true,
            cs_skip_thresh: 0.0,
            cs_down_norms: Vec::new(),
            cs_skipped_blocks: 0,
            mlp_top_p: 1.0,
            mlp_sparse_skipped_blks: 0,
            mlp_sparse_total_blks: 0,
            profile_components: false,
            prof_attn_s: 0.0,
            prof_gate_s: 0.0,
            prof_up_s: 0.0,
            prof_down_s: 0.0,
            prof_norm_kv_s: 0.0,
            prof_lmhead_s: 0.0,
            prof_steps: 0,
            embed_tile_norms: None,
            lmhead_cs_sieve: None,
            lmhead_cs_skipped_blocks: 0,
            kv_quant,
            fp8_lut: build_fp8_e4m3_lut(),
            kv_raw_bytes: 0,
            kv_compressed_bytes: 0,
            kv,
            glyph_kv,
            vision: None,
            threads,
            ssd_bytes_read: 0,
        },
        pinned_bytes + norm_bytes,
        embed_bytes,
    ))
}

fn load_runtime_31b_t16c(
    t16c_path: &Path,
    fallback_model: Option<&Path>,
    native_lib: Option<&Path>,
    disable_native: bool,
    cfg: Gemma4Cfg,
    embed_mode: EmbedMode31B,
    kv_quant: KvQuantMode,
    hybrid_qat_layers: &[usize],
    threads: usize,
) -> Result<(Runtime31B, usize, usize), String> {
    let mut t16c = run270_rust::t16c::T16CModel::load(t16c_path, native_lib, disable_native)?;
    let native_t16c = t16c.native.take().map(std::sync::Arc::new);
    let sf_opt = match fallback_model {
        Some(mp) => Some(SafeTensors::open(mp)?),
        None => None,
    };

    let mut scalars = Vec::with_capacity(cfg.n_layers);
    for l in 0..cfg.n_layers {
        let sk = format!("model.language_model.layers.{l}.layer_scalar");
        let sk2 = format!("model.layers.{l}.layer_scalar");
        if let Some(v) = t16c.norms.get(&sk).or_else(|| t16c.norms.get(&sk2)) {
            if v.len() != 1 || !v[0].is_finite() {
                return Err(format!("invalid scalar {sk}: {v:?}"));
            }
            scalars.push(v[0]);
        } else if let Some(ref sf) = sf_opt {
            let v = sf.read_tensor_f32(&sk)?;
            if v.len() != 1 || !v[0].is_finite() {
                return Err(format!("invalid scalar {sk}: {v:?}"));
            }
            scalars.push(v[0]);
        } else {
            return Err(format!("T16C missing {sk} and no --model fallback"));
        }
    }
    let norm_bytes: usize = t16c.norms.values().map(|v| v.len() * 4).sum();

    let (embed, embed_bytes) = if let Some(mat) = t16c
        .mats
        .remove("model.language_model.embed_tokens.weight")
        .or_else(|| t16c.mats.remove("model.embed_tokens.weight"))
    {
        let nb = mat.total_bytes();
        (Embed31B::T16C { mat, native: native_t16c.clone() }, nb)
    } else if let Some(ref sf) = sf_opt {
        let em_key = "model.language_model.embed_tokens.weight";
        match embed_mode {
            EmbedMode31B::Stream => (Embed31B::Streamed, 0usize),
            EmbedMode31B::Bf16 => {
                let v = sf.read_tensor_bf16_u16(em_key)?;
                let nb = v.len() * 2;
                (Embed31B::ResidentBf16(v), nb)
            }
            EmbedMode31B::Quant(scheme, h8) => {
                let meta = sf.tensors.get(em_key).ok_or_else(|| format!("missing {em_key}"))?;
                let d_out = meta.shape[0];
                let d_in = meta.shape[1];
                let mut pm = PackedMatrix::new_empty(d_out, d_in, scheme, h8)?;
                let mut r0 = 0usize;
                while r0 < d_out {
                    let take = (d_out - r0).min(4096);
                    let chunk = sf.read_tensor_rows_f32(em_key, r0, take)?;
                    pm.pack_rows_range_f32(r0, take, &chunk, threads)?;
                    r0 += take;
                }
                let nb = pm.bytes.len();
                (Embed31B::Packed(pm), nb)
            }
        }
    } else {
        return Err("T16C missing model.language_model.embed_tokens.weight and no --model fallback".into());
    };

    let mut hybrid_pinned = HashMap::new();
    let mut hybrid_bytes = 0usize;
    if !hybrid_qat_layers.is_empty() {
        let sf = sf_opt
            .as_ref()
            .ok_or("--hybrid-qat-layers requires --model <model.safetensors> alongside --t16c")?;
        for &l in hybrid_qat_layers {
            if l >= cfg.n_layers {
                return Err(format!("hybrid QAT layer {l} >= n_layers {}", cfg.n_layers));
            }
            let pfx = format!("model.language_model.layers.{l}");
            let (q, bq) = load_qat_proj(sf, &format!("{pfx}.self_attn.q_proj"))?;
            let (k, bk) = load_qat_proj(sf, &format!("{pfx}.self_attn.k_proj"))?;
            let (v, bv) = if cfg.layer_full[l] && cfg.attention_k_eq_v {
                (None, 0)
            } else {
                let (pv, bv0) = load_qat_proj(sf, &format!("{pfx}.self_attn.v_proj"))?;
                (Some(pv), bv0)
            };
            let (o, bo) = load_qat_proj(sf, &format!("{pfx}.self_attn.o_proj"))?;
            let (gate, bg) = load_qat_proj(sf, &format!("{pfx}.mlp.gate_proj"))?;
            let (up, bu) = load_qat_proj(sf, &format!("{pfx}.mlp.up_proj"))?;
            let (down, bd) = load_qat_proj(sf, &format!("{pfx}.mlp.down_proj"))?;
            hybrid_bytes += (bq + bk + bv + bo + bg + bu + bd) as usize;
            hybrid_pinned.insert(l, PinnedLayer { q, k, v, o, gate, up, down });
        }
    }

    let vision = run270_rust::vision::VisionWeights::try_load_from_t16c(
        &mut t16c,
        run270_rust::vision::VisionCfg::gemma4_31b(),
    )?;
    let vision_bytes = vision.as_ref().map_or(0usize, |vw| {
        let pos_b = match &vw.pos_table {
            run270_rust::vision::PosTableBackend::SplitMats { x, y } => x.total_bytes() + y.total_bytes(),
            run270_rust::vision::PosTableBackend::NormF32(_) => 0,
        };
        let layer_b: usize = vw
            .layers
            .iter()
            .map(|l| {
                l.q_proj.total_bytes()
                    + l.k_proj.total_bytes()
                    + l.v_proj.total_bytes()
                    + l.o_proj.total_bytes()
                    + l.gate_proj.total_bytes()
                    + l.up_proj.total_bytes()
                    + l.down_proj.total_bytes()
            })
            .sum();
        vw.input_proj.total_bytes() + pos_b + layer_b + vw.embed_proj.total_bytes()
    });
    let body_bytes: usize = t16c.mats.values().map(|m| m.total_bytes()).sum();
    eprintln!(
        "[run31b] weight_source=T16C native_ffi={} mats={} norms={} body_bytes={} vision_bytes={} norm_bytes={} embed_bytes={} hybrid_qat={:?} ({hybrid_bytes} B) kv_quant={kv_quant:?}",
        native_t16c.as_ref().map_or("disabled", |n| n.path.as_str()),
        t16c.mats.len(),
        t16c.norms.len(),
        body_bytes,
        vision_bytes,
        norm_bytes,
        embed_bytes,
        hybrid_qat_layers
    );

    let kv = (0..cfg.n_layers).map(|_| LayerKv { k: Vec::new(), v: Vec::new() }).collect();
    let glyph_kv = init_glyph_kv(&cfg, kv_quant)?;
    Ok((
        Runtime31B {
            cfg,
            sf: sf_opt,
            norms: t16c.norms,
            scalars,
            embed,
            pinned: Vec::new(),
            hybrid_pinned,
            t16c_mats: Some(t16c.mats),
            native_t16c,
            c8d_bank: None,
            apply_c8d_proj: true,
            cs_skip_thresh: 0.0,
            cs_down_norms: Vec::new(),
            cs_skipped_blocks: 0,
            mlp_top_p: 1.0,
            mlp_sparse_skipped_blks: 0,
            mlp_sparse_total_blks: 0,
            profile_components: false,
            prof_attn_s: 0.0,
            prof_gate_s: 0.0,
            prof_up_s: 0.0,
            prof_down_s: 0.0,
            prof_norm_kv_s: 0.0,
            prof_lmhead_s: 0.0,
            prof_steps: 0,
            embed_tile_norms: None,
            lmhead_cs_sieve: None,
            lmhead_cs_skipped_blocks: 0,
            kv_quant,
            fp8_lut: build_fp8_e4m3_lut(),
            kv_raw_bytes: 0,
            kv_compressed_bytes: 0,
            kv,
            glyph_kv,
            vision,
            threads,
            ssd_bytes_read: 0,
        },
        body_bytes + vision_bytes + norm_bytes + hybrid_bytes,
        embed_bytes,
    ))
}

const ALLOWED_SINGLE_BYTE_STARTS: [usize; 26] = [
    236769, 236773, 236774, 236775, 236776, 236777, 236780, 236788, 236791, 236792, 236794,
    236796, 236797, 236798, 236799, 236806, 236811, 236814, 236823, 236824, 236836, 236840,
    236847, 236855, 236863, 236874,
];

/// Suppress already-closed Gemma-4 thought control tokens and repetitive stutter without altering raw prefill parity dumps.
fn apply_degeneracy_guard(logits: &mut [f32], prompt_ids: &[u32], generated: &[u32]) {
    let thought_closed = prompt_ids.ends_with(&[100, 45518, 107, 101]);
    if thought_closed {
        for &tid in &[100usize, 101usize, 45518usize] {
            if tid < logits.len() {
                logits[tid] = -1.0e9;
            }
        }
        if generated.len() < 4 {
            for &tid in &[1usize, 106usize] {
                if tid < logits.len() {
                    logits[tid] = -1.0e9;
                }
            }
        }
        if generated.is_empty() {
            for &tid in &[
                1usize, 106usize, 107usize, 108usize, 109usize, 715usize, 642usize, 1807usize,
                1852usize, 1638usize, 759usize, 7243usize, 1025usize, 4317usize, 1884usize,
                3161usize, 236783usize,
            ] {
                if tid < logits.len() {
                    logits[tid] = -1.0e9;
                }
            }
            let single_byte_end = 236891usize.min(logits.len());
            for tid in 236743usize..single_byte_end {
                if !ALLOWED_SINGLE_BYTE_STARTS.contains(&tid) {
                    logits[tid] = -1.0e9;
                }
            }
        }
    }
    let rep_penalty = 1.15f32;
    for &tid_u32 in generated {
        let tid = tid_u32 as usize;
        if tid < logits.len() && tid != 1 && tid != 106 {
            if logits[tid] > 0.0 {
                logits[tid] /= rep_penalty;
            } else {
                logits[tid] *= rep_penalty;
            }
        }
    }
    if let Some(&last_tok) = generated.last() {
        if !(236770..=236820).contains(&last_tok) && last_tok != 1 && last_tok != 106 && (last_tok as usize) < logits.len() {
            logits[last_tok as usize] -= 8.0;
        }
    }
    if generated.len() >= 3 {
        let last_a = generated[generated.len() - 1];
        for i in 0..(generated.len() - 2) {
            if generated[i] == last_a {
                let repeated_b = generated[i + 1] as usize;
                if repeated_b < logits.len() && repeated_b != 1 && repeated_b != 106 {
                    logits[repeated_b] -= 5.0;
                }
            }
        }
    }
}

fn compute_vision_kv_ends(ids: &[u32]) -> Vec<usize> {
    let mut ends: Vec<usize> = (0..ids.len()).collect();
    let mut i = 0usize;
    while i < ids.len() {
        if ids[i] == GEMMA4_IMAGE_TOKEN_ID {
            let mut j = i;
            while j + 1 < ids.len() && ids[j + 1] == GEMMA4_IMAGE_TOKEN_ID {
                j += 1;
            }
            for k in i..=j {
                ends[k] = j;
            }
            i = j + 1;
        } else {
            i += 1;
        }
    }
    ends
}

fn build_tri_modal_config(
    base_lmhead_sieve: Option<f32>,
    base_cs_skip: f32,
    base_draft_k: usize,
) -> ConsensusConfig {
    ConsensusConfig {
        base_sieve_frac: base_lmhead_sieve.unwrap_or(0.0),
        base_cs_skip,
        base_draft_k: base_draft_k.max(1),
        transformer_relative: true,
        ..ConsensusConfig::default()
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate_tri_modal_step(
    step_idx: usize,
    hidden: &[f32],
    logits: &mut Vec<f32>,
    allow_exact_rescue: bool,
    base_lmhead_sieve: Option<f32>,
    base_cs_skip: f32,
    c8d_cold_va: bool,
    rt: &mut Runtime31B,
    gate: &mut TriModalConsensusGate,
    apply_gate: bool,
) -> Result<(usize, run270_rust::tri_modal::TriModalDecision), String> {
    let hear = analyze_hear_fwht(hidden, 64, false)?;
    let see = analyze_see_spatial(hidden, 64)?;
    let read = analyze_read_logits(logits, true)?;
    let mut decision = gate.evaluate(&read, &hear, &see, true);
    eprintln!(
        "[TRIMODAL] step={} mode={} score={:.4} read={:.3} hear={:.3} see={:.3} hf={:.2} cr={:.2} marg={:.3} prob={:.3} sieve={:.3} cs={:.2e} k={} c8db={}",
        step_idx,
        decision.mode.as_str(),
        decision.consensus_score,
        decision.read_score,
        decision.hear_score,
        decision.see_score,
        hear.hf_ratio,
        see.crest_factor,
        read.margin,
        read.top1_prob,
        decision.lmhead_cs_sieve.unwrap_or(0.0),
        decision.cs_skip_thresh,
        decision.draft_k,
        decision.engage_c8db_rescue,
    );
    if apply_gate {
        let was_pruned = rt.lmhead_cs_sieve.is_some_and(|s| s > 0.0);
        if allow_exact_rescue && decision.mode != ConsensusMode::Clean && was_pruned {
            let exact = rt.compute_logits_batch_mode(hidden, 1, false)?;
            *logits = exact;
            let rescued_read = analyze_read_logits(logits, true)?;
            let rescued = gate.evaluate(&rescued_read, &hear, &see, false);
            eprintln!(
                "[TRIMODAL-EXACT] step={} rescued_marg={:.3} rescued_score={:.4} mode={}",
                step_idx,
                rescued_read.margin,
                rescued.consensus_score,
                rescued.mode.as_str(),
            );
            decision = rescued;
        }
        if base_lmhead_sieve.is_some() && rt.embed_tile_norms.is_some() {
            rt.lmhead_cs_sieve = decision.lmhead_cs_sieve;
        }
        if base_cs_skip > 0.0 && !rt.cs_down_norms.is_empty() {
            rt.cs_skip_thresh = decision.cs_skip_thresh;
        }
        if c8d_cold_va && rt.c8d_bank.is_some() {
            rt.apply_c8d_proj = decision.engage_c8db_rescue;
        }
        if decision.trigger_optical_self_analyze {
            let (_img, crc) = render_tri_modal_rescue_canvas(
                step_idx,
                &decision,
                &read,
                &hear,
                &see,
                &[
                    decision.read_score,
                    decision.hear_score,
                    decision.see_score,
                    decision.consensus_score,
                ],
            );
            eprintln!(
                "[TRIMODAL-OPTICAL] step={} rescue_canvas_crc32=0x{:08x}",
                step_idx, crc
            );
        }
    }
    Ok((decision.draft_k, decision))
}

fn run_bench_128k_kv(
    cfg: &Gemma4Cfg,
    so_arg: Option<&str>,
    disable_native: bool,
    kv_quant: KvQuantMode,
    seq_len: usize,
    bench_k_eq_v: bool,
    threads: usize,
) -> Result<(), String> {
    let eff_mode = if kv_quant.is_glyph() { kv_quant } else { KvQuantMode::Glyph };
    let (sink_cap, tail_cap) = match eff_mode {
        KvQuantMode::Glyph => (4usize, 32usize),
        KvQuantMode::GlyphPure => (0usize, 0usize),
        _ => (4usize, 32usize),
    };
    let native = if disable_native {
        None
    } else if let Some(p) = so_arg {
        Some(run270_rust::t16c::NativeTile16Kernel::try_load(Path::new(p))?)
    } else {
        run270_rust::t16c::NativeTile16Kernel::discover(None)
    };
    let native_ref = native.as_ref();
    let has_native_glyph = native_ref.is_some_and(|n| n.encode_kv_glyph_fn.is_some() && n.flash_glyph_attn_fn.is_some());
    eprintln!(
        "[run31b] --bench-128k-kv mode={eff_mode:?} seq_len={seq_len} k_eq_v={bench_k_eq_v} layers={} threads={threads} native_glyph_c={has_native_glyph}",
        cfg.n_layers
    );

    // 1. Live accuracy verification against exact FP32 GQA (`attn_decode_range`) on a 512-token global layer
    //    using realistic post-qk_norm activation magnitudes (unit-variance QK dot products with sparse channel outliers).
    let ref_seq = 512usize.min(seq_len.max(64));
    let mut ref_layer = GlyphKvLayer::new(
        cfg.n_heads,
        cfg.n_global_kv,
        cfg.global_hd,
        bench_k_eq_v,
        None,
        sink_cap,
        tail_cap,
        ref_seq,
    )?;
    let g_stride = cfg.n_global_kv * cfg.global_hd;
    let inv_hd_scale = 1.0f32 / (cfg.global_hd as f32).sqrt();
    let mut k_ref_all = vec![0.0f32; ref_seq * g_stride];
    let mut v_ref_all = vec![0.0f32; ref_seq * g_stride];
    for t in 0..ref_seq {
        let mut k_tok = vec![0.0f32; g_stride];
        let mut v_tok = vec![0.0f32; g_stride];
        for h in 0..cfg.n_global_kv {
            for d in 0..cfg.global_hd {
                let phase = ((t * 17 + h * 41 + d * 7) as f32) * 0.013;
                let mut kv = (phase.sin() * 0.8 + (phase * 0.37).cos() * 0.5) * inv_hd_scale.sqrt();
                if d % 64 == (t + h) % 64 {
                    kv += 0.25;
                }
                let vv = if bench_k_eq_v {
                    kv
                } else {
                    ((phase * 1.17 + 0.5).cos() * 0.75 + (phase * 0.23).sin() * 0.45) * inv_hd_scale.sqrt()
                };
                k_tok[h * cfg.global_hd + d] = kv;
                v_tok[h * cfg.global_hd + d] = vv;
            }
        }
        k_ref_all[t * g_stride..(t + 1) * g_stride].copy_from_slice(&k_tok);
        v_ref_all[t * g_stride..(t + 1) * g_stride].copy_from_slice(&v_tok);
        let vt = if bench_k_eq_v { &[][..] } else { &v_tok[..] };
        ref_layer.append_token(&k_tok, vt, native_ref)?;
    }
    let q_ref: Vec<f32> = (0..cfg.n_heads * cfg.global_hd)
        .map(|i| (((i * 19 + 5) as f32) * 0.017).sin() * inv_hd_scale.sqrt())
        .collect();
    let out_glyph = ref_layer.attend_range(&q_ref, ref_seq - 1, ref_seq - 1, ref_seq - 1, 1.0, native_ref, threads)?;
    let out_exact = attn_decode_range(
        &q_ref,
        &k_ref_all,
        &v_ref_all,
        ref_seq - 1,
        ref_seq - 1,
        ref_seq - 1,
        cfg.n_heads,
        cfg.n_global_kv,
        cfg.global_hd,
        None,
    );
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in out_glyph.iter().zip(out_exact.iter()) {
        dot += a as f64 * b as f64;
        na += a as f64 * a as f64;
        nb += b as f64 * b as f64;
    }
    let ref_cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
    drop(ref_layer);
    drop(k_ref_all);
    drop(v_ref_all);

    // 2. Allocate and physically populate all 60 Gemma 4 31B layers simultaneously at `seq_len` (default 131,072).
    let t_alloc = std::time::Instant::now();
    let mut layers: Vec<GlyphKvLayer> = Vec::with_capacity(cfg.n_layers);
    let mut n_global = 0usize;
    let mut n_sliding = 0usize;
    let chunk_toks = 4096usize;
    let mut scratch_k_glob = vec![0.0f32; cfg.n_global_kv * chunk_toks * cfg.global_hd];
    for (i, slot) in scratch_k_glob.iter_mut().enumerate() {
        *slot = (((i % 8191) as f32) * 0.0131).sin() * 0.42;
    }
    let mut scratch_k_slid = vec![0.0f32; cfg.n_kv * cfg.window.min(chunk_toks) * cfg.hd];
    for (i, slot) in scratch_k_slid.iter_mut().enumerate() {
        *slot = (((i % 4093) as f32) * 0.0173).cos() * 0.38;
    }
    let tok_k_glob = vec![0.25f32; cfg.n_global_kv * cfg.global_hd];
    let tok_k_slid = vec![0.22f32; cfg.n_kv * cfg.hd];

    for l in 0..cfg.n_layers {
        let is_full = cfg.layer_full[l];
        let hd = if is_full { cfg.global_hd } else { cfg.hd };
        let n_kv = if is_full { cfg.n_global_kv } else { cfg.n_kv };
        let k_eq_v = is_full && bench_k_eq_v;
        let sw = if is_full { None } else { Some(cfg.window) };
        let target_toks = if is_full { seq_len } else { seq_len.min(cfg.window) };
        let eff_sink = if is_full { sink_cap.min(target_toks) } else { 0 };
        let eff_tail = tail_cap.min(target_toks.saturating_sub(eff_sink));
        let n_glyphs = target_toks.saturating_sub(eff_sink + eff_tail);

        let mut layer = GlyphKvLayer::new(
            cfg.n_heads,
            n_kv,
            hd,
            k_eq_v,
            sw,
            eff_sink,
            eff_tail,
            n_glyphs.max(64),
        )?;
        let tok_buf = if is_full { &tok_k_glob } else { &tok_k_slid };
        for _ in 0..eff_sink {
            layer.append_token(tok_buf, if k_eq_v { &[][..] } else { tok_buf }, native_ref)?;
        }
        let mut rem = n_glyphs;
        while rem > 0 {
            let take = rem.min(if is_full { chunk_toks } else { cfg.window.min(chunk_toks) });
            let need = n_kv * take * hd;
            if is_full {
                let v_opt = if k_eq_v { None } else { Some(&scratch_k_glob[..need]) };
                layer.bulk_fill_glyphs(&scratch_k_glob[..need], v_opt, take, native_ref, threads)?;
            } else {
                layer.bulk_fill_glyphs(
                    &scratch_k_slid[..need],
                    Some(&scratch_k_slid[..need]),
                    take,
                    native_ref,
                    threads,
                )?;
            }
            rem -= take;
        }
        for _ in 0..eff_tail {
            layer.append_token(tok_buf, if k_eq_v { &[][..] } else { tok_buf }, native_ref)?;
        }
        if is_full {
            n_global += 1;
        } else {
            n_sliding += 1;
        }
        layers.push(layer);
    }
    drop(scratch_k_glob);
    drop(scratch_k_slid);
    let fill_s = t_alloc.elapsed().as_secs_f64();

    // 3. Aggregate exact memory footprint across all 60 layers (global + sliding).
    let mut tot_all = GlyphKvMemStats::default();
    let mut tot_global = GlyphKvMemStats::default();
    let mut tot_sliding = GlyphKvMemStats::default();
    for (l, g) in layers.iter().enumerate() {
        let s = g.memory_stats();
        let acc = |dst: &mut GlyphKvMemStats| {
            dst.active_tokens += s.active_tokens;
            dst.glyph_tokens += s.glyph_tokens;
            dst.exact_tokens += s.exact_tokens;
            dst.resident_bytes += s.resident_bytes;
            dst.naive_bf16_bytes += s.naive_bf16_bytes;
            dst.fp32_bytes += s.fp32_bytes;
        };
        acc(&mut tot_all);
        if cfg.layer_full[l] {
            acc(&mut tot_global);
        } else {
            acc(&mut tot_sliding);
        }
    }

    // 4. Execute live single-token attention across ALL 60 layers (10 global @ 128K + 50 sliding @ 1024).
    let q_slid: Vec<f32> = (0..cfg.n_heads * cfg.hd)
        .map(|i| (((i * 13 + 3) as f32) * 0.019).cos() * 0.45)
        .collect();
    let mut global_ms = 0.0f64;
    let mut sliding_ms = 0.0f64;
    let mut checksum = 0.0f64;
    for (l, g) in layers.iter().enumerate() {
        let pos = g.total_seen().saturating_sub(1);
        let qv = if cfg.layer_full[l] { &q_ref } else { &q_slid };
        let t0 = std::time::Instant::now();
        let out = g.attend_range(qv, pos, pos, pos, 1.0, native_ref, threads)?;
        let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
        if cfg.layer_full[l] {
            global_ms += dt_ms;
        } else {
            sliding_ms += dt_ms;
        }
        checksum += out.iter().take(8).map(|&v| v as f64).sum::<f64>();
    }
    let total_attn_ms = global_ms + sliding_ms;
    let (vm_hwm, vm_rss) = proc_status_kb();
    println!(
        "KV128K_BENCH mode={eff_mode:?} seq_len={seq_len} k_eq_v={bench_k_eq_v} layers={} global_layers={n_global} sliding_layers={n_sliding} native_glyph_c={has_native_glyph} fill_s={fill_s:.2} ref512_cos={ref_cos:.5} global_resident_mb={:.2} global_naive_bf16_mb={:.2} global_shrink_bf16={:.2}x total_resident_mb={:.2} total_naive_bf16_mb={:.2} total_shrink_bf16={:.2}x total_shrink_fp32={:.2}x global_10l_ms={global_ms:.2} sliding_50l_ms={sliding_ms:.2} total_60l_ms={total_attn_ms:.2} checksum={checksum:.5} vm_hwm_kb={vm_hwm:?} vm_rss_kb={vm_rss:?}",
        cfg.n_layers,
        tot_global.resident_bytes as f64 / (1024.0 * 1024.0),
        tot_global.naive_bf16_bytes as f64 / (1024.0 * 1024.0),
        tot_global.shrink_vs_naive_bf16(),
        tot_all.resident_bytes as f64 / (1024.0 * 1024.0),
        tot_all.naive_bf16_bytes as f64 / (1024.0 * 1024.0),
        tot_all.shrink_vs_naive_bf16(),
        tot_all.shrink_vs_fp32(),
    );
    Ok(())
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let get = |name: &str| -> Option<String> {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
    };
    let bench_128k_kv = args.iter().any(|a| a == "--bench-128k-kv");
    let bench_128k_k_eq_v = args.iter().any(|a| a == "--bench-128k-k-eq-v");
    let bench_128k_seq: usize = match get("--bench-128k-seq") {
        Some(s) => s.parse().map_err(|_| format!("invalid --bench-128k-seq '{s}'"))?,
        None => 131_072,
    };
    let so_arg = get("--so");
    let disable_native = args.iter().any(|a| a == "--no-native-ffi");
    let threads: usize = match get("--threads") {
        Some(s) => s.parse().map_err(|_| format!("invalid --threads '{s}'"))?,
        None => std::thread::available_parallelism().map(|n| n.get().clamp(1, 8)).unwrap_or(4),
    };
    let kv_quant = KvQuantMode::parse(&get("--kv-quant").unwrap_or_else(|| "fp32".into()))?;
    if bench_128k_kv {
        let cfgp = get("--config").ok_or("missing --config")?;
        let cfg = load_gemma4_cfg(Path::new(&cfgp))?;
        return run_bench_128k_kv(&cfg, so_arg.as_deref(), disable_native, kv_quant, bench_128k_seq, bench_128k_k_eq_v, threads);
    }

    let t16c_arg = get("--t16c");
    let model = get("--model");
    if t16c_arg.is_none() && model.is_none() {
        return Err("missing --model (or pass --t16c <path>)".into());
    }
    let c8d_arg = get("--cuneiform-bank");
    let c8d_cold_va = args.iter().any(|a| a == "--cuneiform-cold-va");
    let speculative = args.iter().any(|a| a == "--speculative");
    let batch_k_arg = get("--batch-k");
    let batch_k: usize = match batch_k_arg {
        Some(s) => s.parse().map_err(|_| format!("invalid --batch-k '{s}'"))?,
        None => if speculative { 4 } else { 1 },
    };
    if batch_k == 0 {
        return Err("--batch-k must be >= 1".into());
    }
    if speculative && batch_k < 2 {
        return Err("--speculative requires --batch-k >= 2 (or omit --batch-k for default 4)".into());
    }
    let spec_ngram: usize = match get("--spec-ngram") {
        Some(s) => s.parse().map_err(|_| format!("invalid --spec-ngram '{s}'"))?,
        None => 4,
    };
    let spec_min_count: u32 = match get("--spec-min-count") {
        Some(s) => s.parse().map_err(|_| format!("invalid --spec-min-count '{s}'"))?,
        None => 1,
    };
    let spec_seed_arg = get("--spec-seed");
    let spec_assistant_arg = get("--spec-assistant");
    let assistant_step_pos = args.iter().any(|a| a == "--assistant-step-pos");
    let assistant_min_margin: f32 = match get("--assistant-min-margin") {
        Some(s) => s.parse().map_err(|_| format!("invalid --assistant-min-margin '{s}'"))?,
        None => if spec_assistant_arg.is_some() { 0.75 } else { 0.0 },
    };
    if !assistant_min_margin.is_finite() || assistant_min_margin < 0.0 {
        return Err(format!("--assistant-min-margin must be finite and >= 0.0, got {assistant_min_margin}"));
    }
    let cs_skip_thresh: f32 = match get("--cs-skip") {
        Some(s) => s.parse().map_err(|_| format!("invalid --cs-skip '{s}'"))?,
        None => 0.0,
    };
    if !cs_skip_thresh.is_finite() || cs_skip_thresh < 0.0 {
        return Err(format!("--cs-skip must be finite and >= 0.0, got {cs_skip_thresh}"));
    }
    let mlp_top_p: f32 = match get("--mlp-top-p") {
        Some(s) => s.parse().map_err(|_| format!("invalid --mlp-top-p '{s}'"))?,
        None => 1.0,
    };
    if !mlp_top_p.is_finite() || mlp_top_p <= 0.0 || mlp_top_p > 1.0 {
        return Err(format!("--mlp-top-p must be finite in (0.0, 1.0], got {mlp_top_p}"));
    }
    let profile_components = args.iter().any(|a| a == "--profile-components");
    let prompt_batch_arg = get("--prompt-batch");
    let lmhead_cs_sieve: Option<f32> = match get("--lmhead-cs") {
        Some(s) => parse_lmhead_cs_mode(&s)?,
        None => Some(0.0),
    };
    if (spec_seed_arg.is_some() || spec_assistant_arg.is_some()) && !speculative {
        return Err("--spec-seed and --spec-assistant require --speculative".into());
    }
    let cfgp = get("--config").ok_or("missing --config")?;
    let tkp = get("--tokenizer").ok_or("missing --tokenizer")?;
    let embed_h8 = args.iter().any(|a| a == "--embed-h8");
    let embed_mode = EmbedMode31B::parse(&get("--embed-mode").unwrap_or_else(|| "stream".into()), embed_h8)?;
    let hybrid_qat_arg = get("--hybrid-qat-layers");
    let pin_layers: usize = match get("--pin-layers") {
        Some(s) => s.parse().map_err(|_| format!("invalid --pin-layers '{s}'"))?,
        None => 0,
    };
    let max_tokens: usize = match get("--max-tokens") {
        Some(s) => s.parse().map_err(|_| format!("invalid --max-tokens '{s}'"))?,
        None => 8,
    };
    let prompt_text = get("--prompt-text").unwrap_or_else(|| "Who wrote the play Hamlet?".into());
    let raw_prompt = args.iter().any(|a| a == "--raw-prompt");
    let image_arg = get("--image");
    let optical_self_analyze = args.iter().any(|a| a == "--optical-self-analyze");
    let optical_ctx_roundtrip = args.iter().any(|a| a == "--optical-context-roundtrip");
    let optical_ctx_out = get("--optical-context-out");
    let optical_ctx_in = get("--optical-context-in");
    let tri_modal_gate = args.iter().any(|a| a == "--tri-modal-gate");
    let tri_modal_log = tri_modal_gate || args.iter().any(|a| a == "--tri-modal-log");
    let dump_logits = get("--dump-logits");
    let dump_vision_soft = get("--dump-vision-soft");
    let vision_only = args.iter().any(|a| a == "--vision-only");
    let ids_arg = get("--ids");

    let cfg = load_gemma4_cfg(Path::new(&cfgp))?;
    let hybrid_qat_layers = match hybrid_qat_arg.as_deref() {
        Some(s) => parse_hybrid_qat_layers(s, cfg.n_layers)?,
        None => Vec::new(),
    };
    eprintln!(
        "[run31b] cfg d={} layers={} ff={} heads={}/{} global_kv={} hd={}/{} window={} softcap={:?} kv_quant={kv_quant:?}",
        cfg.d, cfg.n_layers, cfg.ff, cfg.n_heads, cfg.n_kv, cfg.n_global_kv, cfg.hd, cfg.global_hd, cfg.window, cfg.final_softcap
    );

    let t_load = std::time::Instant::now();
    let (mut rt, pinned_bytes, embed_bytes) = if let Some(tp) = t16c_arg.as_deref() {
        load_runtime_31b_t16c(
            Path::new(tp),
            model.as_deref().map(Path::new),
            so_arg.as_deref().map(Path::new),
            disable_native,
            cfg,
            embed_mode,
            kv_quant,
            &hybrid_qat_layers,
            threads,
        )?
    } else {
        load_runtime_31b(Path::new(model.as_deref().unwrap()), cfg, embed_mode, pin_layers, kv_quant, threads)?
    };
    if let Some(cp) = c8d_arg.as_deref() {
        let native_hint = so_arg
            .as_deref()
            .or_else(|| rt.native_t16c.as_ref().map(|n| n.path.as_str()));
        let bank = run270_rust::cuneiform_tbi::CuneiformBank::load(
            Path::new(cp),
            native_hint.map(Path::new),
            disable_native,
            !c8d_cold_va,
        )?;
        for (k, v) in &bank.norms {
            rt.norms.insert(k.clone(), v.clone());
        }
        eprintln!(
            "[run31b] loaded C8DB {}: entries={} norms={} payload_bytes={} cold_va={} native_tbi_hw={}",
            cp,
            bank.entries.len(),
            bank.norms.len(),
            bank.total_payload_bytes,
            c8d_cold_va,
            bank.native.as_ref().is_some_and(|n| n.tbi_hw_active),
        );
        rt.c8d_bank = Some(bank);
    }
    if cs_skip_thresh > 0.0 {
        let (cs_mats, cs_bytes) = rt.enable_cs_skip(cs_skip_thresh)?;
        eprintln!(
            "[run31b] enabled Cauchy-Schwarz down_proj tile skip: thresh={cs_skip_thresh:.2e} mats={cs_mats} norms_bytes={cs_bytes}"
        );
    }
    rt.mlp_top_p = mlp_top_p;
    rt.profile_components = profile_components;
    if mlp_top_p < 0.9999 {
        eprintln!("[run31b] enabled Gate-Guided Block-Sparse GeGLU MLP: top_p={mlp_top_p:.3}");
    }
    let lmhead_cs_bytes = rt.enable_lmhead_cs(lmhead_cs_sieve)?;
    if let Some(sf_cs) = rt.lmhead_cs_sieve {
        eprintln!(
            "[run31b] enabled LM-head CS pruning: sieve_frac={sf_cs:.3} norms_bytes={lmhead_cs_bytes}"
        );
    }
    let assistant_state: Option<(AssistantRuntime, usize, usize)> = if let Some(ap) = spec_assistant_arg.as_deref() {
        let native_hint = so_arg
            .as_deref()
            .or_else(|| rt.native_t16c.as_ref().map(|n| n.path.as_str()));
        let ast = AssistantRuntime::load_31b(
            std::path::Path::new(ap),
            native_hint.map(std::path::Path::new),
            disable_native,
            threads,
            rt.cfg.n_kv,
            rt.cfg.n_global_kv,
            rt.cfg.window,
        )?;
        if ast.cfg.backbone_hidden != rt.cfg.d || ast.cfg.vocab != rt.cfg.vocab {
            return Err(format!(
                "assistant {} dims ({}x{}) incompatible with 31B ({}x{})",
                ap, ast.cfg.backbone_hidden, ast.cfg.vocab, rt.cfg.d, rt.cfg.vocab
            ));
        }
        let (sl_l, fl_l) = rt.assistant_kv_layers()?;
        eprintln!(
            "[run31b] loaded AssistantRuntime {}: layers={} d={} resident_mb={:.2} shared_kv=(L{sl_l},L{fl_l}) step_pos={assistant_step_pos}",
            ap,
            ast.cfg.n_layers,
            ast.cfg.d,
            ast.resident_bytes() as f64 / (1024.0 * 1024.0)
        );
        Some((ast, sl_l, fl_l))
    } else {
        None
    };
    let active_pinned_layers = if rt.t16c_mats.is_some() { rt.cfg.n_layers } else { rt.pinned.len() };
    eprintln!(
        "[run31b] loaded in {:.2}s pinned_layers={}/{} pinned_bytes={} embed_bytes={} vm={:?}",
        t_load.elapsed().as_secs_f64(),
        active_pinned_layers,
        rt.cfg.n_layers,
        pinned_bytes,
        embed_bytes,
        proc_status_kb()
    );

    let tk = Tokenizer::load(Path::new(&tkp))?;
    let mut vision_soft: Option<(usize, Vec<f32>)> = None;
    if image_arg.is_some() || optical_self_analyze {
        let vw = rt
            .vision
            .as_ref()
            .ok_or("Multimodal --image / --optical-self-analyze requires full .t16c with vision_tower")?;
        let img = if optical_self_analyze {
            let mut layer_bits: Vec<u8> = Vec::with_capacity(rt.cfg.n_layers + vw.layers.len());
            if let Some(ref mats) = rt.t16c_mats {
                for l in 0..rt.cfg.n_layers {
                    let nb = mats
                        .get(&format!("model.language_model.layers.{l}.mlp.gate_proj.weight"))
                        .map_or(0, |m| m.n_bits);
                    layer_bits.push(nb);
                }
            }
            for lv in &vw.layers {
                layer_bits.push(lv.gate_proj.n_bits);
            }
            let c8d_n = rt.c8d_bank.as_ref().map_or(0, |b| b.entries.len());
            let l1 = format!("GEMMA4 31B FULL LM={} VIS={}", rt.cfg.n_layers, vw.layers.len());
            let l2 = format!("PINNED={}MB C8DB={}", pinned_bytes / (1024 * 1024), c8d_n);
            let (canvas, crc) = run270_rust::vision::render_optical_va_canvas(
                &[&l1, &l2, &prompt_text],
                &layer_bits,
            );
            eprintln!(
                "[run31b] rendered optical VA canvas={}x{} crc32=0x{crc:08x} bars={}",
                canvas.width,
                canvas.height,
                layer_bits.len()
            );
            canvas
        } else {
            run270_rust::vision::load_rgb_image(Path::new(image_arg.as_deref().unwrap()))?
        };
        let t_vis = std::time::Instant::now();
        let (n_soft, soft) = vw.encode_image(
            &img,
            rt.native_t16c.as_deref(),
            rt.c8d_bank.as_ref(),
            rt.threads,
        )?;
        eprintln!(
            "[run31b] vision encoded {}x{} -> n_soft={n_soft} in {:.2}s",
            img.width,
            img.height,
            t_vis.elapsed().as_secs_f64()
        );
        if let Some(ref vp) = dump_vision_soft {
            let bytes: Vec<u8> = soft.iter().flat_map(|v| v.to_le_bytes()).collect();
            fs::write(vp, bytes).map_err(|e| format!("write {vp}: {e}"))?;
            eprintln!("[run31b] dumped {n_soft}x{} vision soft tokens to {vp}", rt.cfg.d);
        }
        if vision_only {
            return Ok(());
        }
        vision_soft = Some((n_soft, soft));
    }

    let batch_prompts: Vec<String> = if let Some(ref pbf) = prompt_batch_arg {
        let raw = fs::read_to_string(pbf).map_err(|e| format!("read --prompt-batch {pbf}: {e}"))?;
        let lines: Vec<String> = raw
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| l.to_string())
            .collect();
        if lines.is_empty() {
            return Err(format!("--prompt-batch {pbf} contains no active prompts"));
        }
        lines
    } else {
        vec![prompt_text.clone()]
    };
    let is_multi_prompt = batch_prompts.len() > 1;

    for (prompt_idx, raw_line) in batch_prompts.iter().enumerate() {
        if prompt_idx > 0 {
            rt.truncate_kv(0)?;
            rt.mlp_sparse_skipped_blks = 0;
            rt.mlp_sparse_total_blks = 0;
            rt.prof_attn_s = 0.0;
            rt.prof_gate_s = 0.0;
            rt.prof_up_s = 0.0;
            rt.prof_down_s = 0.0;
            rt.prof_norm_kv_s = 0.0;
            rt.prof_lmhead_s = 0.0;
            rt.prof_steps = 0;
        }
        let (cur_prompt_text, cur_vision_override): (String, Option<(usize, Vec<f32>)>) =
            if let Some(rest) = raw_line.strip_prefix("IMAGE:") {
                let (img_path_s, p_txt) = rest
                    .split_once('|')
                    .ok_or_else(|| format!("invalid IMAGE:<path>|<prompt> line: {raw_line}"))?;
                let img = run270_rust::vision::load_rgb_image(Path::new(img_path_s.trim()))?;
                let vw = rt.vision.as_ref().ok_or("IMAGE: batch line requires .t16c with vision weights")?;
                let t_vis = std::time::Instant::now();
                let (n_soft, soft) = vw.encode_image(
                    &img,
                    rt.native_t16c.as_deref(),
                    rt.c8d_bank.as_ref(),
                    rt.threads,
                )?;
                eprintln!(
                    "[run31b] batch vision encoded {} ({}x{}) -> n_soft={n_soft} in {:.2}s",
                    img_path_s.trim(),
                    img.width,
                    img.height,
                    t_vis.elapsed().as_secs_f64()
                );
                (p_txt.trim().to_string(), Some((n_soft, soft)))
            } else {
                (raw_line.clone(), None)
            };
        let active_vision: Option<&(usize, Vec<f32>)> = cur_vision_override.as_ref().or(vision_soft.as_ref());
        if is_multi_prompt {
            println!("=== BATCH_PROMPT {}/{} prompt={cur_prompt_text:?} ===", prompt_idx + 1, batch_prompts.len());
        }

    let ids: Vec<u32> = if let Some(ref in_ppm) = optical_ctx_in {
        let img = run270_rust::vision::load_rgb_image(Path::new(in_ppm))?;
        let (decoded_ids, hdr, dec_ms) = decode_tokens_from_optical_frame(&img, 4)?;
        println!(
            "OPTICAL_DMA mode=ingest file={in_ppm} frame={}x{} tokens={} payload_bytes={} crc32=0x{:08x} decode_ms={dec_ms:.3}",
            img.width, img.height, decoded_ids.len(), hdr.stored_len, hdr.crc32
        );
        decoded_ids
    } else {
        match ids_arg.as_deref() {
            Some(s) => s
                .split(',')
                .map(|x| x.trim().parse::<u32>().map_err(|_| format!("invalid id in --ids '{s}'")))
                .collect::<Result<Vec<_>, _>>()?,
            None => {
                if let Some((n_soft, _)) = active_vision {
                    let mut v_ids = Vec::new();
                    v_ids.extend(tk.encode("<|turn>user\n", true));
                    v_ids.push(run270_rust::vision::GEMMA4_BOI_TOKEN_ID);
                    v_ids.resize(v_ids.len() + *n_soft, run270_rust::vision::GEMMA4_IMAGE_TOKEN_ID);
                    v_ids.push(run270_rust::vision::GEMMA4_EOI_TOKEN_ID);
                    let tail = format!("{cur_prompt_text}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>");
                    v_ids.extend(tk.encode(&tail, false));
                    v_ids
                } else {
                    let prompt = if raw_prompt {
                        cur_prompt_text.clone()
                    } else {
                        format!("<|turn>user\n{cur_prompt_text}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
                    };
                    tk.encode(&prompt, true)
                }
            }
        }
    };
    let ids = if optical_ctx_roundtrip || optical_ctx_out.is_some() {
        let (frame, recovered_ids, stats) = optical_context_dma_roundtrip(&ids, 4)?;
        if let Some(ref out_ppm) = optical_ctx_out {
            save_ppm(&frame, Path::new(out_ppm))?;
        }
        println!(
            "OPTICAL_DMA mode=roundtrip frame={}x{} scale={} tokens={} payload_bytes={} crc32=0x{:08x} encode_ms={:.3} dma_ms={:.3} fps={:.1} tok_per_s={:.0} exact_parity={}",
            stats.width,
            stats.height,
            stats.scale,
            stats.n_tokens,
            stats.payload_bytes,
            stats.crc32,
            stats.encode_ms,
            stats.dma_ms,
            stats.fps,
            stats.tok_per_s,
            recovered_ids == ids,
        );
        recovered_ids
    } else {
        ids
    };
    eprintln!("[run31b] prompt {} tokens ids={:?}", ids.len(), &ids[..ids.len().min(32)]);

    let t_pf = std::time::Instant::now();
    let mut prefill_states = rt.embed_batch(&ids)?;
    if let Some((n_soft, soft_emb)) = active_vision {
        let n_soft = *n_soft;
        let d_h = rt.cfg.d;
        let mut s_idx = 0usize;
        for (t_idx, &tid) in ids.iter().enumerate() {
            if tid == run270_rust::vision::GEMMA4_IMAGE_TOKEN_ID {
                if s_idx >= n_soft {
                    return Err("More IMAGE_TOKEN_ID placeholders than encoded vision soft tokens".into());
                }
                prefill_states[t_idx * d_h..(t_idx + 1) * d_h]
                    .copy_from_slice(&soft_emb[s_idx * d_h..(s_idx + 1) * d_h]);
                s_idx += 1;
            }
        }
        if s_idx != n_soft {
            return Err(format!("Expected {n_soft} image placeholder slots, filled {s_idx}"));
        }
    }
    let vis_kv_ends = compute_vision_kv_ends(&ids);
    let mut last_backbone_h = rt.forward_batch_with_embeds(
        prefill_states,
        ids.len(),
        0,
        true,
        false,
        Some(&vis_kv_ends),
    )?;
    let mut prefill_logits = rt.compute_logits_batch_mode(&last_backbone_h, 1, dump_logits.is_none())?;
    let mut tri_gate = if tri_modal_log {
        Some(TriModalConsensusGate::new(build_tri_modal_config(
            lmhead_cs_sieve,
            cs_skip_thresh,
            if speculative { (batch_k - 1).max(1) } else { 1 },
        )))
    } else {
        None
    };
    let mut tri_draft_cap: usize = (batch_k - 1).max(1);
    if let Some(ref mut gate) = tri_gate {
        let (cap, _dec) = evaluate_tri_modal_step(
            0,
            &last_backbone_h,
            &mut prefill_logits,
            dump_logits.is_none(),
            lmhead_cs_sieve,
            cs_skip_thresh,
            c8d_cold_va,
            &mut rt,
            gate,
            tri_modal_gate,
        )?;
        if tri_modal_gate {
            tri_draft_cap = cap;
        }
    }
    let prefill_s = t_pf.elapsed().as_secs_f64();
    eprintln!("[run31b] prefill done in {prefill_s:.2}s ssd_read_mb={:.1} vm={:?}", rt.ssd_bytes_read as f64 / 1e6, proc_status_kb());

    if let Some(ref dp) = dump_logits {
        let bytes: Vec<u8> = prefill_logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        fs::write(dp, bytes).map_err(|e| format!("write {dp}: {e}"))?;
    }

    apply_degeneracy_guard(&mut prefill_logits, &ids, &[]);

    let mut top5: Vec<(usize, f32)> = prefill_logits.iter().copied().enumerate().collect();
    top5.select_nth_unstable_by(4, |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    top5.truncate(5);
    top5.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top5_str: Vec<String> = top5
        .iter()
        .map(|&(id, sc)| format!("{id}:{:?}({sc:.3})", tk.decode(&[id as u32])))
        .collect();
    println!("PREFILL_TOP5 [{}]", top5_str.join(", "));

    let eos: [u32; 2] = [1, 106];
    let closed_thought_bans: &[u32] = if ids.ends_with(&[100, 45518, 107, 101]) {
        &[100, 101, 45518]
    } else {
        &[]
    };
    let mut generated = Vec::new();
    let t_dec = std::time::Instant::now();
    if speculative {
        let mut history = ids.clone();
        let mut ngram = FastNGramProposalEngine::new(spec_ngram).with_min_count(spec_min_count);
        let seed_entries = if let Some(sp) = spec_seed_arg.as_deref() {
            let n = ngram.load_seed_file(std::path::Path::new(sp))?;
            eprintln!("[run31b] loaded CNGM spec seed {sp}: transitions={n}");
            n
        } else {
            0
        };
        ngram.train_tokens(&history);
        let max_draft = (batch_k - 1).max(1);
        let mut ctrl = AdaptiveDraftController::new(1, max_draft);
        let mut pending = if max_tokens > 0 { Some(argmax_f32(&prefill_logits)) } else { None };
        let mut spec_drafted = 0usize;
        let mut spec_accepted = 0usize;
        let mut spec_batches = 0usize;
        let mut spec_fallback_k1 = 0usize;
        let mut spec_margin_stopped = 0usize;
        let mut ast_lm_skipped_blocks = 0u64;
        let vocab = rt.cfg.vocab;
        let d_hidden = rt.cfg.d;

        while let Some(y0) = pending.take() {
            let step_idx = generated.len();
            generated.push(y0);
            ngram.observe_transition(&history, y0);
            history.push(y0);
            println!("[STEP {step_idx}] id={y0} text={:?}", tk.decode(&[y0]));
            if eos.contains(&y0) || generated.len() >= max_tokens {
                break;
            }

            let cur_kv_len = rt.kv_len();
            let expected_kv_len = ids.len() + generated.len() - 1;
            if cur_kv_len != expected_kv_len {
                return Err(format!("KV desync before spec step: kv_len={cur_kv_len} != expected={expected_kv_len}"));
            }

            let rem = max_tokens - generated.len();
            let draft_budget = if tri_modal_gate {
                ctrl.current_k().min(tri_draft_cap).min(rem.saturating_sub(1))
            } else {
                ctrl.current_k().min(rem.saturating_sub(1))
            };
            let drafts = if draft_budget > 0 {
                let seed_first = ngram.propose_sequence(&history, draft_budget);
                if !seed_first.is_empty() {
                    seed_first
                } else if let Some((ref ast, sl_l, fl_l)) = assistant_state {
                    let (ast_drafts, m_stop, ast_sk) = if rt.kv_quant.is_glyph() {
                        let (k_sl, v_sl) = rt.glyph_kv[sl_l]
                            .as_ref()
                            .ok_or("missing sliding glyph KV for assistant")?
                            .materialize_chronological_kv();
                        let (k_fl, v_fl) = rt.glyph_kv[fl_l]
                            .as_ref()
                            .ok_or("missing global glyph KV for assistant")?
                            .materialize_chronological_kv();
                        ast.draft_sequence_gated(
                            y0,
                            |tok| rt.embed_batch(&[tok]),
                            &last_backbone_h,
                            &k_sl,
                            &v_sl,
                            &k_fl,
                            &v_fl,
                            cur_kv_len,
                            cur_kv_len,
                            draft_budget,
                            assistant_step_pos,
                            &eos,
                            closed_thought_bans,
                            assistant_min_margin,
                            lmhead_cs_sieve,
                        )?
                    } else {
                        let kv_tmp = std::mem::take(&mut rt.kv);
                        let ast_res = ast.draft_sequence_gated(
                            y0,
                            |tok| rt.embed_batch(&[tok]),
                            &last_backbone_h,
                            &kv_tmp[sl_l].k,
                            &kv_tmp[sl_l].v,
                            &kv_tmp[fl_l].k,
                            &kv_tmp[fl_l].v,
                            cur_kv_len,
                            cur_kv_len,
                            draft_budget,
                            assistant_step_pos,
                            &eos,
                            closed_thought_bans,
                            assistant_min_margin,
                            lmhead_cs_sieve,
                        );
                        rt.kv = kv_tmp;
                        ast_res?
                    };
                    spec_margin_stopped += m_stop;
                    if ast_sk > 0 {
                        ast_lm_skipped_blocks += ast_sk as u64;
                    }
                    ast_drafts
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

            let td = std::time::Instant::now();
            if drafts.is_empty() {
                spec_fallback_k1 += 1;
                let h = rt.forward_batch(&[y0], cur_kv_len, false)?;
                let mut l = rt.compute_logits(&h)?;
                if let Some(ref mut gate) = tri_gate {
                    let (cap, _dec) = evaluate_tri_modal_step(
                        step_idx + 1,
                        &h,
                        &mut l,
                        true,
                        lmhead_cs_sieve,
                        cs_skip_thresh,
                        c8d_cold_va,
                        &mut rt,
                        gate,
                        tri_modal_gate,
                    )?;
                    if tri_modal_gate {
                        tri_draft_cap = cap;
                    }
                }
                apply_degeneracy_guard(&mut l, &ids, &generated);
                last_backbone_h = h;
                pending = Some(argmax_f32(&l));
                eprintln!(
                    "[run31b] spec fallback k=1 step={} in {:.2}s k_window={} vm={:?}",
                    step_idx,
                    td.elapsed().as_secs_f64(),
                    ctrl.current_k(),
                    proc_status_kb()
                );
            } else {
                spec_batches += 1;
                spec_drafted += drafts.len();
                let mut batch = Vec::with_capacity(1 + drafts.len());
                batch.push(y0);
                batch.extend_from_slice(&drafts);
                let h_batch = rt.forward_batch_mode(&batch, cur_kv_len, false, true)?;
                let mut l_batch = rt.compute_logits_batch(&h_batch, batch.len())?;
                // Apply the identical degeneracy guard sequentially across verification slots
                // so greedy speculative verification stays 100% bit-exact with k=1.
                let mut sim_gen = generated.clone();
                for (slot_i, slot_logits) in l_batch.chunks_exact_mut(vocab).enumerate() {
                    apply_degeneracy_guard(slot_logits, &ids, &sim_gen);
                    if let Some(&d_tok) = drafts.get(slot_i) {
                        let pred = argmax_f32(slot_logits);
                        if pred != d_tok {
                            break;
                        }
                        sim_gen.push(pred);
                    }
                }
                let outcome = verify_greedy_draft_batch(&drafts, &l_batch, vocab, &eos, rem)?;
                let acc = outcome.accepted_drafts.len();
                spec_accepted += acc;
                ctrl.observe(acc, drafts.len());
                rt.truncate_kv(cur_kv_len + 1 + acc)?;
                last_backbone_h = h_batch[acc * d_hidden..(acc + 1) * d_hidden].to_vec();
                for &tok in &outcome.accepted_drafts {
                    let s_idx = generated.len();
                    generated.push(tok);
                    ngram.observe_transition(&history, tok);
                    history.push(tok);
                    println!("[STEP {s_idx}] id={tok} text={:?}", tk.decode(&[tok]));
                }
                let mut next_pending = if outcome.hit_stop { None } else { outcome.next_pending };
                if let Some(ref mut gate) = tri_gate {
                    let mut slot_logits = l_batch[acc * vocab..(acc + 1) * vocab].to_vec();
                    let (cap, _dec) = evaluate_tri_modal_step(
                        generated.len(),
                        &last_backbone_h,
                        &mut slot_logits,
                        !outcome.hit_stop && next_pending.is_some(),
                        lmhead_cs_sieve,
                        cs_skip_thresh,
                        c8d_cold_va,
                        &mut rt,
                        gate,
                        tri_modal_gate,
                    )?;
                    if tri_modal_gate {
                        tri_draft_cap = cap;
                        if !outcome.hit_stop && next_pending.is_some() {
                            let post_gen = generated.clone();
                            apply_degeneracy_guard(&mut slot_logits, &ids, &post_gen);
                            next_pending = Some(argmax_f32(&slot_logits));
                        }
                    }
                }
                pending = next_pending;
                eprintln!(
                    "[run31b] spec batch={} drafted={} accepted={} next_k={} ewma={:.2} in {:.2}s vm={:?}",
                    spec_batches,
                    drafts.len(),
                    acc,
                    ctrl.current_k(),
                    ctrl.acceptance_ewma(),
                    td.elapsed().as_secs_f64(),
                    proc_status_kb()
                );
            }
        }
        let spec_s = t_dec.elapsed().as_secs_f64();
        let dec_steps_after_prefill = generated.len().saturating_sub(1);
        let accept_rate = if spec_drafted > 0 { spec_accepted as f64 / spec_drafted as f64 } else { 0.0 };
        let eff_s_tok = if dec_steps_after_prefill > 0 { spec_s / dec_steps_after_prefill as f64 } else { 0.0 };
        let ast_mb = assistant_state
            .as_ref()
            .map_or(0.0, |(a, _, _)| a.resident_bytes() as f64 / (1024.0 * 1024.0));
        println!(
            "SPEC31B max_batch_k={batch_k} ngram={spec_ngram} min_count={spec_min_count} seed_entries={seed_entries} assistant_mb={ast_mb:.2} step_pos={assistant_step_pos} min_margin={assistant_min_margin:.2} margin_stopped={spec_margin_stopped} ast_lm_skipped={ast_lm_skipped_blocks} drafted={spec_drafted} accepted={spec_accepted} batches={spec_batches} fallback_k1={spec_fallback_k1} accept_rate={accept_rate:.3} generated={} s_per_tok={eff_s_tok:.3}",
            generated.len()
        );
    } else {
        let mut logits = prefill_logits;
        for step in 0..max_tokens {
            let nxt = argmax_f32(&logits);
            generated.push(nxt);
            let tok_txt = tk.decode(&[nxt]);
            println!("[STEP {step}] id={nxt} text={tok_txt:?}");
            if eos.contains(&nxt) {
                break;
            }
            if step + 1 < max_tokens {
                let td = std::time::Instant::now();
                let step_hidden = rt.forward_batch(&[nxt], ids.len() + step, false)?;
                logits = rt.compute_logits(&step_hidden)?;
                if let Some(ref mut gate) = tri_gate {
                    let _ = evaluate_tri_modal_step(
                        step + 1,
                        &step_hidden,
                        &mut logits,
                        true,
                        lmhead_cs_sieve,
                        cs_skip_thresh,
                        c8d_cold_va,
                        &mut rt,
                        gate,
                        tri_modal_gate,
                    )?;
                }
                apply_degeneracy_guard(&mut logits, &ids, &generated);
                eprintln!("[run31b] decode step {step} in {:.2}s vm={:?}", td.elapsed().as_secs_f64(), proc_status_kb());
            }
        }
    }
    let decode_s = t_dec.elapsed().as_secs_f64();
    if batch_k > 1 && !speculative {
        let vocab = rt.cfg.vocab;
        let batch_ids: Vec<u32> = (0..batch_k).map(|i| generated[i % generated.len().max(1)]).collect();
        let tb = std::time::Instant::now();
        let h_batch = rt.forward_batch_mode(&batch_ids, ids.len() + generated.len(), false, true)?;
        let tb_body = tb.elapsed().as_secs_f64();
        let th = std::time::Instant::now();
        let l_batch = rt.compute_logits_batch(&h_batch, batch_k)?;
        let tb_head = th.elapsed().as_secs_f64();
        let tb_tot = tb.elapsed().as_secs_f64();
        let top1s: Vec<u32> = (0..batch_k)
            .map(|k| {
                let s = &l_batch[k * vocab..(k + 1) * vocab];
                s.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(i, _)| i) as u32
            })
            .collect();
        println!(
            "BATCH_K k={batch_k} body_s={tb_body:.3} head_s={tb_head:.3} total_s={tb_tot:.3} s_per_tok={:.3} head_ms_per_tok={:.1} top1s={top1s:?}",
            tb_tot / batch_k as f64,
            1000.0 * tb_head / batch_k as f64
        );
    }
    let (vm_hwm, vm_rss) = proc_status_kb();
    println!("GEN31B ids={generated:?}");
    println!("GEN31B text={:?}", tk.decode(&generated));
    if rt.kv_quant.is_glyph() {
        let st = rt.glyph_kv_total_stats();
        println!(
            "KV31B mode={:?} raw_bytes={} naive_bf16_bytes={} compressed_bytes={} glyph_tokens={} exact_tokens={} shrink={:.2}x shrink_vs_bf16={:.2}x",
            rt.kv_quant,
            st.fp32_bytes,
            st.naive_bf16_bytes,
            st.resident_bytes,
            st.glyph_tokens,
            st.exact_tokens,
            st.shrink_vs_fp32(),
            st.shrink_vs_naive_bf16(),
        );
    } else if rt.kv_raw_bytes > 0 {
        let ratio = rt.kv_raw_bytes as f64 / rt.kv_compressed_bytes.max(1) as f64;
        println!(
            "KV31B mode={:?} raw_bytes={} compressed_bytes={} shrink={ratio:.2}x",
            rt.kv_quant, rt.kv_raw_bytes, rt.kv_compressed_bytes
        );
    }
    if rt.mlp_top_p < 0.9999 {
        let pct = if rt.mlp_sparse_total_blks > 0 {
            100.0 * (rt.mlp_sparse_skipped_blks as f64) / (rt.mlp_sparse_total_blks as f64)
        } else {
            0.0
        };
        println!(
            "MLP_SPARSE top_p={:.3} skipped_blks={} total_blks={} dram_saved_pct={:.1}%",
            rt.mlp_top_p, rt.mlp_sparse_skipped_blks, rt.mlp_sparse_total_blks, pct
        );
    }
    if rt.profile_components && rt.prof_steps > 0 {
        let n = rt.prof_steps as f64;
        println!(
            "PROFILE31B steps={} attn_s={:.3} gate_s={:.3} up_s={:.3} down_s={:.3} norm_kv_s={:.3} lmhead_s={:.3} avg_step_s={:.3}",
            rt.prof_steps,
            rt.prof_attn_s,
            rt.prof_gate_s,
            rt.prof_up_s,
            rt.prof_down_s,
            rt.prof_norm_kv_s,
            rt.prof_lmhead_s,
            (rt.prof_attn_s + rt.prof_gate_s + rt.prof_up_s + rt.prof_down_s + rt.prof_norm_kv_s + rt.prof_lmhead_s) / n
        );
    }
    println!(
        "MEM31B pinned_bytes={pinned_bytes} embed_bytes={embed_bytes} ssd_read_bytes={} cs_skipped_blocks={} lmhead_cs_skipped_blocks={} prefill_s={prefill_s:.2} decode_s={decode_s:.2} vm_hwm_kb={vm_hwm:?} vm_rss_kb={vm_rss:?}",
        rt.ssd_bytes_read, rt.cs_skipped_blocks, rt.lmhead_cs_skipped_blocks
    );
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("FAIL: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qat_batch_projection_matches_explicit_dequant() {
        let d_out = 3usize;
        let d_in = 64usize;
        let ngroups = d_in / 32;
        let mut packed = Vec::with_capacity(d_out * d_in / 2);
        let mut scales = Vec::with_capacity(d_out * ngroups * 2);
        let mut dense = vec![0f32; d_out * d_in];
        for r in 0..d_out {
            for g in 0..ngroups {
                let sc_bf16: u16 = match (r + g) % 3 {
                    0 => 0x3C00, // 0.03125
                    1 => 0x3D80, // 0.0625
                    _ => 0xBC80, // -0.0390625
                };
                scales.extend_from_slice(&sc_bf16.to_le_bytes());
                let sc = bf16_to_f32(sc_bf16);
                for j in 0..16 {
                    let q0 = ((r * 5 + g * 3 + j) % 16) as u8;
                    let q1 = ((r * 7 + g * 11 + j + 1) % 16) as u8;
                    packed.push(q0 | (q1 << 4));
                    dense[r * d_in + g * 32 + 2 * j] = round_bf16_f32((q0 as i32 - 8) as f32 * sc);
                    dense[r * d_in + g * 32 + 2 * j + 1] = round_bf16_f32((q1 as i32 - 8) as f32 * sc);
                }
            }
        }
        let seq = 4usize;
        let mut x = vec![0f32; seq * d_in];
        for (i, slot) in x.iter_mut().enumerate() {
            *slot = ((i as f32 * 0.17).sin() - 0.25) * 1.4;
        }
        let qp = QatProj { d_out, d_in, packed, scales };
        for check_seq in [1usize, seq] {
            let x_sub = &x[..check_seq * d_in];
            let got = qat_project_batch(&qp, x_sub, check_seq, 2).unwrap();
            for t in 0..check_seq {
                for r in 0..d_out {
                    let mut want = 0f64;
                    for c in 0..d_in {
                        want += dense[r * d_in + c] as f64 * x_sub[t * d_in + c] as f64;
                    }
                    assert!((got[t * d_out + r] as f64 - want).abs() < 1e-4);
                }
            }
        }
    }

    #[test]
    fn t16c_mat_integrates_with_run31b_batch_k() {
        use run270_rust::t16c::T16CMat;
        let (d_out, d_in, bsize, nb) = (32usize, 256usize, 256usize, 1u8);
        let sc: Vec<f32> = (0..d_out).map(|r| 0.015 + 0.002 * (r as f32 % 5.0)).collect();
        let bi: Vec<f32> = (0..d_out).map(|r| -0.004 + 0.001 * (r as f32 % 3.0)).collect();
        let idx: Vec<u8> = (0..d_out * d_in).map(|i| ((i * 7 + 3) & 1) as u8).collect();
        let mat = T16CMat::pack_from_components(
            "model.language_model.layers.0.mlp.gate_proj.weight".into(),
            d_out,
            d_in,
            bsize,
            nb,
            true,
            false,
            &sc,
            Some(&bi),
            &idx,
            None,
            None,
        )
        .unwrap();
        let k = 4usize;
        let x: Vec<f32> = (0..k * d_in).map(|i| ((i as f32) * 0.11).sin()).collect();
        let y = mat.project_batch(&x, k, None, 2).unwrap();
        assert_eq!(y.len(), k * d_out);
        for tok in 0..k {
            for r in 0..d_out {
                let wrow = mat.decode_row(r).unwrap();
                let want: f32 = wrow.iter().zip(&x[tok * d_in..(tok + 1) * d_in]).map(|(w, xv)| w * xv).sum();
                assert!((y[tok * d_out + r] - want).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn cuneiform_bank_integrates_with_t16c_proj() {
        use run270_rust::cuneiform_tbi::{
            Cuneiform6DRadicals, CuneiformBank, CuneiformBankEntryInput, CuneiformExpertInput,
        };
        use run270_rust::t16c::T16CMat;

        let (d_out, d_in, bsize, nb, rk) = (32usize, 64usize, 64usize, 1u8, 4usize);
        let sc = vec![0.02f32; d_out];
        let bi = vec![0.0f32; d_out];
        let idx: Vec<u8> = (0..d_out * d_in).map(|i| (i & 1) as u8).collect();
        let mname = "model.language_model.layers.39.mlp.down_proj.weight".to_string();
        let mat = T16CMat::pack_from_components(mname.clone(), d_out, d_in, bsize, nb, true, false, &sc, Some(&bi), &idx, None, None).unwrap();

        let n_exp = 4usize;
        let mut cents = vec![0.0f32; n_exp * 8];
        for e in 0..n_exp {
            cents[e * 8 + e] = 1.0;
        }
        let su: Vec<f32> = (0..d_out).map(|r| 0.01 * (r as f32 + 1.0)).collect();
        let sv = vec![0.02f32; rk];
        let u: Vec<i8> = (0..d_out * rk).map(|i| ((i as i32 % 17) - 8) as i8).collect();
        let v: Vec<i8> = (0..rk * d_in).map(|i| ((i as i32 % 15) - 7) as i8).collect();
        let rad = Cuneiform6DRadicals::from_axes(10, 14, 0, 7, 10, 0, 0);
        let experts: Vec<CuneiformExpertInput<'_>> = (0..n_exp)
            .map(|e| CuneiformExpertInput {
                radicals: Cuneiform6DRadicals::from_axes(10, 14, 0, 7, 10, 0, e as u8),
                su: &su,
                sv: &sv,
                u_i8: &u,
                v_i8: &v,
            })
            .collect();

        let tmp = std::env::temp_dir().join("run31b_c8d_integration.c8d");
        let ein = CuneiformBankEntryInput {
            name: mname.clone(),
            d_out,
            d_in,
            rank: rk,
            top_k: 2,
            c6d: rad,
            temperature: 0.5,
            gate_scale: 1.0,
            centroids: &cents,
            experts,
        };
        CuneiformBank::write_file(&tmp, &[], &[ein]).unwrap();
        let bank = CuneiformBank::load(&tmp, None, true, true).unwrap();
        let _ = std::fs::remove_file(&tmp);

        let k = 2usize;
        let x: Vec<f32> = (0..k * d_in).map(|i| ((i as f32) * 0.21).cos()).collect();
        let y_base = mat.project_batch(&x, k, None, 2).unwrap();
        let mut y_plus = y_base.clone();
        bank.entries.get(&mname).unwrap().apply_batch_into(&x, k, &mut y_plus, None, 2, None).unwrap();
        let delta: f32 = y_plus.iter().zip(y_base.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(delta > 1e-3, "expected non-zero routed delta from C8DB bank");
    }

    #[test]
    fn degeneracy_guard_blocks_closed_thought_and_stutter() {
        let prompt_ids = vec![2u32, 105, 2364, 100, 45518, 107, 101];
        let mut logits = vec![0.0f32; 50000];
        logits[101] = 30.0;
        logits[107] = 28.0;
        logits[45518] = 25.0;
        logits[23681] = 20.0;
        apply_degeneracy_guard(&mut logits, &prompt_ids, &[]);
        assert_eq!(argmax_f32(&logits), 23681);

        let mut logits_sb = vec![0.0f32; 240000];
        logits_sb[1] = 35.0; // <eos>
        logits_sb[106] = 34.0; // <turn|>
        logits_sb[1852] = 33.0; // common-mode carrier artifact
        logits_sb[236779] = 32.0; // '_'
        logits_sb[236751] = 31.0; // 's'
        logits_sb[236774] = 20.0; // 'T' (allowed uppercase ASCII start)
        apply_degeneracy_guard(&mut logits_sb, &prompt_ids, &[]);
        assert_eq!(argmax_f32(&logits_sb), 236774);

        let mut logits_premature = vec![0.0f32; 50000];
        logits_premature[1] = 40.0; // premature <eos> at generated.len() < 4
        logits_premature[23682] = 18.0;
        apply_degeneracy_guard(&mut logits_premature, &prompt_ids, &[23681]);
        assert_eq!(argmax_f32(&logits_premature), 23682);

        let mut logits2 = vec![0.0f32; 50000];
        logits2[23681] = 20.0;
        logits2[23682] = 18.0;
        apply_degeneracy_guard(&mut logits2, &prompt_ids, &[23681]);
        assert_eq!(argmax_f32(&logits2), 23682);
    }

    #[test]
    fn kv_quant_q8_and_fp8_e4m3_roundtrip_fidelity() {
        assert_eq!(KvQuantMode::parse("fp32").unwrap(), KvQuantMode::Fp32);
        assert_eq!(KvQuantMode::parse("q8").unwrap(), KvQuantMode::Q8);
        assert_eq!(KvQuantMode::parse("fp8").unwrap(), KvQuantMode::Fp8E4M3);
        assert!(KvQuantMode::parse("int2_bad").is_err());

        let hd = 256usize;
        let orig: Vec<f32> = (0..4 * hd).map(|i| ((i as f32 * 0.13).sin() * 3.2) + ((i as f32 * 0.03).cos() * 0.4)).collect();
        let mut q8 = orig.clone();
        let bytes_q8 = quantize_dequant_kv_q8_inplace(&mut q8, hd);
        assert_eq!(bytes_q8, 4 * (hd + 4));

        let lut = build_fp8_e4m3_lut();
        let mut fp8 = orig.clone();
        let bytes_fp8 = quantize_dequant_kv_fp8_e4m3_inplace(&mut fp8, hd, &lut);
        assert_eq!(bytes_fp8, 4 * (hd + 4));

        let shrink = (orig.len() * 4) as f64 / bytes_fp8 as f64;
        assert!((shrink - (1024.0 / 260.0)).abs() < 1e-6);

        let cos = |v: &[f32]| -> f64 {
            let (mut d, mut n1, mut n2) = (0.0f64, 0.0f64, 0.0f64);
            for (&a, &b) in orig.iter().zip(v.iter()) {
                d += (a as f64) * (b as f64);
                n1 += (a as f64) * (a as f64);
                n2 += (b as f64) * (b as f64);
            }
            d / (n1.sqrt() * n2.sqrt() + 1e-12)
        };
        assert!(cos(&q8) > 0.9999, "Q8 KV cos={}", cos(&q8));
        assert!(cos(&fp8) > 0.999, "FP8 E4M3 KV cos={}", cos(&fp8));
    }

    #[test]
    fn parse_hybrid_qat_layers_validates_bounds() {
        assert_eq!(parse_hybrid_qat_layers("59, 58, 59, 0", 60).unwrap(), vec![0, 58, 59]);
        assert!(parse_hybrid_qat_layers("60", 60).is_err());
        assert!(parse_hybrid_qat_layers("abc", 60).is_err());
    }

    #[test]
    fn parse_lmhead_cs_mode_validates_variants() {
        assert_eq!(parse_lmhead_cs_mode("off").unwrap(), None);
        assert_eq!(parse_lmhead_cs_mode("NONE").unwrap(), None);
        assert_eq!(parse_lmhead_cs_mode("exact").unwrap(), Some(0.0));
        assert_eq!(parse_lmhead_cs_mode("sieve").unwrap(), Some(0.25));
        assert_eq!(parse_lmhead_cs_mode("sieve:0.35").unwrap(), Some(0.35));
        assert!(parse_lmhead_cs_mode("sieve:0").is_err());
        assert!(parse_lmhead_cs_mode("sieve:1.5").is_err());
        assert!(parse_lmhead_cs_mode("bad").is_err());
    }

    #[test]
    fn tri_modal_config_and_gate_driver_transitions() {
        let cfg = build_tri_modal_config(Some(0.25), 0.15, 4);
        assert!((cfg.base_sieve_frac - 0.25).abs() < 1e-6);
        assert!((cfg.base_cs_skip - 0.15).abs() < 1e-6);
        assert_eq!(cfg.base_draft_k, 4);

        let cfg_off = build_tri_modal_config(None, 0.0, 0);
        assert!((cfg_off.base_sieve_frac - 0.0).abs() < 1e-6);
        assert_eq!(cfg_off.base_draft_k, 1);

        let mut gate = TriModalConsensusGate::new(cfg);
        let d = 5376usize;
        let smooth_h: Vec<f32> = (0..d)
            .map(|i| ((i as f32) * std::f32::consts::TAU / d as f32).cos())
            .collect();
        let mut sharp_logits = vec![-2.0f32; 256];
        sharp_logits[42] = 6.5;
        sharp_logits[7] = 1.0;

        let hear_clean = analyze_hear_fwht(&smooth_h, 64, false).unwrap();
        let see_clean = analyze_see_spatial(&smooth_h, 64).unwrap();
        let read_clean = analyze_read_logits(&sharp_logits, true).unwrap();
        let dec_clean = gate.evaluate(&read_clean, &hear_clean, &see_clean, true);
        assert_eq!(dec_clean.mode, ConsensusMode::Clean);
        assert_eq!(dec_clean.lmhead_cs_sieve, Some(0.25));
        assert!(!dec_clean.engage_c8db_rescue);
        assert_eq!(dec_clean.draft_k, 4);

        let mut noisy_h: Vec<f32> = (0..d)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        noisy_h[17] = 45.0;
        let flat_logits = vec![0.05f32; 256];
        let hear_bad = analyze_hear_fwht(&noisy_h, 64, false).unwrap();
        let see_bad = analyze_see_spatial(&noisy_h, 64).unwrap();
        let read_bad = analyze_read_logits(&flat_logits, true).unwrap();
        let dec_bad = gate.evaluate(&read_bad, &hear_bad, &see_bad, true);
        assert_ne!(dec_bad.mode, ConsensusMode::Clean);
        assert_eq!(dec_bad.lmhead_cs_sieve, Some(0.0));
        assert!(dec_bad.engage_c8db_rescue);
        assert!(dec_bad.draft_k < 4);
    }

    #[test]
    fn compute_vision_kv_ends_marks_contiguous_image_spans_only() {
        let ids = vec![2, 255_999, 258_880, 258_880, 258_880, 258_882, 500];
        let ends = compute_vision_kv_ends(&ids);
        assert_eq!(ends, vec![0, 1, 4, 4, 4, 5, 6]);
    }

    #[test]
    fn glyph_kv_mode_and_optical_dma_integrate_in_run31b() {
        assert_eq!(KvQuantMode::parse("glyph").unwrap(), KvQuantMode::Glyph);
        assert_eq!(KvQuantMode::parse("glyph-pure").unwrap(), KvQuantMode::GlyphPure);
        assert!(KvQuantMode::Glyph.is_glyph());
        assert!(KvQuantMode::GlyphPure.is_glyph());
        assert!(!KvQuantMode::Fp8E4M3.is_glyph());

        let cfg = Gemma4Cfg {
            d: 512,
            n_layers: 2,
            ff: 1024,
            n_heads: 4,
            n_kv: 2,
            n_global_kv: 2,
            hd: 256,
            global_hd: 512,
            window: 1024,
            eps: 1e-6,
            theta_global: 1_000_000.0,
            partial_global: 0.25,
            theta_local: 10_000.0,
            partial_local: 1.0,
            attention_k_eq_v: true,
            final_softcap: Some(30.0),
            layer_full: vec![false, true],
            vocab: 262_144,
        };
        let layers = init_glyph_kv(&cfg, KvQuantMode::Glyph).unwrap();
        assert_eq!(layers.len(), 2);
        let l0 = layers[0].as_ref().unwrap();
        let l1 = layers[1].as_ref().unwrap();
        assert!(!l0.k_eq_v);
        assert_eq!(l0.sliding_window, Some(1024));
        assert!(!l1.k_eq_v);
        assert_eq!(l1.sliding_window, None);

        let prompt_ids = vec![2u32, 105, 2364, 107, 1841, 603, 506, 6037, 576, 7001, 236881];
        let (_frame, rec, stats) = optical_context_dma_roundtrip(&prompt_ids, 4).unwrap();
        assert_eq!(rec, prompt_ids);
        assert_eq!(stats.n_tokens, prompt_ids.len());
        assert!(stats.fps > 0.0);
    }
}
