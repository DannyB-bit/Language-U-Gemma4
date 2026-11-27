//! run31b — bounded-RAM native Rust inference runtime for Gemma 4 31B QAT W4A16 (zero Python).
//!
//! Executes all 60 dense Gemma 4 transformer layers (compressed-tensors symmetric group-32 INT4)
//! directly on 8 GB edge hardware (Raspberry Pi 4 ARM64) using:
//!   - Batched prefill across each layer so an N-token prompt reads SSD weights only once.
//!   - Streaming single-projection lifetime (`<= 90 MB` layer scratch) plus optional `--pin-layers N`.
//!   - Configurable embedding/LM-head residency (`stream` ~44 MB scratch, `bf16` 2.82 GB, or
//!     `q4`/`q6`/`q8` sub-byte packed via `run270_rust::packed::PackedMatrix`).

use run270_rust::alu::evaluate_alu_context;
use run270_rust::assistant::AssistantRuntime;
use run270_rust::byte_head::BpePrefixTrie;
use run270_rust::cca_bridge::{cca_quantile, CcaClusterPlan, CcaLayerPolicy, CCA_PROBE_PAIRS};
use run270_rust::codec::{bf16_to_f32, f32_to_bf16};
use run270_rust::cortical_router::{CorticalRouterBank, DistrictRouterBank};
use run270_rust::glyph_kv::{GlyphKvLayer, GlyphKvMemStats};
use run270_rust::json;
use run270_rust::math::{
    attn_decode_range, attn_decode_tree_batch_into, gelu_tanh_fast_f32, gelu_tanh_inplace_parallel,
    qk_norm_raw_inplace, rmsn_raw, rmsn_raw_into, rope_cos_sin_for_pos, rope_head_with_cos_sin,
    rope_inv_freqs,
};
use run270_rust::optical_carrier::{decode_tokens_from_optical_frame, optical_context_dma_roundtrip, save_ppm};
use run270_rust::packed::{PackedMatrix, QuantScheme};
use run270_rust::safetensors::SafeTensors;
use run270_rust::speculative::{
    argmax_f32, fuse_unverified_prefix_and_tree, prompt_suffix_lookup, verify_greedy_draft_batch,
    verify_greedy_tree_batch, AdaptiveDraftController, FastNGramProposalEngine, SpecTreeBuilder,
};
use run270_rust::t16c::T16CLoadPolicy;
use run270_rust::thread_pool::detect_fast_cpus;
use run270_rust::tokenizer::Tokenizer;
use run270_rust::tri_modal::{
    analyze_hear_fwht, analyze_read_logits, analyze_see_spatial, evaluate_6sense_consensus_gate,
    extract_6sense_pov_features, render_tri_modal_rescue_canvas, ConsensusConfig, ConsensusMode,
    TriModalConsensusGate,
};
use run270_rust::va_mirror::{MirrorProjector, VaMirrorArena, MIRROR_DEPTHS};
use run270_rust::vision::GEMMA4_IMAGE_TOKEN_ID;
use run270_rust::zk_cluster::{
    pack_zk_lora_chirps, reassemble_zk_lora_message, serve_cluster_worker,
    unpack_and_verify_zk_lora_chirp, ClusterWorkerConn, LayerShardBackend, ZKLORA_MTU,
};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::Arc;
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
    GlyphLod,
}

impl KvQuantMode {
    fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "fp32" | "f32" => Ok(Self::Fp32),
            "q8" | "q8_0" => Ok(Self::Q8),
            "fp8" | "e4m3" | "fp8_e4m3" => Ok(Self::Fp8E4M3),
            "glyph" | "glyph_v2" | "opt5" => Ok(Self::Glyph),
            "glyph-pure" | "glyph_pure" | "opt5-pure" => Ok(Self::GlyphPure),
            "glyph-lod" | "glyph_lod" | "lod" => Ok(Self::GlyphLod),
            other => Err(format!("invalid --kv-quant '{other}' (expected fp32|q8|fp8|glyph|glyph-pure|glyph-lod)")),
        }
    }

    fn is_glyph(self) -> bool {
        matches!(self, Self::Glyph | Self::GlyphPure | Self::GlyphLod)
    }
}

fn resolve_kv_lod_caps(
    kv_quant: KvQuantMode,
    sink_ov: Option<usize>,
    near_ov: Option<usize>,
    mid_ov: Option<usize>,
) -> (usize, usize, usize) {
    let (d_sink, d_near, d_mid) = match kv_quant {
        KvQuantMode::Glyph => (4usize, 32usize, 0usize),
        KvQuantMode::GlyphPure => (0usize, 0usize, 0usize),
        KvQuantMode::GlyphLod => (4usize, 32usize, 192usize),
        _ => (0usize, 0usize, 0usize),
    };
    (
        sink_ov.unwrap_or(d_sink),
        near_ov.unwrap_or(d_near),
        mid_ov.unwrap_or(d_mid),
    )
}

fn init_glyph_kv(cfg: &Gemma4Cfg, kv_quant: KvQuantMode, lod_caps: (usize, usize, usize)) -> Result<Vec<Option<GlyphKvLayer>>, String> {
    if !kv_quant.is_glyph() {
        return Ok((0..cfg.n_layers).map(|_| None).collect());
    }
    let (sink_cap, tail_cap, mid_cap) = lod_caps;
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
        out.push(Some(GlyphKvLayer::new_with_lod(
            cfg.n_heads,
            n_kv,
            hd,
            k_eq_v,
            sw,
            sink_cap,
            tail_cap,
            mid_cap,
            256,
        )?));
    }
    Ok(out)
}

fn append_glyph_prefix(
    gkv: &mut GlyphKvLayer,
    k_batch: &[f32],
    v_batch: &[f32],
    stride: usize,
    nat_ref: Option<&run270_rust::t16c::NativeTile16Kernel>,
    appended: &mut usize,
    target: usize,
) -> Result<(), String> {
    while *appended < target {
        let kt = &k_batch[*appended * stride..(*appended + 1) * stride];
        let vt = if gkv.k_eq_v {
            &[][..]
        } else {
            &v_batch[*appended * stride..(*appended + 1) * stride]
        };
        gkv.append_token(kt, vt, nat_ref)?;
        *appended += 1;
    }
    Ok(())
}

/// Causally append K/V states and evaluate attention token-by-token so recent hot-tail
/// tokens are never evicted prematurely by later prompt positions during batched prefill.
#[allow(clippy::too_many_arguments)]
fn prefill_glyph_kv_causal_batch(
    gkv: &mut GlyphKvLayer,
    kv_quant: KvQuantMode,
    q_batch: &[f32],
    k_batch: &[f32],
    v_batch: &[f32],
    seq_len: usize,
    start_pos: usize,
    n_heads: usize,
    n_kv: usize,
    hd: usize,
    sw: Option<usize>,
    is_full: bool,
    vis_kv_ends: Option<&[usize]>,
    nat_ref: Option<&run270_rust::t16c::NativeTile16Kernel>,
    threads: usize,
    ctx_batch: &mut [f32],
) -> Result<(), String> {
    let base_t = gkv.total_seen();
    let q_stride = n_heads * hd;
    let stride = n_kv * hd;
    let mut appended = 0usize;
    for t in 0..seq_len {
        let rel_end = if !is_full {
            vis_kv_ends.map_or(t, |ends| ends[t])
        } else {
            t
        };
        let need_len = (rel_end + 1).min(seq_len);
        append_glyph_prefix(gkv, k_batch, v_batch, stride, nat_ref, &mut appended, need_len)?;
        let q_t = &q_batch[t * q_stride..(t + 1) * q_stride];
        let c_t = if base_t == 0 && rel_end > t && kv_quant != KvQuantMode::GlyphPure {
            attn_decode_range(q_t, k_batch, v_batch, t, rel_end, start_pos + t, n_heads, n_kv, hd, sw)
        } else {
            gkv.attend_range(
                q_t,
                base_t + t,
                base_t + rel_end,
                start_pos + t,
                1.0,
                nat_ref,
                threads,
            )?
        };
        ctx_batch[t * q_stride..(t + 1) * q_stride].copy_from_slice(&c_t);
    }
    append_glyph_prefix(gkv, k_batch, v_batch, stride, nat_ref, &mut appended, seq_len)?;
    Ok(())
}

/// Stateless exact causal FP32 self-attention over `n_trajs` independent trajectories packed contiguously into
/// `seq_len = n_trajs * traj_len` tokens (zero KV-cache allocation, shared weight-pass execution for CCA probes).
#[allow(clippy::too_many_arguments)]
fn causal_self_attn_independent_trajs(
    q_batch: &[f32],
    k_batch: &[f32],
    v_batch: &[f32],
    seq_len: usize,
    traj_len: usize,
    start_pos: usize,
    n_heads: usize,
    n_kv: usize,
    hd: usize,
    sw: Option<usize>,
    threads: usize,
    ctx_batch: &mut [f32],
) -> Result<(), String> {
    if traj_len == 0 || seq_len % traj_len != 0 {
        return Err(format!("causal_self_attn_independent_trajs: seq_len={seq_len} not divisible by traj_len={traj_len}"));
    }
    let q_stride = n_heads * hd;
    let kv_stride = n_kv * hd;
    if q_batch.len() != seq_len * q_stride
        || k_batch.len() != seq_len * kv_stride
        || v_batch.len() != seq_len * kv_stride
        || ctx_batch.len() != seq_len * q_stride
    {
        return Err("causal_self_attn_independent_trajs buffer shape mismatch".into());
    }
    if seq_len > 1 && threads > 1 {
        let tok_chunk = seq_len.div_ceil(threads);
        std::thread::scope(|s| {
            for (ci, dst_chunk) in ctx_batch.chunks_mut(tok_chunk * q_stride).enumerate() {
                let t_base = ci * tok_chunk;
                let n_tok = dst_chunk.len() / q_stride;
                s.spawn(move || {
                    for dt in 0..n_tok {
                        let t = t_base + dt;
                        let b = t / traj_len;
                        let local_t = t % traj_len;
                        let kb = &k_batch[b * traj_len * kv_stride..(b + 1) * traj_len * kv_stride];
                        let vb = &v_batch[b * traj_len * kv_stride..(b + 1) * traj_len * kv_stride];
                        let q_t = &q_batch[t * q_stride..(t + 1) * q_stride];
                        let c_t = attn_decode_range(q_t, kb, vb, local_t, local_t, start_pos + local_t, n_heads, n_kv, hd, sw);
                        dst_chunk[dt * q_stride..(dt + 1) * q_stride].copy_from_slice(&c_t);
                    }
                });
            }
        });
    } else {
        for t in 0..seq_len {
            let b = t / traj_len;
            let local_t = t % traj_len;
            let kb = &k_batch[b * traj_len * kv_stride..(b + 1) * traj_len * kv_stride];
            let vb = &v_batch[b * traj_len * kv_stride..(b + 1) * traj_len * kv_stride];
            let q_t = &q_batch[t * q_stride..(t + 1) * q_stride];
            let c_t = attn_decode_range(q_t, kb, vb, local_t, local_t, start_pos + local_t, n_heads, n_kv, hd, sw);
            ctx_batch[t * q_stride..(t + 1) * q_stride].copy_from_slice(&c_t);
        }
    }
    Ok(())
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

fn parse_layer_spec(spec: &str, n_layers: usize) -> Result<HashSet<usize>, String> {
    let trimmed = spec.trim();
    if trimmed.is_empty() {
        return Err("empty layer spec".into());
    }
    let low = trimmed.to_ascii_lowercase();
    if low == "none" {
        return Ok(HashSet::new());
    }
    if low == "all" {
        return Ok((0..n_layers).collect());
    }
    let mut out = HashSet::new();
    for raw in trimmed.split(',') {
        let t = raw.trim();
        if t.is_empty() {
            continue;
        }
        if let Some((a_s, b_s)) = t.split_once("..=") {
            let a: usize = a_s.trim().parse().map_err(|_| format!("invalid range start '{a_s}' in '{t}'"))?;
            let b: usize = b_s.trim().parse().map_err(|_| format!("invalid range end '{b_s}' in '{t}'"))?;
            if a > b || b >= n_layers {
                return Err(format!("layer range {a}..={b} out of bounds (n_layers={n_layers})"));
            }
            out.extend(a..=b);
        } else if let Some((a_s, b_s)) = t.split_once("..") {
            let a: usize = a_s.trim().parse().map_err(|_| format!("invalid range start '{a_s}' in '{t}'"))?;
            let b: usize = b_s.trim().parse().map_err(|_| format!("invalid range end '{b_s}' in '{t}'"))?;
            if a >= b || b > n_layers {
                return Err(format!("layer range {a}..{b} out of bounds (n_layers={n_layers})"));
            }
            out.extend(a..b);
        } else if let Some((a_s, b_s)) = t.split_once('-') {
            let a: usize = a_s.trim().parse().map_err(|_| format!("invalid range start '{a_s}' in '{t}'"))?;
            let b: usize = b_s.trim().parse().map_err(|_| format!("invalid range end '{b_s}' in '{t}'"))?;
            if a > b || b >= n_layers {
                return Err(format!("layer range {a}-{b} out of bounds (n_layers={n_layers})"));
            }
            out.extend(a..=b);
        } else {
            let idx: usize = t.parse().map_err(|_| format!("invalid layer id '{t}' in layer spec"))?;
            if idx >= n_layers {
                return Err(format!("layer {idx} >= n_layers {n_layers}"));
            }
            out.insert(idx);
        }
    }
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

fn effective_pov_mod_thresh(base: f32, adaptive: Option<f32>, dynamic: bool) -> f32 {
    if dynamic {
        adaptive.unwrap_or(base).clamp(0.0, base)
    } else {
        base
    }
}

fn apply_pov_draft_cap(budget: usize, enabled: bool, cap: usize) -> usize {
    if enabled {
        budget.min(cap.max(1))
    } else {
        budget
    }
}

fn parse_mod_thresh(s: &str) -> Result<f32, String> {
    let v: f32 = s.parse().map_err(|_| format!("invalid --mod-thresh '{s}'"))?;
    if !v.is_finite() || !(0.0..1.0).contains(&v) {
        return Err(format!("--mod-thresh must be finite in [0.0, 1.0), got {v}"));
    }
    Ok(v)
}

fn default_assistant_min_margin(has_assistant: bool, has_byte_head: bool) -> f32 {
    if has_byte_head {
        0.25
    } else if has_assistant {
        0.75
    } else {
        0.0
    }
}

const LGT1_MAGIC: &[u8; 4] = b"LGT1";
const LGT1_VERSION: u32 = 1;
const LGT1_HEADER_BYTES: usize = 24;

fn fnv1a64_ids(ids: &[u32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &id in ids {
        h ^= id as u64;
        h = h.wrapping_mul(0x0100_0000_01b3u64);
    }
    h
}

fn write_lgt1_header(w: &mut BufWriter<fs::File>, n_pos: usize, vocab: usize, ids_hash: u64) -> Result<(), String> {
    w.write_all(LGT1_MAGIC).map_err(|e| format!("LGT1 write magic: {e}"))?;
    w.write_all(&LGT1_VERSION.to_le_bytes()).map_err(|e| format!("LGT1 write ver: {e}"))?;
    w.write_all(&(n_pos as u32).to_le_bytes()).map_err(|e| format!("LGT1 write n_pos: {e}"))?;
    w.write_all(&(vocab as u32).to_le_bytes()).map_err(|e| format!("LGT1 write vocab: {e}"))?;
    w.write_all(&ids_hash.to_le_bytes()).map_err(|e| format!("LGT1 write hash: {e}"))?;
    Ok(())
}

fn write_lgt1_bf16_row(w: &mut BufWriter<fs::File>, logits: &[f32]) -> Result<(), String> {
    let mut row_bytes = Vec::with_capacity(logits.len() * 2);
    for &v in logits {
        row_bytes.extend_from_slice(&f32_to_bf16(v).to_le_bytes());
    }
    w.write_all(&row_bytes).map_err(|e| format!("LGT1 write row: {e}"))
}

fn resolve_eval_nll_start_pos(
    ids: &[u32],
    explicit_start: Option<usize>,
    after_turn: bool,
) -> Result<usize, String> {
    if ids.len() < 2 {
        return Err("--eval-nll requires at least 2 tokens in sequence".into());
    }
    if explicit_start.is_some() && after_turn {
        return Err("--eval-nll-start-pos and --eval-nll-after-turn are mutually exclusive".into());
    }
    let max_pos = ids.len() - 1;
    let start = if let Some(s) = explicit_start {
        if s >= max_pos {
            return Err(format!(
                "--eval-nll-start-pos {s} out of range for {}-token sequence (must be < {max_pos})",
                ids.len()
            ));
        }
        s
    } else if after_turn {
        if let Some(pos) = ids[..max_pos].iter().rposition(|&t| t == 101) {
            pos
        } else if let Some(pos) = ids[..max_pos]
            .windows(3)
            .rposition(|w| w == [105, 4368, 107])
            .map(|i| i + 2)
            .filter(|&p| p < max_pos)
        {
            pos
        } else {
            return Err(
                "--eval-nll-after-turn could not locate `<channel|>` (101) or `<|turn>model\\n` ([105, 4368, 107]) before final token"
                    .into(),
            );
        }
    } else {
        0
    };
    Ok(start)
}

struct Lgt1Reader {
    reader: BufReader<fs::File>,
    vocab: usize,
    raw_row: Vec<u8>,
}

impl Lgt1Reader {
    fn open(path: &Path, expected_n_pos: usize, expected_vocab: usize, expected_ids_hash: u64) -> Result<Self, String> {
        let f = fs::File::open(path).map_err(|e| format!("LGT1 open {}: {e}", path.display()))?;
        let file_len = f.metadata().map_err(|e| format!("LGT1 stat {}: {e}", path.display()))?.len() as usize;
        let expected_len = LGT1_HEADER_BYTES
            + expected_n_pos
                .checked_mul(expected_vocab)
                .and_then(|n| n.checked_mul(2))
                .ok_or("LGT1 size overflow")?;
        if file_len < LGT1_HEADER_BYTES {
            return Err(format!("LGT1 {} truncated: {file_len} < {LGT1_HEADER_BYTES}", path.display()));
        }
        let mut r = BufReader::new(f);
        let mut hdr = [0u8; LGT1_HEADER_BYTES];
        r.read_exact(&mut hdr).map_err(|e| format!("LGT1 read hdr {}: {e}", path.display()))?;
        if &hdr[0..4] != LGT1_MAGIC {
            return Err(format!("LGT1 invalid magic in {}", path.display()));
        }
        let ver = u32::from_le_bytes(hdr[4..8].try_into().unwrap());
        let n_pos = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
        let vocab = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
        let ids_hash = u64::from_le_bytes(hdr[16..24].try_into().unwrap());
        if ver != LGT1_VERSION || n_pos != expected_n_pos || vocab != expected_vocab || ids_hash != expected_ids_hash {
            return Err(format!(
                "LGT1 header mismatch {}: ver={ver}/{LGT1_VERSION} n_pos={n_pos}/{expected_n_pos} vocab={vocab}/{expected_vocab} hash=0x{ids_hash:016x}/0x{expected_ids_hash:016x}",
                path.display()
            ));
        }
        if file_len != expected_len {
            return Err(format!("LGT1 file len {file_len} != expected {expected_len}"));
        }
        Ok(Self {
            reader: r,
            vocab,
            raw_row: vec![0u8; vocab * 2],
        })
    }

    fn read_next_row(&mut self, out: &mut Vec<f32>) -> Result<(), String> {
        self.reader
            .read_exact(&mut self.raw_row)
            .map_err(|e| format!("LGT1 read row: {e}"))?;
        out.clear();
        out.reserve(self.vocab);
        for ch in self.raw_row.chunks_exact(2) {
            out.push(bf16_to_f32(u16::from_le_bytes([ch[0], ch[1]])));
        }
        Ok(())
    }
}

fn eval_student_logits_stats(student_logits: &[f32], gold_next_id: u32) -> Result<(f64, bool, f64, usize), String> {
    let vocab = student_logits.len();
    let gi = gold_next_id as usize;
    if vocab == 0 || gi >= vocab {
        return Err(format!("eval_student_logits_stats invalid gold={gi} vocab={vocab}"));
    }
    let mut max_v = f64::NEG_INFINITY;
    let mut argmax_s = 0usize;
    for (i, &x) in student_logits.iter().enumerate() {
        let xd = x as f64;
        if xd > max_v {
            max_v = xd;
            argmax_s = i;
        }
    }
    let mut sum_exp = 0.0f64;
    let mut sum_x_exp = 0.0f64;
    for &x in student_logits {
        let d = (x as f64) - max_v;
        let e = d.exp();
        sum_exp += e;
        sum_x_exp += d * e;
    }
    let log_z = sum_exp.ln();
    let nll = -(((student_logits[gi] as f64) - max_v) - log_z);
    let entropy = log_z - (sum_x_exp / sum_exp);
    Ok((nll, argmax_s == gi, entropy.max(0.0), argmax_s))
}

fn compute_seq_kl_metrics(
    teacher_logits: &[f32],
    student_logits: &[f32],
    gold_next_id: u32,
) -> Result<(f64, f64, bool, bool, f64, f64), String> {
    let vocab = teacher_logits.len();
    let gi = gold_next_id as usize;
    if vocab == 0 || student_logits.len() != vocab || gi >= vocab {
        return Err(format!(
            "compute_seq_kl_metrics dim mismatch t={} s={} gold={gi}",
            vocab,
            student_logits.len()
        ));
    }
    let mut max_t = f64::NEG_INFINITY;
    let mut max_s = f64::NEG_INFINITY;
    let mut argmax_t = 0usize;
    let mut argmax_s = 0usize;
    for i in 0..vocab {
        let tv = teacher_logits[i] as f64;
        let sv = student_logits[i] as f64;
        if tv > max_t {
            max_t = tv;
            argmax_t = i;
        }
        if sv > max_s {
            max_s = sv;
            argmax_s = i;
        }
    }
    let mut sum_t = 0.0f64;
    let mut sum_s = 0.0f64;
    for i in 0..vocab {
        sum_t += ((teacher_logits[i] as f64) - max_t).exp();
        sum_s += ((student_logits[i] as f64) - max_s).exp();
    }
    let log_zt = max_t + sum_t.ln();
    let log_zs = max_s + sum_s.ln();
    let mut kl_ts = 0.0f64;
    let mut kl_st = 0.0f64;
    for i in 0..vocab {
        let lp_t = (teacher_logits[i] as f64) - log_zt;
        let lp_s = (student_logits[i] as f64) - log_zs;
        let pt = lp_t.exp();
        let ps = lp_s.exp();
        kl_ts += pt * (lp_t - lp_s);
        kl_st += ps * (lp_s - lp_t);
    }
    let s_at_t = student_logits[argmax_t];
    let better_cnt = student_logits.iter().filter(|&&v| v > s_at_t).count();
    let t_nll = -((teacher_logits[gi] as f64) - log_zt);
    let s_nll = -((student_logits[gi] as f64) - log_zs);
    Ok((
        kl_ts.max(0.0),
        kl_st.max(0.0),
        argmax_t == argmax_s,
        better_cnt < 5,
        t_nll,
        s_nll,
    ))
}

fn print_runtime_telemetry(rt: &Runtime31B, pinned_bytes: usize, embed_bytes: usize, prefill_s: f64, decode_s: f64) {
    let (vm_hwm, vm_rss) = proc_status_kb();
    if rt.kv_quant.is_glyph() {
        let st = rt.glyph_kv_total_stats();
        println!(
            "KV31B mode={:?} raw_bytes={} naive_bf16_bytes={} compressed_bytes={} glyph_tokens={} mid_tokens={} exact_tokens={} shrink={:.2}x shrink_vs_bf16={:.2}x",
            rt.kv_quant,
            st.fp32_bytes,
            st.naive_bf16_bytes,
            st.resident_bytes,
            st.glyph_tokens,
            st.mid_tokens,
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
    if rt.cortical_router.is_some() && rt.cortical_gate_total_blks > 0 {
        let pct = 100.0 * (rt.cortical_gate_skipped_blks as f64) / (rt.cortical_gate_total_blks as f64);
        println!(
            "CORTICAL_ROUTER gate_skipped_blks={} gate_total_blks={} gate_saved_pct={:.1}%",
            rt.cortical_gate_skipped_blks, rt.cortical_gate_total_blks, pct
        );
    }
    if rt.attn_sparse_total_heads > 0 {
        let pct = 100.0 * (rt.attn_sparse_skipped_heads as f64) / (rt.attn_sparse_total_heads as f64);
        println!(
            "ATTN_SPARSE skipped_heads={} total_heads={} saved_pct={:.1}%",
            rt.attn_sparse_skipped_heads, rt.attn_sparse_total_heads, pct
        );
    }
    if rt.mod_thresh > 0.0 {
        let pct = if rt.mod_total_layers > 0 {
            100.0 * (rt.mod_skipped_layers as f64) / (rt.mod_total_layers as f64)
        } else {
            0.0
        };
        println!(
            "MOD31B thresh={:.3} skipped_layers={} eligible_layers={} skipped_pct={:.1}%",
            rt.mod_thresh, rt.mod_skipped_layers, rt.mod_total_layers, pct
        );
    }
    if rt.lod_total_districts_sum > 0 {
        let skipped_d = rt.lod_total_districts_sum.saturating_sub(rt.lod_active_districts_sum);
        let pct = 100.0 * (skipped_d as f64) / (rt.lod_total_districts_sum as f64);
        println!(
            "GTALOD31B active_districts={} total_districts={} district_dram_saved_pct={:.1}%",
            rt.lod_active_districts_sum, rt.lod_total_districts_sum, pct
        );
    }
    if rt.pov_layers_evaluated > 0 {
        let mean_c = rt.pov_consensus_sum / (rt.pov_layers_evaluated as f64);
        println!(
            "POV6SENSE31B layers_eval={} mean_consensus={:.4} clean={} caution={} rescue={}",
            rt.pov_layers_evaluated,
            mean_c,
            rt.pov_clean_layers,
            rt.pov_caution_layers,
            rt.pov_rescue_layers
        );
    }
    if rt.profile_components && rt.prof_steps > 0 {
        let n = rt.prof_steps as f64;
        println!(
            "PROFILE31B steps={} attn_s={:.3} gate_s={:.3} up_s={:.3} down_s={:.3} norm_kv_s={:.3} lmhead_s={:.3} avg_step_s={:.3} router_s={:.3}",
            rt.prof_steps,
            rt.prof_attn_s,
            rt.prof_gate_s,
            rt.prof_up_s,
            rt.prof_down_s,
            rt.prof_norm_kv_s,
            rt.prof_lmhead_s,
            (rt.prof_attn_s + rt.prof_gate_s + rt.prof_up_s + rt.prof_down_s + rt.prof_norm_kv_s + rt.prof_lmhead_s + rt.prof_router_s) / n,
            rt.prof_router_s
        );
    }
    println!(
        "MEM31B pinned_bytes={pinned_bytes} embed_bytes={embed_bytes} ssd_read_bytes={} cs_skipped_blocks={} lmhead_cs_skipped_blocks={} prefill_s={prefill_s:.2} decode_s={decode_s:.2} vm_hwm_kb={vm_hwm:?} vm_rss_kb={vm_rss:?}",
        rt.ssd_bytes_read, rt.cs_skipped_blocks, rt.lmhead_cs_skipped_blocks
    );
}

fn expand_head_mask(head_cmask: &[u8], hd: usize, o_bsize: usize) -> (Vec<u8>, Vec<u8>) {
    let n_heads = head_cmask.len();
    let q_tiles_per_head = hd.div_ceil(16);
    let mut q_rmask = vec![0u8; n_heads * q_tiles_per_head];
    let blks_per_head = if o_bsize > 0 && hd >= o_bsize && hd.is_multiple_of(o_bsize) {
        hd / o_bsize
    } else {
        1
    };
    let mut o_cmask = vec![0u8; n_heads * blks_per_head];
    for (h, &active) in head_cmask.iter().enumerate() {
        if active != 0 {
            q_rmask[h * q_tiles_per_head..(h + 1) * q_tiles_per_head].fill(1);
            o_cmask[h * blks_per_head..(h + 1) * blks_per_head].fill(1);
        }
    }
    (q_rmask, o_cmask)
}

fn union_col_masks(tok_masks: &[Vec<u8>], n_blks: usize) -> Vec<u8> {
    let mut cmask = vec![0u8; n_blks];
    for tm in tok_masks {
        for (dst, &src) in cmask.iter_mut().zip(tm.iter()) {
            if src != 0 {
                *dst = 1;
            }
        }
    }
    cmask
}

fn expand_col_to_row_mask(cmask: &[u8], bsize: usize, d_out: usize) -> Vec<u8> {
    let tiles_per_blk = bsize / 16;
    let mut rmask = vec![0u8; d_out.div_ceil(16)];
    for (b, &active) in cmask.iter().enumerate() {
        if active != 0 {
            let t0 = b * tiles_per_blk;
            for dt in 0..tiles_per_blk {
                if t0 + dt < rmask.len() {
                    rmask[t0 + dt] = 1;
                }
            }
        }
    }
    rmask
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
    lod_shared_mats: Option<HashMap<String, run270_rust::t16c::T16CMat>>,
    lod_district_mats: Vec<HashMap<String, run270_rust::t16c::T16CMat>>,
    district_router: Option<DistrictRouterBank>,
    player_pov_6sense: bool,
    // POV target controls are causal for ordinary decode. Speculative batches
    // freeze target-side adaptations and only use POV as a draft-length cap.
    pov_target_dynamic: bool,
    pov_base_draft_k: usize,
    pov_draft_cap: usize,
    pov_consensus_sum: f64,
    pov_layers_evaluated: u64,
    pov_clean_layers: u64,
    pov_caution_layers: u64,
    pov_rescue_layers: u64,
    lod_active_districts_sum: u64,
    lod_total_districts_sum: u64,
    native_t16c: Option<std::sync::Arc<run270_rust::t16c::NativeTile16Kernel>>,
    c8d_bank: Option<run270_rust::cuneiform_tbi::CuneiformBank>,
    apply_c8d_proj: bool,
    cs_skip_thresh: f32,
    cs_down_norms: Vec<Option<Vec<u16>>>,
    cs_skipped_blocks: u64,
    mlp_top_p: f32,
    shared_mlp_blocks: usize,
    mlp_sparse_skipped_blks: u64,
    mlp_sparse_total_blks: u64,
    cortical_router: Option<CorticalRouterBank>,
    head_router: Option<CorticalRouterBank>,
    attn_head_top_p: f32,
    attn_shared_heads: usize,
    mod_thresh: f32,
    mod_zero_kv: bool,
    mod_skipped_layers: u64,
    mod_total_layers: u64,
    attn_sparse_skipped_heads: u64,
    attn_sparse_total_heads: u64,
    cortical_gate_skipped_blks: u64,
    cortical_gate_total_blks: u64,
    profile_components: bool,
    prof_attn_s: f64,
    prof_gate_s: f64,
    prof_up_s: f64,
    prof_down_s: f64,
    prof_norm_kv_s: f64,
    prof_lmhead_s: f64,
    prof_router_s: f64,
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
    ephemeral_tree_kv: Vec<(Vec<f32>, Vec<f32>)>,
    pending_l29_states: Vec<f32>,
    vision: Option<run270_rust::vision::VisionWeights>,
    local_layer_range: (usize, usize),
    cluster_workers: Vec<ClusterWorkerConn>,
    bypass_layers: Vec<bool>,
    cca_layer_policies: Option<Vec<CcaLayerPolicy>>,
    cca_sparse_active: bool,
    threads: usize,
    ssd_bytes_read: u64,
}

impl Runtime31B {
    fn layer_kv_len(&self, l: usize) -> usize {
        if self.kv_quant.is_glyph() {
            return self
                .glyph_kv
                .get(l)
                .and_then(|o| o.as_ref())
                .map_or(0, |g| g.total_seen());
        }
        if l >= self.kv.len() || l >= self.cfg.layer_full.len() {
            return 0;
        }
        let is_full = self.cfg.layer_full[l];
        let hd = if is_full { self.cfg.global_hd } else { self.cfg.hd };
        let n_kv = if is_full { self.cfg.n_global_kv } else { self.cfg.n_kv };
        let stride = n_kv * hd;
        if stride == 0 {
            0
        } else {
            self.kv[l].k.len() / stride
        }
    }

    #[allow(dead_code)]
    fn kv_len(&self) -> usize {
        self.layer_kv_len(self.local_layer_range.0)
    }

    fn glyph_kv_total_stats(&self) -> GlyphKvMemStats {
        let mut tot = GlyphKvMemStats::default();
        for g in self.glyph_kv.iter().flatten() {
            let s = g.memory_stats();
            tot.active_tokens += s.active_tokens;
            tot.glyph_tokens += s.glyph_tokens;
            tot.mid_tokens += s.mid_tokens;
            tot.exact_tokens += s.exact_tokens;
            tot.resident_bytes += s.resident_bytes;
            tot.naive_bf16_bytes += s.naive_bf16_bytes;
            tot.fp32_bytes += s.fp32_bytes;
        }
        tot
    }

    fn truncate_kv(&mut self, keep_tokens: usize) -> Result<(), String> {
        let split_l = 30.min(self.cfg.n_layers);
        let upper_len = if split_l < self.cfg.n_layers {
            self.layer_kv_len(split_l)
        } else {
            self.layer_kv_len(0)
        };
        if keep_tokens <= upper_len {
            self.pending_l29_states.clear();
        } else if !self.pending_l29_states.is_empty() {
            let keep_pend = (keep_tokens - upper_len) * self.cfg.d;
            if keep_pend < self.pending_l29_states.len() {
                self.pending_l29_states.truncate(keep_pend);
            }
        }
        let (ls_own, le_own) = self.local_layer_range;
        for w in &mut self.cluster_workers {
            w.truncate_kv(keep_tokens)?;
        }
        if self.kv_quant.is_glyph() {
            for l in ls_own..le_own.min(self.glyph_kv.len()) {
                if let Some(g) = self.glyph_kv[l].as_mut() {
                    let cur = g.total_seen();
                    if keep_tokens < cur {
                        g.truncate(keep_tokens)?;
                    } else if keep_tokens > cur && l < split_l {
                        return Err(format!("truncate_kv glyph layer {l}: keep_tokens={keep_tokens} > cur={cur}"));
                    }
                }
            }
            return Ok(());
        }
        for l in ls_own..le_own.min(self.kv.len()) {
            let layer_kv = &mut self.kv[l];
            let is_full = self.cfg.layer_full[l];
            let hd = if is_full { self.cfg.global_hd } else { self.cfg.hd };
            let n_kv = if is_full { self.cfg.n_global_kv } else { self.cfg.n_kv };
            let stride = n_kv * hd;
            let cur_toks = if stride > 0 { layer_kv.k.len() / stride } else { 0 };
            let eff_keep = keep_tokens.min(cur_toks);
            if keep_tokens > cur_toks && l < split_l {
                return Err(format!(
                    "truncate_kv layer {l}: keep_tokens={keep_tokens} (elems={}) exceeds kv len ({}, {})",
                    keep_tokens * stride,
                    layer_kv.k.len(),
                    layer_kv.v.len()
                ));
            }
            let target = eff_keep * stride;
            layer_kv.k.truncate(target);
            layer_kv.v.truncate(target);
        }
        Ok(())
    }

    fn flush_upper_layers(&mut self) -> Result<(), String> {
        if self.pending_l29_states.is_empty() || self.cfg.n_layers <= 30 {
            self.pending_l29_states.clear();
            return Ok(());
        }
        let d = self.cfg.d;
        let k_pend = self.pending_l29_states.len() / d;
        if k_pend == 0 {
            self.pending_l29_states.clear();
            return Ok(());
        }
        let upper_pos = self.layer_kv_len(30);
        let states = std::mem::take(&mut self.pending_l29_states);
        let _ = self.forward_batch_with_embeds_layers(
            states,
            k_pend,
            upper_pos,
            false,
            false,
            None,
            None,
            30,
            self.cfg.n_layers,
            None,
        )?;
        Ok(())
    }

    fn commit_tree_slots(&mut self, base_kv_len: usize, accepted_slots: &[usize]) -> Result<(), String> {
        self.commit_tree_slots_upto(base_kv_len, accepted_slots, self.cfg.n_layers, None)
    }

    fn commit_tree_slots_upto(
        &mut self,
        base_kv_len: usize,
        accepted_slots: &[usize],
        layers_upto: usize,
        raw_l29_batch: Option<&[f32]>,
    ) -> Result<(), String> {
        let lim = layers_upto.min(self.cfg.n_layers);
        if lim < self.cfg.n_layers && self.local_layer_range == (0, self.cfg.n_layers) {
            let d = self.cfg.d;
            let raw = raw_l29_batch.ok_or("commit_tree_slots_upto < n_layers requires raw_l29_batch")?;
            for &slot in accepted_slots {
                self.pending_l29_states.extend_from_slice(&raw[slot * d..(slot + 1) * d]);
            }
        }
        let (ls_own, le_own) = self.local_layer_range;
        let loc_end = lim.min(le_own);
        for w in &mut self.cluster_workers {
            if w.layer_start < lim {
                w.commit_tree(base_kv_len, accepted_slots)?;
            }
        }
        if self.kv_quant.is_glyph() {
            if ls_own < loc_end && self.ephemeral_tree_kv.len() < loc_end {
                return Err(format!(
                    "commit_tree_slots: ephemeral_tree_kv len {} < loc_end {loc_end}",
                    self.ephemeral_tree_kv.len(),
                ));
            }
            for l in ls_own..loc_end {
                let is_full = self.cfg.layer_full[l];
                let hd = if is_full { self.cfg.global_hd } else { self.cfg.hd };
                let n_kv = if is_full { self.cfg.n_global_kv } else { self.cfg.n_kv };
                let stride = n_kv * hd;
                let gkv = self.glyph_kv[l]
                    .as_mut()
                    .ok_or_else(|| format!("missing GlyphKvLayer at layer {l}"))?;
                if gkv.total_seen() != base_kv_len {
                    return Err(format!(
                        "commit_tree_slots layer {l}: total_seen {} != base_kv_len {base_kv_len}",
                        gkv.total_seen()
                    ));
                }
                let (ref tk, ref tv) = self.ephemeral_tree_kv[l];
                for &slot in accepted_slots {
                    let kt = &tk[slot * stride..(slot + 1) * stride];
                    let vt = if gkv.k_eq_v {
                        &[][..]
                    } else {
                        &tv[slot * stride..(slot + 1) * stride]
                    };
                    gkv.append_token(kt, vt, self.native_t16c.as_deref())?;
                }
            }
            self.ephemeral_tree_kv.clear();
            return Ok(());
        }
        for l in ls_own..loc_end {
            let layer_kv = &mut self.kv[l];
            let is_full = self.cfg.layer_full[l];
            let hd = if is_full { self.cfg.global_hd } else { self.cfg.hd };
            let n_kv = if is_full { self.cfg.n_global_kv } else { self.cfg.n_kv };
            let stride = n_kv * hd;
            let base_off = base_kv_len * stride;
            for (dst_i, &slot) in accepted_slots.iter().enumerate() {
                if dst_i != slot {
                    let src_off = base_off + slot * stride;
                    let dst_off = base_off + dst_i * stride;
                    layer_kv.k.copy_within(src_off..src_off + stride, dst_off);
                    layer_kv.v.copy_within(src_off..src_off + stride, dst_off);
                }
            }
            let new_len = base_off + accepted_slots.len() * stride;
            layer_kv.k.truncate(new_len);
            layer_kv.v.truncate(new_len);
        }
        Ok(())
    }

    fn assistant_kv_layers(&self) -> Result<(usize, usize), String> {
        let (ls, le) = self.local_layer_range;
        let end_l = le.min(self.cfg.layer_full.len());
        let slide_l = (ls..end_l)
            .rev()
            .find(|&i| !self.cfg.layer_full[i])
            .ok_or("no local sliding_attention layer in shard for assistant KV sharing")?;
        let full_l = (ls..end_l)
            .rev()
            .find(|&i| self.cfg.layer_full[i])
            .ok_or("no local full_attention layer in shard for assistant KV sharing")?;
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

    fn embed_token_ro(&self, tid: u32) -> Result<Vec<f32>, String> {
        let d = self.cfg.d;
        let escale = (d as f32).sqrt();
        let ti = tid as usize;
        if ti >= self.cfg.vocab {
            return Err(format!("token {tid} out of bounds"));
        }
        let mut out = Vec::with_capacity(d);
        let em_key = "model.language_model.embed_tokens.weight";
        match &self.embed {
            Embed31B::Streamed => {
                let sf = self.sf.as_ref().ok_or("streamed embed requires --model")?;
                let meta = sf.tensors.get(em_key).ok_or_else(|| format!("missing {em_key}"))?;
                let raw = sf.read_raw_slice(meta, ti * d * 2, d * 2)?;
                for c in raw.chunks_exact(2) {
                    out.push(bf16_to_f32(u16::from_le_bytes([c[0], c[1]])) * escale);
                }
            }
            Embed31B::ResidentBf16(tbl) => {
                for &b in &tbl[ti * d..(ti + 1) * d] {
                    out.push(bf16_to_f32(b) * escale);
                }
            }
            Embed31B::Packed(pm) => {
                let row = pm.decode_row(ti)?;
                for v in row {
                    out.push(v * escale);
                }
            }
            Embed31B::T16C { mat, .. } => {
                let row = mat.decode_row(ti)?;
                if row.len() != d {
                    return Err(format!("T16C embed decode_row len {} != d {d}", row.len()));
                }
                for v in row {
                    out.push(v * escale);
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

    fn forward_batch_tree(
        &mut self,
        ids: &[u32],
        start_pos: usize,
        tree_depths: &[usize],
        ancestor_masks: &[u32],
    ) -> Result<Vec<f32>, String> {
        let (normed, _raw) = self.forward_batch_tree_upto(
            ids,
            start_pos,
            tree_depths,
            ancestor_masks,
            self.cfg.n_layers,
        )?;
        Ok(normed)
    }

    fn forward_batch_tree_upto(
        &mut self,
        ids: &[u32],
        start_pos: usize,
        tree_depths: &[usize],
        ancestor_masks: &[u32],
        layer_end: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let seq_len = ids.len();
        if seq_len == 0 || seq_len > 32 || tree_depths.len() != seq_len || ancestor_masks.len() != seq_len {
            return Err(format!(
                "invalid forward_batch_tree dims: ids={} depths={} masks={}",
                seq_len,
                tree_depths.len(),
                ancestor_masks.len()
            ));
        }
        let states = self.embed_batch(ids)?;
        let mut raw_out = Vec::new();
        let normed = self.forward_batch_with_embeds_layers(
            states,
            seq_len,
            start_pos,
            false,
            true,
            None,
            Some((tree_depths, ancestor_masks)),
            0,
            layer_end.min(self.cfg.n_layers),
            Some(&mut raw_out),
        )?;
        Ok((normed, raw_out))
    }

    fn forward_batch_upto(&mut self, ids: &[u32], start_pos: usize, layer_end: usize) -> Result<Vec<f32>, String> {
        let seq_len = ids.len();
        if seq_len == 0 {
            return Err("empty token slice".into());
        }
        let lim = layer_end.min(self.cfg.n_layers);
        let states = self.embed_batch(ids)?;
        let mut raw_out = Vec::new();
        let _normed = self.forward_batch_with_embeds_layers(
            states,
            seq_len,
            start_pos,
            false,
            true,
            None,
            None,
            0,
            lim,
            if lim < self.cfg.n_layers { Some(&mut raw_out) } else { None },
        )?;
        if lim < self.cfg.n_layers {
            self.pending_l29_states.extend_from_slice(&raw_out);
            Ok(raw_out)
        } else {
            Ok(_normed)
        }
    }

    fn forward_batch_with_embeds(
        &mut self,
        states: Vec<f32>,
        seq_len: usize,
        start_pos: usize,
        verbose_layers: bool,
        all_final: bool,
        vis_kv_ends: Option<&[usize]>,
    ) -> Result<Vec<f32>, String> {
        self.forward_batch_with_embeds_ext(states, seq_len, start_pos, verbose_layers, all_final, vis_kv_ends, None)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_batch_with_embeds_ext(
        &mut self,
        states: Vec<f32>,
        seq_len: usize,
        start_pos: usize,
        verbose_layers: bool,
        all_final: bool,
        vis_kv_ends: Option<&[usize]>,
        tree_spec: Option<(&[usize], &[u32])>,
    ) -> Result<Vec<f32>, String> {
        let n_layers = self.cfg.n_layers;
        self.forward_batch_with_embeds_layers(
            states,
            seq_len,
            start_pos,
            verbose_layers,
            all_final,
            vis_kv_ends,
            tree_spec,
            0,
            n_layers,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_batch_with_embeds_layers(
        &mut self,
        states: Vec<f32>,
        seq_len: usize,
        start_pos: usize,
        verbose_layers: bool,
        all_final: bool,
        vis_kv_ends: Option<&[usize]>,
        tree_spec: Option<(&[usize], &[u32])>,
        layer_start: usize,
        layer_end: usize,
        raw_states_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<f32>, String> {
        let lim = layer_end.min(self.cfg.n_layers);
        if !self.cluster_workers.is_empty() && layer_start == 0 && lim > self.local_layer_range.1 {
            let loc_end = self.local_layer_range.1.min(lim);
            let mut cur_states = if loc_end > 0 {
                self.forward_batch_with_embeds_layers_raw(
                    states,
                    seq_len,
                    start_pos,
                    verbose_layers,
                    vis_kv_ends,
                    tree_spec,
                    0,
                    loc_end,
                )?
            } else {
                states
            };
            for w in &mut self.cluster_workers {
                if w.layer_start >= lim {
                    break;
                }
                cur_states = if let Some((depths, masks)) = tree_spec {
                    w.forward_tree(&cur_states, seq_len, start_pos, depths, masks)?
                } else {
                    w.forward_seq(&cur_states, seq_len, start_pos)?
                };
            }
            if let Some(out_raw) = raw_states_out {
                *out_raw = cur_states.clone();
            }
            return self.finalize_hidden_states(&cur_states, seq_len, lim, all_final);
        }
        let raw_states = self.forward_batch_with_embeds_layers_raw(
            states,
            seq_len,
            start_pos,
            verbose_layers,
            vis_kv_ends,
            tree_spec,
            layer_start,
            lim,
        )?;
        if let Some(out_raw) = raw_states_out {
            *out_raw = raw_states.clone();
        }
        self.finalize_hidden_states(&raw_states, seq_len, lim, all_final)
    }

    fn finalize_hidden_states(
        &self,
        states: &[f32],
        seq_len: usize,
        lim: usize,
        all_final: bool,
    ) -> Result<Vec<f32>, String> {
        let d = self.cfg.d;
        let eps = self.cfg.eps;
        let fn_key = if lim < self.cfg.n_layers && self.norms.contains_key("model.language_model.early_exit_l29_norm.weight") {
            "model.language_model.early_exit_l29_norm.weight"
        } else if lim < self.cfg.n_layers && self.norms.contains_key("model.early_exit_l29_norm.weight") {
            "model.early_exit_l29_norm.weight"
        } else {
            "model.language_model.norm.weight"
        };
        let fn_gain = self.norm(fn_key)?;
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
        if lim == self.cfg.n_layers && self.apply_c8d_proj {
            if let Some(ref bank) = self.c8d_bank {
                if let Some(entry) = bank.entries.get("model.language_model.norm.post_cell60") {
                    let x_in = out.clone();
                    entry.apply_batch_into(&x_in, n_out_tok, &mut out, bank.native.as_ref(), self.threads, None)?;
                }
            }
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_batch_with_embeds_layers_raw(
        &mut self,
        states: Vec<f32>,
        seq_len: usize,
        start_pos: usize,
        verbose_layers: bool,
        vis_kv_ends: Option<&[usize]>,
        tree_spec: Option<(&[usize], &[u32])>,
        layer_start: usize,
        layer_end: usize,
    ) -> Result<Vec<f32>, String> {
        self.forward_batch_with_embeds_layers_raw_opt(
            states,
            seq_len,
            start_pos,
            verbose_layers,
            vis_kv_ends,
            tree_spec,
            layer_start,
            layer_end,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_batch_with_embeds_layers_raw_opt(
        &mut self,
        mut states: Vec<f32>,
        seq_len: usize,
        start_pos: usize,
        verbose_layers: bool,
        vis_kv_ends: Option<&[usize]>,
        tree_spec: Option<(&[usize], &[u32])>,
        layer_start: usize,
        layer_end: usize,
        independent_traj_len: Option<usize>,
    ) -> Result<Vec<f32>, String> {
        if seq_len == 0 {
            return Err("empty token slice".into());
        }
        let traj_len = independent_traj_len.unwrap_or(seq_len);
        if traj_len == 0 || seq_len % traj_len != 0 {
            return Err(format!("invalid independent_traj_len={traj_len} for seq_len={seq_len}"));
        }
        let lim = layer_end.min(self.cfg.n_layers);
        if layer_start == 0 && lim == self.cfg.n_layers && !self.pending_l29_states.is_empty() {
            self.flush_upper_layers()?;
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
        let lod_shared_mats = &self.lod_shared_mats;
        let lod_district_mats = &self.lod_district_mats;
        let district_router = &self.district_router;
        let player_pov_6sense = self.player_pov_6sense;
        let pov_target_dynamic = self.pov_target_dynamic;
        let pov_base_draft_k = self.pov_base_draft_k;
        let all_pinned = t16c_mats.is_some();
        let native_t16c = &self.native_t16c;
        let hybrid_pinned = &self.hybrid_pinned;
        let cs_down_norms = &self.cs_down_norms;
        let cs_skip_thresh = self.cs_skip_thresh;
        let mlp_top_p = self.mlp_top_p;
        let shared_mlp_blocks = self.shared_mlp_blocks;
        let cortical_router = &self.cortical_router;
        let head_router = &self.head_router;
        let attn_head_top_p = self.attn_head_top_p;
        let attn_shared_heads = self.attn_shared_heads;
        let mod_thresh = self.mod_thresh;
        let mod_zero_kv = self.mod_zero_kv;
        let bypass_layers = &self.bypass_layers;
        let cca_layer_policies = &self.cca_layer_policies;
        let cca_sparse_active = self.cca_sparse_active;
        let profile_components = self.profile_components;
        let kv_quant = self.kv_quant;
        let fp8_lut = &self.fp8_lut;
        let kv = &mut self.kv;
        let glyph_kv = &mut self.glyph_kv;
        let ephemeral_tree_kv = &mut self.ephemeral_tree_kv;
        if tree_spec.is_some() && kv_quant.is_glyph() {
            ephemeral_tree_kv.clear();
            ephemeral_tree_kv.resize_with(cfg.n_layers, || (Vec::new(), Vec::new()));
        }
        let threads = self.threads;
        let mut read_bytes = 0u64;
        let mut cold_read_bytes = 0u64;
        let mut cs_skipped_step = 0u64;
        let mut mlp_sp_skipped_step = 0u64;
        let mut cold_sp_skipped_step = 0u64;
        let mut mlp_sp_total_step = 0u64;
        let mut cold_sp_total_step = 0u64;
        let apply_c8d_proj = self.apply_c8d_proj;
        let c8d_bank = &self.c8d_bank;
        let mut mod_skipped_step = 0u64;
        let mut mod_total_step = 0u64;
        let mut attn_sp_skipped_step = 0u64;
        let mut attn_sp_total_step = 0u64;
        let mut cortical_gate_skipped_step = 0u64;
        let mut cortical_gate_total_step = 0u64;
        let mut pov_consensus_step = 0.0f64;
        let mut pov_layers_step = 0u64;
        let mut pov_clean_step = 0u64;
        let mut pov_caution_step = 0u64;
        let mut pov_rescue_step = 0u64;
        let mut pov_draft_cap_step = usize::MAX;
        let mut lod_active_d_step = 0u64;
        let mut lod_total_d_step = 0u64;
        let mut prof_attn_step = 0.0f64;
        let mut prof_gate_step = 0.0f64;
        let mut prof_up_step = 0.0f64;
        let mut prof_down_step = 0.0f64;
        let mut prof_norm_kv_step = 0.0f64;
        let mut prof_router_step = 0.0f64;
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
                    let mut out = if let Some(mat) = mats.get(&k1).or_else(|| mats.get(&k2)) {
                        if row_mask.is_some() || col_mask.is_some() {
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
                        }
                    } else {
                        /* Native Samsung LittleBit / LittleBit-2 Factored Projection:
                           W ≈ U_q(V_q(x)) [+ optional residual U2_q(V2_q(x))] in pure 1-bit Row-Major AND+VCNT! */
                        let kv1 = format!("{k1}.lb_v");
                        let ku1 = format!("{k1}.lb_u");
                        let kv2 = format!("{k2}.lb_v");
                        let ku2 = format!("{k2}.lb_u");
                        let mat_v = mats
                            .get(&kv1)
                            .or_else(|| mats.get(&kv2))
                            .ok_or_else(|| format!("T16C missing {k1} (and LittleBit {kv1})"))?;
                        let mat_u = mats
                            .get(&ku1)
                            .or_else(|| mats.get(&ku2))
                            .ok_or_else(|| format!("T16C missing LittleBit {ku1}"))?;
                        let (z, sk_v) = if col_mask.is_some() {
                            mat_v.project_batch_sparse_ext(
                                x_batch,
                                seq_len,
                                native_t16c.as_deref(),
                                threads,
                                None,
                                col_mask,
                                include_aux,
                                false,
                            )?
                        } else {
                            (mat_v.project_batch(x_batch, seq_len, native_t16c.as_deref(), threads)?, 0)
                        };
                        let (mut y_lb, sk_u) = if row_mask.is_some() {
                            mat_u.project_batch_sparse_ext(
                                &z,
                                seq_len,
                                native_t16c.as_deref(),
                                threads,
                                row_mask,
                                None,
                                include_aux,
                                false,
                            )?
                        } else {
                            (mat_u.project_batch(&z, seq_len, native_t16c.as_deref(), threads)?, 0)
                        };
                        if row_mask.is_some() || col_mask.is_some() {
                            let passes = (seq_len.div_ceil(8)).max(1) as u64;
                            mlp_sp_total_step += (mat_v.d_out.div_ceil(16) * (mat_v.d_in / mat_v.block_size)
                                + mat_u.d_out.div_ceil(16) * (mat_u.d_in / mat_u.block_size)) as u64
                                * passes;
                            if sk_v > 0 {
                                mlp_sp_skipped_step += sk_v as u64;
                            }
                            if sk_u > 0 {
                                mlp_sp_skipped_step += sk_u as u64;
                            }
                        }
                        let krv1 = format!("{k1}.lb_rv");
                        let kru1 = format!("{k1}.lb_ru");
                        if let (Some(rv), Some(ru)) = (mats.get(&krv1), mats.get(&kru1)) {
                            let zr = if col_mask.is_some() {
                                rv.project_batch_sparse_ext(x_batch, seq_len, native_t16c.as_deref(), threads, None, col_mask, include_aux, false)?.0
                            } else {
                                rv.project_batch(x_batch, seq_len, native_t16c.as_deref(), threads)?
                            };
                            let yr = if row_mask.is_some() {
                                ru.project_batch_sparse_ext(&zr, seq_len, native_t16c.as_deref(), threads, row_mask, None, include_aux, false)?.0
                            } else {
                                ru.project_batch(&zr, seq_len, native_t16c.as_deref(), threads)?
                            };
                            for (dst, &src) in y_lb.iter_mut().zip(yr.iter()) {
                                *dst += src;
                            }
                        }
                        y_lb
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

            // Precompute RoPE (cos, sin) tables ONCE per token position across all 60 layers (eliminating ~3M powf/sin_cos calls/step).
            let rot_local = ((cfg.hd as f32) * cfg.partial_local) as usize;
            let rot_global = ((cfg.global_hd as f32) * cfg.partial_global) as usize;
            let inv_f_local = rope_inv_freqs(cfg.theta_local, rot_local, cfg.hd);
            let inv_f_global = rope_inv_freqs(cfg.theta_global, rot_global, cfg.global_hd);
            let half_rot_local = inv_f_local.len();
            let half_rot_global = inv_f_global.len();
            let rope_len = independent_traj_len.unwrap_or(seq_len);
            let mut rope_cos_local = vec![0.0f32; rope_len * half_rot_local];
            let mut rope_sin_local = vec![0.0f32; rope_len * half_rot_local];
            let mut rope_cos_global = vec![0.0f32; rope_len * half_rot_global];
            let mut rope_sin_global = vec![0.0f32; rope_len * half_rot_global];
            for t in 0..rope_len {
                let pos = (start_pos + tree_spec.map_or(t, |(depths, _)| depths[t])) as u32;
                if half_rot_local > 0 {
                    rope_cos_sin_for_pos(
                        &inv_f_local,
                        pos,
                        &mut rope_cos_local[t * half_rot_local..(t + 1) * half_rot_local],
                        &mut rope_sin_local[t * half_rot_local..(t + 1) * half_rot_local],
                    );
                }
                if half_rot_global > 0 {
                    rope_cos_sin_for_pos(
                        &inv_f_global,
                        pos,
                        &mut rope_cos_global[t * half_rot_global..(t + 1) * half_rot_global],
                        &mut rope_sin_global[t * half_rot_global..(t + 1) * half_rot_global],
                    );
                }
            }

            let mut inp = vec![0.0f32; seq_len * d];
            let mut inp2 = vec![0.0f32; seq_len * d];
            let mut norm_tmp = vec![0.0f32; d];

            for l in layer_start..lim {
                let tl = std::time::Instant::now();
                let pfx = format!("model.language_model.layers.{l}");
                let is_full = cfg.layer_full[l];
                let hd = if is_full { cfg.global_hd } else { cfg.hd };
                let n_kv = if is_full { cfg.n_global_kv } else { cfg.n_kv };
                let sw = if is_full { None } else { Some(cfg.window) };

                let cca_pol_opt = if cca_sparse_active {
                    cca_layer_policies.as_ref().and_then(|v| v.get(l))
                } else {
                    None
                };
                let explicit_bypass = (bypass_layers.get(l).copied().unwrap_or(false)
                    || cca_pol_opt.is_some_and(|p| p.tier2_bypass))
                    && independent_traj_len.is_none();
                if explicit_bypass {
                    mod_total_step += 1;
                    mod_skipped_step += 1;
                    let kv_elems = seq_len * n_kv * hd;
                    let zero_kv = vec![0.0f32; kv_elems];
                    if kv_quant.is_glyph() {
                        if tree_spec.is_some() {
                            ephemeral_tree_kv[l] = (zero_kv.clone(), zero_kv);
                        } else {
                            let gkv = glyph_kv[l]
                                .as_mut()
                                .ok_or_else(|| format!("missing GlyphKvLayer at layer {l}"))?;
                            let stride = n_kv * hd;
                            for t in 0..seq_len {
                                let kt = &zero_kv[t * stride..(t + 1) * stride];
                                let vt = if gkv.k_eq_v { &[][..] } else { kt };
                                gkv.append_token(kt, vt, native_t16c.as_deref())?;
                            }
                        }
                    } else {
                        kv[l].k.extend_from_slice(&zero_kv);
                        kv[l].v.extend_from_slice(&zero_kv);
                    }
                    let ls = scalars[l];
                    if (ls - 1.0).abs() > 1e-7 {
                        for v in states.iter_mut() {
                            *v *= ls;
                        }
                    }
                    continue;
                }

                let layer_mlp_top_p = cca_pol_opt.map_or(mlp_top_p, |p| p.mlp_top_p);
                let layer_head_top_p = cca_pol_opt.map_or(attn_head_top_p, |p| p.head_top_p);
                let layer_mod_thresh = cca_pol_opt.map_or(mod_thresh, |p| p.mod_thresh);

                {
                    let t_n0 = if profile_components { Some(std::time::Instant::now()) } else { None };
                    let g_in = get_norm(&format!("{pfx}.input_layernorm.weight"))?;
                    for t in 0..seq_len {
                        rmsn_raw_into(
                            &states[t * d..(t + 1) * d],
                            Some(g_in),
                            eps,
                            &mut inp[t * d..(t + 1) * d],
                        );
                    }
                    if let Some(t0) = t_n0 {
                        prof_norm_kv_step += t0.elapsed().as_secs_f64();
                    }
                }

                // KV-Safe V7.0 MoD Residual Highway (Option 1): on eligible sliding middle layers, skip
                // q_proj, attention_decode, o_proj, and 100% of MLP (~90.8% of layer DRAM/compute), while
                // refreshing k_proj/v_proj so all 60 layers remain KV-contiguous for truncate_kv & spec rollback.
                // Compute POV before the MoD branch so adaptive_mod_q is live
                // even on layers that may be skipped. The same decision is
                // reused below for CRTR and GTA-LOD; never evaluate it twice.
                let need_6sense = player_pov_6sense
                    || district_router.as_ref().is_some_and(|dr| dr.sense_dim > 0);
                let pov_s32_opt = if need_6sense {
                    let tr0 = if profile_components { Some(std::time::Instant::now()) } else { None };
                    let s32 = extract_6sense_pov_features(&inp, seq_len, d, 64)?;
                    if let Some(t0) = tr0 {
                        prof_router_step += t0.elapsed().as_secs_f64();
                    }
                    Some(s32)
                } else {
                    None
                };
                let base_top_k_l = district_router.as_ref().map_or(2usize, |dr| dr.base_top_k.max(1));
                let pov_decision_opt = pov_s32_opt.as_ref().map(|s32| {
                    let dec = evaluate_6sense_consensus_gate(
                        s32,
                        0.72,
                        0.48,
                        base_top_k_l,
                        if layer_mlp_top_p < 0.9999 { layer_mlp_top_p } else { 0.30 },
                        layer_mod_thresh,
                        pov_base_draft_k,
                    );
                    pov_layers_step += 1;
                    pov_consensus_step += dec.consensus_score as f64;
                    pov_draft_cap_step = pov_draft_cap_step.min(dec.dyn_draft_k.max(1));
                    match dec.mode {
                        ConsensusMode::Clean => pov_clean_step += 1,
                        ConsensusMode::Caution => pov_caution_step += 1,
                        ConsensusMode::Rescue => pov_rescue_step += 1,
                    }
                    dec
                });
                let eff_mod_thresh = effective_pov_mod_thresh(
                    layer_mod_thresh,
                    pov_decision_opt.as_ref().map(|dec| dec.adaptive_mod_q),
                    player_pov_6sense && pov_target_dynamic,
                );
                let mod_eligible = eff_mod_thresh > 0.0
                    && independent_traj_len.is_none()
                    && !is_full
                    && l >= 4
                    && l + 4 < cfg.n_layers
                    && t16c_mats.is_some()
                    && cortical_router.is_some();
                let mod_skip_layer = if mod_eligible {
                    mod_total_step += 1;
                    let tr0 = if profile_components { Some(std::time::Instant::now()) } else { None };
                    let exec_prob = cortical_router
                        .as_ref()
                        .unwrap()
                        .layers[l]
                        .predict_mod_exec_prob(&inp, seq_len)?
                        .ok_or_else(|| format!("MoD gate missing on layer {l}"))?;
                    if let Some(t0) = tr0 {
                        prof_router_step += t0.elapsed().as_secs_f64();
                    }
                    exec_prob < eff_mod_thresh
                } else {
                    false
                };
                if mod_skip_layer {
                    mod_skipped_step += 1;
                    let t_ap0 = if profile_components { Some(std::time::Instant::now()) } else { None };
                    let (mut k_batch, mut v_batch) = if mod_zero_kv {
                        let kv_elems = seq_len * n_kv * hd;
                        (vec![0.0f32; kv_elems], vec![0.0f32; kv_elems])
                    } else {
                        let kb = run_proj_masked(l, "self_attn.k_proj", &inp, None, None, true)?;
                        let vb = if is_full && cfg.attention_k_eq_v {
                            kb.clone()
                        } else {
                            run_proj_masked(l, "self_attn.v_proj", &inp, None, None, true)?
                        };
                        (kb, vb)
                    };
                    if let Some(t0) = t_ap0 {
                        prof_attn_step += t0.elapsed().as_secs_f64();
                    }
                    let t_kv0 = if profile_components { Some(std::time::Instant::now()) } else { None };
                    if !mod_zero_kv {
                        let kn = get_norm(&format!("{pfx}.self_attn.k_norm.weight"))?;
                        let kv_stride = n_kv * hd;
                        for t in 0..seq_len {
                            let cos_t = &rope_cos_local[t * half_rot_local..(t + 1) * half_rot_local];
                            let sin_t = &rope_sin_local[t * half_rot_local..(t + 1) * half_rot_local];
                            let ks = &mut k_batch[t * kv_stride..(t + 1) * kv_stride];
                            qk_norm_raw_inplace(ks, n_kv, hd, Some(kn), eps);
                            for h in 0..n_kv {
                                rope_head_with_cos_sin(&mut ks[h * hd..(h + 1) * hd], cos_t, sin_t, hd);
                            }
                            let vs = &mut v_batch[t * kv_stride..(t + 1) * kv_stride];
                            qk_norm_raw_inplace(vs, n_kv, hd, None, eps);
                        }
                    }
                    if kv_quant.is_glyph() {
                        if tree_spec.is_some() {
                            ephemeral_tree_kv[l] = (k_batch, v_batch);
                        } else {
                            let gkv = glyph_kv[l]
                                .as_mut()
                                .ok_or_else(|| format!("missing GlyphKvLayer at layer {l}"))?;
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
                            KvQuantMode::Glyph | KvQuantMode::GlyphPure | KvQuantMode::GlyphLod => unreachable!(),
                        }
                        kv[l].k.extend_from_slice(&k_batch);
                        kv[l].v.extend_from_slice(&v_batch);
                    }
                    if let Some(t0) = t_kv0 {
                        prof_norm_kv_step += t0.elapsed().as_secs_f64();
                    }
                    let ls = scalars[l];
                    if (ls - 1.0).abs() > 1e-7 {
                        for v in states.iter_mut() {
                            *v *= ls;
                        }
                    }
                    if verbose_layers && (l == 0 || l == 5 || l % 15 == 14 || l + 1 == cfg.n_layers) {
                        eprintln!(
                            "[run31b] layer {l:02}/{} MOD_SKIP in {:.2}s (pinned={})",
                            cfg.n_layers,
                            tl.elapsed().as_secs_f64(),
                            all_pinned || l < pinned.len()
                        );
                    }
                    continue;
                }

                let head_masks_opt: Option<(Vec<u8>, Vec<u8>, Vec<u8>, Option<Vec<Vec<u8>>>)> =
                    if let Some(hr) = head_router.as_ref() {
                        if layer_head_top_p < 0.9999 && l >= 2 && l + 2 < cfg.n_layers {
                            let tr0 = if profile_components { Some(std::time::Instant::now()) } else { None };
                            let tok_masks = hr.layers[l].predict_per_token_col_masks(
                                &inp,
                                seq_len,
                                attn_shared_heads,
                                layer_head_top_p,
                            )?;
                            let head_cmask = union_col_masks(&tok_masks, n_heads);
                            let head_tok_masks = if seq_len > 1 { Some(tok_masks) } else { None };
                            if let Some(t0) = tr0 {
                                prof_router_step += t0.elapsed().as_secs_f64();
                            }
                            let o_bsize = t16c_mats
                                .as_ref()
                                .and_then(|mats| {
                                    mats.get(&format!("{pfx}.self_attn.o_proj.weight"))
                                        .or_else(|| mats.get(&format!("model.layers.{l}.self_attn.o_proj.weight")))
                                        .map(|m| m.block_size)
                                })
                                .unwrap_or(256);
                            let (q_rmask, o_cmask) = expand_head_mask(&head_cmask, hd, o_bsize);
                            attn_sp_total_step += (seq_len * n_heads) as u64;
                            if let Some(ref htm) = head_tok_masks {
                                let sk_h: usize = htm.iter().map(|v| v.iter().filter(|&&b| b == 0).count()).sum();
                                attn_sp_skipped_step += sk_h as u64;
                            } else {
                                let sk_h = head_cmask.iter().filter(|&&b| b == 0).count();
                                attn_sp_skipped_step += (seq_len * sk_h) as u64;
                            }
                            Some((head_cmask, q_rmask, o_cmask, head_tok_masks))
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                let t_ap0 = if profile_components { Some(std::time::Instant::now()) } else { None };
                let q_rm_ref = head_masks_opt.as_ref().map(|(_, qm, _, _)| qm.as_slice());
                let mut q_batch = run_proj_masked(l, "self_attn.q_proj", &inp, q_rm_ref, None, true)?;
                let k_raw = run_proj_masked(l, "self_attn.k_proj", &inp, None, None, true)?;
                let mut v_batch = if is_full && cfg.attention_k_eq_v {
                    k_raw.clone()
                } else {
                    run_proj_masked(l, "self_attn.v_proj", &inp, None, None, true)?
                };
                if let Some(t0) = t_ap0 {
                    prof_attn_step += t0.elapsed().as_secs_f64();
                }
                let t_kv0 = if profile_components { Some(std::time::Instant::now()) } else { None };

                let mut k_batch = k_raw;
                {
                    let qn = get_norm(&format!("{pfx}.self_attn.q_norm.weight"))?;
                    let kn = get_norm(&format!("{pfx}.self_attn.k_norm.weight"))?;
                    let (rcos_all, rsin_all, hrot) = if is_full {
                        (&rope_cos_global[..], &rope_sin_global[..], half_rot_global)
                    } else {
                        (&rope_cos_local[..], &rope_sin_local[..], half_rot_local)
                    };
                    let q_stride = n_heads * hd;
                    let kv_stride = n_kv * hd;
                    if seq_len > 1 && threads > 1 {
                        let tok_chunk = seq_len.div_ceil(threads);
                        std::thread::scope(|s| {
                            for (ci, ((qc, kc), vc)) in q_batch
                                .chunks_mut(tok_chunk * q_stride)
                                .zip(k_batch.chunks_mut(tok_chunk * kv_stride))
                                .zip(v_batch.chunks_mut(tok_chunk * kv_stride))
                                .enumerate()
                            {
                                let t_base = ci * tok_chunk;
                                let n_tok = qc.len() / q_stride;
                                s.spawn(move || {
                                    for dt in 0..n_tok {
                                        let t = t_base + dt;
                                        let rope_t = if independent_traj_len.is_some() { t % rope_len } else { t };
                                        let cos_t = &rcos_all[rope_t * hrot..(rope_t + 1) * hrot];
                                        let sin_t = &rsin_all[rope_t * hrot..(rope_t + 1) * hrot];
                                        let qs = &mut qc[dt * q_stride..(dt + 1) * q_stride];
                                        qk_norm_raw_inplace(qs, n_heads, hd, Some(qn), eps);
                                        for h in 0..n_heads {
                                            rope_head_with_cos_sin(&mut qs[h * hd..(h + 1) * hd], cos_t, sin_t, hd);
                                        }
                                        let ks = &mut kc[dt * kv_stride..(dt + 1) * kv_stride];
                                        qk_norm_raw_inplace(ks, n_kv, hd, Some(kn), eps);
                                        for h in 0..n_kv {
                                            rope_head_with_cos_sin(&mut ks[h * hd..(h + 1) * hd], cos_t, sin_t, hd);
                                        }
                                        let vs = &mut vc[dt * kv_stride..(dt + 1) * kv_stride];
                                        qk_norm_raw_inplace(vs, n_kv, hd, None, eps);
                                    }
                                });
                            }
                        });
                    } else {
                        for t in 0..seq_len {
                            let rope_t = if independent_traj_len.is_some() { t % rope_len } else { t };
                            let cos_t = &rcos_all[rope_t * hrot..(rope_t + 1) * hrot];
                            let sin_t = &rsin_all[rope_t * hrot..(rope_t + 1) * hrot];
                            let qs = &mut q_batch[t * q_stride..(t + 1) * q_stride];
                            qk_norm_raw_inplace(qs, n_heads, hd, Some(qn), eps);
                            for h in 0..n_heads {
                                rope_head_with_cos_sin(&mut qs[h * hd..(h + 1) * hd], cos_t, sin_t, hd);
                            }

                            let ks = &mut k_batch[t * kv_stride..(t + 1) * kv_stride];
                            qk_norm_raw_inplace(ks, n_kv, hd, Some(kn), eps);
                            for h in 0..n_kv {
                                rope_head_with_cos_sin(&mut ks[h * hd..(h + 1) * hd], cos_t, sin_t, hd);
                            }

                            let vs = &mut v_batch[t * kv_stride..(t + 1) * kv_stride];
                            qk_norm_raw_inplace(vs, n_kv, hd, None, eps);
                        }
                    }
                }

                let q_stride = n_heads * hd;
                let mut ctx_batch = vec![0.0f32; seq_len * q_stride];
                if let Some(tlen) = independent_traj_len {
                    causal_self_attn_independent_trajs(
                        &q_batch,
                        &k_batch,
                        &v_batch,
                        seq_len,
                        tlen,
                        start_pos,
                        n_heads,
                        n_kv,
                        hd,
                        sw,
                        threads,
                        &mut ctx_batch,
                    )?;
                    drop(k_batch);
                    drop(v_batch);
                } else if kv_quant.is_glyph() {
                    if let Some((tree_depths, ancestor_masks)) = tree_spec {
                        let gkv_ref = glyph_kv[l]
                            .as_ref()
                            .ok_or_else(|| format!("missing GlyphKvLayer at layer {l}"))?;
                        let base_t = gkv_ref.total_seen();
                        let nat_ref = native_t16c.as_deref();
                        if seq_len > 1 && threads > 1 {
                            let tok_chunk = seq_len.div_ceil(threads);
                            let inner_thr = (threads / seq_len).max(1);
                            let err_slot = std::sync::Mutex::new(None::<String>);
                            let k_ref = &k_batch;
                            let v_ref = &v_batch;
                            let q_ref = &q_batch;
                            std::thread::scope(|s| {
                                for (ci, dst_chunk) in ctx_batch.chunks_mut(tok_chunk * q_stride).enumerate() {
                                    let t_base = ci * tok_chunk;
                                    let n_tok = dst_chunk.len() / q_stride;
                                    let err_r = &err_slot;
                                    s.spawn(move || {
                                        for dt in 0..n_tok {
                                            let t = t_base + dt;
                                            let q_t = &q_ref[t * q_stride..(t + 1) * q_stride];
                                            match gkv_ref.attend_tree_ephemeral(
                                                q_t,
                                                base_t,
                                                k_ref,
                                                v_ref,
                                                seq_len,
                                                ancestor_masks[t],
                                                tree_depths,
                                                start_pos + tree_depths[t],
                                                1.0,
                                                nat_ref,
                                                inner_thr,
                                            ) {
                                                Ok(c_t) => {
                                                    dst_chunk[dt * q_stride..(dt + 1) * q_stride].copy_from_slice(&c_t);
                                                }
                                                Err(e) => {
                                                    let mut guard = err_r.lock().unwrap();
                                                    if guard.is_none() {
                                                        *guard = Some(e);
                                                    }
                                                    return;
                                                }
                                            }
                                        }
                                    });
                                }
                            });
                            if let Some(e) = err_slot.into_inner().unwrap() {
                                return Err(e);
                            }
                        } else {
                            for t in 0..seq_len {
                                let q_t = &q_batch[t * q_stride..(t + 1) * q_stride];
                                let c_t = gkv_ref.attend_tree_ephemeral(
                                    q_t,
                                    base_t,
                                    &k_batch,
                                    &v_batch,
                                    seq_len,
                                    ancestor_masks[t],
                                    tree_depths,
                                    start_pos + tree_depths[t],
                                    1.0,
                                    nat_ref,
                                    threads,
                                )?;
                                ctx_batch[t * q_stride..(t + 1) * q_stride].copy_from_slice(&c_t);
                            }
                        }
                        ephemeral_tree_kv[l] = (k_batch, v_batch);
                    } else {
                        let gkv = glyph_kv[l]
                            .as_mut()
                            .ok_or_else(|| format!("missing GlyphKvLayer at layer {l}"))?;
                        prefill_glyph_kv_causal_batch(
                            gkv,
                            kv_quant,
                            &q_batch,
                            &k_batch,
                            &v_batch,
                            seq_len,
                            start_pos,
                            n_heads,
                            n_kv,
                            hd,
                            sw,
                            is_full,
                            vis_kv_ends,
                            native_t16c.as_deref(),
                            threads,
                            &mut ctx_batch,
                        )?;
                        drop(k_batch);
                        drop(v_batch);
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
                        KvQuantMode::Glyph | KvQuantMode::GlyphPure | KvQuantMode::GlyphLod => unreachable!(),
                    }

                    let base_t = kv[l].k.len() / (n_kv * hd);
                    kv[l].k.extend_from_slice(&k_batch);
                    kv[l].v.extend_from_slice(&v_batch);
                    drop(k_batch);
                    drop(v_batch);
                    if let Some((tree_depths, ancestor_masks)) = tree_spec {
                        attn_decode_tree_batch_into(
                            &q_batch,
                            &kv[l].k,
                            &kv[l].v,
                            base_t,
                            seq_len,
                            ancestor_masks,
                            tree_depths,
                            start_pos,
                            n_heads,
                            n_kv,
                            hd,
                            sw,
                            threads,
                            &mut ctx_batch,
                        );
                    } else if seq_len > 1 && threads > 1 {
                        let tok_chunk = seq_len.div_ceil(threads);
                        let k_seq_ref = &kv[l].k;
                        let v_seq_ref = &kv[l].v;
                        let q_ref = &q_batch;
                        std::thread::scope(|s| {
                            for (ci, dst_chunk) in ctx_batch.chunks_mut(tok_chunk * q_stride).enumerate() {
                                let t_base = ci * tok_chunk;
                                let n_tok = dst_chunk.len() / q_stride;
                                s.spawn(move || {
                                    for dt in 0..n_tok {
                                        let t = t_base + dt;
                                        let q_t = &q_ref[t * q_stride..(t + 1) * q_stride];
                                        let kv_end = if !is_full {
                                            vis_kv_ends.map_or(base_t + t, |ends| base_t + ends[t])
                                        } else {
                                            base_t + t
                                        };
                                        let c_t = attn_decode_range(q_t, k_seq_ref, v_seq_ref, base_t + t, kv_end, start_pos + t, n_heads, n_kv, hd, sw);
                                        dst_chunk[dt * q_stride..(dt + 1) * q_stride].copy_from_slice(&c_t);
                                    }
                                });
                            }
                        });
                    } else {
                        for t in 0..seq_len {
                            let q_t = &q_batch[t * q_stride..(t + 1) * q_stride];
                            let kv_end = if !is_full {
                                vis_kv_ends.map_or(base_t + t, |ends| base_t + ends[t])
                            } else {
                                base_t + t
                            };
                            let c_t = attn_decode_range(q_t, &kv[l].k, &kv[l].v, base_t + t, kv_end, start_pos + t, n_heads, n_kv, hd, sw);
                            ctx_batch[t * q_stride..(t + 1) * q_stride].copy_from_slice(&c_t);
                        }
                    }
                }
                drop(q_batch);

                if let Some((ref hcm, _, _, ref htm_opt)) = head_masks_opt {
                    for t in 0..seq_len {
                        let base_ctx = t * q_stride;
                        for h in 0..n_heads {
                            let active = hcm[h] != 0 && htm_opt.as_ref().is_none_or(|htm| htm[t][h] != 0);
                            if !active {
                                ctx_batch[base_ctx + h * hd..base_ctx + (h + 1) * hd].fill(0.0);
                            }
                        }
                    }
                }
                if let Some(t0) = t_kv0 {
                    prof_norm_kv_step += t0.elapsed().as_secs_f64();
                }
                let t_ao0 = if profile_components { Some(std::time::Instant::now()) } else { None };
                let o_cm_ref = head_masks_opt.as_ref().map(|(_, _, om, _)| om.as_slice());
                let attn_o = run_proj_masked(l, "self_attn.o_proj", &ctx_batch, None, o_cm_ref, true)?;
                if let Some(t0) = t_ao0 {
                    prof_attn_step += t0.elapsed().as_secs_f64();
                }
                drop(ctx_batch);
                let t_n1 = if profile_components { Some(std::time::Instant::now()) } else { None };
                {
                    let g_pa = get_norm(&format!("{pfx}.post_attention_layernorm.weight"))?;
                    for t in 0..seq_len {
                        rmsn_raw_into(&attn_o[t * d..(t + 1) * d], Some(g_pa), eps, &mut norm_tmp);
                        let st = &mut states[t * d..(t + 1) * d];
                        for (dst, &src) in st.iter_mut().zip(norm_tmp.iter()) {
                            *dst += src;
                        }
                    }
                }
                drop(attn_o);

                {
                    let g_pf = get_norm(&format!("{pfx}.pre_feedforward_layernorm.weight"))?;
                    for t in 0..seq_len {
                        rmsn_raw_into(
                            &states[t * d..(t + 1) * d],
                            Some(g_pf),
                            eps,
                            &mut inp2[t * d..(t + 1) * d],
                        );
                    }
                }
                if let Some(t0) = t_n1 {
                    prof_norm_kv_step += t0.elapsed().as_secs_f64();
                }
                let t_g0 = if profile_components { Some(std::time::Instant::now()) } else { None };
                let eff_mlp_top_p = if player_pov_6sense && pov_target_dynamic && layer_mlp_top_p < 0.9999 {
                    pov_decision_opt
                        .as_ref()
                        .map_or(layer_mlp_top_p, |dec| dec.adaptive_crtr_top_p)
                } else {
                    layer_mlp_top_p
                };
                let gta_lod_shared_l = lod_shared_mats.as_ref().and_then(|sh| {
                    let g = sh
                        .get(&format!("{pfx}.mlp.gate_proj.weight"))
                        .or_else(|| sh.get(&format!("model.layers.{l}.mlp.gate_proj.weight")))?;
                    let u = sh
                        .get(&format!("{pfx}.mlp.up_proj.weight"))
                        .or_else(|| sh.get(&format!("model.layers.{l}.mlp.up_proj.weight")))?;
                    let dm = sh
                        .get(&format!("{pfx}.mlp.down_proj.weight"))
                        .or_else(|| sh.get(&format!("model.layers.{l}.mlp.down_proj.weight")))?;
                    Some((g, u, dm))
                });
                let cold_mlp_mats = if gta_lod_shared_l.is_some() {
                    None
                } else {
                    t16c_mats.as_ref().and_then(|mats| {
                        let gm = mats
                            .get(&format!("{pfx}.mlp.gate_proj.weight"))
                            .or_else(|| mats.get(&format!("model.layers.{l}.mlp.gate_proj.weight")));
                        let um = mats
                            .get(&format!("{pfx}.mlp.up_proj.weight"))
                            .or_else(|| mats.get(&format!("model.layers.{l}.mlp.up_proj.weight")));
                        let dm = mats
                            .get(&format!("{pfx}.mlp.down_proj.weight"))
                            .or_else(|| mats.get(&format!("model.layers.{l}.mlp.down_proj.weight")));
                        match (gm, um, dm) {
                            (Some(g), Some(u), Some(d)) if g.is_streamed() || u.is_streamed() || d.is_streamed() => {
                                Some((g, u, d))
                            }
                            _ => None,
                        }
                    })
                };
                let predicted_masks: Option<(Vec<u8>, Vec<u8>, usize, Option<Vec<Vec<u8>>>)> = if eff_mlp_top_p < 0.9999 {
                    if let (Some(cr), Some(mats)) = (cortical_router.as_ref(), t16c_mats.as_ref()) {
                        let check_proj_blk = |suf: &str, l_key: &str| -> Option<usize> {
                            let k1 = format!("model.language_model.layers.{l}.{suf}.weight");
                            let k2 = format!("model.layers.{l}.{suf}.weight");
                            let k1_lb = format!("{k1}.{l_key}");
                            let m = mats.get(&k1).or_else(|| mats.get(&k2)).or_else(|| mats.get(&k1_lb))?;
                            let eb = m.n_bits & 0x1F;
                            if !m.is_streamed() && eb == 8 { None } else { Some(m.block_size) }
                        };
                        let gb = check_proj_blk("mlp.gate_proj", "lb_u");
                        let ub = check_proj_blk("mlp.up_proj", "lb_u");
                        let db = check_proj_blk("mlp.down_proj", "lb_v");
                        let bsize = cr.block_size;
                        if gb == Some(bsize) && ub == Some(bsize) && db == Some(bsize) && bsize > 0 && cfg.ff % bsize == 0 {
                            let tr0 = if profile_components { Some(std::time::Instant::now()) } else { None };
                            let _ = &tree_spec;
                            let tok_masks = cr.layers[l].predict_per_token_col_masks(
                                &inp2,
                                seq_len,
                                shared_mlp_blocks,
                                eff_mlp_top_p,
                            )?;
                            let cmask = union_col_masks(&tok_masks, cr.layers[l].n_blocks);
                            let rmask = expand_col_to_row_mask(&cmask, bsize, cfg.ff);
                            let blk_sz = bsize;
                            let tok_cmasks = if seq_len > 1 { Some(tok_masks) } else { None };
                            if let Some(t0) = tr0 {
                                prof_router_step += t0.elapsed().as_secs_f64();
                            }
                            cortical_gate_total_step += cmask.len() as u64;
                            cortical_gate_skipped_step += cmask.iter().filter(|&&v| v == 0).count() as u64;
                            Some((rmask, cmask, blk_sz, tok_cmasks))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };

                let down: Vec<f32> = if let Some((sh_gate, sh_up, sh_down)) = gta_lod_shared_l {
                    let ff = cfg.ff;
                    let n_dist_shards = lod_district_mats.len();
                    let active_top_k = if pov_target_dynamic {
                        pov_decision_opt
                            .as_ref()
                            .map_or(base_top_k_l, |dec| dec.dyn_top_k)
                    } else {
                        base_top_k_l
                    }
                    .clamp(1, n_dist_shards.max(1));
                    let (sh_slice, active_d_slices): ((usize, usize), Vec<(usize, (usize, usize))>) =
                        if let Some(ref dr) = *district_router {
                            let tr0 = if profile_components { Some(std::time::Instant::now()) } else { None };
                            let dl = &dr.layers[l];
                            let s_ref = pov_s32_opt.as_ref().map(|s| &s.features[..]);
                            let (top_idx, _) = dl.predict_active_districts(&inp, seq_len, s_ref, active_top_k)?;
                            if let Some(t0) = tr0 {
                                prof_router_step += t0.elapsed().as_secs_f64();
                            }
                            let act_s = top_idx
                                .into_iter()
                                .filter(|&di| di < dl.district_slices.len() && di < n_dist_shards)
                                .map(|di| (di, dl.district_slices[di]))
                                .collect();
                            (dl.shared_slice, act_s)
                        } else {
                            let sh_end = sh_gate.d_out.min(ff);
                            let rem = ff.saturating_sub(sh_end);
                            let d_w = if n_dist_shards > 0 { rem / n_dist_shards } else { 0 };
                            let mut act_s = Vec::new();
                            for di in 0..active_top_k.min(n_dist_shards) {
                                let ds = sh_end + di * d_w;
                                let de = (ds + d_w).min(ff);
                                if de > ds {
                                    act_s.push((di, (ds, de)));
                                }
                            }
                            ((0, sh_end), act_s)
                        };
                    lod_total_d_step += n_dist_shards as u64;
                    lod_active_d_step += active_d_slices.len() as u64;

                    let rm_ref = predicted_masks.as_ref().map(|(r, _, _, _)| r.as_slice());
                    let cm_ref = predicted_masks.as_ref().map(|(_, c, _, _)| c.as_slice());
                    let cr_bsize_opt = predicted_masks.as_ref().map(|(_, _, b, _)| *b);
                    let mut gelu_gate = run_proj_masked(l, "mlp.gate_proj", &inp2, rm_ref, None, true)?;
                    let mut add_lod_rows = |mat: &run270_rust::t16c::T16CMat, s0: usize, s1: usize, dst: &mut [f32]| -> Result<(), String> {
                        let w_sl = s1.saturating_sub(s0);
                        if w_sl == 0 || mat.d_out != w_sl {
                            return Err(format!("GTA-LOD row slice ({s0}..{s1}) mismatch vs mat {} d_out={}", mat.name, mat.d_out));
                        }
                        let sub_rm = if s0 % 16 == 0 && s1 % 16 == 0 {
                            rm_ref.and_then(|rm| rm.get(s0 / 16..s1 / 16))
                        } else {
                            None
                        };
                        if sub_rm.is_some_and(|sr| sr.iter().all(|&v| v == 0)) {
                            return Ok(());
                        }
                        let delta = if let Some(sr) = sub_rm {
                            let act_tiles = sr.iter().filter(|&&v| v != 0).count();
                            if mat.is_streamed() {
                                cold_read_bytes += (mat.packed_bytes_len * act_tiles / sr.len().max(1)) as u64;
                            }
                            mat.project_batch_sparse(&inp2, seq_len, native_t16c.as_deref(), threads, Some(sr), None)?.0
                        } else {
                            if mat.is_streamed() {
                                cold_read_bytes += mat.packed_bytes_len as u64;
                            }
                            mat.project_batch(&inp2, seq_len, native_t16c.as_deref(), threads)?
                        };
                        for t in 0..seq_len {
                            let d_row = &delta[t * w_sl..(t + 1) * w_sl];
                            let y_row = &mut dst[t * ff + s0..t * ff + s1];
                            for (yv, &dv) in y_row.iter_mut().zip(d_row.iter()) {
                                *yv += dv;
                            }
                        }
                        Ok(())
                    };
                    if sh_slice.1 > sh_slice.0 {
                        add_lod_rows(sh_gate, sh_slice.0, sh_slice.1, &mut gelu_gate)?;
                    }
                    for &(di, (ds, de)) in &active_d_slices {
                        let d_map = &lod_district_mats[di];
                        if let Some(gm_d) = d_map
                            .get(&format!("{pfx}.mlp.gate_proj.weight"))
                            .or_else(|| d_map.get(&format!("model.layers.{l}.mlp.gate_proj.weight")))
                        {
                            add_lod_rows(gm_d, ds, de, &mut gelu_gate)?;
                        }
                    }
                    if let Some(t0) = t_g0 {
                        prof_gate_step += t0.elapsed().as_secs_f64();
                    }

                    let t_gl0 = if profile_components { Some(std::time::Instant::now()) } else { None };
                    if let Some((_, ref cmask, bsize, ref tok_cmasks_opt)) = predicted_masks {
                        for t in 0..seq_len {
                            let base = t * ff;
                            let tc = tok_cmasks_opt.as_ref().map_or(cmask.as_slice(), |v| v[t].as_slice());
                            for (b, &active) in tc.iter().enumerate() {
                                if active != 0 {
                                    for x in &mut gelu_gate[base + b * bsize..base + (b + 1) * bsize] {
                                        *x = gelu_tanh_fast_f32(*x);
                                    }
                                } else {
                                    gelu_gate[base + b * bsize..base + (b + 1) * bsize].fill(0.0);
                                }
                            }
                        }
                    } else {
                        gelu_tanh_inplace_parallel(&mut gelu_gate, threads);
                    }
                    if let Some(t0) = t_gl0 {
                        prof_norm_kv_step += t0.elapsed().as_secs_f64();
                    }

                    let t_u0 = if profile_components { Some(std::time::Instant::now()) } else { None };
                    let mut up = run_proj_masked(l, "mlp.up_proj", &inp2, rm_ref, None, true)?;
                    if sh_slice.1 > sh_slice.0 {
                        add_lod_rows(sh_up, sh_slice.0, sh_slice.1, &mut up)?;
                    }
                    for &(di, (ds, de)) in &active_d_slices {
                        let d_map = &lod_district_mats[di];
                        if let Some(um_d) = d_map
                            .get(&format!("{pfx}.mlp.up_proj.weight"))
                            .or_else(|| d_map.get(&format!("model.layers.{l}.mlp.up_proj.weight")))
                        {
                            add_lod_rows(um_d, ds, de, &mut up)?;
                        }
                    }
                    if let Some(t0) = t_u0 {
                        prof_up_step += t0.elapsed().as_secs_f64();
                    }

                    let act: Vec<f32> = if let Some((_, ref cmask, bsize, ref tok_cmasks_opt)) = predicted_masks {
                        let mut out_a = vec![0.0f32; seq_len * ff];
                        for t in 0..seq_len {
                            let base = t * ff;
                            let tc = tok_cmasks_opt.as_ref().map_or(cmask.as_slice(), |v| v[t].as_slice());
                            for (b, &active) in tc.iter().enumerate() {
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
                    let mut down_out = run_proj_masked(l, "mlp.down_proj", &act, None, cm_ref, true)?;
                    let mut add_lod_cols = |mat: &run270_rust::t16c::T16CMat, s0: usize, s1: usize, dst: &mut [f32]| -> Result<(), String> {
                        let w_sl = s1.saturating_sub(s0);
                        if w_sl == 0 || mat.d_in != w_sl || mat.d_out != d {
                            return Err(format!("GTA-LOD col slice ({s0}..{s1}) mismatch vs mat {} ({}x{})", mat.name, mat.d_out, mat.d_in));
                        }
                        let sub_cm = match (cm_ref, cr_bsize_opt) {
                            (Some(cm), Some(cb)) if cb == mat.block_size && cb > 0 && s0 % cb == 0 && s1 % cb == 0 => {
                                cm.get(s0 / cb..s1 / cb)
                            }
                            _ => None,
                        };
                        if sub_cm.is_some_and(|sc| sc.iter().all(|&v| v == 0)) {
                            return Ok(());
                        }
                        let mut act_sub = Vec::with_capacity(seq_len * w_sl);
                        for t in 0..seq_len {
                            act_sub.extend_from_slice(&act[t * ff + s0..t * ff + s1]);
                        }
                        let delta = if let Some(sc) = sub_cm {
                            let act_blks = sc.iter().filter(|&&v| v != 0).count();
                            if mat.is_streamed() {
                                cold_read_bytes += (mat.packed_bytes_len * act_blks / sc.len().max(1)) as u64;
                            }
                            mat.project_batch_sparse(&act_sub, seq_len, native_t16c.as_deref(), threads, None, Some(sc))?.0
                        } else {
                            if mat.is_streamed() {
                                cold_read_bytes += mat.packed_bytes_len as u64;
                            }
                            mat.project_batch(&act_sub, seq_len, native_t16c.as_deref(), threads)?
                        };
                        for (yv, &dv) in dst.iter_mut().zip(delta.iter()) {
                            *yv += dv;
                        }
                        Ok(())
                    };
                    if sh_slice.1 > sh_slice.0 {
                        add_lod_cols(sh_down, sh_slice.0, sh_slice.1, &mut down_out)?;
                    }
                    for &(di, (ds, de)) in &active_d_slices {
                        let d_map = &lod_district_mats[di];
                        if let Some(dm_d) = d_map
                            .get(&format!("{pfx}.mlp.down_proj.weight"))
                            .or_else(|| d_map.get(&format!("model.layers.{l}.mlp.down_proj.weight")))
                        {
                            add_lod_cols(dm_d, ds, de, &mut down_out)?;
                        }
                    }
                    if let Some(t0) = t_d0 {
                        prof_down_step += t0.elapsed().as_secs_f64();
                    }
                    drop(act);
                    down_out
                } else if let Some((gate_mat, up_mat, down_mat)) = cold_mlp_mats {
                    let apply_c8db_cold = |proj_suffix: &str, x_in: &[f32], y_out: &mut [f32]| -> Result<(), String> {
                        if apply_c8d_proj {
                            if let Some(ref bank) = *c8d_bank {
                                let k1 = format!("{pfx}.{proj_suffix}.weight");
                                let k2 = format!("model.layers.{l}.{proj_suffix}.weight");
                                if let Some(entry) = bank.entries.get(&k1).or_else(|| bank.entries.get(&k2)) {
                                    entry.apply_batch_into(x_in, seq_len, y_out, bank.native.as_ref(), threads, None)?;
                                }
                            }
                        }
                        Ok(())
                    };
                    let rec_sparse = |mat: &run270_rust::t16c::T16CMat, sk: i64, tot_acc: &mut u64, sk_acc: &mut u64| {
                        let n_tiles_m = mat.d_out.div_ceil(16) as u64;
                        let n_blks_m = (mat.d_in / mat.block_size) as u64;
                        let passes = (seq_len.div_ceil(4)).max(1) as u64;
                        *tot_acc += n_tiles_m * n_blks_m * passes;
                        if sk > 0 {
                            *sk_acc += sk as u64;
                        }
                    };
                    let build_gate_masks = |gg: &[f32]| -> Option<(Vec<u8>, Vec<u8>, usize)> {
                        if mlp_top_p >= 0.9999 {
                            return None;
                        }
                        let ue = up_mat.n_bits & 0x1F;
                        let de = down_mat.n_bits & 0x1F;
                        if (!up_mat.is_streamed() && ue == 8) || (!down_mat.is_streamed() && de == 8) {
                            return None;
                        }
                        let bsize = down_mat.block_size;
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
                                let sl = &gg[t * ff + b * bsize..t * ff + (b + 1) * bsize];
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
                    };
                    let compute_act = |cm_opt: Option<(&[u8], usize)>, gg: &[f32], up: &[f32]| -> Vec<f32> {
                        if let Some((cmask, bsize)) = cm_opt {
                            let ff = cfg.ff;
                            let mut out_a = vec![0.0f32; seq_len * ff];
                            for t in 0..seq_len {
                                let base = t * ff;
                                for (b, &active) in cmask.iter().enumerate() {
                                    if active != 0 {
                                        let off = base + b * bsize;
                                        for j in 0..bsize {
                                            out_a[off + j] = gg[off + j] * up[off + j];
                                        }
                                    }
                                }
                            }
                            out_a
                        } else {
                            gg.iter().zip(up.iter()).map(|(&a, &b)| a * b).collect()
                        }
                    };

                    if let Some((ref rmask, ref cmask, bsize, ref tok_cmasks_opt)) = predicted_masks {
                        let slab_g = gate_mat.read_sparse_slab(Some(rmask), None)?;
                        cold_read_bytes += slab_g.bytes_read;
                        let (gelu_gate, slab_u) = std::thread::scope(|sc| -> Result<(Vec<f32>, run270_rust::t16c::StreamedSparseSlab), String> {
                            let up_io = sc.spawn(|| up_mat.read_sparse_slab(Some(rmask), None));
                            let (mut gg, sk_g) = gate_mat.eval_sparse_slab(
                                &slab_g,
                                &inp2,
                                seq_len,
                                native_t16c.as_deref(),
                                threads,
                                Some(rmask),
                                None,
                                true,
                                true,
                            )?;
                            drop(slab_g);
                            rec_sparse(gate_mat, sk_g, &mut cold_sp_total_step, &mut cold_sp_skipped_step);
                            apply_c8db_cold("mlp.gate_proj", &inp2, &mut gg)?;
                            if let Some(t0) = t_g0 {
                                prof_gate_step += t0.elapsed().as_secs_f64();
                            }
                            let t_gl0 = if profile_components { Some(std::time::Instant::now()) } else { None };
                            let ff = cfg.ff;
                            for t in 0..seq_len {
                                let base = t * ff;
                                let tc = tok_cmasks_opt.as_ref().map_or(cmask.as_slice(), |v| v[t].as_slice());
                                for (b, &active) in tc.iter().enumerate() {
                                    if active != 0 {
                                        for x in &mut gg[base + b * bsize..base + (b + 1) * bsize] {
                                            *x = gelu_tanh_fast_f32(*x);
                                        }
                                    } else {
                                        gg[base + b * bsize..base + (b + 1) * bsize].fill(0.0);
                                    }
                                }
                            }
                            if let Some(t0) = t_gl0 {
                                prof_norm_kv_step += t0.elapsed().as_secs_f64();
                            }
                            let su = up_io.join().map_err(|_| "cold up_proj read thread panicked".to_string())??;
                            Ok((gg, su))
                        })?;
                        cold_read_bytes += slab_u.bytes_read;

                        let t_u0 = if profile_components { Some(std::time::Instant::now()) } else { None };
                        let (act, slab_d) = std::thread::scope(|sc| -> Result<(Vec<f32>, run270_rust::t16c::StreamedSparseSlab), String> {
                            let down_io = sc.spawn(|| down_mat.read_sparse_slab(None, Some(cmask)));
                            let (mut up, sk_u) = up_mat.eval_sparse_slab(
                                &slab_u,
                                &inp2,
                                seq_len,
                                native_t16c.as_deref(),
                                threads,
                                Some(rmask),
                                None,
                                true,
                                false,
                            )?;
                            drop(slab_u);
                            rec_sparse(up_mat, sk_u, &mut cold_sp_total_step, &mut cold_sp_skipped_step);
                            apply_c8db_cold("mlp.up_proj", &inp2, &mut up)?;
                            if let Some(t0) = t_u0 {
                                prof_up_step += t0.elapsed().as_secs_f64();
                            }
                            let act_v = compute_act(Some((cmask, bsize)), &gelu_gate, &up);
                            let sd = down_io.join().map_err(|_| "cold down_proj read thread panicked".to_string())??;
                            Ok((act_v, sd))
                        })?;
                        drop(gelu_gate);
                        cold_read_bytes += slab_d.bytes_read;

                        let t_d0 = if profile_components { Some(std::time::Instant::now()) } else { None };
                        let (mut down_out, sk_d) = down_mat.eval_sparse_slab(
                            &slab_d,
                            &act,
                            seq_len,
                            native_t16c.as_deref(),
                            threads,
                            None,
                            Some(cmask),
                            true,
                            false,
                        )?;
                        drop(slab_d);
                        rec_sparse(down_mat, sk_d, &mut cold_sp_total_step, &mut cold_sp_skipped_step);
                        apply_c8db_cold("mlp.down_proj", &act, &mut down_out)?;
                        if let Some(t0) = t_d0 {
                            prof_down_step += t0.elapsed().as_secs_f64();
                        }
                        down_out
                    } else {
                        let slab_g = gate_mat.read_sparse_slab(None, None)?;
                        cold_read_bytes += slab_g.bytes_read;
                        let (mut gelu_gate, _) = gate_mat.eval_sparse_slab(
                            &slab_g,
                            &inp2,
                            seq_len,
                            native_t16c.as_deref(),
                            threads,
                            None,
                            None,
                            true,
                            false,
                        )?;
                        drop(slab_g);
                        apply_c8db_cold("mlp.gate_proj", &inp2, &mut gelu_gate)?;
                        if let Some(t0) = t_g0 {
                            prof_gate_step += t0.elapsed().as_secs_f64();
                        }
                        let t_gl0 = if profile_components { Some(std::time::Instant::now()) } else { None };
                        gelu_tanh_inplace_parallel(&mut gelu_gate, threads);
                        if let Some(t0) = t_gl0 {
                            prof_norm_kv_step += t0.elapsed().as_secs_f64();
                        }
                        let sp_masks = build_gate_masks(&gelu_gate);
                        let rm_opt = sp_masks.as_ref().map(|(r, _, _)| r.as_slice());
                        let cm_opt = sp_masks.as_ref().map(|(_, c, _)| c.as_slice());
                        let t_u0 = if profile_components { Some(std::time::Instant::now()) } else { None };
                        let slab_u = up_mat.read_sparse_slab(rm_opt, None)?;
                        cold_read_bytes += slab_u.bytes_read;
                        let (act, slab_d) = std::thread::scope(|sc| -> Result<(Vec<f32>, run270_rust::t16c::StreamedSparseSlab), String> {
                            let down_io = sc.spawn(|| down_mat.read_sparse_slab(None, cm_opt));
                            let (mut up, sk_u) = up_mat.eval_sparse_slab(
                                &slab_u,
                                &inp2,
                                seq_len,
                                native_t16c.as_deref(),
                                threads,
                                rm_opt,
                                None,
                                true,
                                false,
                            )?;
                            drop(slab_u);
                            if rm_opt.is_some() {
                                rec_sparse(up_mat, sk_u, &mut cold_sp_total_step, &mut cold_sp_skipped_step);
                            }
                            apply_c8db_cold("mlp.up_proj", &inp2, &mut up)?;
                            if let Some(t0) = t_u0 {
                                prof_up_step += t0.elapsed().as_secs_f64();
                            }
                            let act_v = compute_act(
                                sp_masks.as_ref().map(|(_, c, bs)| (c.as_slice(), *bs)),
                                &gelu_gate,
                                &up,
                            );
                            let sd = down_io.join().map_err(|_| "cold down_proj read thread panicked".to_string())??;
                            Ok((act_v, sd))
                        })?;
                        drop(gelu_gate);
                        cold_read_bytes += slab_d.bytes_read;

                        let t_d0 = if profile_components { Some(std::time::Instant::now()) } else { None };
                        let (mut down_out, sk_d) = down_mat.eval_sparse_slab(
                            &slab_d,
                            &act,
                            seq_len,
                            native_t16c.as_deref(),
                            threads,
                            None,
                            cm_opt,
                            true,
                            false,
                        )?;
                        drop(slab_d);
                        if cm_opt.is_some() {
                            rec_sparse(down_mat, sk_d, &mut cold_sp_total_step, &mut cold_sp_skipped_step);
                        }
                        apply_c8db_cold("mlp.down_proj", &act, &mut down_out)?;
                        if let Some(t0) = t_d0 {
                            prof_down_step += t0.elapsed().as_secs_f64();
                        }
                        down_out
                    }
                } else {
                // V6.1 Predictive Cortical Column Routing (when --cortical-router is loaded) skips `(1 - mlp_top_p)`
                // of `gate_proj` DRAM reads too; otherwise preserves exact 1-pass `gate_proj` routing unchanged.
                let mut gelu_gate = if let Some((ref rmask, _, _, _)) = predicted_masks {
                    run_proj_masked(l, "mlp.gate_proj", &inp2, Some(rmask), None, true)?
                } else {
                    run_proj_masked(l, "mlp.gate_proj", &inp2, None, None, true)?
                };
                if let Some(t0) = t_g0 {
                    prof_gate_step += t0.elapsed().as_secs_f64();
                }
                let t_gl0 = if profile_components { Some(std::time::Instant::now()) } else { None };
                if let Some((_, ref cmask, bsize, ref tok_cm_opt)) = predicted_masks {
                    let ff = cfg.ff;
                    for t in 0..seq_len {
                        let base = t * ff;
                        for (b, &active) in cmask.iter().enumerate() {
                            if active != 0 && tok_cm_opt.as_ref().is_none_or(|tm| tm[t][b] != 0) {
                                for x in &mut gelu_gate[base + b * bsize..base + (b + 1) * bsize] {
                                    *x = gelu_tanh_fast_f32(*x);
                                }
                            } else {
                                gelu_gate[base + b * bsize..base + (b + 1) * bsize].fill(0.0);
                            }
                        }
                    }
                } else {
                    gelu_tanh_inplace_parallel(&mut gelu_gate, threads);
                }
                if let Some(t0) = t_gl0 {
                    prof_norm_kv_step += t0.elapsed().as_secs_f64();
                }
                let pred_tok_cmasks = predicted_masks.as_ref().and_then(|(_, _, _, tm)| tm.clone());
                let sparse_masks: Option<(Vec<u8>, Vec<u8>, usize)> = if let Some((rm, cm, bsz, _)) = predicted_masks {
                    Some((rm, cm, bsz))
                } else if mlp_top_p < 0.9999 {
                    t16c_mats.as_ref().and_then(|mats| {
                        let ku1 = format!("model.language_model.layers.{l}.mlp.up_proj.weight");
                        let ku2 = format!("model.layers.{l}.mlp.up_proj.weight");
                        let ku1_lb = format!("{ku1}.lb_u");
                        let kd1 = format!("model.language_model.layers.{l}.mlp.down_proj.weight");
                        let kd2 = format!("model.layers.{l}.mlp.down_proj.weight");
                        let kd1_lb = format!("{kd1}.lb_v");
                        let um = mats.get(&ku1).or_else(|| mats.get(&ku2)).or_else(|| mats.get(&ku1_lb))?;
                        let dm = mats.get(&kd1).or_else(|| mats.get(&kd2)).or_else(|| mats.get(&kd1_lb))?;
                        let ue = um.n_bits & 0x1F;
                        let de = dm.n_bits & 0x1F;
                        if ue == 8 || de == 8 {
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
                let act: Vec<f32> = if let Some((_, ref cmask, bsize)) = sparse_masks {
                    let ff = cfg.ff;
                    let mut out_a = vec![0.0f32; seq_len * ff];
                    for t in 0..seq_len {
                        let base = t * ff;
                        for (b, &active) in cmask.iter().enumerate() {
                            if active != 0 && pred_tok_cmasks.as_ref().is_none_or(|tm| tm[t][b] != 0) {
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
                down
                };

                let t_n2 = if profile_components { Some(std::time::Instant::now()) } else { None };
                let g_pff = get_norm(&format!("{pfx}.post_feedforward_layernorm.weight"))?;
                let ls = scalars[l];
                for t in 0..seq_len {
                    rmsn_raw_into(&down[t * d..(t + 1) * d], Some(g_pff), eps, &mut norm_tmp);
                    let st = &mut states[t * d..(t + 1) * d];
                    for (dst, &src) in st.iter_mut().zip(norm_tmp.iter()) {
                        *dst = (*dst + src) * ls;
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
        self.ssd_bytes_read += read_bytes + cold_read_bytes;
        self.cs_skipped_blocks += cs_skipped_step;
        self.mlp_sparse_skipped_blks += mlp_sp_skipped_step + cold_sp_skipped_step;
        self.mlp_sparse_total_blks += mlp_sp_total_step + cold_sp_total_step;
        self.mod_skipped_layers += mod_skipped_step;
        self.mod_total_layers += mod_total_step;
        self.attn_sparse_skipped_heads += attn_sp_skipped_step;
        self.attn_sparse_total_heads += attn_sp_total_step;
        self.cortical_gate_skipped_blks += cortical_gate_skipped_step;
        self.cortical_gate_total_blks += cortical_gate_total_step;
        self.pov_consensus_sum += pov_consensus_step;
        self.pov_layers_evaluated += pov_layers_step;
        self.pov_clean_layers += pov_clean_step;
        self.pov_caution_layers += pov_caution_step;
        self.pov_rescue_layers += pov_rescue_step;
        if pov_layers_step > 0 {
            // The most conservative layer sets the next speculative draft cap.
            self.pov_draft_cap = pov_draft_cap_step.max(1);
        }
        self.lod_active_districts_sum += lod_active_d_step;
        self.lod_total_districts_sum += lod_total_d_step;
        if profile_components {
            self.prof_attn_s += prof_attn_step;
            self.prof_gate_s += prof_gate_step;
            self.prof_up_s += prof_up_step;
            self.prof_down_s += prof_down_step;
            self.prof_norm_kv_s += prof_norm_kv_step;
            self.prof_router_s += prof_router_step;
            self.prof_steps += 1;
        }
        self.kv_raw_bytes += kv_raw_step;
        self.kv_compressed_bytes += kv_cmp_step;
        Ok(states)
    }

    fn compute_logits(&mut self, hidden: &[f32]) -> Result<Vec<f32>, String> {
        self.compute_logits_batch(hidden, 1)
    }

    fn compute_logits_batch(&mut self, hidden_batch: &[f32], seq_len: usize) -> Result<Vec<f32>, String> {
        self.compute_logits_batch_mode_seeded(hidden_batch, seq_len, true, None)
    }

    fn compute_logits_batch_seeded(
        &mut self,
        hidden_batch: &[f32],
        seq_len: usize,
        seed_row_ids: Option<&[i32]>,
    ) -> Result<Vec<f32>, String> {
        self.compute_logits_batch_mode_seeded(hidden_batch, seq_len, true, seed_row_ids)
    }

    fn compute_logits_batch_mode(
        &mut self,
        hidden_batch: &[f32],
        seq_len: usize,
        allow_lmhead_cs: bool,
    ) -> Result<Vec<f32>, String> {
        self.compute_logits_batch_mode_seeded(hidden_batch, seq_len, allow_lmhead_cs, None)
    }

    fn compute_logits_batch_mode_seeded(
        &mut self,
        hidden_batch: &[f32],
        seq_len: usize,
        allow_lmhead_cs: bool,
        seed_row_ids: Option<&[i32]>,
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
                        let k_keep = if seed_row_ids.is_some() && sieve_frac > 0.0 { 1 } else { 32 };
                        let (y, sk) = mat.project_lmhead_cs_seeded(
                            hidden_batch,
                            seq_len,
                            native.as_deref(),
                            self.threads,
                            Some(norms),
                            k_keep,
                            sieve_frac,
                            seed_row_ids,
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

    /// Normalize raw backbone hidden state `[k, d]` via host `model.language_model.norm.weight`
    /// plus optional `post_cell60` C8DB correction so Assistant inputs match Stage 3B training bit-for-bit.
    fn normalize_backbone_hidden(&self, raw_h: &[f32]) -> Result<Vec<f32>, String> {
        let d = self.cfg.d;
        if raw_h.len() % d != 0 {
            return Err(format!("normalize_backbone_hidden: len {} not multiple of d={d}", raw_h.len()));
        }
        let n_tok = raw_h.len() / d;
        let fn_gain = self.norm("model.language_model.norm.weight")?;
        let mut hidden = Vec::with_capacity(raw_h.len());
        for t in 0..n_tok {
            let h_t = rmsn_raw(&raw_h[t * d..(t + 1) * d], Some(fn_gain), self.cfg.eps);
            hidden.extend_from_slice(&h_t);
        }
        if self.apply_c8d_proj {
            if let Some(ref bank) = self.c8d_bank {
                if let Some(entry) = bank.entries.get("model.language_model.norm.post_cell60") {
                    entry.apply_batch_into(&hidden.clone(), n_tok, &mut hidden, bank.native.as_ref(), self.threads, None)?;
                }
            }
        }
        Ok(hidden)
    }

    /// Whether loaded T16C/checkpoint contains `early_exit_l29_norm.weight` for 30-layer Tier-2 exit.
    #[allow(dead_code)]
    fn has_early_exit_l29(&self) -> bool {
        self.norms.contains_key("model.language_model.early_exit_l29_norm.weight")
            || self.norms.contains_key("model.early_exit_l29_norm.weight")
    }

    /// Project already-normalized `[d]` hidden state via resident 31B `embed_tokens` and return top-`want_k`
    /// softcapped `(token_id, logit)` pairs, applying `tanh` softcap ONLY to the winning `want_k` logits.
    fn score_normalized_lm_head_top_k(
        &self,
        hidden: &[f32],
        want_k: usize,
        sieve_frac: f32,
        ban_tokens: &[u32],
    ) -> Result<(Vec<(u32, f32)>, i64), String> {
        let d = self.cfg.d;
        if hidden.len() != d {
            return Err(format!("score_normalized_lm_head_top_k: hidden len {} != d {d}", hidden.len()));
        }
        let k_req = want_k.max(1);
        let (mut logits, skipped) = match &self.embed {
            Embed31B::T16C { mat, native } => {
                if let Some(norms) = self.embed_tile_norms.as_deref() {
                    let sf = if sieve_frac <= 0.0 { 0.125 } else { sieve_frac };
                    mat.project_lmhead_cs(
                        hidden,
                        1,
                        native.as_deref(),
                        self.threads,
                        Some(norms),
                        k_req.max(2),
                        sf,
                    )?
                } else {
                    (mat.project_batch(hidden, 1, native.as_deref(), self.threads)?, 0)
                }
            }
            Embed31B::Packed(pm) => (pm.matvec(hidden, self.threads)?, 0),
            Embed31B::ResidentBf16(tbl) => (dot_bf16_rows(tbl, d, self.cfg.vocab, hidden, self.threads), 0),
            Embed31B::Streamed => {
                return Err("Assistant Target-Embed Sharing requires resident --t16c or packed/bf16 embed".into());
            }
        };
        for &tid in ban_tokens {
            let u = tid as usize;
            if u < logits.len() {
                logits[u] = -1.0e9;
            }
        }
        // Softcap x -> cap * tanh(x / cap) is strictly monotonic, so top-K order is invariant;
        // skipping masked sieve rows (<= -1e8) and softcapping only the top-K eliminates 262K tanh() calls.
        let mut best: Vec<(u32, f32)> = Vec::with_capacity(k_req);
        for (i, &v) in logits.iter().enumerate() {
            if v <= -1.0e8 {
                continue;
            }
            if best.len() < k_req {
                best.push((i as u32, v));
                best.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            } else if v > best[k_req - 1].1 {
                best[k_req - 1] = (i as u32, v);
                best.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            }
        }
        let h_sq: f64 = hidden.iter().map(|&x| (x as f64) * (x as f64)).sum();
        let h_rms = ((h_sq / (d.max(1) as f64)).sqrt() as f32).max(1e-6);
        let logit_scale = 16.0f32 / h_rms;
        for (_, v) in &mut best {
            *v *= logit_scale;
        }
        Ok((best, skipped))
    }

    /// Zero-RAM Target-Embed Sharing head for `Assistant-V10` (`has_lm_head == false && byte_head == None`).
    #[allow(dead_code)]
    fn score_shared_lm_head_top_k(
        &self,
        next_backbone_h: &[f32],
        want_k: usize,
        sieve_frac: f32,
        ban_tokens: &[u32],
    ) -> Result<(Vec<(u32, f32)>, i64), String> {
        let hidden = self.normalize_backbone_hidden(next_backbone_h)?;
        self.score_normalized_lm_head_top_k(&hidden, want_k, sieve_frac, ban_tokens)
    }
}

impl LayerShardBackend for Runtime31B {
    fn shard_info(&self) -> (usize, usize, usize) {
        (self.local_layer_range.0, self.local_layer_range.1, self.cfg.d)
    }

    fn forward_hidden_seq(&mut self, hidden_in: &[f32], seq_len: usize, start_pos: usize) -> Result<Vec<f32>, String> {
        let (ls, le) = self.local_layer_range;
        self.forward_batch_with_embeds_layers_raw(
            hidden_in.to_vec(),
            seq_len,
            start_pos,
            false,
            None,
            None,
            ls,
            le,
        )
    }

    fn forward_hidden_tree(
        &mut self,
        hidden_in: &[f32],
        seq_len: usize,
        start_pos: usize,
        tree_depths: &[usize],
        ancestor_masks: &[u32],
    ) -> Result<Vec<f32>, String> {
        let (ls, le) = self.local_layer_range;
        self.forward_batch_with_embeds_layers_raw(
            hidden_in.to_vec(),
            seq_len,
            start_pos,
            false,
            None,
            Some((tree_depths, ancestor_masks)),
            ls,
            le,
        )
    }

    fn commit_tree_slots_shard(&mut self, base_kv_len: usize, accepted_slots: &[usize]) -> Result<(), String> {
        let le = self.local_layer_range.1;
        self.commit_tree_slots_upto(base_kv_len, accepted_slots, le, None)
    }

    fn truncate_kv_shard(&mut self, new_len: usize) -> Result<(), String> {
        self.truncate_kv(new_len)
    }
}

fn load_runtime_31b(
    model_path: &Path,
    cfg: Gemma4Cfg,
    embed_mode: EmbedMode31B,
    pin_layers: usize,
    kv_quant: KvQuantMode,
    kv_lod_caps: (usize, usize, usize),
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
    let glyph_kv = init_glyph_kv(&cfg, kv_quant, kv_lod_caps)?;
        let n_layers_init = cfg.n_layers;
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
            lod_shared_mats: None,
            lod_district_mats: Vec::new(),
            district_router: None,
            player_pov_6sense: false,
            pov_target_dynamic: true,
            pov_base_draft_k: 4,
            pov_draft_cap: 4,
            pov_consensus_sum: 0.0,
            pov_layers_evaluated: 0,
            pov_clean_layers: 0,
            pov_caution_layers: 0,
            pov_rescue_layers: 0,
            lod_active_districts_sum: 0,
            lod_total_districts_sum: 0,
            native_t16c: None,
            c8d_bank: None,
            apply_c8d_proj: true,
            cs_skip_thresh: 0.0,
            cs_down_norms: Vec::new(),
            cs_skipped_blocks: 0,
            mlp_top_p: 1.0,
            shared_mlp_blocks: 0,
            mlp_sparse_skipped_blks: 0,
            mlp_sparse_total_blks: 0,
            cortical_router: None,
            head_router: None,
            attn_head_top_p: 1.0,
            attn_shared_heads: 4,
            mod_thresh: 0.0,
            mod_zero_kv: false,
            mod_skipped_layers: 0,
            mod_total_layers: 0,
            attn_sparse_skipped_heads: 0,
            attn_sparse_total_heads: 0,
            cortical_gate_skipped_blks: 0,
            cortical_gate_total_blks: 0,
            profile_components: false,
            prof_attn_s: 0.0,
            prof_gate_s: 0.0,
            prof_up_s: 0.0,
            prof_down_s: 0.0,
            prof_norm_kv_s: 0.0,
            prof_lmhead_s: 0.0,
            prof_router_s: 0.0,
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
            ephemeral_tree_kv: Vec::new(),
            pending_l29_states: Vec::new(),
            vision: None,
            local_layer_range: (0, n_layers_init),
            cluster_workers: Vec::new(),
            bypass_layers: vec![false; n_layers_init],
            cca_layer_policies: None,
            cca_sparse_active: false,
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
    kv_lod_caps: (usize, usize, usize),
    hybrid_qat_layers: &[usize],
    t16c_policy: &T16CLoadPolicy,
    threads: usize,
) -> Result<(Runtime31B, usize, usize), String> {
    let mut t16c = run270_rust::t16c::T16CModel::load_with_policy(t16c_path, native_lib, disable_native, t16c_policy)?;
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
        let nb = mat.resident_bytes();
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
    let vision_bytes = vision.as_ref().map_or(0usize, |vw| vw.total_bytes());
    let vision_resident_bytes = if t16c_policy.defer_vision_fault { 0usize } else { vision_bytes };
    let body_bytes: usize = t16c.mats.values().map(|m| m.resident_bytes()).sum();
    let streamed_cold_bytes: usize = t16c
        .mats
        .values()
        .map(|m| m.total_bytes().saturating_sub(m.resident_bytes()))
        .sum();
    eprintln!(
        "[run31b] weight_source=T16C native_ffi={} mats={} norms={} body_bytes={} streamed_cold_bytes={} vision_bytes={} vision_deferred={} norm_bytes={} embed_bytes={} hybrid_qat={:?} ({hybrid_bytes} B) kv_quant={kv_quant:?}",
        native_t16c.as_ref().map_or("disabled", |n| n.path.as_str()),
        t16c.mats.len(),
        t16c.norms.len(),
        body_bytes,
        streamed_cold_bytes,
        vision_bytes,
        t16c_policy.defer_vision_fault,
        norm_bytes,
        embed_bytes,
        hybrid_qat_layers
    );

    let loc_range = t16c_policy.active_layer_range.unwrap_or((0, cfg.n_layers));
    let n_layers_init = cfg.n_layers;
    let kv = (0..cfg.n_layers).map(|_| LayerKv { k: Vec::new(), v: Vec::new() }).collect();
    let glyph_kv = init_glyph_kv(&cfg, kv_quant, kv_lod_caps)?;
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
            lod_shared_mats: None,
            lod_district_mats: Vec::new(),
            district_router: None,
            player_pov_6sense: false,
            pov_target_dynamic: true,
            pov_base_draft_k: 4,
            pov_draft_cap: 4,
            pov_consensus_sum: 0.0,
            pov_layers_evaluated: 0,
            pov_clean_layers: 0,
            pov_caution_layers: 0,
            pov_rescue_layers: 0,
            lod_active_districts_sum: 0,
            lod_total_districts_sum: 0,
            native_t16c,
            c8d_bank: None,
            apply_c8d_proj: true,
            cs_skip_thresh: 0.0,
            cs_down_norms: Vec::new(),
            cs_skipped_blocks: 0,
            mlp_top_p: 1.0,
            shared_mlp_blocks: 0,
            mlp_sparse_skipped_blks: 0,
            mlp_sparse_total_blks: 0,
            cortical_router: None,
            head_router: None,
            attn_head_top_p: 1.0,
            attn_shared_heads: 4,
            mod_thresh: 0.0,
            mod_zero_kv: false,
            mod_skipped_layers: 0,
            mod_total_layers: 0,
            attn_sparse_skipped_heads: 0,
            attn_sparse_total_heads: 0,
            cortical_gate_skipped_blks: 0,
            cortical_gate_total_blks: 0,
            profile_components: false,
            prof_attn_s: 0.0,
            prof_gate_s: 0.0,
            prof_up_s: 0.0,
            prof_down_s: 0.0,
            prof_norm_kv_s: 0.0,
            prof_lmhead_s: 0.0,
            prof_router_s: 0.0,
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
            ephemeral_tree_kv: Vec::new(),
            pending_l29_states: Vec::new(),
            vision,
            local_layer_range: loc_range,
            cluster_workers: Vec::new(),
            bypass_layers: vec![false; n_layers_init],
            cca_layer_policies: None,
            cca_sparse_active: false,
            threads,
            ssd_bytes_read: 0,
        },
        body_bytes + vision_resident_bytes + norm_bytes + hybrid_bytes,
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
    kv_lod_caps: (usize, usize, usize),
    seq_len: usize,
    bench_k_eq_v: bool,
    threads: usize,
) -> Result<(), String> {
    let eff_mode = if kv_quant.is_glyph() { kv_quant } else { KvQuantMode::Glyph };
    let (sink_cap, tail_cap, mid_cap) = if kv_quant.is_glyph() {
        kv_lod_caps
    } else {
        (4usize, 32usize, 0usize)
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
        "[run31b] --bench-128k-kv mode={eff_mode:?} seq_len={seq_len} k_eq_v={bench_k_eq_v} layers={} lod=({sink_cap},{tail_cap},{mid_cap}) threads={threads} native_glyph_c={has_native_glyph}",
        cfg.n_layers
    );

    // 1. Live accuracy verification against exact FP32 GQA (`attn_decode_range`) on a 512-token global layer
    //    using realistic post-qk_norm activation magnitudes (unit-variance QK dot products with sparse channel outliers).
    let ref_seq = 512usize.min(seq_len.max(64));
    let mut ref_layer = GlyphKvLayer::new_with_lod(
        cfg.n_heads,
        cfg.n_global_kv,
        cfg.global_hd,
        bench_k_eq_v,
        None,
        sink_cap,
        tail_cap,
        mid_cap,
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
        let eff_mid = mid_cap.min(target_toks.saturating_sub(eff_sink + eff_tail));
        let n_glyphs = target_toks.saturating_sub(eff_sink + eff_mid + eff_tail);

        let mut layer = GlyphKvLayer::new_with_lod(
            cfg.n_heads,
            n_kv,
            hd,
            k_eq_v,
            sw,
            eff_sink,
            eff_tail,
            eff_mid,
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
        for _ in 0..(eff_mid + eff_tail) {
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
            dst.mid_tokens += s.mid_tokens;
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
        "KV128K_BENCH mode={eff_mode:?} seq_len={seq_len} k_eq_v={bench_k_eq_v} layers={} global_layers={n_global} sliding_layers={n_sliding} native_glyph_c={has_native_glyph} fill_s={fill_s:.2} ref512_cos={ref_cos:.5} mid_tokens={} global_resident_mb={:.2} global_naive_bf16_mb={:.2} global_shrink_bf16={:.2}x total_resident_mb={:.2} total_naive_bf16_mb={:.2} total_shrink_bf16={:.2}x total_shrink_fp32={:.2}x global_10l_ms={global_ms:.2} sliding_50l_ms={sliding_ms:.2} total_60l_ms={total_attn_ms:.2} checksum={checksum:.5} vm_hwm_kb={vm_hwm:?} vm_rss_kb={vm_rss:?}",
        cfg.n_layers,
        tot_all.mid_tokens,
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
    let kv_sinks_ov: Option<usize> = match get("--kv-sinks") {
        Some(s) => Some(s.parse().map_err(|_| format!("invalid --kv-sinks '{s}'"))?),
        None => None,
    };
    let kv_near_ov: Option<usize> = match get("--kv-near") {
        Some(s) => Some(s.parse().map_err(|_| format!("invalid --kv-near '{s}'"))?),
        None => None,
    };
    let kv_mid_ov: Option<usize> = match get("--kv-mid") {
        Some(s) => Some(s.parse().map_err(|_| format!("invalid --kv-mid '{s}'"))?),
        None => None,
    };
    let kv_lod_caps = resolve_kv_lod_caps(kv_quant, kv_sinks_ov, kv_near_ov, kv_mid_ov);
    let keep_vision_resident = args.iter().any(|a| a == "--keep-vision-resident");
    let cold_mlp_arg = get("--cold-mlp-layers");
    if bench_128k_kv {
        let cfgp = get("--config").ok_or("missing --config")?;
        let cfg = load_gemma4_cfg(Path::new(&cfgp))?;
        return run_bench_128k_kv(&cfg, so_arg.as_deref(), disable_native, kv_quant, kv_lod_caps, bench_128k_seq, bench_128k_k_eq_v, threads);
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
    let spec_byte_head_arg = get("--spec-byte-head");
    let cortical_cascade_tau: Option<f32> = match get("--cortical-cascade") {
        Some(s) => {
            let v: f32 = s.parse().map_err(|_| format!("invalid --cortical-cascade '{s}'"))?;
            if !v.is_finite() || v < 0.0 {
                return Err(format!("--cortical-cascade tau must be >= 0.0, got {v}"));
            }
            Some(v)
        }
        None => None,
    };
    let cascade_mlp_top_p: f32 = match get("--cascade-mlp-top-p") {
        Some(s) => s.parse().map_err(|_| format!("invalid --cascade-mlp-top-p '{s}'"))?,
        None => 0.143,
    };
    let cascade_head_top_p: f32 = match get("--cascade-head-top-p") {
        Some(s) => s.parse().map_err(|_| format!("invalid --cascade-head-top-p '{s}'"))?,
        None => 0.25,
    };
    let cascade_mod_thresh: f32 = match get("--cascade-mod-thresh") {
        Some(s) => parse_mod_thresh(&s)?,
        None => 0.0,
    };
    let cascade_mod_zero_kv = args.iter().any(|a| a == "--cascade-mod-zero-kv") || cortical_cascade_tau.is_some();
    let cascade_tier1_tau_arg: Option<f32> = match get("--cascade-tier1-tau") {
        Some(s) => {
            let v: f32 = s.parse().map_err(|_| format!("invalid --cascade-tier1-tau '{s}'"))?;
            if !v.is_finite() || v < 0.0 {
                return Err(format!("--cascade-tier1-tau must be >= 0.0, got {v}"));
            }
            Some(v)
        }
        None => None,
    };
    let cascade_max_unverified: usize = match get("--cascade-max-unverified") {
        Some(s) => s.parse().map_err(|_| format!("invalid --cascade-max-unverified '{s}'"))?,
        None => 2,
    };
    let cascade_exit_layer_arg: Option<usize> = match get("--cascade-exit-layer") {
        Some(s) => {
            let v: usize = s.parse().map_err(|_| format!("invalid --cascade-exit-layer '{s}'"))?;
            if v == 0 {
                return Err("--cascade-exit-layer must be >= 1".into());
            }
            Some(v)
        }
        None => None,
    };
    let general_alu = args.iter().any(|a| a == "--general-alu");
    let zplane_compact = args.iter().any(|a| a == "--zplane-compact");
    let fast_cores = args.iter().any(|a| a == "--fast-cores");
    let layers_range_arg = get("--layers");
    let cluster_worker_bind = get("--cluster-worker");
    let cluster_workers_arg = get("--cluster-workers");
    let mut cluster_psk = get("--cluster-psk").unwrap_or_else(|| "lu-v11-default-psk".into());
    let cluster_max_sessions: Option<usize> = match get("--cluster-max-sessions") {
        Some(s) => Some(s.parse().map_err(|_| format!("invalid --cluster-max-sessions '{s}'"))?),
        None => None,
    };
    let zk_lora_chirp_in = get("--zk-lora-chirp-in");
    let zk_lora_chirp_out = args.iter().any(|a| a == "--zk-lora-chirp-out");
    let bypass_layers_arg = get("--bypass-layers");
    let cca_plan_arg = get("--cca-plan");
    let cca_va_plan_arg = get("--cca-va-plan");
    let cca_depth_probe_out = get("--cca-depth-probe");
    let cca_probe_layers_arg = get("--cca-probe-layers");
    let va_mirror_exp_out = get("--va-mirror-exp");
    let va_meaning_cache_depth: Option<usize> = match get("--va-meaning-cache") {
        Some(s) => Some(s.parse().map_err(|_| format!("invalid --va-meaning-cache '{s}'"))?),
        None => None,
    };
    let spec_tree_nodes: Option<usize> = match get("--spec-tree") {
        Some(s) => {
            let n: usize = s.parse().map_err(|_| format!("invalid --spec-tree '{s}'"))?;
            if !(2..=24).contains(&n) {
                return Err(format!("--spec-tree must be in 2..=24, got {n}"));
            }
            Some(n)
        }
        None => if cortical_cascade_tau.is_some() { Some(12) } else { None },
    };
    let spec_tree_depth: usize = match get("--spec-tree-depth") {
        Some(s) => {
            let d: usize = s.parse().map_err(|_| format!("invalid --spec-tree-depth '{s}'"))?;
            if !(1..=16).contains(&d) {
                return Err(format!("--spec-tree-depth must be in 1..=16, got {d}"));
            }
            d
        }
        None => if cortical_cascade_tau.is_some() { 10 } else { 5 },
    };
    let pin_lod_districts: usize = match get("--pin-lod-districts") {
        Some(s) => s.parse().map_err(|_| format!("invalid --pin-lod-districts '{s}'"))?,
        None => 0,
    };
    let eval_nll = args.iter().any(|a| a == "--eval-nll");
    let eval_nll_after_turn = args.iter().any(|a| a == "--eval-nll-after-turn");
    let eval_nll_start_pos: Option<usize> = match get("--eval-nll-start-pos") {
        Some(s) => Some(s.parse().map_err(|_| format!("invalid --eval-nll-start-pos '{s}'"))?),
        None => None,
    };
    let dump_seq_logits = get("--dump-seq-logits");
    let eval_teacher_logits = get("--eval-teacher-logits");
    let assistant_step_pos = args.iter().any(|a| a == "--assistant-step-pos") || spec_tree_nodes.is_some();
    let assistant_min_margin: f32 = match get("--assistant-min-margin") {
        Some(s) => s.parse().map_err(|_| format!("invalid --assistant-min-margin '{s}'"))?,
        None => default_assistant_min_margin(spec_assistant_arg.is_some(), spec_byte_head_arg.is_some()),
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
    let shared_mlp_blocks: usize = match get("--shared-mlp-blocks") {
        Some(s) => s.parse().map_err(|_| format!("invalid --shared-mlp-blocks '{s}'"))?,
        None => 0,
    };
    let cortical_router_arg = get("--cortical-router");
    let head_router_arg = get("--head-router");
    let attn_head_top_p: f32 = match get("--attn-head-top-p") {
        Some(s) => s.parse().map_err(|_| format!("invalid --attn-head-top-p '{s}'"))?,
        None => 1.0,
    };
    if !attn_head_top_p.is_finite() || attn_head_top_p <= 0.0 || attn_head_top_p > 1.0 {
        return Err(format!("--attn-head-top-p must be finite in (0.0, 1.0], got {attn_head_top_p}"));
    }
    let attn_shared_heads: usize = match get("--attn-shared-heads") {
        Some(s) => s.parse().map_err(|_| format!("invalid --attn-shared-heads '{s}'"))?,
        None => 4,
    };
    let mod_zero_kv = args.iter().any(|a| a == "--mod-zero-kv");
    let lod_shared_arg = get("--lod-shared");
    let lod_district_dir_arg = get("--lod-district-dir");
    let district_router_arg = get("--district-router");
    let player_pov_6sense = args.iter().any(|a| a == "--player-pov-6sense");
    let mod_thresh: f32 = match get("--mod-thresh") {
        Some(s) => parse_mod_thresh(&s)?,
        None => 0.0,
    };
    let andvcnt_bits: Option<i32> = match get("--andvcnt-bits") {
        Some(s) => {
            let v: i32 = s.parse().map_err(|_| format!("invalid --andvcnt-bits '{s}'"))?;
            if !(4..=6).contains(&v) {
                return Err(format!("--andvcnt-bits must be 4, 5, or 6, got {v}"));
            }
            Some(v)
        }
        None => None,
    };
    let profile_components = args.iter().any(|a| a == "--profile-components");
    let prompt_batch_arg = get("--prompt-batch");
    let lmhead_cs_sieve: Option<f32> = match get("--lmhead-cs") {
        Some(s) => parse_lmhead_cs_mode(&s)?,
        None => Some(0.0),
    };
    if (spec_seed_arg.is_some() || spec_assistant_arg.is_some() || spec_tree_nodes.is_some() || cortical_cascade_tau.is_some()) && !speculative {
        return Err("--spec-seed, --spec-assistant, --spec-tree, and --cortical-cascade require --speculative".into());
    }
    if cortical_cascade_tau.is_some() && cortical_router_arg.is_none() {
        return Err("--cortical-cascade requires --cortical-router for Tier-2 fast verification".into());
    }
    if spec_byte_head_arg.is_some() && (!speculative || spec_assistant_arg.is_none()) {
        return Err("--spec-byte-head requires --speculative and --spec-assistant".into());
    }
    if (dump_seq_logits.is_some() || eval_teacher_logits.is_some()) && !eval_nll {
        return Err("--dump-seq-logits and --eval-teacher-logits require --eval-nll".into());
    }
    if (eval_nll_start_pos.is_some() || eval_nll_after_turn) && !eval_nll {
        return Err("--eval-nll-start-pos and --eval-nll-after-turn require --eval-nll".into());
    }
    if eval_nll_start_pos.is_some() && eval_nll_after_turn {
        return Err("--eval-nll-start-pos and --eval-nll-after-turn are mutually exclusive".into());
    }
    if dump_seq_logits.is_some() && eval_teacher_logits.is_some() {
        return Err("--dump-seq-logits and --eval-teacher-logits are mutually exclusive".into());
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
    let mut prompt_text = get("--prompt-text").unwrap_or_else(|| "Who wrote the play Hamlet?".into());
    if let Some(ref hex_in) = zk_lora_chirp_in {
        let clean: String = hex_in.chars().filter(|c| !c.is_whitespace()).collect();
        if clean.len() % (ZKLORA_MTU * 2) != 0 || clean.is_empty() {
            return Err(format!("--zk-lora-chirp-in hex len {} is not a multiple of 510 (255B frame)", clean.len()));
        }
        let mut raw_bytes = Vec::with_capacity(clean.len() / 2);
        for i in (0..clean.len()).step_by(2) {
            raw_bytes.push(u8::from_str_radix(&clean[i..i + 2], 16).map_err(|e| format!("bad hex in --zk-lora-chirp-in: {e}"))?);
        }
        let mut decoded_chirps = Vec::new();
        for ch in raw_bytes.chunks_exact(ZKLORA_MTU) {
            let mut arr = [0u8; ZKLORA_MTU];
            arr.copy_from_slice(ch);
            let dc = unpack_and_verify_zk_lora_chirp(&arr, None, None)?;
            println!(
                "ZKLORA_INGRESS seq={} frag={}/{} coords_6d={:?} nullifier={:02x}{:02x}{:02x}{:02x}",
                dc.seq_id, dc.frag_idx + 1, dc.frag_total, dc.coords_6d,
                dc.nullifier_hash[0], dc.nullifier_hash[1], dc.nullifier_hash[2], dc.nullifier_hash[3]
            );
            decoded_chirps.push(dc);
        }
        prompt_text = reassemble_zk_lora_message(&decoded_chirps)?;
    }
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
    let cold_mlp_layers = match cold_mlp_arg.as_deref() {
        Some(s) => parse_layer_spec(s, cfg.n_layers)?,
        None => HashSet::new(),
    };
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
    let batch_has_images = batch_prompts.iter().any(|l| l.starts_with("IMAGE:"));
    if is_multi_prompt && (dump_seq_logits.is_some() || eval_teacher_logits.is_some()) {
        return Err("--dump-seq-logits and --eval-teacher-logits require a single prompt sequence".into());
    }
    let active_layer_range: Option<(usize, usize)> = if let Some(ref lr_s) = layers_range_arg {
        let (a_s, b_s) = lr_s
            .split_once("..")
            .or_else(|| lr_s.split_once('-'))
            .ok_or_else(|| format!("invalid --layers '{lr_s}': expected L_start..L_end"))?;
        let ls: usize = a_s.trim().parse().map_err(|_| format!("invalid --layers start '{a_s}'"))?;
        let le: usize = b_s.trim().parse().map_err(|_| format!("invalid --layers end '{b_s}'"))?;
        if ls >= le || le > cfg.n_layers {
            return Err(format!("invalid --layers {ls}..{le} for n_layers={}", cfg.n_layers));
        }
        Some((ls, le))
    } else {
        None
    };
    let threads = if fast_cores {
        let fc = detect_fast_cpus();
        let n_fc = fc.len().max(1);
        eprintln!("[run31b] --fast-cores detected {} performance CPUs: {:?}", n_fc, fc);
        threads.min(n_fc).max(1)
    } else {
        threads
    };
    let is_worker_only = cluster_worker_bind.is_some();
    let need_vision_ever = keep_vision_resident
        || image_arg.is_some()
        || optical_self_analyze
        || batch_has_images
        || vision_only;
    // Intra-chip VA plan (`--cca-va-plan`): evidence-parked layers stay VA/file-addressable but are
    // never faulted into DRAM and are always residual-bypassed (prefill, verify, and dense rescue alike).
    let cca_va_plan: Option<CcaClusterPlan> = match cca_va_plan_arg.as_deref() {
        Some(p) => {
            let plan = CcaClusterPlan::load(Path::new(p))?;
            if plan.num_layers != cfg.n_layers || plan.d_model != cfg.d {
                return Err(format!(
                    "--cca-va-plan {p} dims (L={} d={}) mismatch model (L={} d={})",
                    plan.num_layers, plan.d_model, cfg.n_layers, cfg.d
                ));
            }
            for &li in &plan.tier2_bypass_layers {
                if cfg.layer_full[li] {
                    return Err(format!("--cca-va-plan refuses to park global full-attention layer {li}"));
                }
            }
            Some(plan)
        }
        None => None,
    };
    let parked_layers: HashSet<usize> = cca_va_plan
        .as_ref()
        .map(|p| p.tier2_bypass_layers.iter().copied().collect())
        .unwrap_or_default();
    let t16c_policy = T16CLoadPolicy {
        defer_vision_fault: !keep_vision_resident,
        cold_mlp_layers,
        active_layer_range,
        skip_embed_tokens: is_worker_only,
        parked_layers,
    };
    eprintln!(
        "[run31b] cfg d={} layers={} ff={} heads={}/{} global_kv={} hd={}/{} window={} softcap={:?} kv_quant={kv_quant:?} kv_lod={kv_lod_caps:?} keep_vis={keep_vision_resident} cold_mlps={}",
        cfg.d, cfg.n_layers, cfg.ff, cfg.n_heads, cfg.n_kv, cfg.n_global_kv, cfg.hd, cfg.global_hd, cfg.window, cfg.final_softcap, t16c_policy.cold_mlp_layers.len()
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
            kv_lod_caps,
            &hybrid_qat_layers,
            &t16c_policy,
            threads,
        )?
    } else {
        load_runtime_31b(Path::new(model.as_deref().unwrap()), cfg, embed_mode, pin_layers, kv_quant, kv_lod_caps, threads)?
    };
    // Evict Vision metadata & pages BEFORE allocating Routers + Assistant on text-only runs so lifetime VmHWM stays <= 6.50 GiB!
    if !need_vision_ever {
        if let Some(vw_early) = rt.vision.take() {
            let ev = vw_early.evict_pages();
            drop(vw_early);
            if ev > 0 {
                eprintln!("[run31b] pre-router text-only vision evicted {ev} B (vm={:?})", proc_status_kb());
            }
        }
    }
    // Tag each layer matrix handle's unused ARM64 TBI top byte with its evidence tier, force-bypass
    // parked layers, and prefault measured Cognitive-Core layers.
    if let Some(ref plan) = cca_va_plan {
        let mut parked_va_bytes = 0usize;
        let mut core_touched_bytes = 0usize;
        let mut tagged_handles_xor = 0u64;
        let mut tagged_handles = 0usize;
        if let Some(ref mats) = rt.t16c_mats {
            for (mname, mat) in mats {
                if let Some(li) = run270_rust::t16c::parse_any_layer_idx(mname) {
                    if let Some(pol) = plan.layer_policies.get(li) {
                        let tag = pol.intra_chip_tbi_tag();
                        let raw_va = mat as *const run270_rust::t16c::T16CMat as usize as u64;
                        let tbi_va = tag.tag_va(raw_va);
                        let (dec_tag, dec_va) = run270_rust::cca_bridge::TbiCognitiveTag::from_tagged_va(tbi_va);
                        if dec_va != raw_va || dec_tag != tag {
                            return Err(format!("TBI tag roundtrip failed on {mname}"));
                        }
                        tagged_handles_xor ^= tbi_va;
                        tagged_handles += 1;
                        if pol.tier2_bypass {
                            parked_va_bytes += mat.total_bytes().saturating_sub(mat.resident_bytes());
                        } else if pol.selected_cognitive_core {
                            core_touched_bytes += mat.advise_willneed_and_touch();
                        }
                    }
                }
            }
        }
        for &li in &plan.tier2_bypass_layers {
            rt.bypass_layers[li] = true;
        }
        println!(
            "CCA_VA_REALLOC tbi_hw={} tagged_handles={tagged_handles} parked_layers={:?} parked_va_bytes={parked_va_bytes} core_layers={:?} core_touched_bytes={core_touched_bytes} tbi_xor=0x{tagged_handles_xor:016x} vm={:?}",
            cfg!(all(target_arch = "aarch64", target_os = "linux")),
            plan.tier2_bypass_layers,
            plan.selected_cognitive_core_layers,
            proc_status_kb()
        );
    }
    if zplane_compact {
        if let Some(ref mut mats) = rt.t16c_mats {
            let zp_saved: usize = mats.values_mut().map(|m| m.compact_zplane()).sum();
            if zp_saved > 0 {
                eprintln!("[run31b] --zplane-compact saved {:.2} MiB on Owned backbone tiles", zp_saved as f64 / (1024.0 * 1024.0));
            }
        }
    }
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
    rt.shared_mlp_blocks = shared_mlp_blocks;
    if let Some(ref crp) = cortical_router_arg {
        if rt.t16c_mats.is_none() {
            return Err("--cortical-router requires --t16c resident quantized runtime".into());
        }
        let cr = CorticalRouterBank::load(Path::new(crp))?;
        if cr.n_layers != rt.cfg.n_layers || cr.d_model != rt.cfg.d || cr.ff != rt.cfg.ff {
            return Err(format!(
                "cortical router {} dims (L={} d={} ff={}) incompatible with model (L={} d={} ff={})",
                crp, cr.n_layers, cr.d_model, cr.ff, rt.cfg.n_layers, rt.cfg.d, rt.cfg.ff
            ));
        }
        eprintln!(
            "[run31b] loaded CorticalRouter {}: layers={} rank={} bsize={} blocks={} mod_gate={} ({:.2} MB)",
            crp,
            cr.n_layers,
            cr.rank,
            cr.block_size,
            cr.n_blocks,
            cr.has_mod_gate(),
            cr.byte_len() as f64 / (1024.0 * 1024.0)
        );
        rt.cortical_router = Some(cr);
    }
    rt.attn_head_top_p = attn_head_top_p;
    rt.attn_shared_heads = attn_shared_heads;
    rt.mod_zero_kv = mod_zero_kv;
    if let Some(ref hrp) = head_router_arg {
        if rt.t16c_mats.is_none() {
            return Err("--head-router requires --t16c resident quantized runtime".into());
        }
        let hr = CorticalRouterBank::load(Path::new(hrp))?;
        if hr.n_layers != rt.cfg.n_layers || hr.d_model != rt.cfg.d || hr.n_blocks != rt.cfg.n_heads {
            return Err(format!(
                "head router {} dims (L={} d={} heads={}) incompatible with model (L={} d={} heads={})",
                hrp, hr.n_layers, hr.d_model, hr.n_blocks, rt.cfg.n_layers, rt.cfg.d, rt.cfg.n_heads
            ));
        }
        eprintln!(
            "[run31b] loaded HeadRouter {}: layers={} rank={} heads={} top_p={:.3} shared={} ({:.2} MB)",
            hrp,
            hr.n_layers,
            hr.rank,
            hr.n_blocks,
            attn_head_top_p,
            attn_shared_heads,
            hr.byte_len() as f64 / (1024.0 * 1024.0)
        );
        rt.head_router = Some(hr);
    }
    if let Some(ref lsh_p) = lod_shared_arg {
        let sh_m = run270_rust::t16c::T16CModel::load(Path::new(lsh_p), None, true)?;
        eprintln!(
            "[run31b] loaded GTA-LOD Zone 0 Shared Core {}: mats={} ({:.2} MiB pinned)",
            lsh_p,
            sh_m.mats.len(),
            sh_m.total_mat_bytes as f64 / (1024.0 * 1024.0)
        );
        rt.lod_shared_mats = Some(sh_m.mats);
    }
    if let Some(ref ldd_p) = lod_district_dir_arg {
        let base_p = Path::new(ldd_p);
        let cold_all = T16CLoadPolicy {
            defer_vision_fault: true,
            cold_mlp_layers: (0..rt.cfg.n_layers).collect(),
            ..Default::default()
        };
        let mut d_shards = Vec::new();
        for di in 0..16 {
            let cand1 = base_p.join(format!("gemma4_31b_v8_0_lod_d{di}.t16c"));
            let cand2 = base_p.join(format!("gemma4_31b_v7_5_lod_d{di}.t16c"));
            let cand3 = Path::new(&format!("{ldd_p}_d{di}.t16c")).to_path_buf();
            let chosen = [cand1, cand2, cand3].into_iter().find(|p| p.is_file());
            if let Some(cp) = chosen {
                let dm = if di < pin_lod_districts {
                    run270_rust::t16c::T16CModel::load(&cp, None, true)?
                } else {
                    run270_rust::t16c::T16CModel::load_with_policy(&cp, None, true, &cold_all)?
                };
                d_shards.push(dm.mats);
            } else if di == 0 {
                return Err(format!("--lod-district-dir {ldd_p}: could not find lod_d0.t16c shard"));
            } else {
                break;
            }
        }
        eprintln!(
            "[run31b] mapped {} GTA-LOD Spatial District shards from {} (on-demand COLBLOCK streaming)",
            d_shards.len(),
            ldd_p
        );
        rt.lod_district_mats = d_shards;
    }
    if let Some(ref drp) = district_router_arg {
        let dr = DistrictRouterBank::load(Path::new(drp))?;
        if dr.n_layers != rt.cfg.n_layers || dr.d_model != rt.cfg.d || dr.ff != rt.cfg.ff {
            return Err(format!(
                "district router {} dims (L={} d={} ff={}) incompatible with model (L={} d={} ff={})",
                drp, dr.n_layers, dr.d_model, dr.ff, rt.cfg.n_layers, rt.cfg.d, rt.cfg.ff
            ));
        }
        eprintln!(
            "[run31b] loaded DistrictRouter {}: layers={} districts={} rank={} sense_dim={} base_top_k={} ({:.2} MB)",
            drp,
            dr.n_layers,
            dr.n_districts,
            dr.rank,
            dr.sense_dim,
            dr.base_top_k,
            dr.byte_len() as f64 / (1024.0 * 1024.0)
        );
        if dr.sense_dim > 0 {
            rt.player_pov_6sense = true;
        }
        rt.district_router = Some(dr);
    }
    if player_pov_6sense {
        rt.player_pov_6sense = true;
        eprintln!("[run31b] enabled V8.0 32D 6-Sense Player POV Cortex + Active Consensus Gate");
    }
    if mod_thresh > 0.0 {
        let cr = rt
            .cortical_router
            .as_ref()
            .ok_or("--mod-thresh > 0 requires --cortical-router with MoD gates")?;
        if !cr.has_mod_gate() {
            return Err("--mod-thresh > 0 requires --cortical-router built with CRTR_FLAG_MOD_GATE".into());
        }
        rt.mod_thresh = mod_thresh;
        eprintln!("[run31b] enabled KV-Safe V7.0 MoD residual highway: thresh={mod_thresh:.3}");
    }
    rt.profile_components = profile_components;
    rt.pov_base_draft_k = (batch_k - 1).max(1);
    rt.pov_draft_cap = rt.pov_base_draft_k;
    rt.pov_target_dynamic = !speculative;
    if rt.player_pov_6sense && speculative {
        eprintln!(
            "[run31b] speculative POV safety: target mod/top-p/top-k adaptations frozen; dyn_draft_k=cap base_k={}",
            rt.pov_base_draft_k
        );
    }
    if let (Some(ab), Some(ref nat)) = (andvcnt_bits, rt.native_t16c.as_ref()) {
        if let Some(active) = nat.set_andvcnt_abits(ab) {
            eprintln!("[run31b] configured Row-Major AND+VCNT activation bits = {active}");
        }
    }
    if mlp_top_p < 0.9999 {
        eprintln!("[run31b] enabled Gate-Guided Block-Sparse GeGLU MLP: top_p={mlp_top_p:.3}");
    }
    let lmhead_cs_bytes = rt.enable_lmhead_cs(lmhead_cs_sieve)?;
    if rt.embed_tile_norms.is_none() && (cortical_cascade_tau.is_some() || spec_assistant_arg.is_some()) {
        if let Embed31B::T16C { ref mat, .. } = rt.embed {
            rt.embed_tile_norms = Some(mat.compute_tile_norms_fp16());
        }
    }
    if let Some(sf_cs) = rt.lmhead_cs_sieve {
        eprintln!(
            "[run31b] enabled LM-head CS pruning: sieve_frac={sf_cs:.3} norms_bytes={lmhead_cs_bytes}"
        );
    }
    if let Some(ref bp_spec) = bypass_layers_arg {
        let bp_set = parse_layer_spec(bp_spec, rt.cfg.n_layers)?;
        for &li in &bp_set {
            rt.bypass_layers[li] = true;
        }
        eprintln!("[run31b] enabled explicit --bypass-layers ({} layers): {bp_spec}", bp_set.len());
    }
    if let Some(ref cca_p) = cca_plan_arg {
        let plan = CcaClusterPlan::load(Path::new(cca_p))?;
        if plan.num_layers != rt.cfg.n_layers || plan.d_model != rt.cfg.d {
            return Err(format!(
                "--cca-plan {} dims (L={} d={}) mismatch model (L={} d={})",
                cca_p, plan.num_layers, plan.d_model, rt.cfg.n_layers, rt.cfg.d
            ));
        }
        cluster_psk = plan.bind_psk(&cluster_psk);
        eprintln!(
            "[run31b] loaded Cognitive-Core-Atlas plan {}: core_layers={:?} tier2_bypass={:?} sha256={}",
            cca_p,
            plan.selected_cognitive_core_layers,
            plan.tier2_bypass_layers,
            &plan.cca_manifest_sha256[..16]
        );
        rt.cca_layer_policies = Some(plan.layer_policies);
        rt.cca_sparse_active = cortical_cascade_tau.is_none();
    }
    if let Some(ref probe_out) = cca_depth_probe_out {
        let tk = Tokenizer::load(Path::new(&tkp))?;
        let mut probe_layers: Vec<usize> = if let Some(ref pl_s) = cca_probe_layers_arg {
            let mut v: Vec<usize> = parse_layer_spec(pl_s, rt.cfg.n_layers)?.into_iter().collect();
            v.sort_unstable();
            v
        } else {
            (0..rt.cfg.n_layers).collect()
        };
        probe_layers.dedup();
        if probe_layers.is_empty() {
            return Err("--cca-probe-layers resolved to 0 layers".into());
        }
        let t_probe = std::time::Instant::now();
        let n_p = probe_layers.len();
        let d_h = rt.cfg.d;
        let vocab = rt.cfg.vocab;
        let n_layers_tot = rt.cfg.n_layers;
        let mut probe_idx_by_layer: Vec<Option<usize>> = vec![None; n_layers_tot];
        for (pi, &pl) in probe_layers.iter().enumerate() {
            probe_idx_by_layer[pl] = Some(pi);
        }
        let mut base_records: Vec<String> = Vec::new();
        let mut r_deltas_by_l: Vec<Vec<f64>> = vec![Vec::with_capacity(CCA_PROBE_PAIRS.len()); n_p];
        let mut c_deltas_by_l: Vec<Vec<f64>> = vec![Vec::with_capacity(CCA_PROBE_PAIRS.len()); n_p];
        let model_src = t16c_arg.as_deref().or(model.as_deref()).unwrap_or("gemma-4-31B");

        let write_depth_report = |complete: bool,
                                  elapsed_s: f64,
                                  b_recs: &[String],
                                  r_by_l: &[Vec<f64>],
                                  c_by_l: &[Vec<f64>]|
         -> Result<Vec<usize>, String> {
            let mut selected_layers: Vec<usize> = Vec::new();
            let mut layer_json_rows: Vec<String> = Vec::with_capacity(n_p);
            for (pi, &pl) in probe_layers.iter().enumerate() {
                let r_vec = &r_by_l[pi];
                let c_vec = &c_by_l[pi];
                let mean_r = r_vec.iter().sum::<f64>() / (r_vec.len().max(1) as f64);
                let mean_c = c_vec.iter().sum::<f64>() / (c_vec.len().max(1) as f64);
                let q95_c = cca_quantile(c_vec, 0.95)?;
                let stab = (r_vec.iter().filter(|&&d| d > q95_c).count() as f64) / (r_vec.len().max(1) as f64);
                let sel = mean_r > q95_c && stab >= 0.8;
                if sel {
                    selected_layers.push(pl);
                }
                layer_json_rows.push(format!(
                    "    {{\"layer\": {pl}, \"mean_reasoning_nll_delta\": {mean_r:.6}, \"mean_control_nll_delta\": {mean_c:.6}, \"control_q95_threshold\": {q95_c:.6}, \"reasoning_stability\": {stab:.4}, \"selected\": {sel}}}"
                ));
            }
            let sel_str = selected_layers.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ");
            let doc = format!(
                "{{\n  \"schema_version\": \"cca.depth.v1\",\n  \"complete\": {complete},\n  \"model_id\": \"google/gemma-4-31B-it\",\n  \"model_dir\": {model_src:?},\n  \"dtype\": \"t16c_v1\",\n  \"real_weights\": true,\n  \"probe_elapsed_s\": {elapsed_s:.2},\n  \"total_layers\": {n_layers_tot},\n  \"probed_layer_count\": {n_p},\n  \"selected_count\": {},\n  \"selected_layers\": [{sel_str}],\n  \"baseline\": [\n{}\n  ],\n  \"layers\": [\n{}\n  ]\n}}\n",
                selected_layers.len(),
                b_recs.join(",\n"),
                layer_json_rows.join(",\n")
            );
            let tmp_p = format!("{probe_out}.tmp");
            fs::write(&tmp_p, doc).map_err(|e| format!("write --cca-depth-probe tmp {tmp_p}: {e}"))?;
            fs::rename(&tmp_p, probe_out).map_err(|e| format!("rename {tmp_p} -> {probe_out}: {e}"))?;
            Ok(selected_layers)
        };

        let orig_cca_sp = rt.cca_sparse_active;
        rt.cca_sparse_active = false;
        const MAX_TRAJ_BATCH_TOKS: usize = 384;

        for (pair_idx, pair) in CCA_PROBE_PAIRS.iter().enumerate() {
            for &(dom, q_txt) in &[("reasoning", pair.reasoning), ("control", pair.control)] {
                let chat = format!("<|turn>user\n{q_txt}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>");
                let p_ids = tk.encode(&chat, true);
                let mut a_ids = tk.encode(&format!(" {}", pair.answer), false);
                if a_ids.is_empty() {
                    a_ids = tk.encode(pair.answer, false);
                }
                if p_ids.is_empty() || a_ids.is_empty() {
                    rt.cca_sparse_active = orig_cca_sp;
                    return Err(format!("empty prompt/answer encoding on CCA pair {}", pair.id));
                }
                let start_p = p_ids.len() - 1;
                let cnt = a_ids.len();
                let mut eval_ids = Vec::with_capacity(start_p + cnt);
                eval_ids.extend_from_slice(&p_ids);
                if cnt > 1 {
                    eval_ids.extend_from_slice(&a_ids[..cnt - 1]);
                }
                let seq_len = eval_ids.len();
                let max_trajs_per_chunk = (MAX_TRAJ_BATCH_TOKS / seq_len).max(1);

                let mut base_h = rt.embed_batch(&eval_ids)?;
                let mut active_trajs: Vec<(usize, Vec<f32>)> = Vec::with_capacity(n_p);
                let mut final_raw_by_pi: Vec<Option<Vec<f32>>> = vec![None; n_p];

                for l in 0..n_layers_tot {
                    let spawned: Option<(usize, Vec<f32>)> = if let Some(pi) = probe_idx_by_layer[l] {
                        let ls = rt.scalars[l];
                        let mut skipped_h = base_h.clone();
                        if (ls - 1.0).abs() > 1e-7 {
                            for v in &mut skipped_h {
                                *v *= ls;
                            }
                        }
                        if l + 1 == n_layers_tot {
                            final_raw_by_pi[pi] = Some(skipped_h[start_p * d_h..(start_p + cnt) * d_h].to_vec());
                            None
                        } else {
                            Some((pi, skipped_h))
                        }
                    } else {
                        None
                    };

                    // Pack `[base_h] + active_trajs` into bounded token micro-batches so weights stream once
                    // across trajectories while staying safely inside the <= 6.50 GiB Pi 4 memory envelope.
                    let total_trajs = 1 + active_trajs.len();
                    let mut c0 = 0usize;
                    while c0 < total_trajs {
                        let c1 = (c0 + max_trajs_per_chunk).min(total_trajs);
                        let n_c = c1 - c0;
                        let mut packed = Vec::with_capacity(n_c * seq_len * d_h);
                        for tidx in c0..c1 {
                            if tidx == 0 {
                                packed.extend_from_slice(&base_h);
                            } else {
                                packed.extend_from_slice(&active_trajs[tidx - 1].1);
                            }
                        }
                        let updated = rt.forward_batch_with_embeds_layers_raw_opt(
                            packed,
                            n_c * seq_len,
                            0,
                            false,
                            None,
                            None,
                            l,
                            l + 1,
                            Some(seq_len),
                        )?;
                        let stride_t = seq_len * d_h;
                        for (off_i, tidx) in (c0..c1).enumerate() {
                            let sl = &updated[off_i * stride_t..(off_i + 1) * stride_t];
                            if tidx == 0 {
                                base_h.copy_from_slice(sl);
                            } else {
                                active_trajs[tidx - 1].1.copy_from_slice(sl);
                            }
                        }
                        c0 = c1;
                    }

                    if let Some((pi, skipped_h)) = spawned {
                        active_trajs.push((pi, skipped_h));
                    }
                }

                for (pi, traj_h) in active_trajs {
                    final_raw_by_pi[pi] = Some(traj_h[start_p * d_h..(start_p + cnt) * d_h].to_vec());
                }
                let base_ans_raw = base_h[start_p * d_h..(start_p + cnt) * d_h].to_vec();

                let mut all_normed = Vec::with_capacity((1 + n_p) * cnt * d_h);
                all_normed.extend_from_slice(&rt.finalize_hidden_states(&base_ans_raw, cnt, n_layers_tot, true)?);
                for pi in 0..n_p {
                    let sl = final_raw_by_pi[pi]
                        .as_deref()
                        .ok_or_else(|| format!("missing probed output for probe index {pi}"))?;
                    all_normed.extend_from_slice(&rt.finalize_hidden_states(sl, cnt, n_layers_tot, true)?);
                }

                let score_logits_nll = |logits_slice: &[f32]| -> Result<f64, String> {
                    let mut sum_nll = 0.0f64;
                    for (ci, &gold_id) in a_ids.iter().enumerate() {
                        let row = &logits_slice[ci * vocab..(ci + 1) * vocab];
                        let (nll, _, _, _) = eval_student_logits_stats(row, gold_id)?;
                        sum_nll += nll;
                    }
                    Ok(sum_nll / (cnt as f64))
                };

                let total_evals = 1 + n_p;
                let mut nlls = Vec::with_capacity(total_evals);
                let mut e0 = 0usize;
                while e0 < total_evals {
                    let e1 = (e0 + 16).min(total_evals);
                    let ec = e1 - e0;
                    let sub_h = &all_normed[e0 * cnt * d_h..e1 * cnt * d_h];
                    let logits_b = rt.compute_logits_batch_mode(sub_h, ec * cnt, false)?;
                    for k in 0..ec {
                        let ls = &logits_b[k * cnt * vocab..(k + 1) * cnt * vocab];
                        nlls.push(score_logits_nll(ls)?);
                    }
                    e0 = e1;
                }

                let base_nll = nlls[0];
                base_records.push(format!(
                    "    {{\"id\": \"{}\", \"domain\": \"{}\", \"answer\": \"{}\", \"baseline_nll\": {:.6}}}",
                    pair.id, dom, pair.answer, base_nll
                ));
                for pi in 0..n_p {
                    let delta = nlls[1 + pi] - base_nll;
                    if dom == "reasoning" {
                        r_deltas_by_l[pi].push(delta);
                    } else {
                        c_deltas_by_l[pi].push(delta);
                    }
                }
                eprintln!(
                    "[cca-probe] finished {}:{} ({}/{}) base_nll={base_nll:.4} probed={} elapsed={:.1}s",
                    pair.id,
                    dom,
                    pair_idx + 1,
                    CCA_PROBE_PAIRS.len(),
                    n_p,
                    t_probe.elapsed().as_secs_f64()
                );
            }
            let is_last = pair_idx + 1 == CCA_PROBE_PAIRS.len();
            let sel = write_depth_report(is_last, t_probe.elapsed().as_secs_f64(), &base_records, &r_deltas_by_l, &c_deltas_by_l)?;
            if is_last {
                rt.cca_sparse_active = orig_cca_sp;
                println!(
                    "CCA_DEPTH_PROBE probed={} selected_count={} selected_layers={:?} elapsed_s={:.2} out={probe_out}",
                    n_p,
                    sel.len(),
                    sel,
                    t_probe.elapsed().as_secs_f64()
                );
                print_runtime_telemetry(&rt, pinned_bytes, embed_bytes, t_probe.elapsed().as_secs_f64(), 0.0);
                return Ok(());
            }
        }
        rt.cca_sparse_active = orig_cca_sp;
        return Ok(());
    }
    if let Some(ref mirror_out_p) = va_mirror_exp_out {
        let tk = Tokenizer::load(Path::new(&tkp))?;
        let d_h = rt.cfg.d;
        let vocab = rt.cfg.vocab;
        let n_layers_tot = rt.cfg.n_layers;
        let projector = MirrorProjector::new(d_h);
        let mut arena = VaMirrorArena::reserve_default()?;
        println!(
            "VA_MIRROR_ARENA_INIT va_reserved_gib={:.2} hardware_mmap={} hardware_tbi={} initial_faulted_kb={}",
            arena.va_reserved_gib(),
            arena.hardware_mmap,
            arena.hardware_tbi,
            arena.faulted_ram_kb()
        );

        let warm_passages: [(&str, &str); 3] = [
            ("warm_systems", "Virtual memory page tables translate virtual addresses to physical RAM frames via TLBs."),
            ("warm_code", "def merge_sorted(a, b): return sorted(a + b)"),
            ("warm_math", "Solve the linear system 2a + b = 11 and a - b = 1 step by step."),
        ];
        let unseen_passage = "Hardware caches store recently translated physical frames to cut memory latency.";
        let shifted_passage = "Solve the linear system 2a + b = 13 and a - b = 1 step by step.";

        let mut eval_seq_mirrors = |arena_mut: &mut VaMirrorArena, text: &str, store_in_arena: bool, collect_out_mirrors: bool| -> Result<(Vec<[run270_rust::va_mirror::VaMirrorCode; 4]>, Vec<u32>), String> {
            let chat = format!("<|turn>user\n{text}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>");
            let ids = tk.encode(&chat, true);
            let seq_len = ids.len();
            if seq_len < 4 {
                return Err("VA mirror sequence too short".into());
            }
            rt.truncate_kv(0)?;
            let raw_embeds = rt.embed_batch(&ids)?;
            let (mir_vecs, in_codes) = projector.convert_input_sequence(&raw_embeds, seq_len)?;
            if store_in_arena {
                let last_p = seq_len - 1;
                let mc_last = in_codes[last_p];
                println!(
                    "VA_INPUT_MIRROR_SAMPLE tok={} in_mir0={:.4} mirror_code=0x{:016x} tbi_tag=0x{:02x} c6d=0x{:02x}",
                    ids[last_p],
                    mir_vecs[last_p * d_h],
                    mc_last.code64,
                    mc_last.tbi_tag(0, 1),
                    mc_last.cuneiform_u8
                );
            }

            let h8 = rt.forward_batch_with_embeds_layers_raw(raw_embeds, seq_len, 0, false, None, None, 0, 8)?;
            let h20 = rt.forward_batch_with_embeds_layers_raw(h8.clone(), seq_len, 0, false, None, None, 8, 20)?;
            let h32 = rt.forward_batch_with_embeds_layers_raw(h20.clone(), seq_len, 0, false, None, None, 20, 32)?;
            let h60_raw = rt.forward_batch_with_embeds_layers_raw(h32.clone(), seq_len, 0, false, None, None, 32, n_layers_tot)?;
            let h60_norm = rt.finalize_hidden_states(&h60_raw, seq_len, n_layers_tot, true)?;
            drop(h60_raw);

            let mut codes_per_pos: Vec<[run270_rust::va_mirror::VaMirrorCode; 4]> = Vec::with_capacity(seq_len);
            let mut teacher_toks: Vec<u32> = Vec::with_capacity(seq_len);
            let chunk_t = 16usize;
            let mut p0 = 0usize;
            while p0 < seq_len {
                let cnt = (seq_len - p0).min(chunk_t);
                let l_chunk = rt.compute_logits_batch_mode(&h60_norm[p0 * d_h..(p0 + cnt) * d_h], cnt, false)?;
                for c in 0..cnt {
                    let pos = p0 + c;
                    let row = &l_chunk[c * vocab..(c + 1) * vocab];
                    let pred_tok = argmax_f32(row);
                    teacher_toks.push(pred_tok);

                    let c0 = in_codes[pos];
                    let c8 = projector.encode_state(&h8[pos * d_h..(pos + 1) * d_h])?;
                    let c20 = projector.encode_state(&h20[pos * d_h..(pos + 1) * d_h])?;
                    let c32 = projector.encode_state(&h32[pos * d_h..(pos + 1) * d_h])?;
                    let tup = [c0, c8, c20, c32];
                    if store_in_arena {
                        let out_sig = if collect_out_mirrors {
                            let oe = rt.embed_token_ro(pred_tok)?;
                            projector.encode_state(&oe)?.sig24
                        } else {
                            c0.sig24
                        };
                        for (di, &mc) in tup.iter().enumerate() {
                            arena_mut.insert(di as u8, mc, pred_tok, out_sig);
                        }
                    }
                    codes_per_pos.push(tup);
                }
                p0 += cnt;
            }
            Ok((codes_per_pos, teacher_toks))
        };

        let t_exp = std::time::Instant::now();
        let mut warm_tokens_tot = 0usize;
        for (wname, wtxt) in warm_passages {
            let (_codes, ttoks) = eval_seq_mirrors(&mut arena, wtxt, true, true)?;
            warm_tokens_tot += ttoks.len();
            println!(
                "VA_MIRROR_WARM tag={wname} tokens={} total_writes={} faulted_kb={}",
                ttoks.len(),
                arena.total_writes,
                arena.faulted_ram_kb()
            );
        }

        let probe_split = |arena_ref: &VaMirrorArena, split_name: &str, codes: &[[run270_rust::va_mirror::VaMirrorCode; 4]], ttoks: &[u32]| -> Vec<String> {
            let mut rows_json = Vec::new();
            for (di, &depth_l) in MIRROR_DEPTHS.iter().enumerate() {
                let mut per_h_json = Vec::new();
                for &max_h in &[0u32, 4u32, 8u32, 12u32] {
                    let mut hits = 0usize;
                    let mut agrees = 0usize;
                    let mut sum_dist = 0u64;
                    let n_eval = ttoks.len().saturating_sub(1);
                    for p in 1..ttoks.len() {
                        if let Some(hit) = arena_ref.query(di as u8, codes[p][di], max_h) {
                            hits += 1;
                            sum_dist += hit.hamming_dist as u64;
                            if hit.out_tok == ttoks[p] {
                                agrees += 1;
                            }
                        }
                    }
                    let hit_rate = if n_eval > 0 { hits as f64 / n_eval as f64 } else { 0.0 };
                    let agree_on_hit = if hits > 0 { agrees as f64 / hits as f64 } else { 0.0 };
                    let overall_agree = if n_eval > 0 { agrees as f64 / n_eval as f64 } else { 0.0 };
                    println!(
                        "VA_MIRROR_EVAL split={split_name} depth={depth_l} max_hamming={max_h} n={n_eval} hits={hits} ({:.1}%) agrees={agrees} agree_on_hit={:.1}% overall_exact={:.1}%",
                        hit_rate * 100.0,
                        agree_on_hit * 100.0,
                        overall_agree * 100.0
                    );
                    per_h_json.push(format!(
                        "{{\"max_hamming\": {max_h}, \"n_eval\": {n_eval}, \"hits\": {hits}, \"hit_rate\": {:.4}, \"agrees\": {agrees}, \"agree_on_hit\": {:.4}, \"overall_exact\": {:.4}, \"mean_hamming_on_hit\": {:.2}}}",
                        hit_rate,
                        agree_on_hit,
                        overall_agree,
                        if hits > 0 { sum_dist as f64 / hits as f64 } else { 0.0 }
                    ));
                }
                rows_json.push(format!(
                    "    {{\"depth\": {depth_l}, \"role\": \"{}\", \"thresholds\": [{}]}}",
                    if depth_l == 0 { "input_mirror" } else { "hidden_mirror" },
                    per_h_json.join(", ")
                ));
            }
            rows_json
        };

        let (unseen_codes, unseen_toks) = eval_seq_mirrors(&mut arena, unseen_passage, false, false)?;
        let unseen_rows = probe_split(&arena, "strict_unseen", &unseen_codes, &unseen_toks);

        let (shifted_codes, shifted_toks) = eval_seq_mirrors(&mut arena, shifted_passage, false, false)?;
        let shifted_rows = probe_split(&arena, "shifted_domain", &shifted_codes, &shifted_toks);

        // Phase 3: Physical Autoregressive Speed & Exact Parity Test on Pi 4 (`16` tokens):
        //   Pass A (Dense 60L Warmup Decode): populates Layer-0 Input Mirror & Layer-8 Hidden Mirror
        //   into the 32 GiB VA arena during generation.
        //   Pass B (VA-Mirror Early-Exit L=8 Decode on perturbed question): skips layers `8..60` on
        //   hits while preserving exact 60L KV via deferred batched upper-layer flush on miss.
        let decode_prompt_warm = "Solve the linear system 2a + b = 11 and a - b = 1 step by step.";
        let decode_prompt_test = "Solve the linear system 2a + b = 11 and a - b = 1 step by step.";
        let dec_steps = 16usize;

        let mut run_dense_populate = |p_txt: &str| -> Result<(Vec<u32>, f64, f64), String> {
            let chat = format!("<|turn>user\n{p_txt}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>");
            let ids = tk.encode(&chat, true);
            rt.truncate_kv(0)?;
            let t_pf0 = std::time::Instant::now();
            let pf_embeds = rt.embed_batch(&ids)?;
            let (_mv, pf_in_codes) = projector.convert_input_sequence(&pf_embeds, ids.len())?;
            let pf_h8 = rt.forward_batch_with_embeds_layers_raw(pf_embeds, ids.len(), 0, false, None, None, 0, 8)?;
            let pf_h60 = rt.forward_batch_with_embeds_layers_raw(pf_h8, ids.len(), 0, false, None, None, 8, n_layers_tot)?;
            let pf_norm = rt.finalize_hidden_states(&pf_h60, ids.len(), n_layers_tot, false)?;
            let prefill_s = t_pf0.elapsed().as_secs_f64();
            let mut logits = rt.compute_logits(&pf_norm)?;
            apply_degeneracy_guard(&mut logits, &ids, &[]);
            let mut gen: Vec<u32> = Vec::with_capacity(dec_steps);
            let mut recent_e: Vec<Vec<f32>> = vec![
                rt.embed_token_ro(ids[ids.len().saturating_sub(2)])?,
                rt.embed_token_ro(ids[ids.len().saturating_sub(1)])?,
            ];
            let t_dec0 = std::time::Instant::now();
            let mut cur_tok = argmax_f32(&logits);
            let oe0 = rt.embed_token_ro(cur_tok)?;
            arena.insert(0, *pf_in_codes.last().unwrap(), cur_tok, projector.encode_state(&oe0)?.sig24);
            gen.push(cur_tok);
            let mut pos = ids.len();
            while gen.len() < dec_steps {
                let step_e = rt.embed_batch(&[cur_tok])?;
                recent_e.push(step_e.clone());
                if recent_e.len() > 3 {
                    recent_e.remove(0);
                }
                let mut mix = vec![0.0f32; d_h];
                let e_len = recent_e.len();
                for j in 0..d_h {
                    mix[j] = if e_len >= 3 {
                        0.65 * recent_e[e_len - 1][j] + 0.25 * recent_e[e_len - 2][j] + 0.10 * recent_e[e_len - 3][j]
                    } else {
                        0.72 * recent_e[e_len - 1][j] + 0.28 * recent_e[0][j]
                    };
                }
                let c0 = projector.encode_state(&mix)?;
                let h8 = rt.forward_batch_with_embeds_layers_raw(step_e, 1, pos, false, None, None, 0, 8)?;
                let c8 = projector.encode_state(&h8)?;
                let h60 = rt.forward_batch_with_embeds_layers_raw(h8, 1, pos, false, None, None, 8, n_layers_tot)?;
                let h_norm = rt.finalize_hidden_states(&h60, 1, n_layers_tot, false)?;
                let mut l = rt.compute_logits(&h_norm)?;
                apply_degeneracy_guard(&mut l, &ids, &gen);
                let nxt = argmax_f32(&l);
                let osig = projector.encode_state(&rt.embed_token_ro(nxt)?)?.sig24;
                arena.insert(0, c0, nxt, osig);
                arena.insert(1, c8, nxt, osig);
                gen.push(nxt);
                cur_tok = nxt;
                pos += 1;
            }
            let decode_s = t_dec0.elapsed().as_secs_f64();
            Ok((gen, prefill_s, decode_s))
        };

        let (base_ids, base_pf_s, base_dec_s) = run_dense_populate(decode_prompt_warm)?;
        let base_tps = (base_ids.len() as f64) / base_dec_s.max(1e-6);

        let chat_t = format!("<|turn>user\n{decode_prompt_test}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>");
        let ids_t = tk.encode(&chat_t, true);
        rt.truncate_kv(0)?;
        let pf_e = rt.embed_batch(&ids_t)?;
        let pf_h60 = rt.forward_batch_with_embeds_layers_raw(pf_e, ids_t.len(), 0, false, None, None, 0, n_layers_tot)?;
        let pf_norm = rt.finalize_hidden_states(&pf_h60, ids_t.len(), n_layers_tot, false)?;
        let mut l0 = rt.compute_logits(&pf_norm)?;
        apply_degeneracy_guard(&mut l0, &ids_t, &[]);
        let mut cur_t = argmax_f32(&l0);
        let mut mir_gen: Vec<u32> = vec![cur_t];
        let mut pending_l8_states: Vec<f32> = Vec::new();
        let mut upper_pos = ids_t.len();
        let mut pos_l = ids_t.len();
        let mut mirror_hits = 0usize;
        let mut mirror_misses = 0usize;
        let t_mir0 = std::time::Instant::now();
        while mir_gen.len() < dec_steps {
            let step_e = rt.embed_batch(&[cur_t])?;
            let h8 = rt.forward_batch_with_embeds_layers_raw(step_e, 1, pos_l, false, None, None, 0, 8)?;
            let c8 = projector.encode_state(&h8)?;
            pending_l8_states.extend_from_slice(&h8);
            pos_l += 1;

            let hit_opt = arena.query(1, c8, 4).filter(|_| pending_l8_states.len() / d_h < 4);
            if let Some(hit) = hit_opt {
                mirror_hits += 1;
                cur_t = hit.out_tok;
                mir_gen.push(cur_t);
            } else {
                mirror_misses += 1;
                let k_pend = pending_l8_states.len() / d_h;
                let pend_buf = std::mem::take(&mut pending_l8_states);
                let h60 = rt.forward_batch_with_embeds_layers_raw(
                    pend_buf,
                    k_pend,
                    upper_pos,
                    false,
                    None,
                    None,
                    8,
                    n_layers_tot,
                )?;
                upper_pos += k_pend;
                let last_h60 = &h60[(k_pend - 1) * d_h..k_pend * d_h];
                let h_norm = rt.finalize_hidden_states(last_h60, 1, n_layers_tot, false)?;
                let mut l = rt.compute_logits(&h_norm)?;
                apply_degeneracy_guard(&mut l, &ids_t, &mir_gen);
                cur_t = argmax_f32(&l);
                mir_gen.push(cur_t);
            }
        }
        let mir_dec_s = t_mir0.elapsed().as_secs_f64();
        let mir_tps = (mir_gen.len() as f64) / mir_dec_s.max(1e-6);
        let exact_matches = base_ids.iter().zip(mir_gen.iter()).filter(|(a, b)| a == b).count();

        // Pass C: Live Layer-0 Direct Input-Mirror Decode (skips ALL 60 layers on hits, flushes 0..60 batched on miss!)
        rt.truncate_kv(0)?;
        let pf_e0 = rt.embed_batch(&ids_t)?;
        let pf_h60_0 = rt.forward_batch_with_embeds_layers_raw(pf_e0, ids_t.len(), 0, false, None, None, 0, n_layers_tot)?;
        let pf_norm0 = rt.finalize_hidden_states(&pf_h60_0, ids_t.len(), n_layers_tot, false)?;
        let mut l0_c = rt.compute_logits(&pf_norm0)?;
        apply_degeneracy_guard(&mut l0_c, &ids_t, &[]);
        let mut cur_c0 = argmax_f32(&l0_c);
        let mut mir0_gen: Vec<u32> = vec![cur_c0];
        let mut recent_e0: Vec<Vec<f32>> = vec![
            rt.embed_token_ro(ids_t[ids_t.len().saturating_sub(2)])?,
            rt.embed_token_ro(ids_t[ids_t.len().saturating_sub(1)])?,
        ];
        let mut pending_l0_ids: Vec<u32> = Vec::new();
        let mut pos_l0 = ids_t.len();
        let mut mir0_hits = 0usize;
        let mut mir0_misses = 0usize;
        let t_mir0_start = std::time::Instant::now();
        while mir0_gen.len() < dec_steps {
            let step_e = rt.embed_batch(&[cur_c0])?;
            recent_e0.push(step_e);
            if recent_e0.len() > 3 {
                recent_e0.remove(0);
            }
            let mut mix = vec![0.0f32; d_h];
            let el = recent_e0.len();
            for j in 0..d_h {
                mix[j] = if el >= 3 {
                    0.65 * recent_e0[el - 1][j] + 0.25 * recent_e0[el - 2][j] + 0.10 * recent_e0[el - 3][j]
                } else {
                    0.72 * recent_e0[el - 1][j] + 0.28 * recent_e0[0][j]
                };
            }
            let c0 = projector.encode_state(&mix)?;
            pending_l0_ids.push(cur_c0);
            let hit0 = arena.query(0, c0, 4).filter(|_| pending_l0_ids.len() < 6);
            if let Some(h) = hit0 {
                mir0_hits += 1;
                cur_c0 = h.out_tok;
                mir0_gen.push(cur_c0);
            } else {
                mir0_misses += 1;
                let batch_pend = std::mem::take(&mut pending_l0_ids);
                let bcnt = batch_pend.len();
                let h60 = rt.forward_batch_mode(&batch_pend, pos_l0, false, false)?;
                pos_l0 += bcnt;
                let mut l = rt.compute_logits(&h60)?;
                apply_degeneracy_guard(&mut l, &ids_t, &mir0_gen);
                cur_c0 = argmax_f32(&l);
                mir0_gen.push(cur_c0);
            }
        }
        let mir0_dec_s = t_mir0_start.elapsed().as_secs_f64();
        let mir0_tps = (mir0_gen.len() as f64) / mir0_dec_s.max(1e-6);
        let exact_matches_l0 = base_ids.iter().zip(mir0_gen.iter()).filter(|(a, b)| a == b).count();
        let (vm_hwm_end, vm_rss_end) = proc_status_kb();

        let doc = format!(
            "{{\n\
              \"schema_version\": \"lu.va_mirror_exp.v1\",\n\
              \"va_reserved_gib\": {:.2},\n\
              \"va_faulted_pages_4k\": {},\n\
              \"va_faulted_kb\": {},\n\
              \"hardware_mmap\": {},\n\
              \"hardware_tbi\": {},\n\
              \"warm_tokens\": {warm_tokens_tot},\n\
              \"arena_total_writes\": {},\n\
              \"strict_unseen_split\": [\n{}\n  ],\n\
              \"shifted_domain_split\": [\n{}\n  ],\n\
              \"live_decode_benchmark\": {{\n\
                \"tokens\": {},\n\
                \"dense_60l_prefill_s\": {base_pf_s:.2},\n\
                \"dense_60l_decode_s\": {base_dec_s:.2},\n\
                \"dense_60l_tok_per_s\": {base_tps:.3},\n\
                \"va_mirror_l0_decode_s\": {mir0_dec_s:.2},\n\
                \"va_mirror_l0_tok_per_s\": {mir0_tps:.3},\n\
                \"speedup_l0_vs_dense_60l\": {:.2},\n\
                \"l0_hits\": {mir0_hits},\n\
                \"l0_misses\": {mir0_misses},\n\
                \"l0_exact_token_matches\": {exact_matches_l0},\n\
                \"va_mirror_l8_decode_s\": {mir_dec_s:.2},\n\
                \"va_mirror_l8_tok_per_s\": {mir_tps:.3},\n\
                \"speedup_l8_vs_dense_60l\": {:.2},\n\
                \"l8_hits\": {mirror_hits},\n\
                \"l8_misses\": {mirror_misses},\n\
                \"l8_exact_token_matches\": {exact_matches},\n\
                \"base_text\": {:?},\n\
                \"va_mirror_l0_text\": {:?},\n\
                \"va_mirror_l8_text\": {:?}\n\
              }},\n\
              \"vm_hwm_kb\": {:?},\n\
              \"vm_rss_kb\": {:?},\n\
              \"ssd_read_bytes\": {},\n\
              \"elapsed_s\": {:.2}\n\
            }}\n",
            arena.va_reserved_gib(),
            arena.faulted_pages_4k(),
            arena.faulted_ram_kb(),
            arena.hardware_mmap,
            arena.hardware_tbi,
            arena.total_writes,
            unseen_rows.join(",\n"),
            shifted_rows.join(",\n"),
            base_ids.len(),
            mir0_tps / base_tps.max(1e-6),
            mir_tps / base_tps.max(1e-6),
            tk.decode(&base_ids),
            tk.decode(&mir0_gen),
            tk.decode(&mir_gen),
            vm_hwm_end,
            vm_rss_end,
            rt.ssd_bytes_read,
            t_exp.elapsed().as_secs_f64()
        );
        fs::write(mirror_out_p, &doc).map_err(|e| format!("write --va-mirror-exp {mirror_out_p}: {e}"))?;
        println!(
            "VA_MIRROR_EXP_COMPLETE va_reserved_gib={:.2} faulted_kb={} base_tps={base_tps:.3} va_l0_tps={mir0_tps:.3} ({:.2}x) va_l8_tps={mir_tps:.3} ({:.2}x) l0_exact={exact_matches_l0}/{} l8_exact={exact_matches}/{} vm_hwm_kb={vm_hwm_end:?} out={mirror_out_p}",
            arena.va_reserved_gib(),
            arena.faulted_ram_kb(),
            mir0_tps / base_tps.max(1e-6),
            mir_tps / base_tps.max(1e-6),
            base_ids.len(),
            base_ids.len()
        );
        let _ = va_meaning_cache_depth;
        return Ok(());
    }
    if let Some(ref bind_ep) = cluster_worker_bind {
        eprintln!(
            "[run31b] starting V11 ZK-Cortical-Cluster Worker on {bind_ep} layers={}..{} d={} vm={:?}",
            rt.local_layer_range.0,
            rt.local_layer_range.1,
            rt.cfg.d,
            proc_status_kb()
        );
        return serve_cluster_worker(bind_ep, &cluster_psk, &mut rt, cluster_max_sessions);
    }
    if let Some(ref w_list) = cluster_workers_arg {
        let mut conns = Vec::new();
        let mut expected_next = rt.local_layer_range.1;
        for raw_ep in w_list.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
            let conn = ClusterWorkerConn::connect_and_attest(raw_ep, &cluster_psk, rt.cfg.d)?;
            if conn.layer_start != expected_next {
                return Err(format!(
                    "cluster shard gap/overlap at {raw_ep}: expected layer_start={expected_next}, worker reported {}..{}",
                    conn.layer_start, conn.layer_end
                ));
            }
            expected_next = conn.layer_end;
            eprintln!(
                "[run31b] attested ZK-Cluster Worker {} -> layers {}..{} (ChaCha20-Poly1305 + FWHT-INT8)",
                raw_ep, conn.layer_start, conn.layer_end
            );
            conns.push(conn);
        }
        if expected_next != rt.cfg.n_layers {
            return Err(format!(
                "cluster pipeline ends at layer {expected_next} != n_layers {}",
                rt.cfg.n_layers
            ));
        }
        rt.cluster_workers = conns;
    }

    let tk = Tokenizer::load(Path::new(&tkp))?;
    let assistant_state: Option<(AssistantRuntime, usize, usize)> = if let Some(ap) = spec_assistant_arg.as_deref() {
        let native_hint = so_arg
            .as_deref()
            .or_else(|| rt.native_t16c.as_ref().map(|n| n.path.as_str()));
        let trie_opt = if spec_byte_head_arg.is_some() {
            Some(Arc::new(BpePrefixTrie::from_tokenizer(&tk)))
        } else {
            None
        };
        let mut ast = AssistantRuntime::load_31b_with_byte_head(
            std::path::Path::new(ap),
            spec_byte_head_arg.as_deref().map(std::path::Path::new),
            trie_opt,
            native_hint.map(std::path::Path::new),
            disable_native,
            threads,
            rt.cfg.n_kv,
            rt.cfg.n_global_kv,
            rt.cfg.window,
            rt.cfg.vocab,
        )?;
        let ast_zp_saved = if zplane_compact || cortical_cascade_tau.is_some() {
            ast.compact_zplane()
        } else {
            0
        };
        if ast.cfg.backbone_hidden != rt.cfg.d || ast.cfg.vocab != rt.cfg.vocab {
            return Err(format!(
                "assistant {} dims ({}x{}) incompatible with 31B ({}x{})",
                ap, ast.cfg.backbone_hidden, ast.cfg.vocab, rt.cfg.d, rt.cfg.vocab
            ));
        }
        let (sl_l, fl_l) = rt.assistant_kv_layers()?;
        eprintln!(
            "[run31b] loaded AssistantRuntime {}: layers={} d={} resident_mb={:.2} zplane_saved_kb={} byte_head={} target_shared_head={} shared_kv=(L{sl_l},L{fl_l}) step_pos={assistant_step_pos}",
            ap,
            ast.cfg.n_layers,
            ast.cfg.d,
            ast.resident_bytes() as f64 / (1024.0 * 1024.0),
            ast_zp_saved / 1024,
            ast.has_byte_head(),
            ast.uses_target_shared_head(),
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

    let mut vision_soft: Option<(usize, Vec<f32>)> = None;
    if image_arg.is_some() || optical_self_analyze {
        let vm_before_vis = proc_status_kb();
        let vw = rt
            .vision
            .as_ref()
            .ok_or("Multimodal --image / --optical-self-analyze requires full .t16c with vision_tower")?;
        let vis_warmed = if keep_vision_resident { 0 } else { vw.warm_pages() };
        let vm_after_warm = proc_status_kb();
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
        let vm_after_enc = proc_status_kb();
        eprintln!(
            "[run31b] vision encoded {}x{} -> n_soft={n_soft} in {:.2}s warmed={} vm_before={vm_before_vis:?} vm_warm={vm_after_warm:?} vm_enc={vm_after_enc:?}",
            img.width,
            img.height,
            t_vis.elapsed().as_secs_f64(),
            vis_warmed
        );
        if let Some(ref vp) = dump_vision_soft {
            let bytes: Vec<u8> = soft.iter().flat_map(|v| v.to_le_bytes()).collect();
            fs::write(vp, bytes).map_err(|e| format!("write {vp}: {e}"))?;
            eprintln!("[run31b] dumped {n_soft}x{} vision soft tokens to {vp}", rt.cfg.d);
        }
        if !keep_vision_resident && !batch_has_images && !vision_only {
            if let Some(vw_drop) = rt.vision.take() {
                let ev = vw_drop.evict_pages();
                drop(vw_drop);
                eprintln!(
                    "[run31b] vision on-demand evicted {ev} B after prefill (vm_after_evict={:?})",
                    proc_status_kb()
                );
            }
        }
        if vision_only {
            return Ok(());
        }
        vision_soft = Some((n_soft, soft));
    } else if !keep_vision_resident && !batch_has_images {
        if let Some(vw_drop) = rt.vision.take() {
            let ev = vw_drop.evict_pages();
            drop(vw_drop);
            eprintln!(
                "[run31b] vision on-demand: text-only run evicted {ev} B (vm={:?})",
                proc_status_kb()
            );
        }
    }

    for (prompt_idx, raw_line) in batch_prompts.iter().enumerate() {
        if prompt_idx > 0 {
            rt.truncate_kv(0)?;
            rt.cs_skipped_blocks = 0;
            rt.lmhead_cs_skipped_blocks = 0;
            rt.mlp_sparse_skipped_blks = 0;
            rt.mlp_sparse_total_blks = 0;
            rt.cortical_gate_skipped_blks = 0;
            rt.cortical_gate_total_blks = 0;
            rt.mod_skipped_layers = 0;
            rt.mod_total_layers = 0;
            rt.attn_sparse_skipped_heads = 0;
            rt.attn_sparse_total_heads = 0;
            rt.prof_attn_s = 0.0;
            rt.prof_gate_s = 0.0;
            rt.prof_up_s = 0.0;
            rt.prof_down_s = 0.0;
            rt.prof_norm_kv_s = 0.0;
            rt.prof_lmhead_s = 0.0;
            rt.prof_router_s = 0.0;
            rt.prof_steps = 0;
            rt.ssd_bytes_read = 0;
        }
        let mut cur_max_tokens = max_tokens;
        let mut line_body: &str = raw_line.as_str();
        if let Some(rest_mt) = line_body.strip_prefix("MAX_TOKENS:") {
            let (mt_s, after_mt) = rest_mt
                .split_once('|')
                .ok_or_else(|| format!("invalid MAX_TOKENS:<n>|<prompt> line: {raw_line}"))?;
            cur_max_tokens = mt_s
                .trim()
                .parse::<usize>()
                .map_err(|_| format!("invalid MAX_TOKENS '{mt_s}'"))?;
            line_body = after_mt;
        }
        let (cur_prompt_text, cur_vision_override): (String, Option<(usize, Vec<f32>)>) =
            if let Some(rest) = line_body.strip_prefix("IMAGE:") {
                let (img_path_s, p_txt) = rest
                    .split_once('|')
                    .ok_or_else(|| format!("invalid IMAGE:<path>|<prompt> line: {raw_line}"))?;
                let img = run270_rust::vision::load_rgb_image(Path::new(img_path_s.trim()))?;
                let vw = rt.vision.as_ref().ok_or("IMAGE: batch line requires .t16c with vision weights")?;
                let warmed = if keep_vision_resident { 0 } else { vw.warm_pages() };
                let t_vis = std::time::Instant::now();
                let (n_soft, soft) = vw.encode_image(
                    &img,
                    rt.native_t16c.as_deref(),
                    rt.c8d_bank.as_ref(),
                    rt.threads,
                )?;
                let evicted = if keep_vision_resident { 0 } else { vw.evict_pages() };
                eprintln!(
                    "[run31b] batch vision encoded {} ({}x{}) -> n_soft={n_soft} in {:.2}s warmed={warmed} evicted={evicted} vm={:?}",
                    img_path_s.trim(),
                    img.width,
                    img.height,
                    t_vis.elapsed().as_secs_f64(),
                    proc_status_kb()
                );
                (p_txt.trim().to_string(), Some((n_soft, soft)))
            } else {
                (line_body.to_string(), None)
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
    if eval_nll {
        let eval_start = resolve_eval_nll_start_pos(&ids, eval_nll_start_pos, eval_nll_after_turn)?;
        let n_pos = (ids.len() - 1) - eval_start;
        let ids_hash = if eval_start == 0 {
            fnv1a64_ids(&ids)
        } else {
            fnv1a64_ids(&ids) ^ (eval_start as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        };
        let vocab = rt.cfg.vocab;
        let d_h = rt.cfg.d;

        let mut lgt1_writer = if let Some(ref dp) = dump_seq_logits {
            let f = fs::File::create(dp).map_err(|e| format!("LGT1 create {dp}: {e}"))?;
            let mut bw = BufWriter::new(f);
            write_lgt1_header(&mut bw, n_pos, vocab, ids_hash)?;
            Some(bw)
        } else {
            None
        };
        let mut lgt1_reader = if let Some(ref tp_lgt) = eval_teacher_logits {
            Some(Lgt1Reader::open(Path::new(tp_lgt), n_pos, vocab, ids_hash)?)
        } else {
            None
        };

        let all_h = rt.forward_batch_with_embeds(
            prefill_states,
            ids.len(),
            0,
            true,
            true,
            Some(&vis_kv_ends),
        )?;
        let mut sum_nll = 0.0f64;
        let mut sum_ent = 0.0f64;
        let mut hits1 = 0usize;
        let mut sum_kl_ts = 0.0f64;
        let mut sum_kl_st = 0.0f64;
        let mut agree1 = 0usize;
        let mut agree5 = 0usize;
        let mut sum_t_nll = 0.0f64;
        let mut teacher_row = Vec::new();
        let chunk_t = 16usize;
        let end_pos = ids.len() - 1;
        let mut p0 = eval_start;
        while p0 < end_pos {
            let cnt = (end_pos - p0).min(chunk_t);
            let h_slice = &all_h[p0 * d_h..(p0 + cnt) * d_h];
            let logits_c = rt.compute_logits_batch_mode(h_slice, cnt, false)?;
            for c in 0..cnt {
                let pos = p0 + c;
                let s_logits = &logits_c[c * vocab..(c + 1) * vocab];
                let gold_next = ids[pos + 1];
                let (nll, h1, ent, _) = eval_student_logits_stats(s_logits, gold_next)?;
                sum_nll += nll;
                sum_ent += ent;
                if h1 {
                    hits1 += 1;
                }
                if let Some(ref mut wr) = lgt1_writer {
                    write_lgt1_bf16_row(wr, s_logits)?;
                }
                if let Some(ref mut rd) = lgt1_reader {
                    rd.read_next_row(&mut teacher_row)?;
                    let (kts, kst, a1, a5, tnll, _) = compute_seq_kl_metrics(&teacher_row, s_logits, gold_next)?;
                    sum_kl_ts += kts;
                    sum_kl_st += kst;
                    if a1 {
                        agree1 += 1;
                    }
                    if a5 {
                        agree5 += 1;
                    }
                    sum_t_nll += tnll;
                }
            }
            p0 += cnt;
        }
        if let Some(ref mut wr) = lgt1_writer {
            wr.flush().map_err(|e| format!("LGT1 flush: {e}"))?;
        }
        let eval_s = t_pf.elapsed().as_secs_f64();
        let mean_nll = sum_nll / n_pos as f64;
        let mean_ent = sum_ent / n_pos as f64;
        println!(
            "EVAL_NLL tokens={} eval_start={eval_start} eval_pos={n_pos} mean_nll={mean_nll:.5} ppl={:.3} top1_acc={:.4} mean_entropy={mean_ent:.4} eval_s={eval_s:.2}",
            ids.len(),
            mean_nll.exp(),
            hits1 as f64 / n_pos as f64,
        );
        if lgt1_reader.is_some() {
            let n_f = n_pos as f64;
            let t_nll = sum_t_nll / n_f;
            println!(
                "EVAL_TEACHER_KL pos={n_pos} kl_t_to_s={:.5} kl_s_to_t={:.5} top1_agree={:.4} top5_agree={:.4} teacher_nll={t_nll:.5} teacher_ppl={:.3}",
                sum_kl_ts / n_f,
                sum_kl_st / n_f,
                agree1 as f64 / n_f,
                agree5 as f64 / n_f,
                t_nll.exp(),
            );
        }
        print_runtime_telemetry(&rt, pinned_bytes, embed_bytes, eval_s, 0.0);
        continue;
    }
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
        } else if let Some(tp) = t16c_arg.as_deref() {
            // Auto-discover co-located V10 `.cngm` Macro-Glyph transition seed if present
            let auto_p = Path::new(tp)
                .parent()
                .map(|d| d.join("gemma4_v10_seed.cngm"))
                .filter(|p| p.is_file());
            if let Some(ap) = auto_p {
                match ngram.load_seed_file(&ap) {
                    Ok(n) => {
                        eprintln!("[run31b] auto-loaded co-located CNGM Macro-Glyph seed {}: transitions={n}", ap.display());
                        n
                    }
                    Err(_) => 0,
                }
            } else {
                0
            }
        } else {
            0
        };
        ngram.train_tokens(&history);
        let max_draft = (batch_k - 1).max(1);
        let mut ctrl = AdaptiveDraftController::new(1, max_draft);
        let mut pending = if cur_max_tokens > 0 { Some(argmax_f32(&prefill_logits)) } else { None };
        let mut spec_drafted = 0usize;
        let mut spec_accepted = 0usize;
        let mut spec_batches = 0usize;
        let mut spec_fallback_k1 = 0usize;
        let mut spec_margin_stopped = 0usize;
        let mut ast_lm_skipped_blocks = 0u64;
        let mut cascade_tier1_fast = 0usize;
        let mut cascade_tier2_accepted = 0usize;
        let mut cascade_tier3_rescues = 0usize;
        let mut cascade_draft_s = 0.0f64;
        let mut cascade_tier2_s = 0.0f64;
        let mut cascade_tier3_s = 0.0f64;
        let cascade_tier2_layers = if let Some(el) = cascade_exit_layer_arg {
            el.min(rt.cfg.n_layers)
        } else {
            rt.cfg.n_layers
        };
        let cascade_tier1_tau: f32 = cascade_tier1_tau_arg.unwrap_or_else(|| {
            if cortical_cascade_tau.is_some() {
                if assistant_state.as_ref().is_some_and(|(a, _, _)| a.has_byte_head()) {
                    0.85
                } else {
                    1.60
                }
            } else {
                0.0
            }
        });
        let mut tier1_unverified: Vec<u32> = Vec::new();
        let vocab = rt.cfg.vocab;
        let d_hidden = rt.cfg.d;

        while let Some(y0) = pending.take() {
            let step_idx = generated.len();
            generated.push(y0);
            ngram.observe_transition(&history, y0);
            history.push(y0);
            println!("[STEP {step_idx}] id={y0} text={:?}", tk.decode(&[y0]));
            if eos.contains(&y0) || generated.len() >= cur_max_tokens {
                break;
            }

            let cur_kv_len = rt.layer_kv_len(0);
            let eff_kv_with_tier1 = cur_kv_len + tier1_unverified.len();
            let expected_kv_len = ids.len() + generated.len() - 1;
            if eff_kv_with_tier1 != expected_kv_len {
                return Err(format!(
                    "KV desync before spec step: kv_len={cur_kv_len} + tier1_unverified={} != expected={expected_kv_len}",
                    tier1_unverified.len()
                ));
            }

            let rem = cur_max_tokens - generated.len();
            let td = std::time::Instant::now();

            if let Some(max_tree_nodes) = spec_tree_nodes {
                let max_depth_step = spec_tree_depth.min(rem.saturating_sub(1));
                let raw_tree_cap = max_tree_nodes.min(1 + rem.saturating_sub(1).saturating_mul(3)).max(1);
                let max_prefix_cap = 24usize.saturating_sub(tier1_unverified.len()).max(1);
                let eff_tree_cap = raw_tree_cap.min(max_prefix_cap);
                let cascade_fast_budget = if cortical_cascade_tau.is_some()
                    && cascade_tier1_tau > 0.0
                    && tier1_unverified.len() < cascade_max_unverified
                {
                    rem.min(cascade_max_unverified - tier1_unverified.len())
                } else {
                    0
                };
                let tree_opt: Option<(SpecTreeBuilder, Vec<Option<Vec<f32>>>)> = if (max_depth_step == 0 || eff_tree_cap <= 1) && cascade_fast_budget == 0 {
                    None
                } else {
                    let t_dr0 = std::time::Instant::now();
                    let suffix_hint = prompt_suffix_lookup(&history, 2, max_depth_step);
                    let ngram_branches = ngram.propose_multi_branches(&history, max_depth_step, 2);
                    let ngram_hint = ngram_branches.first().map(|v| v.as_slice()).unwrap_or(&[]);
                    let ast_depth = if cortical_cascade_tau.is_some() {
                        4usize.min(max_depth_step)
                    } else {
                        max_depth_step
                    };
                    let ast_sieve = if cortical_cascade_tau.is_some() {
                        Some(match lmhead_cs_sieve {
                            Some(s) if s > 0.0 => s,
                            _ => 0.125,
                        })
                    } else {
                        lmhead_cs_sieve
                    };
                    let tier1_tau_opt = if cascade_fast_budget > 0 {
                        Some(cascade_tier1_tau)
                    } else {
                        None
                    };
                    let res_tree: Result<Option<(SpecTreeBuilder, Vec<Option<Vec<f32>>>)>, String> = if let Some((ref ast, sl_l, fl_l)) = assistant_state {
                        let kv_bound_len = rt.layer_kv_len(sl_l).min(rt.layer_kv_len(fl_l));
                        let max_tt = kv_bound_len.saturating_sub(1);
                        let draft_q_pos = kv_bound_len + tier1_unverified.len();
                        let (mut tree_b, m_stop, ast_sk, slot_bhs) = ast.draft_cascade_fast_or_tree(
                            y0,
                            |tok| rt.embed_token_ro(tok),
                            &last_backbone_h,
                            draft_q_pos,
                            ast_depth,
                            eff_tree_cap,
                            ngram_hint,
                            &suffix_hint,
                            &eos,
                            closed_thought_bans,
                            assistant_min_margin,
                            ast_sieve,
                            tier1_tau_opt,
                            cascade_fast_budget,
                            |is_full, q, q_pos| {
                                let l_idx = if is_full { fl_l } else { sl_l };
                                if rt.kv_quant.is_glyph() {
                                    let gkv = rt.glyph_kv[l_idx]
                                        .as_ref()
                                        .ok_or_else(|| format!("missing glyph KV at layer {l_idx}"))?;
                                    gkv.attend_range(
                                        q,
                                        max_tt,
                                        max_tt,
                                        q_pos,
                                        1.0,
                                        rt.native_t16c.as_deref(),
                                        rt.threads,
                                    )
                                } else {
                                    let hd_ast = if is_full { rt.cfg.global_hd } else { rt.cfg.hd };
                                    let n_kv_ast = if is_full { rt.cfg.n_global_kv } else { rt.cfg.n_kv };
                                    let sw = if is_full { None } else { Some(rt.cfg.window) };
                                    Ok(attn_decode_range(
                                        q,
                                        &rt.kv[l_idx].k,
                                        &rt.kv[l_idx].v,
                                        max_tt,
                                        max_tt,
                                        q_pos,
                                        ast.cfg.n_heads,
                                        n_kv_ast,
                                        hd_ast,
                                        sw,
                                    ))
                                }
                            },
                            |_h_final, normed_next_bh, _prev_tok, want_k, bans| {
                                rt.score_normalized_lm_head_top_k(
                                    normed_next_bh,
                                    want_k,
                                    ast_sieve.unwrap_or(0.035),
                                    bans,
                                )
                            },
                            |raw_bh| rt.normalize_backbone_hidden(raw_bh),
                        )?;
                        for alt_seq in ngram_branches.iter().skip(1) {
                            tree_b.graft_sequence_with_margin(alt_seq, &eos, 4.0);
                        }
                        // Extend EAGLE-3 leaves with zero-DRAM Macro-Glyph / suffix spans!
                        let hist_before_y0 = &history[..history.len().saturating_sub(1)];
                        let max_span_ext = max_depth_step.saturating_sub(ast_depth).clamp(2, 5);
                        tree_b.extend_leaves_with_macro_spans(hist_before_y0, &ngram, &eos, max_span_ext);
                        spec_margin_stopped += m_stop;
                        if ast_sk > 0 {
                            ast_lm_skipped_blocks += ast_sk as u64;
                        }
                        Ok(if tree_b.nodes.len() > 1 { Some((tree_b, slot_bhs)) } else { None })
                    } else if max_depth_step > 0 && eff_tree_cap > 1 && (!suffix_hint.is_empty() || !ngram_branches.is_empty()) {
                        let mut tb = SpecTreeBuilder::new(y0, eff_tree_cap);
                        if !suffix_hint.is_empty() {
                            tb.graft_sequence_with_margin(&suffix_hint, &eos, 12.0);
                        }
                        for br in &ngram_branches {
                            tb.graft_sequence_with_margin(br, &eos, 5.0);
                        }
                        let empty_bhs = vec![None; tb.nodes.len()];
                        Ok(if tb.nodes.len() > 1 { Some((tb, empty_bhs)) } else { None })
                    } else {
                        Ok(None)
                    };
                    cascade_draft_s += t_dr0.elapsed().as_secs_f64();
                    res_tree?
                };

                if let Some((tree_b, slot_bhs)) = tree_opt {
                    let n_nodes = tree_b.nodes.len();
                    let batch_ids: Vec<u32> = tree_b.nodes.iter().map(|n| n.token_id).collect();
                    let tree_depths: Vec<usize> = tree_b.nodes.iter().map(|n| n.depth).collect();
                    let ancestor_masks: Vec<u32> = tree_b.nodes.iter().map(|n| n.ancestor_mask).collect();
                    let max_d = tree_depths.iter().copied().max().unwrap_or(1).max(1);
                    spec_batches += 1;
                    spec_drafted += max_d;

                    if let Some(tau) = cortical_cascade_tau {
                        let fast_cap = rem.min(cascade_max_unverified.saturating_sub(tier1_unverified.len()));
                        if cascade_tier1_tau > 0.0 && fast_cap >= 1 {
                            let mut fast_slots: Vec<usize> = Vec::new();
                            let mut cur_s = 0usize;
                            while fast_slots.len() < fast_cap {
                                let next_opt = tree_b
                                    .nodes
                                    .iter()
                                    .enumerate()
                                    .skip(cur_s + 1)
                                    .find(|&(idx, n)| {
                                        n.parent == cur_s as i32
                                            && !eos.contains(&n.token_id)
                                            && tree_b.slot_margin(idx) >= cascade_tier1_tau
                                            && slot_bhs.get(idx).and_then(|o| o.as_ref()).is_some()
                                    })
                                    .map(|(idx, _)| idx);
                                let Some(cs) = next_opt else { break; };
                                fast_slots.push(cs);
                                cur_s = cs;
                                if eos.contains(&tree_b.nodes[cs].token_id) {
                                    break;
                                }
                            }
                            if !fast_slots.is_empty() {
                                tier1_unverified.push(y0);
                                for &cs in &fast_slots[..fast_slots.len() - 1] {
                                    let tok = tree_b.nodes[cs].token_id;
                                    let s_idx = generated.len();
                                    generated.push(tok);
                                    ngram.observe_transition(&history, tok);
                                    history.push(tok);
                                    tier1_unverified.push(tok);
                                    cascade_tier1_fast += 1;
                                    println!("[STEP {s_idx}] id={tok} text={:?}", tk.decode(&[tok]));
                                }
                                let last_fast_slot = *fast_slots.last().unwrap();
                                cascade_tier1_fast += 1;
                                if let Some(ref bh_next) = slot_bhs[last_fast_slot] {
                                    last_backbone_h = bh_next.clone();
                                }
                                pending = Some(tree_b.nodes[last_fast_slot].token_id);
                                continue;
                            }
                        }

                        let orig_mlp = rt.mlp_top_p;
                        let orig_head = rt.attn_head_top_p;
                        let orig_mod = rt.mod_thresh;
                        let orig_mod_z = rt.mod_zero_kv;
                        let orig_sieve = rt.lmhead_cs_sieve;
                        let orig_cca_sp = rt.cca_sparse_active;

                        rt.mlp_top_p = cascade_mlp_top_p;
                        if rt.head_router.is_some() {
                            rt.attn_head_top_p = cascade_head_top_p;
                        }
                        if rt.cortical_router.as_ref().is_some_and(|c| c.has_mod_gate()) {
                            rt.mod_thresh = cascade_mod_thresh;
                            rt.mod_zero_kv = cascade_mod_zero_kv;
                        }
                        if rt.cca_layer_policies.is_some() {
                            rt.cca_sparse_active = true;
                        }
                        rt.lmhead_cs_sieve = orig_sieve;

                        let t_t2 = std::time::Instant::now();
                        let flush_ids = std::mem::take(&mut tier1_unverified);
                        let m_prefix = flush_ids.len();
                        let (fused_nodes, _) = fuse_unverified_prefix_and_tree(&flush_ids, &tree_b)?;
                        let fused_ids: Vec<u32> = fused_nodes.iter().map(|n| n.token_id).collect();
                        let fused_depths: Vec<usize> = fused_nodes.iter().map(|n| n.depth).collect();
                        let fused_masks: Vec<u32> = fused_nodes.iter().map(|n| n.ancestor_mask).collect();

                        let (verify_kv_len, h_tree, raw_l29_batch, mut l_batch) = (|| -> Result<(usize, Vec<f32>, Vec<f32>, Vec<f32>), String> {
                            let base_pos = rt.kv_len();
                            // Single fused 60-layer sweep across `flush_ids + tree_b` (eliminates double 6.27 GiB DRAM read!)
                            let (hb_all, raw_all) = rt.forward_batch_tree_upto(
                                &fused_ids,
                                base_pos,
                                &fused_depths,
                                &fused_masks,
                                cascade_tier2_layers,
                            )?;
                            let hb_tree = hb_all[m_prefix * d_hidden..].to_vec();
                            drop(hb_all);
                            let mut seed_row_ids = vec![-1i32; n_nodes];
                            for p_idx in 0..n_nodes {
                                if let Some(child) = tree_b.nodes.iter().skip(p_idx + 1).find(|c| c.parent == p_idx as i32) {
                                    seed_row_ids[p_idx] = child.token_id as i32;
                                }
                            }
                            let lb = rt.compute_logits_batch_seeded(&hb_tree, n_nodes, Some(&seed_row_ids))?;
                            Ok((base_pos, hb_tree, raw_all, lb))
                        })()?;
                        cascade_tier2_s += t_t2.elapsed().as_secs_f64();

                        rt.mlp_top_p = orig_mlp;
                        rt.attn_head_top_p = orig_head;
                        rt.mod_thresh = orig_mod;
                        rt.mod_zero_kv = orig_mod_z;
                        rt.lmhead_cs_sieve = orig_sieve;
                        rt.cca_sparse_active = orig_cca_sp;

                        let mut accepted_drafts: Vec<u32> = Vec::with_capacity(max_d);
                        let mut accepted_slots: Vec<usize> = Vec::with_capacity(1 + max_d);
                        accepted_slots.push(0);
                        let mut cur_slot = 0usize;
                        let mut sim_gen = generated.clone();
                        let mut hit_stop = false;
                        let final_tok: u32;
                        let final_margin: f32;

                        loop {
                            let sl = &mut l_batch[cur_slot * vocab..(cur_slot + 1) * vocab];
                            apply_degeneracy_guard(sl, &ids, &sim_gen);
                            let mut best_i = 0u32;
                            let mut best_v = f32::NEG_INFINITY;
                            let mut sec_v = f32::NEG_INFINITY;
                            for (idx_l, &val_l) in sl.iter().enumerate() {
                                if val_l > best_v {
                                    sec_v = best_v;
                                    best_v = val_l;
                                    best_i = idx_l as u32;
                                } else if val_l > sec_v {
                                    sec_v = val_l;
                                }
                            }
                            let margin = (best_v - sec_v).max(0.0);
                            let matched_child = tree_b
                                .nodes
                                .iter()
                                .enumerate()
                                .skip(cur_slot + 1)
                                .find(|&(_, n)| n.parent == cur_slot as i32 && n.token_id == best_i)
                                .map(|(idx, _)| idx);

                            if let Some(cslot) = matched_child {
                                accepted_drafts.push(best_i);
                                accepted_slots.push(cslot);
                                sim_gen.push(best_i);
                                if eos.contains(&best_i) || accepted_drafts.len() >= rem {
                                    hit_stop = true;
                                    final_tok = best_i;
                                    final_margin = margin;
                                    break;
                                }
                                cur_slot = cslot;
                            } else {
                                final_tok = best_i;
                                final_margin = margin;
                                break;
                            }
                        }

                        let acc = accepted_drafts.len();
                        spec_accepted += acc;
                        cascade_tier2_accepted += acc;
                        ctrl.observe(acc, max_d);
                        let mut commit_slots: Vec<usize> = (0..m_prefix).collect();
                        for &s in &accepted_slots {
                            commit_slots.push(m_prefix + s);
                        }
                        rt.commit_tree_slots_upto(
                            verify_kv_len,
                            &commit_slots,
                            cascade_tier2_layers,
                            if cascade_tier2_layers < rt.cfg.n_layers { Some(&raw_l29_batch) } else { None },
                        )?;
                        for &tok in &accepted_drafts {
                            let s_idx = generated.len();
                            generated.push(tok);
                            ngram.observe_transition(&history, tok);
                            history.push(tok);
                            println!("[STEP {s_idx}] id={tok} text={:?}", tk.decode(&[tok]));
                        }
                        let last_slot = *accepted_slots.last().unwrap_or(&0);
                        let next_pending: Option<u32>;
                        if hit_stop {
                            last_backbone_h = h_tree[last_slot * d_hidden..(last_slot + 1) * d_hidden].to_vec();
                            next_pending = None;
                        } else if final_margin >= tau {
                            last_backbone_h = h_tree[last_slot * d_hidden..(last_slot + 1) * d_hidden].to_vec();
                            next_pending = Some(final_tok);
                        } else {
                            // Tier-3 Exact Dense 60/60 Rescue on uncertain fork node (syncing all 60 layers cleanly)
                            let t_t3 = std::time::Instant::now();
                            let rescue_pos = verify_kv_len + commit_slots.len() - 1;
                            let sync_pos = rt.layer_kv_len(rt.cfg.n_layers - 1).min(rescue_pos);
                            rt.truncate_kv(sync_pos)?;
                            rt.mlp_top_p = 1.0;
                            rt.attn_head_top_p = 1.0;
                            rt.mod_thresh = 0.0;
                            rt.mod_zero_kv = false;
                            rt.lmhead_cs_sieve = None;
                            rt.cca_sparse_active = false;
                            let dense_res = (|| -> Result<(Vec<f32>, Vec<f32>), String> {
                                let hd = rt.forward_batch(&history[sync_pos..=rescue_pos], sync_pos, false)?;
                                let ld = rt.compute_logits_batch_mode(&hd, 1, false)?;
                                Ok((hd, ld))
                            })();
                            rt.mlp_top_p = orig_mlp;
                            rt.attn_head_top_p = orig_head;
                            rt.mod_thresh = orig_mod;
                            rt.mod_zero_kv = orig_mod_z;
                            rt.lmhead_cs_sieve = orig_sieve;
                            rt.cca_sparse_active = orig_cca_sp;
                            let (h_dense, mut l_dense) = dense_res?;
                            apply_degeneracy_guard(&mut l_dense, &ids, &generated);
                            last_backbone_h = h_dense;
                            next_pending = Some(argmax_f32(&l_dense));
                            cascade_tier3_rescues += 1;
                            cascade_tier3_s += t_t3.elapsed().as_secs_f64();
                        }
                        pending = next_pending;
                        eprintln!(
                            "[run31b] cascade batch={} l_exit={} nodes={} max_d={} accepted={} (+1={}) fast={} rescues={} ewma={:.2} in {:.2}s vm={:?}",
                            spec_batches,
                            cascade_tier2_layers,
                            n_nodes,
                            max_d,
                            acc,
                            acc + 1,
                            cascade_tier1_fast,
                            cascade_tier3_rescues,
                            ctrl.acceptance_ewma(),
                            td.elapsed().as_secs_f64(),
                            proc_status_kb()
                        );
                        continue;
                    }

                    let h_batch = rt.forward_batch_tree(&batch_ids, cur_kv_len, &tree_depths, &ancestor_masks)?;
                    let mut seed_row_ids = vec![-1i32; n_nodes];
                    for p_idx in 0..n_nodes {
                        if let Some(child) = tree_b.nodes.iter().skip(p_idx + 1).find(|c| c.parent == p_idx as i32) {
                            seed_row_ids[p_idx] = child.token_id as i32;
                        }
                    }
                    let mut l_batch = rt.compute_logits_batch_seeded(&h_batch, n_nodes, Some(&seed_row_ids))?;
                    let base_gen = generated.clone();
                    let outcome = verify_greedy_tree_batch(
                        &tree_b.nodes,
                        &mut l_batch,
                        vocab,
                        &eos,
                        rem,
                        |slot_logits, acc_so_far| {
                            let mut sim_gen = Vec::with_capacity(base_gen.len() + acc_so_far.len());
                            sim_gen.extend_from_slice(&base_gen);
                            sim_gen.extend_from_slice(acc_so_far);
                            apply_degeneracy_guard(slot_logits, &ids, &sim_gen);
                        },
                    )?;
                    let acc = outcome.accepted_drafts.len();
                    let last_slot = *outcome.accepted_slots.last().unwrap_or(&0);
                    spec_accepted += acc;
                    ctrl.observe(acc, max_d);
                    rt.commit_tree_slots(cur_kv_len, &outcome.accepted_slots)?;
                    last_backbone_h = h_batch[last_slot * d_hidden..(last_slot + 1) * d_hidden].to_vec();
                    for &tok in &outcome.accepted_drafts {
                        let s_idx = generated.len();
                        generated.push(tok);
                        ngram.observe_transition(&history, tok);
                        history.push(tok);
                        println!("[STEP {s_idx}] id={tok} text={:?}", tk.decode(&[tok]));
                    }
                    let mut next_pending = if outcome.hit_stop { None } else { outcome.next_pending };
                    if let Some(ref mut gate) = tri_gate {
                        let mut slot_logits = l_batch[last_slot * vocab..(last_slot + 1) * vocab].to_vec();
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
                        "[run31b] spec_tree batch={} nodes={} max_d={} accepted={} (+1={}) ewma={:.2} in {:.2}s vm={:?}",
                        spec_batches,
                        n_nodes,
                        max_d,
                        acc,
                        acc + 1,
                        ctrl.acceptance_ewma(),
                        td.elapsed().as_secs_f64(),
                        proc_status_kb()
                    );
                    continue;
                }
            }

            let mut draft_budget = if tri_modal_gate {
                ctrl.current_k().min(tri_draft_cap).min(rem.saturating_sub(1))
            } else {
                ctrl.current_k().min(rem.saturating_sub(1))
            };
            // POV is only a performance-side cap here. It never changes the
            // EWMA controller state or the KV rollback boundary.
            draft_budget = apply_pov_draft_cap(
                draft_budget,
                rt.player_pov_6sense,
                rt.pov_draft_cap,
            );
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
            if drafts.is_empty() {
                spec_fallback_k1 += 1;
                if !tier1_unverified.is_empty() {
                    let flush_ids = std::mem::take(&mut tier1_unverified);
                    let f_pos = rt.layer_kv_len(0);
                    rt.forward_batch_upto(&flush_ids, f_pos, rt.cfg.n_layers)?;
                }
                let h = rt.forward_batch(&[y0], rt.layer_kv_len(0), false)?;
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
                if !tier1_unverified.is_empty() {
                    let flush_ids = std::mem::take(&mut tier1_unverified);
                    let f_pos = rt.layer_kv_len(0);
                    rt.forward_batch_upto(&flush_ids, f_pos, rt.cfg.n_layers)?;
                }
                let eff_kv_len = rt.layer_kv_len(0);
                let mut batch = Vec::with_capacity(1 + drafts.len());
                batch.push(y0);
                batch.extend_from_slice(&drafts);
                let h_batch = rt.forward_batch_mode(&batch, eff_kv_len, false, true)?;
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
                rt.truncate_kv(eff_kv_len + 1 + acc)?;
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
        let ast_bh = assistant_state
            .as_ref()
            .is_some_and(|(a, _, _)| a.has_byte_head());
        let ast_tsh = assistant_state
            .as_ref()
            .is_some_and(|(a, _, _)| a.uses_target_shared_head());
        println!(
            "SPEC31B max_batch_k={batch_k} ngram={spec_ngram} min_count={spec_min_count} seed_entries={seed_entries} assistant_mb={ast_mb:.2} byte_head={ast_bh} step_pos={assistant_step_pos} min_margin={assistant_min_margin:.2} margin_stopped={spec_margin_stopped} ast_lm_skipped={ast_lm_skipped_blocks} drafted={spec_drafted} accepted={spec_accepted} batches={spec_batches} fallback_k1={spec_fallback_k1} accept_rate={accept_rate:.3} generated={} s_per_tok={eff_s_tok:.3} target_shared_head={ast_tsh}",
            generated.len()
        );
        if let Some(tau) = cortical_cascade_tau {
            let tok_s = if eff_s_tok > 0.0 { 1.0 / eff_s_tok } else { 0.0 };
            println!(
                "CASCADE31B tau={tau:.2} tier2_accepted={cascade_tier2_accepted} tier3_rescues={cascade_tier3_rescues} tok_per_s={tok_s:.2} ms_per_tok={:.1} tier1_fast={cascade_tier1_fast} l_exit={cascade_tier2_layers} draft_ms={:.1} tier2_ms={:.1} tier3_ms={:.1}",
                eff_s_tok * 1000.0,
                cascade_draft_s * 1000.0,
                cascade_tier2_s * 1000.0,
                cascade_tier3_s * 1000.0,
            );
        }
    } else {
        let mut logits = prefill_logits;
        for step in 0..cur_max_tokens {
            let nxt = argmax_f32(&logits);
            generated.push(nxt);
            let tok_txt = tk.decode(&[nxt]);
            println!("[STEP {step}] id={nxt} text={tok_txt:?}");
            if eos.contains(&nxt) {
                break;
            }
            if step + 1 < cur_max_tokens {
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
    if zk_lora_chirp_out {
        let gen_txt = tk.decode(&generated);
        let secret = run270_rust::zk_cluster::derive_cluster_key(&cluster_psk);
        let shard_tag = run270_rust::zk_cluster::compute_shard_attestation(
            &secret,
            rt.local_layer_range.0,
            rt.cfg.n_layers,
            rt.cfg.d,
        );
        let chirps = pack_zk_lora_chirps(&gen_txt, &secret, 1, &shard_tag);
        let mut hex = String::with_capacity(chirps.len() * ZKLORA_MTU * 2);
        for ch in &chirps {
            for &b in ch {
                hex.push_str(&format!("{b:02x}"));
            }
        }
        println!(
            "ZKLORA_EGRESS chirps={} total_rf_bytes={} text_bytes={} hex={hex}",
            chirps.len(),
            chirps.len() * ZKLORA_MTU,
            gen_txt.len()
        );
    }
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
    println!("GEN31B ids={generated:?}");
    let gen_text = tk.decode(&generated);
    println!("GEN31B text={gen_text:?}");
    if general_alu {
        if let Some(alu_ans) = evaluate_alu_context(&cur_prompt_text, &gen_text) {
            println!("ALU31B expr_result={alu_ans:?}");
        }
    }
    print_runtime_telemetry(&rt, pinned_bytes, embed_bytes, prefill_s, decode_s);
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
        assert_eq!(KvQuantMode::parse("glyph-lod").unwrap(), KvQuantMode::GlyphLod);
        assert!(KvQuantMode::Glyph.is_glyph());
        assert!(KvQuantMode::GlyphPure.is_glyph());
        assert!(KvQuantMode::GlyphLod.is_glyph());
        assert!(!KvQuantMode::Fp8E4M3.is_glyph());
        assert_eq!(resolve_kv_lod_caps(KvQuantMode::GlyphLod, None, None, None), (4, 32, 192));
        assert_eq!(resolve_kv_lod_caps(KvQuantMode::GlyphLod, Some(2), Some(16), Some(64)), (2, 16, 64));

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
        let layers = init_glyph_kv(&cfg, KvQuantMode::GlyphLod, (4, 32, 192)).unwrap();
        assert_eq!(layers.len(), 2);
        let l0 = layers[0].as_ref().unwrap();
        let l1 = layers[1].as_ref().unwrap();
        assert!(!l0.k_eq_v);
        assert_eq!(l0.sliding_window, Some(1024));
        assert_eq!(l0.mid_cap, 192);
        assert!(!l1.k_eq_v);
        assert_eq!(l1.sliding_window, None);
        assert_eq!(l1.mid_cap, 192);

        let prompt_ids = vec![2u32, 105, 2364, 107, 1841, 603, 506, 6037, 576, 7001, 236881];
        let (_frame, rec, stats) = optical_context_dma_roundtrip(&prompt_ids, 4).unwrap();
        assert_eq!(rec, prompt_ids);
        assert_eq!(stats.n_tokens, prompt_ids.len());
        assert!(stats.fps > 0.0);
    }

    #[test]
    fn parse_layer_spec_handles_ranges_lists_and_rejects_oob() {
        let s1 = parse_layer_spec("6-10, 20, 22..24", 60).unwrap();
        assert_eq!(s1, [6usize, 7, 8, 9, 10, 20, 22, 23].into_iter().collect());
        let s_all = parse_layer_spec("all", 4).unwrap();
        assert_eq!(s_all, [0usize, 1, 2, 3].into_iter().collect());
        let s_none = parse_layer_spec("none", 4).unwrap();
        assert!(s_none.is_empty());
        assert!(parse_layer_spec("", 60).is_err());
        assert!(parse_layer_spec("5-60", 60).is_err());
        assert!(parse_layer_spec("10-5", 60).is_err());
        assert!(parse_layer_spec("6..6", 60).is_err());
    }

    #[test]
    fn pov_controls_are_bounded_and_frozen_for_speculative_targets() {
        assert!((effective_pov_mod_thresh(0.35, Some(0.21), true) - 0.21).abs() < 1e-6);
        assert!((effective_pov_mod_thresh(0.35, Some(0.21), false) - 0.35).abs() < 1e-6);
        assert!((effective_pov_mod_thresh(0.35, Some(0.80), true) - 0.35).abs() < 1e-6);
        assert!((effective_pov_mod_thresh(0.35, Some(-1.0), true)).abs() < 1e-6);
        assert_eq!(apply_pov_draft_cap(4, true, 2), 2);
        assert_eq!(apply_pov_draft_cap(4, true, 0), 1);
        assert_eq!(apply_pov_draft_cap(4, false, 1), 4);
        assert_eq!(apply_pov_draft_cap(1, true, 4), 1);
    }

    #[test]
    fn parse_mod_thresh_validates_bounds() {
        assert!((parse_mod_thresh("0").unwrap() - 0.0).abs() < 1e-6);
        assert!((parse_mod_thresh("0.35").unwrap() - 0.35).abs() < 1e-6);
        assert!(parse_mod_thresh("-0.1").is_err());
        assert!(parse_mod_thresh("1.0").is_err());
        assert!(parse_mod_thresh("NaN").is_err());
        assert!(parse_mod_thresh("bad").is_err());
    }

    #[test]
    fn default_assistant_min_margin_matches_head_scale() {
        assert!((default_assistant_min_margin(false, false) - 0.0).abs() < 1e-6);
        assert!((default_assistant_min_margin(true, false) - 0.75).abs() < 1e-6);
        assert!((default_assistant_min_margin(true, true) - 0.25).abs() < 1e-6);
        assert!((default_assistant_min_margin(false, true) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn predictive_cortical_row_mask_matches_dense_on_active_t16c_blocks() {
        use run270_rust::cortical_router::CorticalRouterLayer;
        use run270_rust::t16c::{T16CMat, T16C_ROWMAJOR_BIT};

        let d_out = 256usize; // ff = 4 blocks * 64 rows (4 row tiles per block)
        let d_in = 128usize;
        let block_size = 64usize;
        let n_blks_in = d_in / block_size;
        let mut sc = vec![0.125f32; d_out * n_blks_in];
        let mut idx = vec![0u8; d_out * d_in];
        for r in 0..d_out {
            sc[r * n_blks_in + (r % n_blks_in)] = 0.25 + ((r % 5) as f32) * 0.05;
            for c in 0..d_in {
                idx[r * d_in + c] = ((r * 7 + c * 11) % 4) as u8;
            }
        }
        let mat = T16CMat::pack_from_components(
            "test.mlp.gate_proj.weight".into(),
            d_out,
            d_in,
            block_size,
            2 | T16C_ROWMAJOR_BIT,
            false,
            false,
            &sc,
            None,
            &idx,
            None,
            None,
        )
        .unwrap();

        let n_blocks = 4usize;
        let rank = 4usize;
        let mut v_i8 = vec![0i8; rank * d_in];
        let mut u_i8 = vec![0i8; n_blocks * rank];
        for k in 0..rank {
            for j in 0..8 {
                v_i8[k * d_in + k * 8 + j] = 100i8;
            }
            u_i8[k * rank + k] = 100i8;
        }
        let cr = CorticalRouterLayer {
            d_model: d_in,
            ff: d_out,
            block_size,
            n_blocks,
            rank,
            v_i8,
            sv: vec![0.02f32; rank],
            b1: vec![-0.2f32; rank],
            u_i8,
            su: vec![0.15f32; n_blocks],
            b2: vec![-2.5f32; n_blocks],
            mod_w_i8: None,
            mod_scale: 0.0,
            mod_bias: 0.0,
        };

        let mut x = vec![0.0f32; 2 * d_in];
        x[8..16].fill(2.5); // token 0 triggers block 1
        x[d_in + 24..d_in + 32].fill(2.5); // token 1 triggers block 3
        let (rmask, cmask, bsz) = cr.predict_masks(&x, 2, 0.50).unwrap();
        assert_eq!(bsz, 64);
        assert_eq!(cmask[1], 1);
        assert_eq!(cmask[3], 1);

        let y_dense = mat.project_batch(&x, 2, None, 2).unwrap();
        let (y_sparse, skipped) = mat
            .project_batch_sparse_ext(&x, 2, None, 2, Some(&rmask), None, true, true)
            .unwrap();
        assert!(skipped > 0, "expected skipped > 0 tiles on masked gate_proj");
        for tok in 0..2 {
            for (b, &active) in cmask.iter().enumerate() {
                let s0 = tok * d_out + b * block_size;
                let s1 = s0 + block_size;
                if active != 0 {
                    let max_err = y_dense[s0..s1]
                        .iter()
                        .zip(&y_sparse[s0..s1])
                        .map(|(a, c)| (a - c).abs())
                        .fold(0.0f32, f32::max);
                    assert!(max_err < 1e-5, "tok {tok} blk {b} max_err={max_err}");
                } else {
                    assert!(y_sparse[s0..s1].iter().all(|&v| v == 0.0));
                }
            }
        }
    }

    #[test]
    fn lgt1_bf16_roundtrip_and_hash_mismatch_fails_closed() {
        let ids = vec![2u32, 105, 818, 1354, 563];
        let hash = fnv1a64_ids(&ids);
        assert_ne!(hash, fnv1a64_ids(&[2u32, 105, 818, 1355, 563]));
        let n_pos = 2usize;
        let vocab = 8usize;
        let row0 = vec![-2.5f32, 0.0, 1.25, 3.5, -0.5, 4.0, -1.0, 0.75];
        let row1 = vec![0.125f32, -3.0, 2.5, -0.25, 1.0, 0.0, 5.25, -4.5];
        let path = std::env::temp_dir().join(format!("test_lgt1_{}.lgt1", std::process::id()));
        {
            let f = fs::File::create(&path).unwrap();
            let mut bw = BufWriter::new(f);
            write_lgt1_header(&mut bw, n_pos, vocab, hash).unwrap();
            write_lgt1_bf16_row(&mut bw, &row0).unwrap();
            write_lgt1_bf16_row(&mut bw, &row1).unwrap();
            bw.flush().unwrap();
        }
        assert!(Lgt1Reader::open(&path, n_pos, vocab, hash ^ 1).is_err());
        assert!(Lgt1Reader::open(&path, n_pos + 1, vocab, hash).is_err());
        let mut rd = Lgt1Reader::open(&path, n_pos, vocab, hash).unwrap();
        let mut buf = Vec::new();
        rd.read_next_row(&mut buf).unwrap();
        for (a, b) in buf.iter().zip(&row0) {
            assert!((a - b).abs() < 0.05);
        }
        rd.read_next_row(&mut buf).unwrap();
        for (a, b) in buf.iter().zip(&row1) {
            assert!((a - b).abs() < 0.05);
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn exact_teacher_kl_metrics_zero_self_and_positive_shifted() {
        let t = vec![0.5f32, 3.2, -1.0, 1.8, 0.0, 2.4, -0.3];
        let (kts_self, kst_self, a1_s, a5_s, tnll_s, snll_s) =
            compute_seq_kl_metrics(&t, &t, 1).unwrap();
        assert!(kts_self < 1e-10 && kst_self < 1e-10);
        assert!(a1_s && a5_s);
        assert!((tnll_s - snll_s).abs() < 1e-10);
        let (nll_only, hit1, ent, argmax_s) = eval_student_logits_stats(&t, 1).unwrap();
        assert!(hit1 && argmax_s == 1);
        assert!((nll_only - tnll_s).abs() < 1e-10);
        assert!(ent > 0.5);

        let s_shift = vec![3.5f32, 2.0, -1.0, 1.8, 0.0, 2.4, -0.3];
        let (kts, kst, a1, a5, _, _) = compute_seq_kl_metrics(&t, &s_shift, 1).unwrap();
        assert!(kts > 0.05 && kst > 0.05);
        assert!(!a1 && a5);
    }

    #[test]
    fn resolve_eval_nll_start_pos_handles_chat_turn_and_bounds() {
        let chat_ids = vec![
            2u32, 105, 2364, 107, 3689, 106, 107, 105, 4368, 107, 100, 45518, 107, 101, 818, 3890, 106,
        ];
        assert_eq!(resolve_eval_nll_start_pos(&chat_ids, None, false).unwrap(), 0);
        assert_eq!(resolve_eval_nll_start_pos(&chat_ids, None, true).unwrap(), 13);
        assert_eq!(resolve_eval_nll_start_pos(&chat_ids, Some(5), false).unwrap(), 5);

        let no_thought_chat = vec![2u32, 105, 2364, 107, 3689, 106, 107, 105, 4368, 107, 818, 106];
        assert_eq!(resolve_eval_nll_start_pos(&no_thought_chat, None, true).unwrap(), 9);

        assert!(resolve_eval_nll_start_pos(&[2], None, false).is_err());
        assert!(resolve_eval_nll_start_pos(&chat_ids, Some(1), true).is_err());
        assert!(resolve_eval_nll_start_pos(&chat_ids, Some(chat_ids.len() - 1), false).is_err());
        assert!(resolve_eval_nll_start_pos(&[2, 818, 5279, 529], None, true).is_err());
    }

    #[test]
    fn head_cmask_expansion_covers_sliding_and_global_head_dims() {
        let mut hcm = vec![0u8; 32];
        hcm[0] = 1;
        hcm[5] = 1;
        // Sliding layer: hd=256, o_bsize=256 -> 16 q-tiles/head, 1 o-blk/head
        let (q_s, o_s) = expand_head_mask(&hcm, 256, 256);
        assert_eq!(q_s.len(), 32 * 16);
        assert_eq!(o_s.len(), 32);
        assert!(q_s[0..16].iter().all(|&v| v == 1));
        assert!(q_s[16..80].iter().all(|&v| v == 0));
        assert!(q_s[80..96].iter().all(|&v| v == 1));
        assert_eq!(o_s[0], 1);
        assert_eq!(o_s[1], 0);
        assert_eq!(o_s[5], 1);
        // Global layer: hd=512, o_bsize=256 -> 32 q-tiles/head, 2 o-blks/head
        let (q_g, o_g) = expand_head_mask(&hcm, 512, 256);
        assert_eq!(q_g.len(), 32 * 32);
        assert_eq!(o_g.len(), 32 * 2);
        assert!(q_g[5 * 32..6 * 32].iter().all(|&v| v == 1));
        assert_eq!(&o_g[10..12], &[1u8, 1u8]);
        assert!(&o_g[2..4] == &[0u8, 0u8]);
    }

    #[test]
    fn union_col_masks_and_row_expansion_cover_every_k1_token_mask() {
        let n_blks = 84usize;
        let bsize = 256usize;
        let ff = n_blks * bsize;
        for seq_len in [1usize, 4, 8, 16] {
            let mut tok_masks = Vec::with_capacity(seq_len);
            for t in 0..seq_len {
                let mut m = vec![0u8; n_blks];
                m[0..4].fill(1);
                m[(4 + t * 5) % n_blks] = 1;
                m[(17 + t * 11) % n_blks] = 1;
                tok_masks.push(m);
            }
            let cm = union_col_masks(&tok_masks, n_blks);
            let rm = expand_col_to_row_mask(&cm, bsize, ff);
            assert_eq!(rm.len(), ff / 16);
            let tiles_per_blk = bsize / 16;
            for (t, tm) in tok_masks.iter().enumerate() {
                for (b, &act) in tm.iter().enumerate() {
                    if act != 0 {
                        assert_eq!(cm[b], 1, "union missed active block {b} at seq_len={seq_len} tok={t}");
                        assert!(rm[b * tiles_per_blk..(b + 1) * tiles_per_blk].iter().all(|&r| r == 1));
                    }
                }
            }
        }
    }

    #[test]
    fn glyph_kv_causal_prefill_matches_step_by_step_decode() {
        let seq_len = 80usize;
        let n_heads = 4usize;
        let n_kv = 2usize;
        let hd = 256usize;
        let q_stride = n_heads * hd;
        let kv_stride = n_kv * hd;
        let q_batch: Vec<f32> = (0..seq_len * q_stride)
            .map(|i| (((i * 17 + 5) % 997) as f32) * 0.002 - 1.0)
            .collect();
        let k_batch: Vec<f32> = (0..seq_len * kv_stride)
            .map(|i| (((i * 31 + 11) % 991) as f32) * 0.002 - 1.0)
            .collect();
        let v_batch: Vec<f32> = (0..seq_len * kv_stride)
            .map(|i| (((i * 43 + 19) % 983) as f32) * 0.002 - 1.0)
            .collect();

        let mut gkv_seq = GlyphKvLayer::new_with_lod(n_heads, n_kv, hd, false, None, 4, 16, 0, 256).unwrap();
        let mut expected_ctx = Vec::with_capacity(seq_len * q_stride);
        for t in 0..seq_len {
            let kt = &k_batch[t * kv_stride..(t + 1) * kv_stride];
            let vt = &v_batch[t * kv_stride..(t + 1) * kv_stride];
            gkv_seq.append_token(kt, vt, None).unwrap();
            let q_t = &q_batch[t * q_stride..(t + 1) * q_stride];
            let c_t = gkv_seq.attend_range(q_t, t, t, t, 1.0, None, 1).unwrap();
            expected_ctx.extend_from_slice(&c_t);
        }

        let mut gkv_batch = GlyphKvLayer::new_with_lod(n_heads, n_kv, hd, false, None, 4, 16, 0, 256).unwrap();
        let mut actual_ctx = vec![0.0f32; seq_len * q_stride];
        prefill_glyph_kv_causal_batch(
            &mut gkv_batch,
            KvQuantMode::Glyph,
            &q_batch,
            &k_batch,
            &v_batch,
            seq_len,
            0,
            n_heads,
            n_kv,
            hd,
            None,
            true,
            None,
            None,
            2,
            &mut actual_ctx,
        )
        .unwrap();

        assert_eq!(actual_ctx.len(), expected_ctx.len());
        for (idx, (&a, &b)) in actual_ctx.iter().zip(expected_ctx.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "bit mismatch at float index {idx}: {a} != {b}");
        }
        assert_eq!(gkv_batch.total_seen(), seq_len);
    }

    #[test]
    fn causal_self_attn_independent_trajs_matches_single_trajectory_bit_for_bit() {
        let traj_len = 7usize;
        let n_trajs = 3usize;
        let seq_len = n_trajs * traj_len;
        let n_heads = 4usize;
        let n_kv = 2usize;
        let hd = 16usize;
        let sw = Some(4usize);
        let q_stride = n_heads * hd;
        let kv_stride = n_kv * hd;
        let q_batch: Vec<f32> = (0..seq_len * q_stride)
            .map(|i| (((i * 29 + 3) % 887) as f32) * 0.003 - 1.2)
            .collect();
        let k_batch: Vec<f32> = (0..seq_len * kv_stride)
            .map(|i| (((i * 37 + 7) % 881) as f32) * 0.003 - 1.1)
            .collect();
        let v_batch: Vec<f32> = (0..seq_len * kv_stride)
            .map(|i| (((i * 41 + 13) % 877) as f32) * 0.003 - 1.0)
            .collect();

        let mut packed_ctx = vec![0.0f32; seq_len * q_stride];
        causal_self_attn_independent_trajs(
            &q_batch,
            &k_batch,
            &v_batch,
            seq_len,
            traj_len,
            0,
            n_heads,
            n_kv,
            hd,
            sw,
            2,
            &mut packed_ctx,
        )
        .unwrap();

        for b in 0..n_trajs {
            let kb = &k_batch[b * traj_len * kv_stride..(b + 1) * traj_len * kv_stride];
            let vb = &v_batch[b * traj_len * kv_stride..(b + 1) * traj_len * kv_stride];
            for lt in 0..traj_len {
                let q_t = &q_batch[(b * traj_len + lt) * q_stride..(b * traj_len + lt + 1) * q_stride];
                let exp = attn_decode_range(q_t, kb, vb, lt, lt, lt, n_heads, n_kv, hd, sw);
                let got = &packed_ctx[(b * traj_len + lt) * q_stride..(b * traj_len + lt + 1) * q_stride];
                for (i, (&g, &e)) in got.iter().zip(exp.iter()).enumerate() {
                    assert_eq!(g.to_bits(), e.to_bits(), "traj {b} tok {lt} dim {i}: {g} != {e}");
                }
            }
        }
        assert!(causal_self_attn_independent_trajs(&q_batch, &k_batch, &v_batch, seq_len, 4, 0, n_heads, n_kv, hd, sw, 1, &mut packed_ctx).is_err());
    }
}
