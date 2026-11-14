//! run31b — bounded-RAM native Rust inference runtime for Gemma 4 31B QAT W4A16 (zero Python).
//!
//! Executes all 60 dense Gemma 4 transformer layers (compressed-tensors symmetric group-32 INT4)
//! directly on 8 GB edge hardware (Raspberry Pi 4 ARM64) using:
//!   - Batched prefill across each layer so an N-token prompt reads SSD weights only once.
//!   - Streaming single-projection lifetime (`<= 90 MB` layer scratch) plus optional `--pin-layers N`.
//!   - Configurable embedding/LM-head residency (`stream` ~44 MB scratch, `bf16` 2.82 GB, or
//!     `q4`/`q6`/`q8` sub-byte packed via `run270_rust::packed::PackedMatrix`).

use run270_rust::codec::bf16_to_f32;
use run270_rust::json;
use run270_rust::math::{attn_decode, gelu_tanh_scalar, qk_norm_raw, rmsn_raw, rope_head};
use run270_rust::packed::{PackedMatrix, QuantScheme};
use run270_rust::safetensors::SafeTensors;
use run270_rust::tokenizer::Tokenizer;
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

struct Runtime31B {
    cfg: Gemma4Cfg,
    sf: Option<SafeTensors>,
    norms: HashMap<String, Vec<f32>>,
    scalars: Vec<f32>,
    embed: Embed31B,
    pinned: Vec<PinnedLayer>,
    t16c_mats: Option<HashMap<String, run270_rust::t16c::T16CMat>>,
    native_t16c: Option<std::sync::Arc<run270_rust::t16c::NativeTile16Kernel>>,
    kv: Vec<LayerKv>,
    threads: usize,
    ssd_bytes_read: u64,
}

impl Runtime31B {
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
        let d = self.cfg.d;
        let eps = self.cfg.eps;
        let n_heads = self.cfg.n_heads;
        let mut states = self.embed_batch(ids)?;

        let cfg = &self.cfg;
        let sf_opt = &self.sf;
        let norms = &self.norms;
        let scalars = &self.scalars;
        let pinned = &self.pinned;
        let t16c_mats = &self.t16c_mats;
        let native_t16c = &self.native_t16c;
        let kv = &mut self.kv;
        let threads = self.threads;
        let mut read_bytes = 0u64;

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

            let mut run_proj = |layer: usize, proj_suffix: &'static str, x_batch: &[f32]| -> Result<Vec<f32>, String> {
                if let Some(mats) = t16c_mats {
                    let k1 = format!("model.language_model.layers.{layer}.{proj_suffix}.weight");
                    let mat = mats
                        .get(&k1)
                        .or_else(|| mats.get(&format!("model.layers.{layer}.{proj_suffix}.weight")))
                        .ok_or_else(|| format!("T16C missing {k1}"))?;
                    return mat.project_batch(x_batch, seq_len, native_t16c.as_deref(), threads);
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

                let mut q_batch = run_proj(l, "self_attn.q_proj", &inp)?;
                let k_raw = run_proj(l, "self_attn.k_proj", &inp)?;
                let mut v_batch = if is_full && cfg.attention_k_eq_v {
                    k_raw.clone()
                } else {
                    run_proj(l, "self_attn.v_proj", &inp)?
                };
                drop(inp);

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

                let base_t = kv[l].k.len() / (n_kv * hd);
                kv[l].k.extend_from_slice(&k_batch);
                kv[l].v.extend_from_slice(&v_batch);
                let mut ctx_batch = Vec::with_capacity(seq_len * n_heads * hd);
                for t in 0..seq_len {
                    let q_t = &q_batch[t * n_heads * hd..(t + 1) * n_heads * hd];
                    let c_t = attn_decode(q_t, &kv[l].k, &kv[l].v, base_t + t, start_pos + t, n_heads, n_kv, hd, sw);
                    ctx_batch.extend_from_slice(&c_t);
                }
                drop(q_batch);
                drop(k_batch);
                drop(v_batch);

                let attn_o = run_proj(l, "self_attn.o_proj", &ctx_batch)?;
                drop(ctx_batch);
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
                let gate = run_proj(l, "mlp.gate_proj", &inp2)?;
                let up = run_proj(l, "mlp.up_proj", &inp2)?;
                drop(inp2);
                let act: Vec<f32> = gate.iter().zip(up.iter()).map(|(&a, &b)| gelu_tanh_scalar(a) * b).collect();
                drop(gate);
                drop(up);
                let down = run_proj(l, "mlp.down_proj", &act)?;
                drop(act);

                let g_pff = get_norm(&format!("{pfx}.post_feedforward_layernorm.weight"))?;
                let ls = scalars[l];
                for t in 0..seq_len {
                    let norm_d = rmsn_raw(&down[t * d..(t + 1) * d], Some(g_pff), eps);
                    for i in 0..d {
                        states[t * d + i] = (states[t * d + i] + norm_d[i]) * ls;
                    }
                }
                if verbose_layers && (l == 0 || l == 5 || l % 15 == 14 || l + 1 == cfg.n_layers) {
                    eprintln!("[run31b] layer {l:02}/{} done in {:.2}s (pinned={})", cfg.n_layers, tl.elapsed().as_secs_f64(), l < pinned.len());
                }
            }
            Ok(())
        })?;
        self.ssd_bytes_read += read_bytes;

        let fn_gain = self.norm("model.language_model.norm.weight")?;
        if all_final {
            let mut all_out = Vec::with_capacity(seq_len * d);
            for t in 0..seq_len {
                all_out.extend_from_slice(&rmsn_raw(&states[t * d..(t + 1) * d], Some(fn_gain), eps));
            }
            Ok(all_out)
        } else {
            let last_state = &states[(seq_len - 1) * d..seq_len * d];
            Ok(rmsn_raw(last_state, Some(fn_gain), eps))
        }
    }

    fn compute_logits(&mut self, hidden: &[f32]) -> Result<Vec<f32>, String> {
        self.compute_logits_batch(hidden, 1)
    }

    fn compute_logits_batch(&mut self, hidden_batch: &[f32], seq_len: usize) -> Result<Vec<f32>, String> {
        let d = self.cfg.d;
        let vocab = self.cfg.vocab;
        if hidden_batch.len() != seq_len * d {
            return Err(format!("hidden_batch len {} != {seq_len}*{d}", hidden_batch.len()));
        }
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
            Embed31B::T16C { mat, native } => mat.project_batch(hidden_batch, seq_len, native.as_deref(), self.threads)?,
        };
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
    Ok((
        Runtime31B {
            cfg,
            sf: Some(sf),
            norms,
            scalars,
            embed,
            pinned,
            t16c_mats: None,
            native_t16c: None,
            kv,
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

    let body_bytes: usize = t16c.mats.values().map(|m| m.total_bytes()).sum();
    eprintln!(
        "[run31b] weight_source=T16C native_ffi={} mats={} norms={} body_bytes={} norm_bytes={} embed_bytes={}",
        native_t16c.as_ref().map_or("disabled", |n| n.path.as_str()),
        t16c.mats.len(),
        t16c.norms.len(),
        body_bytes,
        norm_bytes,
        embed_bytes
    );

    let kv = (0..cfg.n_layers).map(|_| LayerKv { k: Vec::new(), v: Vec::new() }).collect();
    Ok((
        Runtime31B {
            cfg,
            sf: sf_opt,
            norms: t16c.norms,
            scalars,
            embed,
            pinned: Vec::new(),
            t16c_mats: Some(t16c.mats),
            native_t16c,
            kv,
            threads,
            ssd_bytes_read: 0,
        },
        body_bytes + norm_bytes,
        embed_bytes,
    ))
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let get = |name: &str| -> Option<String> {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
    };
    let t16c_arg = get("--t16c");
    let model = get("--model");
    if t16c_arg.is_none() && model.is_none() {
        return Err("missing --model (or pass --t16c <path>)".into());
    }
    let so_arg = get("--so");
    let disable_native = args.iter().any(|a| a == "--no-native-ffi");
    let batch_k: usize = match get("--batch-k") {
        Some(s) => s.parse().map_err(|_| format!("invalid --batch-k '{s}'"))?,
        None => 1,
    };
    if batch_k == 0 {
        return Err("--batch-k must be >= 1".into());
    }
    let cfgp = get("--config").ok_or("missing --config")?;
    let tkp = get("--tokenizer").ok_or("missing --tokenizer")?;
    let threads: usize = match get("--threads") {
        Some(s) => s.parse().map_err(|_| format!("invalid --threads '{s}'"))?,
        None => std::thread::available_parallelism().map(|n| n.get().clamp(1, 8)).unwrap_or(4),
    };
    let embed_h8 = args.iter().any(|a| a == "--embed-h8");
    let embed_mode = EmbedMode31B::parse(&get("--embed-mode").unwrap_or_else(|| "stream".into()), embed_h8)?;
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
    let dump_logits = get("--dump-logits");
    let ids_arg = get("--ids");

    let cfg = load_gemma4_cfg(Path::new(&cfgp))?;
    eprintln!(
        "[run31b] cfg d={} layers={} ff={} heads={}/{} global_kv={} hd={}/{} window={} softcap={:?}",
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
            threads,
        )?
    } else {
        load_runtime_31b(Path::new(model.as_deref().unwrap()), cfg, embed_mode, pin_layers, threads)?
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
    let ids: Vec<u32> = match ids_arg {
        Some(s) => s
            .split(',')
            .map(|x| x.trim().parse::<u32>().map_err(|_| format!("invalid id in --ids '{s}'")))
            .collect::<Result<Vec<_>, _>>()?,
        None => {
            let prompt = if raw_prompt {
                prompt_text.clone()
            } else {
                format!("<|turn>user\n{prompt_text}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
            };
            tk.encode(&prompt, true)
        }
    };
    eprintln!("[run31b] prompt {} tokens ids={ids:?}", ids.len());

    let t_pf = std::time::Instant::now();
    let mut hidden = rt.forward_batch(&ids, 0, true)?;
    let mut logits = rt.compute_logits(&hidden)?;
    let prefill_s = t_pf.elapsed().as_secs_f64();
    eprintln!("[run31b] prefill done in {prefill_s:.2}s ssd_read_mb={:.1} vm={:?}", rt.ssd_bytes_read as f64 / 1e6, proc_status_kb());

    if let Some(dp) = dump_logits {
        let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        fs::write(&dp, bytes).map_err(|e| format!("write {dp}: {e}"))?;
    }

    let mut top5: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    top5.select_nth_unstable_by(4, |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    top5.truncate(5);
    top5.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top5_str: Vec<String> = top5
        .iter()
        .map(|&(id, sc)| format!("{id}:{:?}({sc:.3})", tk.decode(&[id as u32])))
        .collect();
    println!("PREFILL_TOP5 [{}]", top5_str.join(", "));

    let eos: [u32; 2] = [1, 106];
    let mut generated = Vec::new();
    let t_dec = std::time::Instant::now();
    for step in 0..max_tokens {
        let nxt = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i) as u32;
        generated.push(nxt);
        let tok_txt = tk.decode(&[nxt]);
        println!("[STEP {step}] id={nxt} text={tok_txt:?}");
        if eos.contains(&nxt) {
            break;
        }
        if step + 1 < max_tokens {
            let td = std::time::Instant::now();
            hidden = rt.forward_batch(&[nxt], ids.len() + step, false)?;
            logits = rt.compute_logits(&hidden)?;
            eprintln!("[run31b] decode step {step} in {:.2}s vm={:?}", td.elapsed().as_secs_f64(), proc_status_kb());
        }
    }
    let decode_s = t_dec.elapsed().as_secs_f64();
    if batch_k > 1 {
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
    println!(
        "MEM31B pinned_bytes={pinned_bytes} embed_bytes={embed_bytes} ssd_read_bytes={} prefill_s={prefill_s:.2} decode_s={decode_s:.2} vm_hwm_kb={vm_hwm:?} vm_rss_kb={vm_rss:?}",
        rt.ssd_bytes_read
    );
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
}
