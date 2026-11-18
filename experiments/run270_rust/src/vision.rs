//! Gemma 4 Multimodal Vision Tower (`Gemma4VisionModel` + `Gemma4MultimodalEmbedder`)
//! and Optical VA Self-Analysis Frame renderer/loader for native `.t16c` + `C8DB` inference.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::Command;

use crate::cuneiform_tbi::CuneiformBank;
use crate::math::{gelu_tanh_scalar, qk_norm_raw, rmsn_raw};
use crate::t16c::{NativeTile16Kernel, T16CMat, T16CModel};

/// Gemma 4 Multimodal special token IDs (`google/gemma-4-31B-it-qat-w4a16-ct/config.json`).
pub const GEMMA4_BOI_TOKEN_ID: u32 = 255_999;
pub const GEMMA4_IMAGE_TOKEN_ID: u32 = 258_880;
pub const GEMMA4_EOI_TOKEN_ID: u32 = 258_882;

/// Architecture constants for Gemma 4 31B's 27-layer ViT encoder and multimodal projector.
#[derive(Clone, Debug)]
pub struct VisionCfg {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub max_soft_tokens: usize,
    pub position_embedding_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub standardize: bool,
    pub text_hidden_size: usize,
}

impl VisionCfg {
    pub fn gemma4_31b() -> Self {
        Self {
            hidden_size: 1152,
            intermediate_size: 4304,
            num_hidden_layers: 27,
            num_heads: 16,
            head_dim: 72,
            patch_size: 16,
            pooling_kernel_size: 3,
            max_soft_tokens: 280,
            position_embedding_size: 10240,
            rms_norm_eps: 1e-6,
            rope_theta: 100.0,
            standardize: true,
            text_hidden_size: 5376,
        }
    }

    pub fn max_patches(&self) -> usize {
        self.max_soft_tokens * self.pooling_kernel_size * self.pooling_kernel_size
    }
}

/// Raw 8-bit RGB image buffer (`height * width * 3` row-major).
#[derive(Clone, Debug)]
pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

/// Position table storage extracted from `.t16c` (quantized matrices or f32 norm fallback).
pub enum PosTableBackend {
    SplitMats { x: T16CMat, y: T16CMat },
    NormF32(Vec<f32>),
}

/// Quantized weights for one Gemma 4 Vision encoder layer (`0..26`).
pub struct VisionLayerWeights {
    pub in_norm: Vec<f32>,
    pub post_attn_norm: Vec<f32>,
    pub pre_ff_norm: Vec<f32>,
    pub post_ff_norm: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub q_proj: T16CMat,
    pub k_proj: T16CMat,
    pub v_proj: T16CMat,
    pub o_proj: T16CMat,
    pub gate_proj: T16CMat,
    pub up_proj: T16CMat,
    pub down_proj: T16CMat,
}

/// Complete quantized Vision Tower + `embed_vision` weights extracted from `.t16c`.
pub struct VisionWeights {
    pub cfg: VisionCfg,
    pub input_proj: T16CMat,
    pub pos_table: PosTableBackend,
    pub layers: Vec<VisionLayerWeights>,
    pub std_bias: Vec<f32>,
    pub std_scale: Vec<f32>,
    pub embed_proj: T16CMat,
}

fn take_mat_any(mats: &mut HashMap<String, T16CMat>, keys: &[&str]) -> Result<T16CMat, String> {
    for &k in keys {
        if let Some(m) = mats.remove(k) {
            return Ok(m);
        }
    }
    Err(format!("Missing vision T16C matrix: tried {:?}", keys))
}

fn clone_norm_any(norms: &HashMap<String, Vec<f32>>, keys: &[&str]) -> Result<Vec<f32>, String> {
    for &k in keys {
        if let Some(v) = norms.get(k) {
            return Ok(v.clone());
        }
    }
    Err(format!("Missing vision T16C norm: tried {:?}", keys))
}

impl VisionWeights {
    /// Extract all Vision Tower + `embed_vision` matrices from `t16c` if present.
    /// Returns `Ok(None)` cleanly for text-only `.t16c` archives.
    pub fn try_load_from_t16c(t16c: &mut T16CModel, cfg: VisionCfg) -> Result<Option<Self>, String> {
        let in_key = "model.vision_tower.patch_embedder.input_proj.weight";
        let Some(input_proj) = t16c.mats.remove(in_key) else {
            return Ok(None);
        };

        let pos_table = if let (Some(mx), Some(my)) = (
            t16c.mats.remove("model.vision_tower.patch_embedder.position_embedding_table.0"),
            t16c.mats.remove("model.vision_tower.patch_embedder.position_embedding_table.1"),
        ) {
            PosTableBackend::SplitMats { x: mx, y: my }
        } else {
            let v = clone_norm_any(
                &t16c.norms,
                &["model.vision_tower.patch_embedder.position_embedding_table"],
            )?;
            let expected = 2 * cfg.position_embedding_size * cfg.hidden_size;
            if v.len() != expected {
                return Err(format!(
                    "position_embedding_table norm len mismatch: got {}, expected {expected}",
                    v.len()
                ));
            }
            PosTableBackend::NormF32(v)
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let pfx = format!("model.vision_tower.encoder.layers.{l}");
            layers.push(VisionLayerWeights {
                in_norm: clone_norm_any(&t16c.norms, &[&format!("{pfx}.input_layernorm.weight")])?,
                post_attn_norm: clone_norm_any(&t16c.norms, &[&format!("{pfx}.post_attention_layernorm.weight")])?,
                pre_ff_norm: clone_norm_any(&t16c.norms, &[&format!("{pfx}.pre_feedforward_layernorm.weight")])?,
                post_ff_norm: clone_norm_any(&t16c.norms, &[&format!("{pfx}.post_feedforward_layernorm.weight")])?,
                q_norm: clone_norm_any(&t16c.norms, &[&format!("{pfx}.self_attn.q_norm.weight")])?,
                k_norm: clone_norm_any(&t16c.norms, &[&format!("{pfx}.self_attn.k_norm.weight")])?,
                q_proj: take_mat_any(
                    &mut t16c.mats,
                    &[&format!("{pfx}.self_attn.q_proj.linear.weight"), &format!("{pfx}.self_attn.q_proj.weight")],
                )?,
                k_proj: take_mat_any(
                    &mut t16c.mats,
                    &[&format!("{pfx}.self_attn.k_proj.linear.weight"), &format!("{pfx}.self_attn.k_proj.weight")],
                )?,
                v_proj: take_mat_any(
                    &mut t16c.mats,
                    &[&format!("{pfx}.self_attn.v_proj.linear.weight"), &format!("{pfx}.self_attn.v_proj.weight")],
                )?,
                o_proj: take_mat_any(
                    &mut t16c.mats,
                    &[&format!("{pfx}.self_attn.o_proj.linear.weight"), &format!("{pfx}.self_attn.o_proj.weight")],
                )?,
                gate_proj: take_mat_any(
                    &mut t16c.mats,
                    &[&format!("{pfx}.mlp.gate_proj.linear.weight"), &format!("{pfx}.mlp.gate_proj.weight")],
                )?,
                up_proj: take_mat_any(
                    &mut t16c.mats,
                    &[&format!("{pfx}.mlp.up_proj.linear.weight"), &format!("{pfx}.mlp.up_proj.weight")],
                )?,
                down_proj: take_mat_any(
                    &mut t16c.mats,
                    &[&format!("{pfx}.mlp.down_proj.linear.weight"), &format!("{pfx}.mlp.down_proj.weight")],
                )?,
            });
        }

        let (std_bias, std_scale) = if cfg.standardize {
            (
                clone_norm_any(&t16c.norms, &["model.vision_tower.std_bias"])?,
                clone_norm_any(&t16c.norms, &["model.vision_tower.std_scale"])?,
            )
        } else {
            (vec![0.0; cfg.hidden_size], vec![1.0; cfg.hidden_size])
        };
        let embed_proj = take_mat_any(&mut t16c.mats, &["model.embed_vision.embedding_projection.weight"])?;

        Ok(Some(Self {
            cfg,
            input_proj,
            pos_table,
            layers,
            std_bias,
            std_scale,
            embed_proj,
        }))
    }

    fn apply_c8d_if_present(
        c8d: Option<&CuneiformBank>,
        keys: &[&str],
        x: &[f32],
        y: &mut [f32],
        k_tokens: usize,
        threads: usize,
    ) -> Result<(), String> {
        let Some(bank) = c8d else {
            return Ok(());
        };
        for &k in keys {
            if let Some(entry) = bank.entries.get(k) {
                if x.len() == k_tokens * entry.d_in && y.len() == k_tokens * entry.d_out {
                    entry.apply_batch_into(x, k_tokens, y, bank.native.as_ref(), threads, None)?;
                    break;
                }
            }
        }
        Ok(())
    }

    fn proj_with_c8d(
        mat: &T16CMat,
        c8d: Option<&CuneiformBank>,
        c8d_keys: &[&str],
        x: &[f32],
        k_tokens: usize,
        native: Option<&NativeTile16Kernel>,
        threads: usize,
    ) -> Result<Vec<f32>, String> {
        let mut y = mat.project_batch(x, k_tokens, native, threads)?;
        Self::apply_c8d_if_present(c8d, c8d_keys, x, &mut y, k_tokens, threads)?;
        Ok(y)
    }

    /// Encode `img` into `[n_soft_tokens * text_hidden_size]` float32 soft-token embeddings
    /// matching HuggingFace `Gemma4VisionModel` + `Gemma4MultimodalEmbedder`.
    pub fn encode_image(
        &self,
        img: &RgbImage,
        native: Option<&NativeTile16Kernel>,
        c8d: Option<&CuneiformBank>,
        threads: usize,
    ) -> Result<(usize, Vec<f32>), String> {
        let (patches, pos_xy, pw, ph) = prepare_vision_patches(img, &self.cfg)?;
        let n_patches = patches.len() / (self.cfg.patch_size * self.cfg.patch_size * 3);
        let d = self.cfg.hidden_size;

        let scaled_pix: Vec<f32> = patches.iter().map(|&v| 2.0 * (v - 0.5)).collect();
        let mut hidden = Self::proj_with_c8d(
            &self.input_proj,
            c8d,
            &["model.vision_tower.patch_embedder.input_proj.weight"],
            &scaled_pix,
            n_patches,
            native,
            threads,
        )?;
        self.add_position_embeddings(&mut hidden, &pos_xy, pw, ph)?;

        let rope = precompute_axial_2d_rope(&pos_xy, self.cfg.head_dim, self.cfg.rope_theta);

        let n_heads = self.cfg.num_heads;
        let hd = self.cfg.head_dim;
        let eps = self.cfg.rms_norm_eps;
        for (l, lw) in self.layers.iter().enumerate() {
            let res_attn = hidden.clone();
            let mut h_norm = Vec::with_capacity(n_patches * d);
            for t in 0..n_patches {
                h_norm.extend_from_slice(&rmsn_raw(&hidden[t * d..(t + 1) * d], Some(&lw.in_norm), eps));
            }
            let pfx = format!("model.vision_tower.encoder.layers.{l}");
            let mut q = Self::proj_with_c8d(
                &lw.q_proj,
                c8d,
                &[&format!("{pfx}.self_attn.q_proj.linear.weight"), &format!("{pfx}.self_attn.q_proj.weight")],
                &h_norm,
                n_patches,
                native,
                threads,
            )?;
            let mut k = Self::proj_with_c8d(
                &lw.k_proj,
                c8d,
                &[&format!("{pfx}.self_attn.k_proj.linear.weight"), &format!("{pfx}.self_attn.k_proj.weight")],
                &h_norm,
                n_patches,
                native,
                threads,
            )?;
            let mut v = Self::proj_with_c8d(
                &lw.v_proj,
                c8d,
                &[&format!("{pfx}.self_attn.v_proj.linear.weight"), &format!("{pfx}.self_attn.v_proj.weight")],
                &h_norm,
                n_patches,
                native,
                threads,
            )?;

            for t in 0..n_patches {
                let s = t * d;
                let e = s + d;
                let qt = qk_norm_raw(&q[s..e], n_heads, hd, Some(&lw.q_norm), eps);
                let kt = qk_norm_raw(&k[s..e], n_heads, hd, Some(&lw.k_norm), eps);
                let vt = qk_norm_raw(&v[s..e], n_heads, hd, None, eps);
                q[s..e].copy_from_slice(&qt);
                k[s..e].copy_from_slice(&kt);
                v[s..e].copy_from_slice(&vt);
                apply_2d_rope_token(&mut q[s..e], &rope[t], n_heads, hd);
                apply_2d_rope_token(&mut k[s..e], &rope[t], n_heads, hd);
            }

            let ctx = bidirectional_attention(&q, &k, &v, n_patches, n_heads, hd, threads);
            let o_out = Self::proj_with_c8d(
                &lw.o_proj,
                c8d,
                &[&format!("{pfx}.self_attn.o_proj.linear.weight"), &format!("{pfx}.self_attn.o_proj.weight")],
                &ctx,
                n_patches,
                native,
                threads,
            )?;
            for t in 0..n_patches {
                let pn = rmsn_raw(&o_out[t * d..(t + 1) * d], Some(&lw.post_attn_norm), eps);
                for i in 0..d {
                    hidden[t * d + i] = res_attn[t * d + i] + pn[i];
                }
            }

            let res_mlp = hidden.clone();
            let mut h_pf = Vec::with_capacity(n_patches * d);
            for t in 0..n_patches {
                h_pf.extend_from_slice(&rmsn_raw(&hidden[t * d..(t + 1) * d], Some(&lw.pre_ff_norm), eps));
            }
            let gate = Self::proj_with_c8d(
                &lw.gate_proj,
                c8d,
                &[&format!("{pfx}.mlp.gate_proj.linear.weight"), &format!("{pfx}.mlp.gate_proj.weight")],
                &h_pf,
                n_patches,
                native,
                threads,
            )?;
            let up = Self::proj_with_c8d(
                &lw.up_proj,
                c8d,
                &[&format!("{pfx}.mlp.up_proj.linear.weight"), &format!("{pfx}.mlp.up_proj.weight")],
                &h_pf,
                n_patches,
                native,
                threads,
            )?;
            let inter = self.cfg.intermediate_size;
            let down_in = lw.down_proj.d_in;
            if down_in < inter {
                return Err(format!("vision down_proj d_in={down_in} < intermediate_size={inter}"));
            }
            let mut mid = vec![0f32; n_patches * down_in];
            for t in 0..n_patches {
                for j in 0..inter {
                    mid[t * down_in + j] = gelu_tanh_scalar(gate[t * inter + j]) * up[t * inter + j];
                }
            }
            let down_out = Self::proj_with_c8d(
                &lw.down_proj,
                c8d,
                &[&format!("{pfx}.mlp.down_proj.linear.weight"), &format!("{pfx}.mlp.down_proj.weight")],
                &mid,
                n_patches,
                native,
                threads,
            )?;
            for t in 0..n_patches {
                let pf = rmsn_raw(&down_out[t * d..(t + 1) * d], Some(&lw.post_ff_norm), eps);
                for i in 0..d {
                    hidden[t * d + i] = res_mlp[t * d + i] + pf[i];
                }
            }
        }

        let k_pool = self.cfg.pooling_kernel_size;
        let sw = pw / k_pool;
        let sh = ph / k_pool;
        let n_soft = sw * sh;
        let k2_inv = 1.0f32 / (k_pool * k_pool) as f32;
        let root_d = (d as f32).sqrt();
        let mut pooled = vec![0f32; n_soft * d];
        for (p_idx, &[px_i, py_i]) in pos_xy.iter().enumerate() {
            let bx = (px_i as usize) / k_pool;
            let by = (py_i as usize) / k_pool;
            let b_idx = bx + sw * by;
            let src = &hidden[p_idx * d..(p_idx + 1) * d];
            let dst = &mut pooled[b_idx * d..(b_idx + 1) * d];
            for c in 0..d {
                dst[c] += src[c] * k2_inv;
            }
        }
        for s in 0..n_soft {
            for c in 0..d {
                let mut v = pooled[s * d + c] * root_d;
                if self.cfg.standardize {
                    v = (v - self.std_bias[c]) * self.std_scale[c];
                }
                pooled[s * d + c] = v;
            }
        }

        let mut pre_proj = Vec::with_capacity(n_soft * d);
        for s in 0..n_soft {
            pre_proj.extend_from_slice(&rmsn_raw(&pooled[s * d..(s + 1) * d], None, eps));
        }
        let soft_tokens = Self::proj_with_c8d(
            &self.embed_proj,
            c8d,
            &["model.embed_vision.embedding_projection.weight"],
            &pre_proj,
            n_soft,
            native,
            threads,
        )?;
        Ok((n_soft, soft_tokens))
    }

    fn add_position_embeddings(
        &self,
        hidden: &mut [f32],
        pos_xy: &[[i32; 2]],
        pw: usize,
        ph: usize,
    ) -> Result<(), String> {
        let d = self.cfg.hidden_size;
        match &self.pos_table {
            PosTableBackend::SplitMats { x, y } => {
                let mut row_x = Vec::with_capacity(pw);
                for px in 0..pw {
                    row_x.push(x.decode_row(px)?);
                }
                let mut row_y = Vec::with_capacity(ph);
                for py in 0..ph {
                    row_y.push(y.decode_row(py)?);
                }
                for (t, &[px_i, py_i]) in pos_xy.iter().enumerate() {
                    let rx = &row_x[px_i as usize];
                    let ry = &row_y[py_i as usize];
                    let dst = &mut hidden[t * d..(t + 1) * d];
                    for c in 0..d {
                        dst[c] += rx[c] + ry[c];
                    }
                }
            }
            PosTableBackend::NormF32(tbl) => {
                let stride = self.cfg.position_embedding_size * d;
                for (t, &[px_i, py_i]) in pos_xy.iter().enumerate() {
                    let ox = (px_i as usize) * d;
                    let oy = stride + (py_i as usize) * d;
                    let dst = &mut hidden[t * d..(t + 1) * d];
                    for c in 0..d {
                        dst[c] += tbl[ox + c] + tbl[oy + c];
                    }
                }
            }
        }
        Ok(())
    }
}

/// Exact `get_aspect_ratio_preserving_size` from `Gemma4ImageProcessor`.
pub fn aspect_ratio_preserving_size(
    height: usize,
    width: usize,
    patch_size: usize,
    max_patches: usize,
    pooling_kernel_size: usize,
) -> Result<(usize, usize), String> {
    if height == 0 || width == 0 {
        return Err(format!("Invalid zero-area image {width}x{height}"));
    }
    let total_px = (height as f64) * (width as f64);
    let target_px = (max_patches * patch_size * patch_size) as f64;
    let factor = (target_px / total_px).sqrt();
    let side_mult = pooling_kernel_size * patch_size;
    let mut target_h = ((factor * height as f64) / side_mult as f64).floor() as usize * side_mult;
    let mut target_w = ((factor * width as f64) / side_mult as f64).floor() as usize * side_mult;
    if target_h == 0 && target_w == 0 {
        return Err(format!("Image {width}x{height} too small to resize to multiple of {side_mult}"));
    }
    let max_side = (max_patches / (pooling_kernel_size * pooling_kernel_size)) * side_mult;
    if target_h == 0 {
        target_h = side_mult;
        target_w = (((width as f64) / (height as f64)).floor() as usize * side_mult).min(max_side).max(side_mult);
    } else if target_w == 0 {
        target_w = side_mult;
        target_h = (((height as f64) / (width as f64)).floor() as usize * side_mult).min(max_side).max(side_mult);
    }
    if (target_h / patch_size) * (target_w / patch_size) > max_patches {
        return Err(format!("Resized {target_h}x{target_w} exceeds max_patches={max_patches}"));
    }
    Ok((target_h, target_w))
}

#[inline]
fn cubic_weight(x: f32) -> f32 {
    let a = -0.5f32;
    let ax = x.abs();
    if ax <= 1.0 {
        ((a + 2.0) * ax - (a + 3.0)) * ax * ax + 1.0
    } else if ax < 2.0 {
        ((a * ax - 5.0 * a) * ax + 8.0 * a) * ax - 4.0 * a
    } else {
        0.0
    }
}

/// Resize `img` to `(target_h, target_w)` via bicubic resampling in `[0.0, 1.0]` RGB f32.
pub fn resize_bicubic_f32(img: &RgbImage, target_h: usize, target_w: usize) -> Vec<f32> {
    if img.height == target_h && img.width == target_w {
        return img.data.iter().map(|&b| b as f32 / 255.0).collect();
    }
    let mut out = vec![0f32; target_h * target_w * 3];
    let sy = img.height as f32 / target_h as f32;
    let sx = img.width as f32 / target_w as f32;
    for oy in 0..target_h {
        let fy = (oy as f32 + 0.5) * sy - 0.5;
        let iy0 = fy.floor() as isize;
        let wy: [f32; 4] = core::array::from_fn(|k| cubic_weight(fy - (iy0 + k as isize - 1) as f32));
        for ox in 0..target_w {
            let fx = (ox as f32 + 0.5) * sx - 0.5;
            let ix0 = fx.floor() as isize;
            let wx: [f32; 4] = core::array::from_fn(|k| cubic_weight(fx - (ix0 + k as isize - 1) as f32));
            let mut acc = [0f32; 3];
            let mut wsum = 0f32;
            for ky in 0..4 {
                let sy_i = (iy0 + ky as isize - 1).clamp(0, img.height as isize - 1) as usize;
                for kx in 0..4 {
                    let sx_i = (ix0 + kx as isize - 1).clamp(0, img.width as isize - 1) as usize;
                    let w = wy[ky] * wx[kx];
                    wsum += w;
                    let idx = (sy_i * img.width + sx_i) * 3;
                    for c in 0..3 {
                        acc[c] += w * (img.data[idx + c] as f32 / 255.0);
                    }
                }
            }
            let norm = if wsum.abs() > 1e-8 { 1.0 / wsum } else { 1.0 };
            let oidx = (oy * target_w + ox) * 3;
            for c in 0..3 {
                out[oidx + c] = (acc[c] * norm).clamp(0.0, 1.0);
            }
        }
    }
    out
}

/// Resize & patchify into `(patches_flat, pixel_position_ids, pw, ph)`.
pub fn prepare_vision_patches(img: &RgbImage, cfg: &VisionCfg) -> Result<(Vec<f32>, Vec<[i32; 2]>, usize, usize), String> {
    let (th, tw) = aspect_ratio_preserving_size(
        img.height,
        img.width,
        cfg.patch_size,
        cfg.max_patches(),
        cfg.pooling_kernel_size,
    )?;
    let f_img = resize_bicubic_f32(img, th, tw);
    let ps = cfg.patch_size;
    let ph = th / ps;
    let pw = tw / ps;
    let patch_dim = ps * ps * 3;
    let mut patches = Vec::with_capacity(ph * pw * patch_dim);
    let mut pos_xy = Vec::with_capacity(ph * pw);
    for py_b in 0..ph {
        for px_b in 0..pw {
            pos_xy.push([px_b as i32, py_b as i32]);
            for iy in 0..ps {
                let y = py_b * ps + iy;
                for ix in 0..ps {
                    let x = px_b * ps + ix;
                    let idx = (y * tw + x) * 3;
                    patches.push(f_img[idx]);
                    patches.push(f_img[idx + 1]);
                    patches.push(f_img[idx + 2]);
                }
            }
        }
    }
    Ok((patches, pos_xy, pw, ph))
}

struct Rope2DPatch {
    cos: Vec<f32>,
    sin: Vec<f32>,
}

fn precompute_axial_2d_rope(pos_xy: &[[i32; 2]], head_dim: usize, theta: f32) -> Vec<Rope2DPatch> {
    let spatial = head_dim / 2;
    let hr = spatial / 2;
    let inv_freq: Vec<f32> = (0..hr)
        .map(|k| 1.0f32 / theta.powf((2 * k) as f32 / spatial as f32))
        .collect();
    pos_xy
        .iter()
        .map(|&[px, py]| {
            let mut cos = vec![0f32; head_dim];
            let mut sin = vec![0f32; head_dim];
            for k in 0..hr {
                let (sx, cx) = (px as f32 * inv_freq[k]).sin_cos();
                cos[k] = cx;
                cos[hr + k] = cx;
                sin[k] = sx;
                sin[hr + k] = sx;
                let (sy, cy) = (py as f32 * inv_freq[k]).sin_cos();
                cos[spatial + k] = cy;
                cos[spatial + hr + k] = cy;
                sin[spatial + k] = sy;
                sin[spatial + hr + k] = sy;
            }
            Rope2DPatch { cos, sin }
        })
        .collect()
}

fn apply_2d_rope_token(tok: &mut [f32], rope: &Rope2DPatch, n_heads: usize, head_dim: usize) {
    let spatial = head_dim / 2;
    let hr = spatial / 2;
    for h in 0..n_heads {
        let base = h * head_dim;
        for part in 0..2 {
            let p = part * spatial;
            for k in 0..hr {
                let x0 = tok[base + p + k];
                let x1 = tok[base + p + hr + k];
                let c = rope.cos[p + k];
                let s = rope.sin[p + k];
                tok[base + p + k] = x0 * c - x1 * s;
                tok[base + p + hr + k] = x1 * c + x0 * s;
            }
        }
    }
}

fn compute_head_bidir_attn(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_tok: usize,
    d: usize,
    hd: usize,
    h: usize,
) -> Vec<f32> {
    let h_off = h * hd;
    let mut head_ctx = vec![0f32; n_tok * hd];
    let mut scores = vec![0f32; n_tok];
    for ti in 0..n_tok {
        let qi = &q[ti * d + h_off..ti * d + h_off + hd];
        let mut smax = f32::NEG_INFINITY;
        for tj in 0..n_tok {
            let kj = &k[tj * d + h_off..tj * d + h_off + hd];
            let mut dot = 0.0f32;
            for i in 0..hd {
                dot += qi[i] * kj[i];
            }
            scores[tj] = dot;
            if dot > smax {
                smax = dot;
            }
        }
        let mut sum_w = 0.0f32;
        for tj in 0..n_tok {
            let w = (scores[tj] - smax).exp();
            scores[tj] = w;
            sum_w += w;
        }
        let inv_w = if sum_w > 0.0 { 1.0 / sum_w } else { 0.0 };
        let out = &mut head_ctx[ti * hd..(ti + 1) * hd];
        for tj in 0..n_tok {
            let w = scores[tj] * inv_w;
            let vj = &v[tj * d + h_off..tj * d + h_off + hd];
            for i in 0..hd {
                out[i] += w * vj[i];
            }
        }
    }
    head_ctx
}

fn bidirectional_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_tok: usize,
    n_heads: usize,
    hd: usize,
    threads: usize,
) -> Vec<f32> {
    let d = n_heads * hd;
    let mut ctx = vec![0f32; n_tok * d];
    let workers = threads.max(1).min(n_heads.max(1));
    if workers <= 1 || n_tok <= 16 {
        for h in 0..n_heads {
            let h_off = h * hd;
            let hctx = compute_head_bidir_attn(q, k, v, n_tok, d, hd, h);
            for ti in 0..n_tok {
                ctx[ti * d + h_off..ti * d + h_off + hd].copy_from_slice(&hctx[ti * hd..(ti + 1) * hd]);
            }
        }
        return ctx;
    }
    let heads_per_worker = n_heads.div_ceil(workers);
    let mut chunk_results: Vec<(usize, Vec<Vec<f32>>)> = Vec::new();
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for w_idx in 0..workers {
            let h0 = w_idx * heads_per_worker;
            let h1 = ((w_idx + 1) * heads_per_worker).min(n_heads);
            if h0 >= h1 {
                continue;
            }
            handles.push(s.spawn(move || {
                let mut res = Vec::with_capacity(h1 - h0);
                for h in h0..h1 {
                    res.push(compute_head_bidir_attn(q, k, v, n_tok, d, hd, h));
                }
                (h0, res)
            }));
        }
        for hnd in handles {
            chunk_results.push(hnd.join().expect("vision head thread"));
        }
    });
    for (h0, res) in chunk_results {
        for (dh, hctx) in res.into_iter().enumerate() {
            let h_off = (h0 + dh) * hd;
            for ti in 0..n_tok {
                ctx[ti * d + h_off..ti * d + h_off + hd].copy_from_slice(&hctx[ti * hd..(ti + 1) * hd]);
            }
        }
    }
    ctx
}

/// Load a P6/P5 PPM, uncompressed 24/32-bit BMP, or delegate `.png`/`.jpg` decoding via Python PIL.
pub fn load_rgb_image(path: &Path) -> Result<RgbImage, String> {
    let bytes = fs::read(path).map_err(|e| format!("read image {}: {e}", path.display()))?;
    if bytes.starts_with(b"P6") || bytes.starts_with(b"P5") {
        return decode_pnm(&bytes);
    }
    if bytes.starts_with(b"BM") {
        return decode_bmp(&bytes);
    }
    let out = Command::new("python3")
        .arg("-c")
        .arg(concat!(
            "import sys; from PIL import Image; im=Image.open(sys.argv[1]).convert('RGB'); ",
            "sys.stdout.buffer.write(f'P6\\n{im.width} {im.height}\\n255\\n'.encode() + im.tobytes())",
        ))
        .arg(path)
        .output()
        .map_err(|e| format!("Unsupported image format {} and python3 PIL helper failed: {e}", path.display()))?;
    if !out.status.success() {
        return Err(format!(
            "python3 PIL decode failed for {}: {}",
            path.display(),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    decode_pnm(&out.stdout)
}

pub fn decode_pnm(bytes: &[u8]) -> Result<RgbImage, String> {
    let mut i = 0usize;
    let next_tok = |idx: &mut usize| -> Result<String, String> {
        while *idx < bytes.len() {
            if bytes[*idx] == b'#' {
                while *idx < bytes.len() && bytes[*idx] != b'\n' {
                    *idx += 1;
                }
            } else if bytes[*idx].is_ascii_whitespace() {
                *idx += 1;
            } else {
                break;
            }
        }
        let s = *idx;
        while *idx < bytes.len() && !bytes[*idx].is_ascii_whitespace() {
            *idx += 1;
        }
        if s == *idx {
            return Err("Truncated PNM header".into());
        }
        Ok(String::from_utf8_lossy(&bytes[s..*idx]).to_string())
    };
    let magic = next_tok(&mut i)?;
    if magic != "P6" && magic != "P5" {
        return Err(format!("Unsupported PNM magic: {magic}"));
    }
    let width: usize = next_tok(&mut i)?.parse().map_err(|e| format!("PNM width: {e}"))?;
    let height: usize = next_tok(&mut i)?.parse().map_err(|e| format!("PNM height: {e}"))?;
    let maxval: usize = next_tok(&mut i)?.parse().map_err(|e| format!("PNM maxval: {e}"))?;
    if maxval == 0 || maxval > 255 {
        return Err(format!("Unsupported PNM maxval: {maxval}"));
    }
    if i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    let raw = &bytes[i..];
    if magic == "P6" {
        if raw.len() < width * height * 3 {
            return Err(format!("PNM P6 truncated: {} < {}", raw.len(), width * height * 3));
        }
        Ok(RgbImage {
            width,
            height,
            data: raw[..width * height * 3].to_vec(),
        })
    } else {
        if raw.len() < width * height {
            return Err(format!("PNM P5 truncated: {} < {}", raw.len(), width * height));
        }
        let mut rgb = Vec::with_capacity(width * height * 3);
        for &g in &raw[..width * height] {
            rgb.push(g);
            rgb.push(g);
            rgb.push(g);
        }
        Ok(RgbImage { width, height, data: rgb })
    }
}

pub fn decode_bmp(bytes: &[u8]) -> Result<RgbImage, String> {
    if bytes.len() < 54 || &bytes[0..2] != b"BM" {
        return Err("Invalid BMP header".into());
    }
    let px_off = u32::from_le_bytes([bytes[10], bytes[11], bytes[12], bytes[13]]) as usize;
    let w_i = i32::from_le_bytes([bytes[18], bytes[19], bytes[20], bytes[21]]);
    let h_i = i32::from_le_bytes([bytes[22], bytes[23], bytes[24], bytes[25]]);
    let bpp = u16::from_le_bytes([bytes[28], bytes[29]]) as usize;
    let comp = u32::from_le_bytes([bytes[30], bytes[31], bytes[32], bytes[33]]);
    if comp != 0 || w_i <= 0 || h_i == 0 || (bpp != 24 && bpp != 32) {
        return Err(format!("Unsupported BMP w={w_i} h={h_i} bpp={bpp} comp={comp}"));
    }
    let width = w_i as usize;
    let top_down = h_i < 0;
    let height = h_i.unsigned_abs() as usize;
    let row_stride = ((bpp * width + 31) / 32) * 4;
    if px_off + row_stride * height > bytes.len() {
        return Err("BMP pixel array truncated".into());
    }
    let ch = bpp / 8;
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        let src_y = if top_down { y } else { height - 1 - y };
        let r_base = px_off + src_y * row_stride;
        for x in 0..width {
            let b = bytes[r_base + x * ch];
            let g = bytes[r_base + x * ch + 1];
            let r = bytes[r_base + x * ch + 2];
            let dst = (y * width + x) * 3;
            rgb[dst] = r;
            rgb[dst + 1] = g;
            rgb[dst + 2] = b;
        }
    }
    Ok(RgbImage { width, height, data: rgb })
}

/// Standard IEEE 802.3 CRC32 (matches `OpticalCarrierV1` / `AIQC` checksumming).
pub fn crc32_ieee(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320u32 & mask);
        }
    }
    !crc
}

/// Encode `img` as a standalone binary P6 PPM byte vector.
pub fn encode_ppm_p6(img: &RgbImage) -> Vec<u8> {
    let hdr = format!("P6\n{} {}\n255\n", img.width, img.height);
    let mut out = Vec::with_capacity(hdr.len() + img.data.len());
    out.extend_from_slice(hdr.as_bytes());
    out.extend_from_slice(&img.data);
    out
}

fn glyph_row_5x7(ch: char, row: usize) -> u8 {
    match ch {
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E][row],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E][row],
        '2' => [0x0E, 0x11, 0x01, 0x06, 0x08, 0x10, 0x1F][row],
        '3' => [0x1F, 0x02, 0x04, 0x06, 0x01, 0x11, 0x0E][row],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02][row],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E][row],
        '6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E][row],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08][row],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E][row],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C][row],
        'A' => [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11][row],
        'B' => [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E][row],
        'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E][row],
        'D' => [0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C][row],
        'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F][row],
        'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10][row],
        'G' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F][row],
        'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11][row],
        'I' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E][row],
        'K' => [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11][row],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F][row],
        'M' => [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11][row],
        'N' => [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11][row],
        'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E][row],
        'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10][row],
        'Q' => [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D][row],
        'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11][row],
        'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E][row],
        'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04][row],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E][row],
        'V' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04][row],
        'W' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11][row],
        'X' => [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11][row],
        'Y' => [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04][row],
        'Z' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F][row],
        ':' => [0x00, 0x04, 0x04, 0x00, 0x04, 0x04, 0x00][row],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x06][row],
        '-' => [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00][row],
        '/' => [0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10][row],
        '=' => [0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00][row],
        _ => 0u8,
    }
}

/// Render a high-contrast 48px-patch-aligned `Optical VA Self-Analysis Frame` containing legible
/// telemetry text, VA/bit-budget utilization bars, and an AIQC-style binary checksum strip.
pub fn render_optical_va_canvas(lines: &[&str], layer_bits: &[u8]) -> (RgbImage, u32) {
    let w = 672usize; // 14 * 48 -> exact 3x3 Vision pooling alignment (14x14 = 196 soft tokens)
    let h = 672usize;
    let mut data = vec![248u8; w * h * 3];
    let set_px = |buf: &mut [u8], x: usize, y: usize, r: u8, g: u8, b: u8| {
        if x < w && y < h {
            let p = (y * w + x) * 3;
            buf[p] = r;
            buf[p + 1] = g;
            buf[p + 2] = b;
        }
    };

    for (li, &line) in lines.iter().take(18).enumerate() {
        let y0 = 18 + li * 22;
        for (ci, ch) in line.to_ascii_uppercase().chars().take(52).enumerate() {
            let x0 = 18 + ci * 12;
            for r in 0..7 {
                let bits = glyph_row_5x7(ch, r);
                for c in 0..5 {
                    if (bits >> (4 - c)) & 1 == 1 {
                        for dy in 0..2 {
                            for dx in 0..2 {
                                set_px(&mut data, x0 + c * 2 + dx, y0 + r * 2 + dy, 12, 18, 28);
                            }
                        }
                    }
                }
            }
        }
    }

    let bar_y0 = 430usize;
    let bar_h = 120usize;
    for (idx, &nb) in layer_bits.iter().take(90).enumerate() {
        let bx = 24 + idx * 7;
        let fill_h = ((nb as usize).min(5) * bar_h) / 5;
        let (cr, cg, cb) = match nb {
            1 => (210u8, 65u8, 45u8),
            2 => (220u8, 150u8, 40u8),
            3 => (35u8, 115u8, 210u8),
            _ => (30u8, 165u8, 95u8),
        };
        for y in (bar_y0 + bar_h - fill_h)..(bar_y0 + bar_h) {
            for dx in 0..5 {
                set_px(&mut data, bx + dx, y, cr, cg, cb);
            }
        }
    }

    let summary_bytes: Vec<u8> = lines
        .iter()
        .flat_map(|s| s.as_bytes().iter().copied())
        .chain(layer_bits.iter().copied())
        .collect();
    let crc = crc32_ieee(&summary_bytes);
    let mut payload = Vec::with_capacity(8 + layer_bits.len());
    payload.extend_from_slice(b"AIQC");
    payload.extend_from_slice(&crc.to_be_bytes());
    payload.extend_from_slice(layer_bits);
    for (bit_i, byte) in payload.iter().enumerate() {
        for b in 0..8 {
            let bit = (byte >> (7 - b)) & 1;
            let px_x = 24 + ((bit_i * 8 + b) % 156) * 4;
            let px_y = 576 + ((bit_i * 8 + b) / 156) * 6;
            let col = if bit == 1 { 16u8 } else { 236u8 };
            for dy in 0..4 {
                for dx in 0..3 {
                    set_px(&mut data, px_x + dx, px_y + dy, col, col, col);
                }
            }
        }
    }

    (RgbImage { width: w, height: h, data }, crc)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack_zero_or_pattern_mat(name: &str, d_out: usize, d_in: usize, val_u8: u8, sc_val: f32) -> T16CMat {
        let bsize = 64usize;
        let n_blks = d_in / bsize;
        let sc = vec![sc_val; d_out * n_blks];
        let idx = vec![val_u8; d_out * d_in];
        T16CMat::pack_from_components(name.to_string(), d_out, d_in, bsize, 2, false, false, &sc, None, &idx, None, None).unwrap()
    }

    #[test]
    fn test_aspect_ratio_preserving_size_alignment() {
        for &(w, h) in &[(640usize, 480usize), (1920, 1080), (512, 512), (2400, 320), (96, 96)] {
            let (th, tw) = aspect_ratio_preserving_size(h, w, 16, 2520, 3).unwrap();
            assert_eq!(th % 48, 0);
            assert_eq!(tw % 48, 0);
            assert!((th / 16) * (tw / 16) <= 2520);
        }
    }

    #[test]
    fn test_toy_vision_forward_and_padding() {
        let cfg = VisionCfg {
            hidden_size: 64,
            intermediate_size: 100,
            num_hidden_layers: 1,
            num_heads: 4,
            head_dim: 16,
            patch_size: 16,
            pooling_kernel_size: 3,
            max_soft_tokens: 4,
            position_embedding_size: 32,
            rms_norm_eps: 1e-6,
            rope_theta: 100.0,
            standardize: true,
            text_hidden_size: 64,
        };
        let layer = VisionLayerWeights {
            in_norm: vec![1.0; 64],
            post_attn_norm: vec![1.0; 64],
            pre_ff_norm: vec![1.0; 64],
            post_ff_norm: vec![1.0; 64],
            q_norm: vec![1.0; 16],
            k_norm: vec![1.0; 16],
            q_proj: pack_zero_or_pattern_mat("q", 64, 64, 2, 0.05),
            k_proj: pack_zero_or_pattern_mat("k", 64, 64, 1, 0.05),
            v_proj: pack_zero_or_pattern_mat("v", 64, 64, 2, 0.05),
            o_proj: pack_zero_or_pattern_mat("o", 64, 64, 2, 0.05),
            gate_proj: pack_zero_or_pattern_mat("g", 100, 64, 2, 0.05),
            up_proj: pack_zero_or_pattern_mat("u", 100, 64, 2, 0.05),
            down_proj: pack_zero_or_pattern_mat("d", 64, 128, 2, 0.05),
        };
        let vw = VisionWeights {
            cfg: cfg.clone(),
            input_proj: pack_zero_or_pattern_mat("in", 64, 768, 2, 0.05),
            pos_table: PosTableBackend::NormF32(vec![0.01; 2 * 32 * 64]),
            layers: vec![layer],
            std_bias: vec![0.0; 64],
            std_scale: vec![1.0; 64],
            embed_proj: pack_zero_or_pattern_mat("emb", 64, 64, 2, 0.1),
        };
        let img = RgbImage { width: 48, height: 48, data: vec![180u8; 48 * 48 * 3] };
        let (n_soft, soft) = vw.encode_image(&img, None, None, 1).unwrap();
        assert_eq!(n_soft, 4);
        assert_eq!(soft.len(), 4 * 64);
        assert!(soft.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_ppm_and_optical_va_canvas_roundtrip() {
        let bits = vec![3u8, 5, 5, 1, 2, 5, 3];
        let (img, crc) = render_optical_va_canvas(&["GEMMA4 31B FULL", "LM=30.7B VIS=0.58B"], &bits);
        assert_eq!(img.width, 672);
        assert_eq!(img.height, 672);
        assert_ne!(crc, 0);
        let ppm = encode_ppm_p6(&img);
        let dec = decode_pnm(&ppm).unwrap();
        assert_eq!(dec.width, 672);
        assert_eq!(dec.data, img.data);
    }

    #[test]
    fn test_parallel_bidirectional_attention_matches_single_thread() {
        let (n_tok, n_heads, hd) = (24, 8, 16);
        let d = n_heads * hd;
        let q: Vec<f32> = (0..n_tok * d).map(|i| ((i as f32) * 0.13).sin() * 0.5).collect();
        let k: Vec<f32> = (0..n_tok * d).map(|i| ((i as f32) * 0.07).cos() * 0.5).collect();
        let v: Vec<f32> = (0..n_tok * d).map(|i| ((i as f32) * 0.19).sin()).collect();
        let c1 = bidirectional_attention(&q, &k, &v, n_tok, n_heads, hd, 1);
        let c4 = bidirectional_attention(&q, &k, &v, n_tok, n_heads, hd, 4);
        let max_diff = c1.iter().zip(c4.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 1e-6, "max_diff={max_diff}");
    }
}
