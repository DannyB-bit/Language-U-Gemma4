//! Native T16C KV-sharing EAGLE-3 Gemma4Assistant speculative draft cell.
//!
//! Executes the 4-layer KV-sharing `Gemma4AssistantForCausalLM` architecture
//! (`pre_projection -> 4 decoder layers reusing target L58 sliding + L59 full KV ->
//! model.norm -> post_projection & lm_head`) packed in 4-bit `.t16c` format (~247 MB)
//! via `libneon_stream_accelerator.so` or bit-accurate Rust fallback.

use crate::byte_head::{BpePrefixTrie, ByteHeadDecoder};
use crate::math::{
    attn_decode, gelu_tanh_fast_f32, qk_norm_raw_inplace, rmsn_raw, rope_cos_sin_for_pos,
    rope_head_with_cos_sin, rope_inv_freqs,
};
use crate::speculative::argmax_f32;
use crate::t16c::T16CModel;
use std::path::Path;
use std::sync::Arc;

/// Structural hyperparameters for a KV-sharing Gemma4Assistant drafter.
#[derive(Debug, Clone)]
pub struct AssistantCfg {
    pub backbone_hidden: usize,
    pub d: usize,
    pub ff: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_slide: usize,
    pub n_kv_full: usize,
    pub hd_slide: usize,
    pub hd_full: usize,
    pub window: usize,
    pub eps: f32,
    pub theta_local: f32,
    pub partial_local: f32,
    pub theta_global: f32,
    pub partial_global: f32,
    pub vocab: usize,
    pub layer_full: Vec<bool>,
    pub has_lm_head: bool,
}

impl AssistantCfg {
    /// Canonical hyperparameters matching Google `Gemma4AssistantForCausalLM` (31B).
    pub fn gemma4_31b_default() -> Self {
        Self {
            backbone_hidden: 5376,
            d: 1024,
            ff: 8192,
            n_layers: 4,
            n_heads: 32,
            n_kv_slide: 16,
            n_kv_full: 4,
            hd_slide: 256,
            hd_full: 512,
            window: 1024,
            eps: 1e-6,
            theta_local: 10_000.0,
            partial_local: 1.0,
            theta_global: 1_000_000.0,
            partial_global: 0.25,
            vocab: 262_144,
            layer_full: vec![false, false, false, true],
            has_lm_head: true,
        }
    }

    /// Infer core dimensions from an already-loaded `T16CModel` while preserving KV head ratios.
    pub fn infer_from_model(
        model: &T16CModel,
        n_kv_slide: usize,
        n_kv_full: usize,
        window: usize,
    ) -> Result<Self, String> {
        Self::infer_from_model_with_vocab(model, n_kv_slide, n_kv_full, window, 0)
    }

    /// Infer core dimensions from `T16CModel`, allowing embed-free `Assistant-V7` bodies when `fallback_vocab > 0`.
    pub fn infer_from_model_with_vocab(
        model: &T16CModel,
        n_kv_slide: usize,
        n_kv_full: usize,
        window: usize,
        fallback_vocab: usize,
    ) -> Result<Self, String> {
        let pre = model.mat("pre_projection.weight")?;
        if pre.d_in % 2 != 0 {
            return Err(format!("pre_projection.weight d_in={} not even", pre.d_in));
        }
        let d = pre.d_out;
        let backbone_hidden = pre.d_in / 2;
        let post = model.mat("post_projection.weight")?;
        if post.d_out != backbone_hidden || post.d_in != d {
            return Err(format!(
                "post_projection.weight dims {}x{} != expected {}x{}",
                post.d_out, post.d_in, backbone_hidden, d
            ));
        }
        let (vocab, has_lm_head) = if let Ok(emb) = model.mat("model.embed_tokens.weight") {
            if emb.d_in != d {
                return Err(format!("assistant embed d_in {} != d {d}", emb.d_in));
            }
            (emb.d_out, true)
        } else if fallback_vocab > 0 {
            (fallback_vocab, false)
        } else {
            return Err("assistant T16CModel missing model.embed_tokens.weight and fallback_vocab == 0".into());
        };

        let mut layer_full = Vec::new();
        let mut n_heads = 0usize;
        let mut hd_slide = 0usize;
        let mut hd_full = 0usize;
        let mut ff = 0usize;
        let mut l = 0usize;
        while let Ok(qp) = model.mat(&format!("model.layers.{l}.self_attn.q_proj.weight")) {
            let qn = model.norm(&format!("model.layers.{l}.self_attn.q_norm.weight"))?;
            let gp = model.mat(&format!("model.layers.{l}.mlp.gate_proj.weight"))?;
            let hd = qn.len();
            if hd == 0 || qp.d_out % hd != 0 {
                return Err(format!("assistant layer {l}: q_proj d_out={} not divisible by q_norm len={hd}", qp.d_out));
            }
            let nh = qp.d_out / hd;
            if l == 0 {
                n_heads = nh;
                ff = gp.d_out;
            } else if nh != n_heads || gp.d_out != ff {
                return Err(format!("assistant layer {l}: inconsistent heads/ff ({nh}/{}) != ({n_heads}/{ff})", gp.d_out));
            }
            // Full attention layers in Gemma4Assistant use larger global head dim (2x sliding) or sit last.
            if l == 0 {
                hd_slide = hd;
            }
            let is_full = hd > hd_slide;
            if is_full {
                hd_full = hd;
            }
            layer_full.push(is_full);
            l += 1;
        }
        if layer_full.is_empty() {
            return Err("assistant T16CModel has zero decoder layers".into());
        }
        if hd_full == 0 {
            // All layers had identical head_dim; keep last layer flagged full only if n_kv_full matches.
            hd_full = hd_slide;
        }
        Ok(Self {
            backbone_hidden,
            d,
            ff,
            n_layers: layer_full.len(),
            n_heads,
            n_kv_slide,
            n_kv_full,
            hd_slide,
            hd_full,
            window,
            eps: 1e-6,
            theta_local: 10_000.0,
            partial_local: 1.0,
            theta_global: 1_000_000.0,
            partial_global: 0.25,
            vocab,
            layer_full,
            has_lm_head,
        })
    }

    pub fn validate(&self, model: &T16CModel) -> Result<(), String> {
        if self.n_layers == 0 || self.layer_full.len() != self.n_layers {
            return Err(format!("invalid assistant n_layers={} layer_full={}", self.n_layers, self.layer_full.len()));
        }
        let pre = model.mat("pre_projection.weight")?;
        if pre.d_out != self.d || pre.d_in != 2 * self.backbone_hidden {
            return Err(format!("pre_projection shape {}x{} != {}x{}", pre.d_out, pre.d_in, self.d, 2 * self.backbone_hidden));
        }
        let post = model.mat("post_projection.weight")?;
        if post.d_out != self.backbone_hidden || post.d_in != self.d {
            return Err(format!("post_projection shape {}x{} != {}x{}", post.d_out, post.d_in, self.backbone_hidden, self.d));
        }
        if self.has_lm_head {
            let emb = model.mat("model.embed_tokens.weight")?;
            if emb.d_out != self.vocab || emb.d_in != self.d {
                return Err(format!("assistant embed shape {}x{} != {}x{}", emb.d_out, emb.d_in, self.vocab, self.d));
            }
        }
        let fn_w = model.norm("model.norm.weight")?;
        if fn_w.len() != self.d {
            return Err(format!("assistant model.norm.weight len {} != {}", fn_w.len(), self.d));
        }
        for (l, &is_full) in self.layer_full.iter().enumerate() {
            let hd = if is_full { self.hd_full } else { self.hd_slide };
            let n_kv = if is_full { self.n_kv_full } else { self.n_kv_slide };
            if self.n_heads % n_kv != 0 {
                return Err(format!("assistant layer {l}: n_heads {} not divisible by n_kv {n_kv}", self.n_heads));
            }
            let p = format!("model.layers.{l}.");
            let qp = model.mat(&format!("{p}self_attn.q_proj.weight"))?;
            let op = model.mat(&format!("{p}self_attn.o_proj.weight"))?;
            let gp = model.mat(&format!("{p}mlp.gate_proj.weight"))?;
            let up = model.mat(&format!("{p}mlp.up_proj.weight"))?;
            let dp = model.mat(&format!("{p}mlp.down_proj.weight"))?;
            if qp.d_out != self.n_heads * hd || qp.d_in != self.d {
                return Err(format!("{p}q_proj shape {}x{} != {}x{}", qp.d_out, qp.d_in, self.n_heads * hd, self.d));
            }
            if op.d_out != self.d || op.d_in != self.n_heads * hd {
                return Err(format!("{p}o_proj shape {}x{} != {}x{}", op.d_out, op.d_in, self.d, self.n_heads * hd));
            }
            if gp.d_out != self.ff || gp.d_in != self.d || up.d_out != self.ff || up.d_in != self.d {
                return Err(format!("{p}mlp gate/up mismatch against ff={} d={}", self.ff, self.d));
            }
            if dp.d_out != self.d || dp.d_in != self.ff {
                return Err(format!("{p}mlp.down_proj shape {}x{} != {}x{}", dp.d_out, dp.d_in, self.d, self.ff));
            }
            let qn = model.norm(&format!("{p}self_attn.q_norm.weight"))?;
            if qn.len() != hd {
                return Err(format!("{p}q_norm len {} != {hd}", qn.len()));
            }
            for nk in ["input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight"] {
                let nv = model.norm(&format!("{p}{nk}"))?;
                if nv.len() != self.d {
                    return Err(format!("{p}{nk} len {} != {}", nv.len(), self.d));
                }
            }
            let ls = model.norm(&format!("{p}layer_scalar"))?;
            if ls.is_empty() {
                return Err(format!("{p}layer_scalar empty"));
            }
        }
        Ok(())
    }
}

/// Return `(argmax_idx, top1_val, top2_val)` over a non-empty logit slice.
#[inline]
fn argmax_with_top2(logits: &[f32]) -> (u32, f32, f32) {
    let mut best_i = 0u32;
    let mut best_v = f32::NEG_INFINITY;
    let mut second_v = f32::NEG_INFINITY;
    for (idx, &v) in logits.iter().enumerate() {
        if v > best_v {
            second_v = best_v;
            best_v = v;
            best_i = idx as u32;
        } else if v > second_v {
            second_v = v;
        }
    }
    (best_i, best_v, second_v)
}

/// Resident KV-sharing Gemma4Assistant speculative draft model backed by `T16CModel`.
pub struct AssistantRuntime {
    pub model: T16CModel,
    pub cfg: AssistantCfg,
    pub threads: usize,
    pub embed_tile_norms_fp16: Vec<u16>,
    pub byte_head: Option<(ByteHeadDecoder, Arc<BpePrefixTrie>)>,
}

impl AssistantRuntime {
    /// Load a `.t16c` packed Gemma4Assistant file targeting a 31B host (`n_kv_slide=16, n_kv_full=4`).
    pub fn load_31b(
        path: &Path,
        native_lib: Option<&Path>,
        disable_native: bool,
        threads: usize,
        n_kv_slide: usize,
        n_kv_full: usize,
        window: usize,
    ) -> Result<Self, String> {
        Self::load_31b_with_byte_head(
            path,
            None,
            None,
            native_lib,
            disable_native,
            threads,
            n_kv_slide,
            n_kv_full,
            window,
            0,
        )
    }

    /// Load an Assistant backbone `.t16c` alongside an optional `BHDR` v1 Byte-Head decoder.
    #[allow(clippy::too_many_arguments)]
    pub fn load_31b_with_byte_head(
        path: &Path,
        byte_head_path: Option<&Path>,
        trie: Option<Arc<BpePrefixTrie>>,
        native_lib: Option<&Path>,
        disable_native: bool,
        threads: usize,
        n_kv_slide: usize,
        n_kv_full: usize,
        window: usize,
        target_vocab: usize,
    ) -> Result<Self, String> {
        let model = T16CModel::load(path, native_lib, disable_native)?;
        let cfg = AssistantCfg::infer_from_model_with_vocab(&model, n_kv_slide, n_kv_full, window, target_vocab)?;
        let byte_head = match byte_head_path {
            Some(bp) => {
                let bh = ByteHeadDecoder::load(bp)?;
                if bh.d_in != cfg.d {
                    return Err(format!(
                        "byte head {} d_in={} != assistant d={}",
                        bp.display(),
                        bh.d_in,
                        cfg.d
                    ));
                }
                let tr = trie.ok_or("byte_head requires a loaded BpePrefixTrie")?;
                Some((bh, tr))
            }
            None => None,
        };
        if !cfg.has_lm_head && byte_head.is_none() && target_vocab == 0 {
            return Err(format!(
                "assistant {} lacks model.embed_tokens.weight; provide --spec-byte-head <file.bhdr> or enable target-embed sharing (target_vocab > 0)",
                path.display()
            ));
        }
        Self::from_model_with_byte_head(model, cfg, threads, byte_head)
    }

    /// Construct from a pre-loaded `T16CModel` and validated config.
    pub fn from_model(model: T16CModel, cfg: AssistantCfg, threads: usize) -> Result<Self, String> {
        Self::from_model_with_byte_head(model, cfg, threads, None)
    }

    /// Construct from a pre-loaded `T16CModel`, validated config, and optional `ByteHeadDecoder`.
    pub fn from_model_with_byte_head(
        model: T16CModel,
        cfg: AssistantCfg,
        threads: usize,
        byte_head: Option<(ByteHeadDecoder, Arc<BpePrefixTrie>)>,
    ) -> Result<Self, String> {
        cfg.validate(&model)?;
        let embed_tile_norms_fp16 = if cfg.has_lm_head && byte_head.is_none() {
            model.mat("model.embed_tokens.weight")?.compute_tile_norms_fp16()
        } else {
            Vec::new()
        };
        Ok(Self {
            model,
            cfg,
            threads: threads.max(1),
            embed_tile_norms_fp16,
            byte_head,
        })
    }

    /// Whether this runtime uses a byte-head micro-decoder instead of the 262K LM head.
    pub fn has_byte_head(&self) -> bool {
        self.byte_head.is_some()
    }

    /// Whether this runtime shares the host 31B `embed_tokens` table (0 MB extra LM-head RAM).
    pub fn uses_target_shared_head(&self) -> bool {
        !self.cfg.has_lm_head && self.byte_head.is_none()
    }

    /// Total compressed weight + norm + cached CS norm (+ optional byte head/trie) bytes resident.
    pub fn resident_bytes(&self) -> usize {
        let mat_b: usize = self.model.mats.values().map(|m| m.total_bytes()).sum();
        let norm_b: usize = self.model.norms.values().map(|v| v.len() * 4).sum();
        let bh_b: usize = self
            .byte_head
            .as_ref()
            .map_or(0, |(bh, tr)| bh.byte_len() + tr.byte_len());
        mat_b + norm_b + self.embed_tile_norms_fp16.len() * 2 + bh_b
    }

    /// Execute the 4-layer KV-sharing assistant backbone, returning `(h_final, next_backbone_h)`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_backbone_step(
        &self,
        tok_embed: &[f32],
        backbone_h: &[f32],
        kv_slide_k: &[f32],
        kv_slide_v: &[f32],
        kv_full_k: &[f32],
        kv_full_v: &[f32],
        kv_len: usize,
        q_pos: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let bh = self.cfg.backbone_hidden;
        if tok_embed.len() != bh || backbone_h.len() != bh {
            return Err(format!(
                "assistant embed/backbone len ({}, {}) != {bh}",
                tok_embed.len(),
                backbone_h.len()
            ));
        }
        if kv_len == 0 || q_pos < kv_len - 1 {
            return Err(format!("invalid assistant kv_len={kv_len} q_pos={q_pos}"));
        }
        let need_slide = kv_len * self.cfg.n_kv_slide * self.cfg.hd_slide;
        let need_full = kv_len * self.cfg.n_kv_full * self.cfg.hd_full;
        if kv_slide_k.len() < need_slide || kv_slide_v.len() < need_slide {
            return Err(format!("assistant slide KV len < required {need_slide}"));
        }
        if kv_full_k.len() < need_full || kv_full_v.len() < need_full {
            return Err(format!("assistant full KV len < required {need_full}"));
        }

        let mut concat = Vec::with_capacity(2 * bh);
        concat.extend_from_slice(tok_embed);
        concat.extend_from_slice(backbone_h);

        let native = self.model.native.as_ref();
        let pre = self.model.mat("pre_projection.weight")?;
        let mut state = pre.project_batch(&concat, 1, native, self.threads)?;
        let eps = self.cfg.eps;
        let d = self.cfg.d;

        for (l, &is_full) in self.cfg.layer_full.iter().enumerate() {
            let p = format!("model.layers.{l}.");
            let hd = if is_full { self.cfg.hd_full } else { self.cfg.hd_slide };
            let n_kv = if is_full { self.cfg.n_kv_full } else { self.cfg.n_kv_slide };
            let theta = if is_full { self.cfg.theta_global } else { self.cfg.theta_local };
            let partial = if is_full { self.cfg.partial_global } else { self.cfg.partial_local };
            let rot_dims = ((hd as f32) * partial).round() as usize;
            let sw = if is_full { None } else { Some(self.cfg.window) };
            let (sk, sv) = if is_full {
                (&kv_full_k[..need_full], &kv_full_v[..need_full])
            } else {
                (&kv_slide_k[..need_slide], &kv_slide_v[..need_slide])
            };

            let g_in = self.model.norm(&format!("{p}input_layernorm.weight"))?;
            let inp = rmsn_raw(&state, Some(g_in), eps);

            let mut q = self.model.mat(&format!("{p}self_attn.q_proj.weight"))?.project_batch(&inp, 1, native, self.threads)?;
            let qn = self.model.norm(&format!("{p}self_attn.q_norm.weight"))?;
            qk_norm_raw_inplace(&mut q, self.cfg.n_heads, hd, Some(qn), eps);
            let inv_f = rope_inv_freqs(theta, rot_dims, hd);
            let mut rcos = vec![0.0f32; inv_f.len()];
            let mut rsin = vec![0.0f32; inv_f.len()];
            rope_cos_sin_for_pos(&inv_f, q_pos as u32, &mut rcos, &mut rsin);
            for h in 0..self.cfg.n_heads {
                rope_head_with_cos_sin(&mut q[h * hd..(h + 1) * hd], &rcos, &rsin, hd);
            }

            let ctx = attn_decode(&q, sk, sv, kv_len - 1, q_pos, self.cfg.n_heads, n_kv, hd, sw);
            let attn_o = self.model.mat(&format!("{p}self_attn.o_proj.weight"))?.project_batch(&ctx, 1, native, self.threads)?;
            let g_pa = self.model.norm(&format!("{p}post_attention_layernorm.weight"))?;
            let norm_a = rmsn_raw(&attn_o, Some(g_pa), eps);
            for i in 0..d {
                state[i] += norm_a[i];
            }

            let g_pf = self.model.norm(&format!("{p}pre_feedforward_layernorm.weight"))?;
            let inp2 = rmsn_raw(&state, Some(g_pf), eps);
            let gate = self.model.mat(&format!("{p}mlp.gate_proj.weight"))?.project_batch(&inp2, 1, native, self.threads)?;
            let up = self.model.mat(&format!("{p}mlp.up_proj.weight"))?.project_batch(&inp2, 1, native, self.threads)?;
            let act: Vec<f32> = gate.iter().zip(up.iter()).map(|(&a, &b)| gelu_tanh_fast_f32(a) * b).collect();
            let down = self.model.mat(&format!("{p}mlp.down_proj.weight"))?.project_batch(&act, 1, native, self.threads)?;
            let g_pff = self.model.norm(&format!("{p}post_feedforward_layernorm.weight"))?;
            let norm_d = rmsn_raw(&down, Some(g_pff), eps);
            let ls = self.model.norm(&format!("{p}layer_scalar"))?[0];
            for i in 0..d {
                state[i] = (state[i] + norm_d[i]) * ls;
            }
        }

        let fn_w = self.model.norm("model.norm.weight")?;
        let h_final = rmsn_raw(&state, Some(fn_w), eps);
        let next_backbone = self.model.mat("post_projection.weight")?.project_batch(&h_final, 1, native, self.threads)?;
        Ok((h_final, next_backbone))
    }

    /// Execute the 4-layer KV-sharing assistant backbone using a caller-provided zero-copy
    /// cross-attention closure `attend_fn(is_full, q, q_pos) -> Result<Vec<f32>, String>`,
    /// avoiding `materialize_chronological_kv()` full-sequence heap decompression on `GlyphKV`!
    pub fn forward_backbone_step_with_attend<A>(
        &self,
        tok_embed: &[f32],
        backbone_h: &[f32],
        q_pos: usize,
        mut attend_fn: A,
    ) -> Result<(Vec<f32>, Vec<f32>), String>
    where
        A: FnMut(bool, &[f32], usize) -> Result<Vec<f32>, String>,
    {
        let bh = self.cfg.backbone_hidden;
        if tok_embed.len() != bh || backbone_h.len() != bh {
            return Err(format!(
                "assistant embed/backbone len ({}, {}) != {bh}",
                tok_embed.len(),
                backbone_h.len()
            ));
        }
        let mut concat = Vec::with_capacity(2 * bh);
        concat.extend_from_slice(tok_embed);
        concat.extend_from_slice(backbone_h);

        let native = self.model.native.as_ref();
        let pre = self.model.mat("pre_projection.weight")?;
        let mut state = pre.project_batch(&concat, 1, native, self.threads)?;
        let eps = self.cfg.eps;
        let d = self.cfg.d;

        for l in 0..self.cfg.n_layers {
            let p = format!("model.layers.{l}.");
            let is_full = self.cfg.layer_full[l];
            let hd = if is_full { self.cfg.hd_full } else { self.cfg.hd_slide };
            let theta = if is_full { self.cfg.theta_global } else { self.cfg.theta_local };
            let partial = if is_full { self.cfg.partial_global } else { self.cfg.partial_local };
            let rot_dims = ((hd as f32) * partial).round() as usize;

            let g_in = self.model.norm(&format!("{p}input_layernorm.weight"))?;
            let inp = rmsn_raw(&state, Some(g_in), eps);

            let mut q = self.model.mat(&format!("{p}self_attn.q_proj.weight"))?.project_batch(&inp, 1, native, self.threads)?;
            let qn = self.model.norm(&format!("{p}self_attn.q_norm.weight"))?;
            qk_norm_raw_inplace(&mut q, self.cfg.n_heads, hd, Some(qn), eps);
            let inv_f = rope_inv_freqs(theta, rot_dims, hd);
            let mut rcos = vec![0.0f32; inv_f.len()];
            let mut rsin = vec![0.0f32; inv_f.len()];
            rope_cos_sin_for_pos(&inv_f, q_pos as u32, &mut rcos, &mut rsin);
            for h in 0..self.cfg.n_heads {
                rope_head_with_cos_sin(&mut q[h * hd..(h + 1) * hd], &rcos, &rsin, hd);
            }

            let ctx = attend_fn(is_full, &q, q_pos)?;
            let attn_o = self.model.mat(&format!("{p}self_attn.o_proj.weight"))?.project_batch(&ctx, 1, native, self.threads)?;
            let g_pa = self.model.norm(&format!("{p}post_attention_layernorm.weight"))?;
            let norm_a = rmsn_raw(&attn_o, Some(g_pa), eps);
            for i in 0..d {
                state[i] += norm_a[i];
            }

            let g_pf = self.model.norm(&format!("{p}pre_feedforward_layernorm.weight"))?;
            let inp2 = rmsn_raw(&state, Some(g_pf), eps);
            let gate = self.model.mat(&format!("{p}mlp.gate_proj.weight"))?.project_batch(&inp2, 1, native, self.threads)?;
            let up = self.model.mat(&format!("{p}mlp.up_proj.weight"))?.project_batch(&inp2, 1, native, self.threads)?;
            let act: Vec<f32> = gate.iter().zip(up.iter()).map(|(&a, &b)| gelu_tanh_fast_f32(a) * b).collect();
            let down = self.model.mat(&format!("{p}mlp.down_proj.weight"))?.project_batch(&act, 1, native, self.threads)?;
            let g_pff = self.model.norm(&format!("{p}post_feedforward_layernorm.weight"))?;
            let norm_d = rmsn_raw(&down, Some(g_pff), eps);
            let ls = self.model.norm(&format!("{p}layer_scalar"))?[0];
            for i in 0..d {
                state[i] = (state[i] + norm_d[i]) * ls;
            }
        }

        let fn_w = self.model.norm("model.norm.weight")?;
        let h_final = rmsn_raw(&state, Some(fn_w), eps);
        let next_backbone = self.model.mat("post_projection.weight")?.project_batch(&h_final, 1, native, self.threads)?;
        Ok((h_final, next_backbone))
    }

    /// Draft an EAGLE-3 speculative prefix tree (`SpecTreeBuilder`) supporting ALL THREE heads:
    /// 1) `BHDR` compact byte-head micro-decoder (`self.byte_head`),
    /// 2) Assistant's own quantized `model.embed_tokens.weight` (`has_lm_head`), or
    /// 3) Zero-RAM Target-Embed Sharing via `shared_score_fn(h_final, next_backbone, prev_tok, want_k, bans)`.
    #[allow(clippy::too_many_arguments)]
    pub fn draft_tree_with_attend<F, A>(
        &self,
        first_tok: u32,
        embed_fn: F,
        first_backbone_h: &[f32],
        start_q_pos: usize,
        max_depth: usize,
        max_nodes: usize,
        ngram_hint: &[u32],
        eos: &[u32],
        ban_tokens: &[u32],
        min_logit_margin: f32,
        lmhead_cs_sieve: Option<f32>,
        attend_fn: A,
    ) -> Result<(crate::speculative::SpecTreeBuilder, usize, i64), String>
    where
        F: FnMut(u32) -> Result<Vec<f32>, String>,
        A: FnMut(bool, &[f32], usize) -> Result<Vec<f32>, String>,
    {
        self.draft_tree_with_target_head(
            first_tok,
            embed_fn,
            first_backbone_h,
            start_q_pos,
            max_depth,
            max_nodes,
            ngram_hint,
            &[],
            eos,
            ban_tokens,
            min_logit_margin,
            lmhead_cs_sieve,
            attend_fn,
            |_h_final, _next_bh, _prev_tok, _want_k, _bans| {
                Err("assistant has no byte_head, no embed_tokens, and no shared_score_fn".into())
            },
        )
    }

    /// Unified EAGLE-3 tree drafter supporting Tier-0 suffix hints + Target-Embed Sharing (`0 MB` extra head RAM).
    #[allow(clippy::too_many_arguments)]
    pub fn draft_tree_with_target_head<F, A, S>(
        &self,
        first_tok: u32,
        mut embed_fn: F,
        first_backbone_h: &[f32],
        start_q_pos: usize,
        max_depth: usize,
        max_nodes: usize,
        ngram_hint: &[u32],
        suffix_hint: &[u32],
        eos: &[u32],
        ban_tokens: &[u32],
        min_logit_margin: f32,
        lmhead_cs_sieve: Option<f32>,
        mut attend_fn: A,
        mut shared_score_fn: S,
    ) -> Result<(crate::speculative::SpecTreeBuilder, usize, i64), String>
    where
        F: FnMut(u32) -> Result<Vec<f32>, String>,
        A: FnMut(bool, &[f32], usize) -> Result<Vec<f32>, String>,
        S: FnMut(&[f32], &[f32], u32, usize, &[u32]) -> Result<(Vec<(u32, f32)>, i64), String>,
    {
        let mut tree = crate::speculative::SpecTreeBuilder::new(first_tok, max_nodes);
        if !suffix_hint.is_empty() {
            tree.graft_sequence_with_margin(suffix_hint, eos, 12.0);
        }
        if !ngram_hint.is_empty() {
            tree.graft_sequence_with_margin(ngram_hint, eos, 5.0);
        }
        let mut cur_slot = 0usize;
        let mut cur_tok = first_tok;
        let mut cur_backbone = first_backbone_h.to_vec();
        let mut margin_stopped = 0usize;
        let mut total_skipped = 0i64;
        let mut alt_branch_root: Option<(usize, u32, Vec<f32>)> = None;

        let mut predict_top_cands = |h_final: &[f32],
                                     next_bh: &[f32],
                                     prev_tok: u32,
                                     want_k: usize,
                                     tot_sk: &mut i64|
         -> Result<Vec<(u32, f32)>, String> {
            let mut effective_bans = ban_tokens.to_vec();
            if !ban_tokens.is_empty() && !(236770..=236820).contains(&prev_tok) && !effective_bans.contains(&prev_tok) {
                effective_bans.push(prev_tok);
            }
            if let Some((ref bh, ref trie)) = self.byte_head {
                bh.decode_top_k_tokens(h_final, trie, &effective_bans, want_k.max(4), want_k)
            } else if self.cfg.has_lm_head {
                let native = self.model.native.as_ref();
                let (mut logits, sk) = if let Some(sf) = lmhead_cs_sieve {
                    self.model.mat("model.embed_tokens.weight")?.project_lmhead_cs(
                        h_final,
                        1,
                        native,
                        self.threads,
                        Some(&self.embed_tile_norms_fp16),
                        want_k.max(16),
                        sf,
                    )?
                } else {
                    (
                        self.model.mat("model.embed_tokens.weight")?.project_batch(h_final, 1, native, self.threads)?,
                        0,
                    )
                };
                *tot_sk += sk;
                for &tid in &effective_bans {
                    let u = tid as usize;
                    if u < logits.len() {
                        logits[u] = -1.0e9;
                    }
                }
                let mut best: Vec<(u32, f32)> = Vec::with_capacity(want_k);
                for (i, &v) in logits.iter().enumerate() {
                    if best.len() < want_k {
                        best.push((i as u32, v));
                        best.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
                    } else if v > best[want_k - 1].1 {
                        best[want_k - 1] = (i as u32, v);
                        best.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
                    }
                }
                Ok(best)
            } else {
                let (cands, sk) = shared_score_fn(h_final, next_bh, prev_tok, want_k, &effective_bans)?;
                *tot_sk += sk;
                Ok(cands)
            }
        };

        for step in 0..max_depth {
            if tree.nodes.len() >= tree.max_nodes {
                break;
            }
            let tok_emb = embed_fn(cur_tok)?;
            let q_pos = start_q_pos + step;
            let (h_final, next_h) =
                self.forward_backbone_step_with_attend(&tok_emb, &cur_backbone, q_pos, &mut attend_fn)?;
            let branch_k = if step == 0 {
                4
            } else if step <= 2 {
                3
            } else {
                1
            };
            let cands = predict_top_cands(&h_final, &next_h, cur_tok, branch_k, &mut total_skipped)?;
            if cands.is_empty() {
                break;
            }
            let (top1_tok, top1_v) = cands[0];
            let top2_v = cands.get(1).map_or(top1_v - 30.0, |c| c.1);
            let step_margin = (top1_v - top2_v).max(0.0);
            if step >= 2 && min_logit_margin > 0.0 && step_margin < min_logit_margin {
                margin_stopped += 1;
                break;
            }
            let Some(primary_slot) = tree.insert_child(cur_slot, top1_tok) else {
                break;
            };
            if step_margin > tree.slot_margin(primary_slot) {
                tree.set_slot_margin(primary_slot, step_margin);
            }
            // Attach high-likelihood sibling branches at early depths (`step <= 2`).
            for (ci, &(alt_tok, alt_lp)) in cands.iter().enumerate().skip(1) {
                if top1_v - alt_lp <= 3.5 {
                    if let Some(alt_slot) = tree.insert_child(cur_slot, alt_tok) {
                        if step == 0 && ci == 1 && !eos.contains(&alt_tok) {
                            alt_branch_root = Some((alt_slot, alt_tok, next_h.clone()));
                        }
                    }
                }
            }
            if eos.contains(&top1_tok) {
                break;
            }
            cur_slot = primary_slot;
            cur_tok = top1_tok;
            cur_backbone = next_h;
        }

        // Expand 2-step lookahead on the strongest depth-1 alternate sibling if node budget allows!
        if let Some((mut alt_slot, mut alt_tok, mut alt_h)) = alt_branch_root {
            for sub_step in 1..=2usize.min(max_depth.saturating_sub(1)) {
                if tree.nodes.len() >= tree.max_nodes {
                    break;
                }
                let tok_emb = embed_fn(alt_tok)?;
                let q_pos = start_q_pos + sub_step;
                let (h_final, next_h) =
                    self.forward_backbone_step_with_attend(&tok_emb, &alt_h, q_pos, &mut attend_fn)?;
                let cands = predict_top_cands(&h_final, &next_h, alt_tok, 2, &mut total_skipped)?;
                if cands.is_empty() {
                    break;
                }
                let (pred_tok, lp1) = cands[0];
                let lp2 = cands.get(1).map_or(lp1 - 30.0, |c| c.1);
                let Some(ns) = tree.insert_child(alt_slot, pred_tok) else {
                    break;
                };
                if (lp1 - lp2) > tree.slot_margin(ns) {
                    tree.set_slot_margin(ns, (lp1 - lp2).max(0.0));
                }
                if let Some(&(alt2, _)) = cands.get(1) {
                    if sub_step == 1 && lp1 - lp2 <= 2.5 {
                        let _ = tree.insert_child(alt_slot, alt2);
                    }
                }
                if eos.contains(&pred_tok) {
                    break;
                }
                alt_slot = ns;
                alt_tok = pred_tok;
                alt_h = next_h;
            }
        }

        Ok((tree, margin_stopped, total_skipped))
    }

    /// Execute one autoregressive speculative draft step reusing target `L58`/`L59` KV states.
    ///
    /// Returns `(predicted_next_token, next_backbone_hidden_state, logits)`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_step_logits(
        &self,
        tok_embed: &[f32],
        backbone_h: &[f32],
        kv_slide_k: &[f32],
        kv_slide_v: &[f32],
        kv_full_k: &[f32],
        kv_full_v: &[f32],
        kv_len: usize,
        q_pos: usize,
    ) -> Result<(u32, Vec<f32>, Vec<f32>), String> {
        self.forward_step_logits_cs(
            tok_embed, backbone_h, kv_slide_k, kv_slide_v, kv_full_k, kv_full_v, kv_len, q_pos, None,
        )
        .map(|(tok, next_h, logits, _)| (tok, next_h, logits))
    }

    /// Execute one autoregressive speculative draft step with optional Cauchy-Schwarz LM-head pruning.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_step_logits_cs(
        &self,
        tok_embed: &[f32],
        backbone_h: &[f32],
        kv_slide_k: &[f32],
        kv_slide_v: &[f32],
        kv_full_k: &[f32],
        kv_full_v: &[f32],
        kv_len: usize,
        q_pos: usize,
        lmhead_cs_sieve: Option<f32>,
    ) -> Result<(u32, Vec<f32>, Vec<f32>, i64), String> {
        if !self.cfg.has_lm_head {
            return Err("forward_step_logits_cs requires assistant model.embed_tokens.weight".into());
        }
        let (h_final, next_backbone) = self.forward_backbone_step(
            tok_embed, backbone_h, kv_slide_k, kv_slide_v, kv_full_k, kv_full_v, kv_len, q_pos,
        )?;
        let native = self.model.native.as_ref();
        let (logits, lm_skipped) = if let Some(sieve_frac) = lmhead_cs_sieve {
            self.model.mat("model.embed_tokens.weight")?.project_lmhead_cs(
                &h_final,
                1,
                native,
                self.threads,
                Some(&self.embed_tile_norms_fp16),
                16,
                sieve_frac,
            )?
        } else {
            (
                self.model.mat("model.embed_tokens.weight")?.project_batch(&h_final, 1, native, self.threads)?,
                0,
            )
        };
        let next_tok = argmax_f32(&logits);
        Ok((next_tok, next_backbone, logits, lm_skipped))
    }

    /// Draft up to `draft_len` greedy candidate tokens with optional logit-margin cutoff
    /// and CS LM-head pruning (or Assistant-V7 `BHDR` trie-constrained byte beam search).
    #[allow(clippy::too_many_arguments)]
    pub fn draft_sequence_gated<F>(
        &self,
        first_tok: u32,
        mut embed_fn: F,
        first_backbone_h: &[f32],
        kv_slide_k: &[f32],
        kv_slide_v: &[f32],
        kv_full_k: &[f32],
        kv_full_v: &[f32],
        kv_len: usize,
        start_q_pos: usize,
        draft_len: usize,
        advance_pos: bool,
        eos: &[u32],
        ban_tokens: &[u32],
        min_logit_margin: f32,
        lmhead_cs_sieve: Option<f32>,
    ) -> Result<(Vec<u32>, usize, i64), String>
    where
        F: FnMut(u32) -> Result<Vec<f32>, String>,
    {
        let mut drafts = Vec::with_capacity(draft_len);
        let mut cur_tok = first_tok;
        let mut cur_backbone = first_backbone_h.to_vec();
        let mut margin_stopped = 0usize;
        let mut total_skipped = 0i64;
        for step in 0..draft_len {
            let tok_emb = embed_fn(cur_tok)?;
            let q_pos = if advance_pos { start_q_pos + step } else { start_q_pos };
            let (pred_tok, next_h, top1_v, top2_v) = if let Some((ref bh, ref trie)) = self.byte_head {
                let (h_final, next_h) = self.forward_backbone_step(
                    &tok_emb,
                    &cur_backbone,
                    kv_slide_k,
                    kv_slide_v,
                    kv_full_k,
                    kv_full_v,
                    kv_len,
                    q_pos,
                )?;
                let mut effective_bans = ban_tokens.to_vec();
                if !ban_tokens.is_empty() && !(236770..=236820).contains(&cur_tok) && !effective_bans.contains(&cur_tok) {
                    effective_bans.push(cur_tok);
                }
                let (tok, lp1, lp2, _) = bh.decode_token_beam(&h_final, trie, &effective_bans, 2)?;
                (tok, next_h, lp1, lp2)
            } else {
                let (raw_tok, next_h, mut logits, sk) = self.forward_step_logits_cs(
                    &tok_emb,
                    &cur_backbone,
                    kv_slide_k,
                    kv_slide_v,
                    kv_full_k,
                    kv_full_v,
                    kv_len,
                    q_pos,
                    lmhead_cs_sieve,
                )?;
                total_skipped += sk;
                let (ptok, t1, t2) = if ban_tokens.is_empty() && min_logit_margin <= 0.0 {
                    (raw_tok, 0.0f32, 0.0f32)
                } else {
                    if !ban_tokens.is_empty() {
                        for &tid in ban_tokens {
                            let u = tid as usize;
                            if u < logits.len() {
                                logits[u] = -1.0e9;
                            }
                        }
                        if !(236770..=236820).contains(&cur_tok) && (cur_tok as usize) < logits.len() {
                            logits[cur_tok as usize] = -1.0e9;
                        }
                    }
                    argmax_with_top2(&logits)
                };
                (ptok, next_h, t1, t2)
            };
            if min_logit_margin > 0.0 && (top1_v - top2_v) < min_logit_margin {
                margin_stopped += 1;
                break;
            }
            drafts.push(pred_tok);
            if eos.contains(&pred_tok) {
                break;
            }
            cur_tok = pred_tok;
            cur_backbone = next_h;
        }
        Ok((drafts, margin_stopped, total_skipped))
    }

    /// Draft up to `draft_len` greedy candidate tokens while optionally suppressing closed-thought
    /// control tokens (`ban_tokens`) and immediate non-digit consecutive stutters.
    #[allow(clippy::too_many_arguments)]
    pub fn draft_sequence_masked<F>(
        &self,
        first_tok: u32,
        embed_fn: F,
        first_backbone_h: &[f32],
        kv_slide_k: &[f32],
        kv_slide_v: &[f32],
        kv_full_k: &[f32],
        kv_full_v: &[f32],
        kv_len: usize,
        start_q_pos: usize,
        draft_len: usize,
        advance_pos: bool,
        eos: &[u32],
        ban_tokens: &[u32],
    ) -> Result<Vec<u32>, String>
    where
        F: FnMut(u32) -> Result<Vec<f32>, String>,
    {
        self.draft_sequence_gated(
            first_tok,
            embed_fn,
            first_backbone_h,
            kv_slide_k,
            kv_slide_v,
            kv_full_k,
            kv_full_v,
            kv_len,
            start_q_pos,
            draft_len,
            advance_pos,
            eos,
            ban_tokens,
            0.0,
            None,
        )
        .map(|(drafts, _, _)| drafts)
    }

    /// Draft up to `draft_len` greedy candidate tokens using target embedding lookup `embed_fn`.
    #[allow(clippy::too_many_arguments)]
    pub fn draft_sequence<F>(
        &self,
        first_tok: u32,
        embed_fn: F,
        first_backbone_h: &[f32],
        kv_slide_k: &[f32],
        kv_slide_v: &[f32],
        kv_full_k: &[f32],
        kv_full_v: &[f32],
        kv_len: usize,
        start_q_pos: usize,
        draft_len: usize,
        advance_pos: bool,
        eos: &[u32],
    ) -> Result<Vec<u32>, String>
    where
        F: FnMut(u32) -> Result<Vec<f32>, String>,
    {
        self.draft_sequence_masked(
            first_tok,
            embed_fn,
            first_backbone_h,
            kv_slide_k,
            kv_slide_v,
            kv_full_k,
            kv_full_v,
            kv_len,
            start_q_pos,
            draft_len,
            advance_pos,
            eos,
            &[],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{gelu_tanh_scalar, qk_norm_raw, rope_head};
    use crate::t16c::T16CMat;

    fn pack_4bit_sym(name: &str, w: &[f32], d_out: usize, d_in: usize, bsize: usize) -> T16CMat {
        let n_blks = d_in / bsize;
        let mut sc = Vec::with_capacity(d_out * n_blks);
        let mut idx = vec![0u8; d_out * d_in];
        for r in 0..d_out {
            for b in 0..n_blks {
                let s0 = r * d_in + b * bsize;
                let max_a = w[s0..s0 + bsize].iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                let step = if max_a > 0.0 { max_a / 15.0 } else { 1.0 };
                sc.push(step);
                for j in 0..bsize {
                    let q = ((w[s0 + j] / step + 15.0) * 0.5).round().clamp(0.0, 15.0) as u8;
                    idx[s0 + j] = q;
                }
            }
        }
        T16CMat::pack_from_components(name.to_string(), d_out, d_in, bsize, 4, false, false, &sc, None, &idx, None, None).unwrap()
    }

    fn det_mat(seed: u64, rows: usize, cols: usize, scale: f32) -> Vec<f32> {
        let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut out = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = ((s & 0xFFFF) as f32) / 32768.0 - 1.0;
            out.push(u * scale);
        }
        out
    }

    fn dense_mv(w: &[f32], d_out: usize, d_in: usize, x: &[f32]) -> Vec<f32> {
        let mut y = vec![0f32; d_out];
        for r in 0..d_out {
            let row = &w[r * d_in..(r + 1) * d_in];
            y[r] = row.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
        }
        y
    }

    fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0f64;
        let mut na = 0f64;
        let mut nb = 0f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            dot += (x as f64) * (y as f64);
            na += (x as f64) * (x as f64);
            nb += (y as f64) * (y as f64);
        }
        (dot / ((na * nb).sqrt() + 1e-18)) as f32
    }

    #[test]
    fn assistant_4bit_kv_sharing_matches_fp32_reference_and_roundtrips_t16c() {
        let bh = 64usize;
        let d = 64usize;
        let ff = 64usize;
        let bsize = 64usize;
        let vocab = 32usize;
        let n_heads = 4usize;
        let n_kv_slide = 2usize;
        let n_kv_full = 1usize;
        let hd_slide = 64usize;
        let hd_full = 128usize;

        let w_pre = det_mat(1, d, 2 * bh, 0.25);
        let w_post = det_mat(2, bh, d, 0.25);
        let mut w_emb = det_mat(3, vocab, d, 0.30);
        // Make one row strongly favored in synthetic embedding space for deterministic top-1
        for j in 0..d {
            w_emb[17 * d + j] *= 3.0;
        }
        let fn_w = vec![1.2f32; d];

        struct LayerW {
            q: Vec<f32>,
            o: Vec<f32>,
            g: Vec<f32>,
            u: Vec<f32>,
            dn: Vec<f32>,
            qn: Vec<f32>,
            in_n: Vec<f32>,
            pa_n: Vec<f32>,
            pf_n: Vec<f32>,
            pff_n: Vec<f32>,
            ls: f32,
            is_full: bool,
        }

        let specs = [false, true];
        let mut lws = Vec::new();
        let mut mats = vec![
            pack_4bit_sym("pre_projection.weight", &w_pre, d, 2 * bh, bsize),
            pack_4bit_sym("post_projection.weight", &w_post, bh, d, bsize),
            pack_4bit_sym("model.embed_tokens.weight", &w_emb, vocab, d, bsize),
        ];
        let mut norms = vec![("model.norm.weight".to_string(), fn_w.clone())];

        for (l, &is_full) in specs.iter().enumerate() {
            let hd = if is_full { hd_full } else { hd_slide };
            let lw = LayerW {
                q: det_mat(10 + l as u64 * 10, n_heads * hd, d, 0.2),
                o: det_mat(11 + l as u64 * 10, d, n_heads * hd, 0.2),
                g: det_mat(12 + l as u64 * 10, ff, d, 0.2),
                u: det_mat(13 + l as u64 * 10, ff, d, 0.2),
                dn: det_mat(14 + l as u64 * 10, d, ff, 0.2),
                qn: vec![1.05f32; hd],
                in_n: vec![1.1f32; d],
                pa_n: vec![0.95f32; d],
                pf_n: vec![1.0f32; d],
                pff_n: vec![1.1f32; d],
                ls: 0.75,
                is_full,
            };
            let p = format!("model.layers.{l}.");
            mats.push(pack_4bit_sym(&format!("{p}self_attn.q_proj.weight"), &lw.q, n_heads * hd, d, bsize));
            mats.push(pack_4bit_sym(&format!("{p}self_attn.o_proj.weight"), &lw.o, d, n_heads * hd, bsize));
            mats.push(pack_4bit_sym(&format!("{p}mlp.gate_proj.weight"), &lw.g, ff, d, bsize));
            mats.push(pack_4bit_sym(&format!("{p}mlp.up_proj.weight"), &lw.u, ff, d, bsize));
            mats.push(pack_4bit_sym(&format!("{p}mlp.down_proj.weight"), &lw.dn, d, ff, bsize));
            norms.push((format!("{p}self_attn.q_norm.weight"), lw.qn.clone()));
            norms.push((format!("{p}input_layernorm.weight"), lw.in_n.clone()));
            norms.push((format!("{p}post_attention_layernorm.weight"), lw.pa_n.clone()));
            norms.push((format!("{p}pre_feedforward_layernorm.weight"), lw.pf_n.clone()));
            norms.push((format!("{p}post_feedforward_layernorm.weight"), lw.pff_n.clone()));
            norms.push((format!("{p}layer_scalar"), vec![lw.ls]));
            lws.push(lw);
        }

        let tmp = std::env::temp_dir().join("assistant_test_4bit.t16c");
        T16CModel::write_file(&tmp, 1, &norms, &mats).unwrap();
        let rt = AssistantRuntime::load_31b(&tmp, None, true, 2, n_kv_slide, n_kv_full, 1024).unwrap();
        let _ = std::fs::remove_file(&tmp);

        let tok_emb = det_mat(90, 1, bh, 0.5);
        let back_h = det_mat(91, 1, bh, 0.5);
        let kv_len = 3usize;
        let kv_sk = det_mat(92, kv_len, n_kv_slide * hd_slide, 0.4);
        let kv_sv = det_mat(93, kv_len, n_kv_slide * hd_slide, 0.4);
        let kv_fk = det_mat(94, kv_len, n_kv_full * hd_full, 0.4);
        let kv_fv = det_mat(95, kv_len, n_kv_full * hd_full, 0.4);

        let dec_rows = |name: &str, rows: usize, cols: usize| -> Vec<f32> {
            let m = rt.model.mat(name).unwrap();
            let mut buf = Vec::with_capacity(rows * cols);
            for r in 0..rows {
                buf.extend_from_slice(&m.decode_row(r).unwrap());
            }
            buf
        };
        let eval_fp32 = |pre_w: &[f32], post_w: &[f32], emb_w: &[f32], layer_ws: &[(&[f32], &[f32], &[f32], &[f32], &[f32], &LayerW)]| -> (Vec<f32>, Vec<f32>) {
            let mut cat = Vec::with_capacity(2 * bh);
            cat.extend_from_slice(&tok_emb);
            cat.extend_from_slice(&back_h);
            let mut st = dense_mv(pre_w, d, 2 * bh, &cat);
            let eps = 1e-6f32;
            for &(q_w, o_w, g_w, u_w, d_w, lw) in layer_ws {
                let hd = if lw.is_full { hd_full } else { hd_slide };
                let n_kv = if lw.is_full { n_kv_full } else { n_kv_slide };
                let theta = if lw.is_full { 1_000_000.0 } else { 10_000.0 };
                let rot_d = if lw.is_full { hd / 4 } else { hd };
                let sw = if lw.is_full { None } else { Some(1024usize) };
                let (sk, sv) = if lw.is_full { (&kv_fk[..], &kv_fv[..]) } else { (&kv_sk[..], &kv_sv[..]) };
                let inp = rmsn_raw(&st, Some(&lw.in_n), eps);
                let mut q = dense_mv(q_w, n_heads * hd, d, &inp);
                q = qk_norm_raw(&q, n_heads, hd, Some(&lw.qn), eps);
                for h in 0..n_heads {
                    rope_head(&mut q[h * hd..(h + 1) * hd], 3, theta, rot_d, hd);
                }
                let ctx = attn_decode(&q, sk, sv, kv_len - 1, 3, n_heads, n_kv, hd, sw);
                let ao = dense_mv(o_w, d, n_heads * hd, &ctx);
                let na = rmsn_raw(&ao, Some(&lw.pa_n), eps);
                for i in 0..d {
                    st[i] += na[i];
                }
                let inp2 = rmsn_raw(&st, Some(&lw.pf_n), eps);
                let gt = dense_mv(g_w, ff, d, &inp2);
                let up = dense_mv(u_w, ff, d, &inp2);
                let act: Vec<f32> = gt.iter().zip(up.iter()).map(|(&a, &b)| gelu_tanh_scalar(a) * b).collect();
                let dn = dense_mv(d_w, d, ff, &act);
                let nd = rmsn_raw(&dn, Some(&lw.pff_n), eps);
                for i in 0..d {
                    st[i] = (st[i] + nd[i]) * lw.ls;
                }
            }
            let hf = rmsn_raw(&st, Some(&fn_w), eps);
            (dense_mv(post_w, bh, d, &hf), dense_mv(emb_w, vocab, d, &hf))
        };

        // 1. Exact architecture parity against dense FP32 evaluated on the quantized weights
        let d_pre = dec_rows("pre_projection.weight", d, 2 * bh);
        let d_post = dec_rows("post_projection.weight", bh, d);
        let d_emb = dec_rows("model.embed_tokens.weight", vocab, d);
        let mut d_layers: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::new();
        for (l, lw) in lws.iter().enumerate() {
            let hd = if lw.is_full { hd_full } else { hd_slide };
            let p = format!("model.layers.{l}.");
            d_layers.push((
                dec_rows(&format!("{p}self_attn.q_proj.weight"), n_heads * hd, d),
                dec_rows(&format!("{p}self_attn.o_proj.weight"), d, n_heads * hd),
                dec_rows(&format!("{p}mlp.gate_proj.weight"), ff, d),
                dec_rows(&format!("{p}mlp.up_proj.weight"), ff, d),
                dec_rows(&format!("{p}mlp.down_proj.weight"), d, ff),
            ));
        }
        let d_refs: Vec<_> = lws.iter().enumerate().map(|(l, lw)| {
            let x = &d_layers[l];
            (x.0.as_slice(), x.1.as_slice(), x.2.as_slice(), x.3.as_slice(), x.4.as_slice(), lw)
        }).collect();
        let (exact_post, exact_logits) = eval_fp32(&d_pre, &d_post, &d_emb, &d_refs);

        // 2. End-to-end 4-bit fidelity against unquantized FP32 weights
        let raw_refs: Vec<_> = lws.iter().map(|lw| {
            (lw.q.as_slice(), lw.o.as_slice(), lw.g.as_slice(), lw.u.as_slice(), lw.dn.as_slice(), lw)
        }).collect();
        let (ref_post, ref_logits) = eval_fp32(&w_pre, &w_post, &w_emb, &raw_refs);

        let (tok_got, got_post, got_logits) = rt
            .forward_step_logits(&tok_emb, &back_h, &kv_sk, &kv_sv, &kv_fk, &kv_fv, kv_len, 3)
            .unwrap();
        let c_exact_post = cos_sim(&got_post, &exact_post);
        let c_exact_logits = cos_sim(&got_logits, &exact_logits);
        assert!(c_exact_post > 0.99999, "decoded-weight post_projection cos={c_exact_post}");
        assert!(c_exact_logits > 0.99999, "decoded-weight logits cos={c_exact_logits}");
        assert_eq!(tok_got, argmax_f32(&exact_logits));

        let c_post = cos_sim(&got_post, &ref_post);
        let c_logits = cos_sim(&got_logits, &ref_logits);
        assert!(c_post > 0.97, "raw FP32 post_projection cos={c_post}");
        assert!(c_logits > 0.97, "raw FP32 logits cos={c_logits}");
        assert_eq!(tok_got, argmax_f32(&ref_logits));

        // Draft sequence honors EOS early stop & requested length
        let seq = rt
            .draft_sequence(
                1,
                |_| Ok(tok_emb.clone()),
                &back_h,
                &kv_sk,
                &kv_sv,
                &kv_fk,
                &kv_fv,
                kv_len,
                3,
                3,
                false,
                &[tok_got],
            )
            .unwrap();
        assert_eq!(seq, vec![tok_got]);

        // Margin gate stops drafting immediately when top1 - top2 falls below threshold,
        // while keeping exact match when threshold <= 0.
        let (gated_stop, m_stops, _) = rt
            .draft_sequence_gated(
                1,
                |_| Ok(tok_emb.clone()),
                &back_h,
                &kv_sk,
                &kv_sv,
                &kv_fk,
                &kv_fv,
                kv_len,
                3,
                3,
                false,
                &[],
                &[],
                1.0e6,
                Some(0.0),
            )
            .unwrap();
        assert!(gated_stop.is_empty());
        assert_eq!(m_stops, 1);
    }

    #[test]
    fn assistant_v7_byte_head_runs_without_embed_tokens_and_emits_valid_token() {
        use crate::byte_head::{BHDR_VOCAB_SYM, SYM_EOT, SYM_SP};

        let bh = 64usize;
        let d = 64usize;
        let ff = 64usize;
        let bsize = 64usize;
        let n_heads = 4usize;
        let n_kv_slide = 2usize;
        let n_kv_full = 1usize;
        let hd_slide = 64usize;
        let hd_full = 128usize;

        let mut mats = vec![
            pack_4bit_sym("pre_projection.weight", &det_mat(21, d, 2 * bh, 0.25), d, 2 * bh, bsize),
            pack_4bit_sym("post_projection.weight", &det_mat(22, bh, d, 0.25), bh, d, bsize),
        ];
        let mut norms = vec![("model.norm.weight".to_string(), vec![1.0f32; d])];
        for (l, &is_full) in [false, true].iter().enumerate() {
            let hd = if is_full { hd_full } else { hd_slide };
            let p = format!("model.layers.{l}.");
            mats.push(pack_4bit_sym(&format!("{p}self_attn.q_proj.weight"), &det_mat(30 + l as u64 * 10, n_heads * hd, d, 0.2), n_heads * hd, d, bsize));
            mats.push(pack_4bit_sym(&format!("{p}self_attn.o_proj.weight"), &det_mat(31 + l as u64 * 10, d, n_heads * hd, 0.2), d, n_heads * hd, bsize));
            mats.push(pack_4bit_sym(&format!("{p}mlp.gate_proj.weight"), &det_mat(32 + l as u64 * 10, ff, d, 0.2), ff, d, bsize));
            mats.push(pack_4bit_sym(&format!("{p}mlp.up_proj.weight"), &det_mat(33 + l as u64 * 10, ff, d, 0.2), ff, d, bsize));
            mats.push(pack_4bit_sym(&format!("{p}mlp.down_proj.weight"), &det_mat(34 + l as u64 * 10, d, ff, 0.2), d, ff, bsize));
            norms.push((format!("{p}self_attn.q_norm.weight"), vec![1.0f32; hd]));
            norms.push((format!("{p}input_layernorm.weight"), vec![1.0f32; d]));
            norms.push((format!("{p}post_attention_layernorm.weight"), vec![1.0f32; d]));
            norms.push((format!("{p}pre_feedforward_layernorm.weight"), vec![1.0f32; d]));
            norms.push((format!("{p}post_feedforward_layernorm.weight"), vec![1.0f32; d]));
            norms.push((format!("{p}layer_scalar"), vec![0.8f32]));
        }

        let tmp_t16c = std::env::temp_dir().join(format!("assistant_v7_body_{}.t16c", std::process::id()));
        let tmp_bhdr = std::env::temp_dir().join(format!("assistant_v7_head_{}.bhdr", std::process::id()));
        T16CModel::write_file(&tmp_t16c, 1, &norms, &mats).unwrap();

        // Fail closed when embed_tokens is missing and no byte head is provided
        assert!(
            AssistantRuntime::load_31b(&tmp_t16c, None, true, 2, n_kv_slide, n_kv_full, 1024).is_err()
        );

        let mut dec = ByteHeadDecoder {
            d_in: d,
            d_byte: 32,
            vocab_sym: BHDR_VOCAB_SYM,
            max_bytes: 16,
            flags: 0,
            w_init: vec![0i8; 32 * d],
            s_init: vec![0.02f32; 32],
            b_init: vec![0.0f32; 32],
            w_ctx: vec![0i8; 96 * d],
            s_ctx: vec![0.01f32; 96],
            e_sym: vec![0i8; BHDR_VOCAB_SYM * 32],
            s_sym: vec![0.02f32; BHDR_VOCAB_SYM],
            w_ih: vec![0i8; 96 * 32],
            s_ih: vec![0.02f32; 96],
            b_ih: vec![0.0f32; 96],
            w_hh: vec![0i8; 96 * 32],
            s_hh: vec![0.02f32; 96],
            b_hh: vec![0.0f32; 96],
            w_out: vec![0i8; BHDR_VOCAB_SYM * 32],
            s_out: vec![0.1f32; BHDR_VOCAB_SYM],
            b_out: vec![-2.0f32; BHDR_VOCAB_SYM],
        };
        dec.b_out[SYM_SP as usize] = 4.0;
        dec.b_out[b'y' as usize] = 4.0;
        dec.b_out[b'a' as usize] = 3.5;
        dec.b_out[SYM_EOT as usize] = 5.0;
        for j in 0..32 {
            dec.b_ih[32 + j] = -3.0;
        }
        dec.e_sym[(SYM_SP as usize) * 32] = 127;
        dec.save(&tmp_bhdr).unwrap();

        let mut id_to_tok = vec![String::new(); 120];
        id_to_tok[1] = "<eos>".into();
        id_to_tok[77] = "\u{2581}y".into();
        id_to_tok[78] = "\u{2581}a".into();
        id_to_tok[106] = "<turn|>".into();
        let trie = Arc::new(BpePrefixTrie::from_id_to_token(&id_to_tok));

        let rt = AssistantRuntime::load_31b_with_byte_head(
            &tmp_t16c,
            Some(&tmp_bhdr),
            Some(trie),
            None,
            true,
            2,
            n_kv_slide,
            n_kv_full,
            1024,
            120,
        )
        .unwrap();
        assert!(!rt.cfg.has_lm_head);
        assert!(rt.has_byte_head());
        assert_eq!(rt.cfg.vocab, 120);
        assert!(rt.resident_bytes() > 0);

        let tok_emb = det_mat(90, 1, bh, 0.5);
        let back_h = det_mat(91, 1, bh, 0.5);
        let kv_sk = det_mat(92, 3, n_kv_slide * hd_slide, 0.4);
        let kv_sv = det_mat(93, 3, n_kv_slide * hd_slide, 0.4);
        let kv_fk = det_mat(94, 3, n_kv_full * hd_full, 0.4);
        let kv_fv = det_mat(95, 3, n_kv_full * hd_full, 0.4);

        let (drafts, m_stops, skipped) = rt
            .draft_sequence_gated(
                2,
                |_| Ok(tok_emb.clone()),
                &back_h,
                &kv_sk,
                &kv_sv,
                &kv_fk,
                &kv_fv,
                3,
                3,
                2,
                true,
                &[1, 106],
                &[100, 101],
                0.0,
                Some(0.25),
            )
            .unwrap();
        assert_eq!(skipped, 0);
        assert_eq!(m_stops, 0);
        assert_eq!(drafts, vec![77, 78]);

        // Verify zero-RAM Target-Embed Sharing (`has_lm_head=false`, `byte_head=None`, `target_vocab > 0`)
        let rt_shared = AssistantRuntime::load_31b_with_byte_head(
            &tmp_t16c,
            None,
            None,
            None,
            true,
            2,
            n_kv_slide,
            n_kv_full,
            1024,
            120,
        )
        .unwrap();
        assert!(rt_shared.uses_target_shared_head());
        let (tree_sh, _, _) = rt_shared
            .draft_tree_with_target_head(
                2,
                |_| Ok(tok_emb.clone()),
                &back_h,
                3,
                3,
                8,
                &[],
                &[],
                &[1, 106],
                &[100, 101],
                0.0,
                None,
                |is_full, q, _qp| {
                    let out_dim = if is_full { n_heads * hd_full } else { n_heads * hd_slide };
                    let _ = q;
                    Ok(vec![0.05f32; out_dim])
                },
                |_hf, next_bh, _prev, _wk, _bans| {
                    assert_eq!(next_bh.len(), bh);
                    Ok((vec![(42, 9.5), (43, 4.0)], 0))
                },
            )
            .unwrap();
        assert!(tree_sh.nodes.len() >= 2);
        assert_eq!(tree_sh.nodes[1].token_id, 42);
        assert!((tree_sh.slot_margin(1) - 5.5).abs() < 1e-4);

        let _ = std::fs::remove_file(&tmp_t16c);
        let _ = std::fs::remove_file(&tmp_bhdr);
    }
}
