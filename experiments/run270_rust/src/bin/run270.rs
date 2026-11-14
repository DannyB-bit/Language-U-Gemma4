//! run270 — standalone native Rust inference runtime for Gemma 3 270M (zero Python).
//!
//! Supports three weight modes:
//!   1. `--exact`: reads BF16/F32 2D weights, embeddings, and 1D norms directly from `model.safetensors`.
//!   2. `--quantize <scheme> [--h8]`: packs 2D projections directly in RAM (`b1`, `q2..q12`,
//!      optional Sylvester-Hadamard `H8` rotation) with zero unpacked weight buffers, executing
//!      4-thread direct dequant-in-matvec (`--embed-mode bf16|f32|<scheme>`).
//!   3. Glyph-stream mode (default when neither `--exact` nor `--quantize` is set): decodes the
//!      132 B OPT4 stream + sidecars into RAM.

use run270_rust::codec::{bf16_to_f32, decode_opt4_glyph, CB_CLASSES, CB_DIM, GLYPH_BYTES, GLYPH_WEIGHTS, LANU_MAGIC, STREAM_HEADER_BYTES};
use run270_rust::json;
use run270_rust::math::{attn_decode, gelu_tanh_scalar, qk_norm_gemma, rmsn_gemma, rope_head, top_k_top_p_sample, Rng};
use run270_rust::packed::{PackedMatrix, QuantScheme};
use run270_rust::safetensors::SafeTensors;
use run270_rust::tokenizer::Tokenizer;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

struct Cfg {
    d: usize,
    n_layers: usize,
    ff: usize,
    n_heads: usize,
    n_kv: usize,
    hd: usize,
    window: usize,
    eps: f32,
    theta_global: f32,
    theta_local: f32,
    layer_sliding: Vec<bool>,
    vocab: usize,
}

fn load_cfg(path: &Path) -> Result<Cfg, String> {
    let b = fs::read(path).map_err(|e| format!("read config {}: {e}", path.display()))?;
    let j = json::parse(&b)?;
    let gu = |k: &str| j.get(k).and_then(|v| v.as_usize()).ok_or_else(|| format!("config missing {k}"));
    let gf = |k: &str| j.get(k).and_then(|v| v.as_f64()).ok_or_else(|| format!("config missing {k}")).map(|v| v as f32);
    let lt = j.get("layer_types").and_then(|v| v.as_arr()).ok_or("config missing layer_types")?;
    let layer_sliding: Vec<bool> = lt.iter().map(|v| v.as_str() == Some("sliding_attention")).collect();
    let cfg = Cfg {
        d: gu("hidden_size")?,
        n_layers: gu("num_hidden_layers")?,
        ff: gu("intermediate_size")?,
        n_heads: gu("num_attention_heads")?,
        n_kv: gu("num_key_value_heads")?,
        hd: gu("head_dim")?,
        window: gu("sliding_window")?,
        eps: gf("rms_norm_eps")?,
        theta_global: gf("rope_theta")?,
        theta_local: gf("rope_local_base_freq")?,
        layer_sliding,
        vocab: gu("vocab_size")?,
    };
    if cfg.layer_sliding.len() != cfg.n_layers {
        return Err(format!(
            "config layer_types length {} != num_hidden_layers {}",
            cfg.layer_sliding.len(),
            cfg.n_layers
        ));
    }
    Ok(cfg)
}

enum ProjBackend {
    ExactF32 {
        d_out: usize,
        d_in: usize,
        w: Vec<f32>,
    },
    Packed(PackedMatrix),
    T16C {
        mat: run270_rust::t16c::T16CMat,
        native: Option<std::sync::Arc<run270_rust::t16c::NativeTile16Kernel>>,
    },
}

impl ProjBackend {
    fn dims(&self) -> (usize, usize) {
        match self {
            Self::ExactF32 { d_out, d_in, .. } => (*d_out, *d_in),
            Self::Packed(pm) => (pm.d_out, pm.d_in),
            Self::T16C { mat, .. } => (mat.d_out, mat.d_in),
        }
    }

    fn byte_len(&self) -> usize {
        match self {
            Self::ExactF32 { w, .. } => w.len() * 4,
            Self::Packed(pm) => pm.bytes.len(),
            Self::T16C { mat, .. } => mat.total_bytes(),
        }
    }

    fn matvec(&self, x: &[f32], threads: usize) -> Result<Vec<f32>, String> {
        match self {
            Self::ExactF32 { d_out, d_in, w } => {
                if x.len() != *d_in {
                    return Err(format!("x len {} != d_in {d_in}", x.len()));
                }
                let mut y = vec![0f32; *d_out];
                let nthreads = threads.max(1);
                std::thread::scope(|s| {
                    let chunk = d_out.div_ceil(nthreads).max(1);
                    let mut handles = Vec::new();
                    for (t, yb) in y.chunks_mut(chunk).enumerate() {
                        let r0 = t * chunk;
                        handles.push(s.spawn(move || {
                            for (r, slot) in yb.iter_mut().enumerate() {
                                let row = &w[(r0 + r) * *d_in..(r0 + r + 1) * *d_in];
                                let mut acc = 0f64;
                                for j in 0..*d_in {
                                    acc += row[j] as f64 * x[j] as f64;
                                }
                                *slot = acc as f32;
                            }
                        }));
                    }
                    for h in handles {
                        h.join().unwrap();
                    }
                });
                Ok(y)
            }
            Self::Packed(pm) => pm.matvec(x, threads),
            Self::T16C { mat, native } => mat.project_batch(x, 1, native.as_deref(), threads),
        }
    }
}

#[derive(Clone, Copy)]
enum EmbedMode {
    F32,
    Bf16,
    Quant { scheme: QuantScheme, use_h8: bool },
}

impl EmbedMode {
    fn parse(s: &str, use_h8: bool) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "f32" | "exact" => {
                if use_h8 {
                    return Err("--embed-h8 is only valid when --embed-mode uses a quant scheme".into());
                }
                Ok(Self::F32)
            }
            "bf16" => {
                if use_h8 {
                    return Err("--embed-h8 is only valid when --embed-mode uses a quant scheme".into());
                }
                Ok(Self::Bf16)
            }
            other => Ok(Self::Quant {
                scheme: QuantScheme::parse(other)?,
                use_h8,
            }),
        }
    }

    fn label(self) -> String {
        match self {
            Self::F32 => "f32".into(),
            Self::Bf16 => "bf16".into(),
            Self::Quant { scheme, use_h8 } => {
                if use_h8 {
                    format!("{}+h8", scheme.name())
                } else {
                    scheme.name().to_string()
                }
            }
        }
    }
}

enum EmbedBackend {
    ExactF32(Vec<f32>),
    PackedBf16(Vec<u16>),
    Packed(PackedMatrix),
    T16C {
        mat: run270_rust::t16c::T16CMat,
        native: Option<std::sync::Arc<run270_rust::t16c::NativeTile16Kernel>>,
    },
}

impl EmbedBackend {
    fn byte_len(&self) -> usize {
        match self {
            Self::ExactF32(v) => v.len() * 4,
            Self::PackedBf16(v) => v.len() * 2,
            Self::Packed(pm) => pm.bytes.len(),
            Self::T16C { mat, .. } => mat.total_bytes(),
        }
    }

    fn lookup_scaled(&self, ti: usize, d: usize, vocab: usize, escale: f32) -> Result<Vec<f32>, String> {
        if ti >= vocab {
            return Err(format!("token id {ti} out of bounds for vocab {vocab}"));
        }
        match self {
            Self::ExactF32(v) => {
                if (ti + 1) * d > v.len() {
                    return Err(format!("token id {ti} exceeds embed f32 length"));
                }
                Ok(v[ti * d..(ti + 1) * d].iter().map(|&val| val * escale).collect())
            }
            Self::PackedBf16(v) => {
                if (ti + 1) * d > v.len() {
                    return Err(format!("token id {ti} exceeds embed bf16 length"));
                }
                Ok(v[ti * d..(ti + 1) * d].iter().map(|&b| bf16_to_f32(b) * escale).collect())
            }
            Self::Packed(pm) => {
                let mut row = pm.decode_row(ti)?;
                for v in &mut row {
                    *v *= escale;
                }
                Ok(row)
            }
            Self::T16C { mat, .. } => {
                let mut row = mat.decode_row(ti)?;
                if row.len() != d {
                    return Err(format!("T16C embed decode_row len {} != d {d}", row.len()));
                }
                for v in &mut row {
                    *v *= escale;
                }
                Ok(row)
            }
        }
    }

    fn compute_logits(&self, x: &[f32], d: usize, vocab: usize, threads: usize) -> Result<Vec<f32>, String> {
        match self {
            Self::ExactF32(embed) => {
                let mut logits = vec![0f32; vocab];
                let nthreads = threads.max(1);
                let chunk = vocab.div_ceil(nthreads).max(1);
                std::thread::scope(|s| {
                    let mut handles = Vec::new();
                    for (tid, out_chunk) in logits.chunks_mut(chunk).enumerate() {
                        let t0 = tid * chunk;
                        handles.push(s.spawn(move || {
                            for (local_t, slot) in out_chunk.iter_mut().enumerate() {
                                let row = &embed[(t0 + local_t) * d..(t0 + local_t + 1) * d];
                                let mut acc = 0f64;
                                for i in 0..d {
                                    acc += row[i] as f64 * x[i] as f64;
                                }
                                *slot = acc as f32;
                            }
                        }));
                    }
                    for h in handles {
                        h.join().unwrap();
                    }
                });
                Ok(logits)
            }
            Self::PackedBf16(embed) => {
                let mut logits = vec![0f32; vocab];
                let nthreads = threads.max(1);
                let chunk = vocab.div_ceil(nthreads).max(1);
                std::thread::scope(|s| {
                    let mut handles = Vec::new();
                    for (tid, out_chunk) in logits.chunks_mut(chunk).enumerate() {
                        let t0 = tid * chunk;
                        handles.push(s.spawn(move || {
                            for (local_t, slot) in out_chunk.iter_mut().enumerate() {
                                let row = &embed[(t0 + local_t) * d..(t0 + local_t + 1) * d];
                                let mut acc = 0f64;
                                let mut blk = 0usize;
                                while blk + 32 <= d {
                                    let mut sub = 0f32;
                                    for i in 0..32 {
                                        sub += bf16_to_f32(row[blk + i]) * x[blk + i];
                                    }
                                    acc += sub as f64;
                                    blk += 32;
                                }
                                if blk < d {
                                    let mut sub = 0f32;
                                    for i in blk..d {
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
                Ok(logits)
            }
            Self::Packed(pm) => pm.matvec(x, threads),
            Self::T16C { mat, native } => mat.project_batch(x, 1, native.as_deref(), threads),
        }
    }
}

struct Art {
    projs: HashMap<String, ProjBackend>,
    embed: EmbedBackend,
    norms: HashMap<String, Vec<f32>>,
    threads: usize,
}

impl Art {
    fn proj(&self, key: &str, x: &[f32]) -> Result<Vec<f32>, String> {
        let p = self.projs.get(key).ok_or_else(|| format!("missing projection {key}"))?;
        let (_, d_in) = p.dims();
        if x.len() != d_in {
            return Err(format!("{key}: x len {} != d_in {d_in}", x.len()));
        }
        p.matvec(x, self.threads).map_err(|e| format!("{key}: {e}"))
    }

    fn norm(&self, key: &str) -> Result<&[f32], String> {
        self.norms.get(key).map(|v| v.as_slice()).ok_or_else(|| format!("missing norm {key}"))
    }

    fn memory_bytes(&self) -> (usize, usize, usize, usize) {
        let proj_bytes: usize = self.projs.values().map(ProjBackend::byte_len).sum();
        let proj_weights: usize = self.projs.values().map(|p| {
            let (o, i) = p.dims();
            o * i
        }).sum();
        let embed_bytes = self.embed.byte_len();
        let norm_bytes: usize = self.norms.values().map(|v| v.len() * 4).sum();
        (proj_bytes, proj_weights, embed_bytes, norm_bytes)
    }
}

struct Kv {
    k: Vec<f32>,
    v: Vec<f32>,
}

struct Model270 {
    art: Art,
    cfg: Cfg,
    kv: Vec<Kv>,
}

impl Model270 {
    fn pfx(l: usize) -> String {
        format!("model.layers.{l}")
    }

    fn layer_theta(&self, l: usize) -> f32 {
        if self.cfg.layer_sliding[l] {
            self.cfg.theta_local
        } else {
            self.cfg.theta_global
        }
    }

    fn sw(&self, l: usize) -> Option<usize> {
        if self.cfg.layer_sliding[l] {
            Some(self.cfg.window)
        } else {
            None
        }
    }

    fn reset_kv(&mut self) {
        for layer_kv in &mut self.kv {
            layer_kv.k.clear();
            layer_kv.v.clear();
        }
    }

    fn forward(&mut self, tid: u32, pos: usize) -> Result<Vec<f32>, String> {
        let c = &self.cfg;
        let d = c.d;
        let escale = (d as f32).sqrt();
        let mut x = self.art.embed.lookup_scaled(tid as usize, d, c.vocab, escale)?;

        for l in 0..c.n_layers {
            let pfx = Self::pfx(l);
            let h = rmsn_gemma(&x, self.art.norm(&format!("{pfx}.input_layernorm.weight"))?, c.eps);
            let q_raw = self.art.proj(&format!("{pfx}.self_attn.q_proj.weight"), &h)?;
            let k_raw = self.art.proj(&format!("{pfx}.self_attn.k_proj.weight"), &h)?;
            let v = self.art.proj(&format!("{pfx}.self_attn.v_proj.weight"), &h)?;

            let mut q = qk_norm_gemma(&q_raw, c.n_heads, c.hd, self.art.norm(&format!("{pfx}.self_attn.q_norm.weight"))?, c.eps);
            let mut k = qk_norm_gemma(&k_raw, c.n_kv, c.hd, self.art.norm(&format!("{pfx}.self_attn.k_norm.weight"))?, c.eps);

            let qs = 1.0 / (c.hd as f32).sqrt();
            for vq in &mut q {
                *vq *= qs;
            }
            let theta = self.layer_theta(l);
            for hq in 0..c.n_heads {
                rope_head(&mut q[hq * c.hd..(hq + 1) * c.hd], pos as u32, theta, c.hd, c.hd);
            }
            for kh in 0..c.n_kv {
                rope_head(&mut k[kh * c.hd..(kh + 1) * c.hd], pos as u32, theta, c.hd, c.hd);
            }

            let t = self.kv[l].k.len() / (c.n_kv * c.hd);
            self.kv[l].k.extend_from_slice(&k);
            self.kv[l].v.extend_from_slice(&v);
            let ctx = attn_decode(&q, &self.kv[l].k, &self.kv[l].v, t, pos, c.n_heads, c.n_kv, c.hd, self.sw(l));

            let attn = self.art.proj(&format!("{pfx}.self_attn.o_proj.weight"), &ctx)?;
            let attn = rmsn_gemma(&attn, self.art.norm(&format!("{pfx}.post_attention_layernorm.weight"))?, c.eps);
            for i in 0..d {
                x[i] += attn[i];
            }

            let h2 = rmsn_gemma(&x, self.art.norm(&format!("{pfx}.pre_feedforward_layernorm.weight"))?, c.eps);
            let g = self.art.proj(&format!("{pfx}.mlp.gate_proj.weight"), &h2)?;
            let u = self.art.proj(&format!("{pfx}.mlp.up_proj.weight"), &h2)?;
            let gu: Vec<f32> = g.iter().zip(&u).map(|(&a, &b)| gelu_tanh_scalar(a) * b).collect();
            let mlp = self.art.proj(&format!("{pfx}.mlp.down_proj.weight"), &gu)?;
            let mlp = rmsn_gemma(&mlp, self.art.norm(&format!("{pfx}.post_feedforward_layernorm.weight"))?, c.eps);
            for i in 0..d {
                x[i] += mlp[i];
            }
        }
        Ok(rmsn_gemma(&x, self.art.norm("model.norm.weight")?, c.eps))
    }

    fn compute_logits(&self, x: &[f32]) -> Result<Vec<f32>, String> {
        self.art.embed.compute_logits(x, self.cfg.d, self.cfg.vocab, self.art.threads)
    }

    fn generate_sequence(
        &mut self,
        ids: &[u32],
        max_tokens: usize,
        sample_cfg: Option<(usize, f64)>,
        rng: &mut Rng,
        verbose_tk: Option<&Tokenizer>,
    ) -> Result<(Vec<f32>, Vec<u32>, f64, f64, usize), String> {
        self.reset_kv();
        let t0 = std::time::Instant::now();
        let mut x = vec![0f32; self.cfg.d];
        for (pos, &t) in ids.iter().enumerate() {
            x = self.forward(t, pos)?;
        }
        let prefill_s = t0.elapsed().as_secs_f64();
        let mut forwards = ids.len();
        let first_logits = self.compute_logits(&x)?;
        let mut logits = first_logits.clone();

        let mut generated: Vec<u32> = Vec::new();
        let eos: [u32; 2] = [1, 106];
        let t1 = std::time::Instant::now();
        let top_k = sample_cfg.map_or(1, |(k, _)| k.max(8));
        for step in 0..max_tokens {
            let (cand, top_logits) = self.head_topk(&logits, top_k);
            let nxt = if let Some((k, p)) = sample_cfg {
                let mut lcopy = top_logits.clone();
                let ci = top_k_top_p_sample(&mut lcopy, &cand, 1.0, k, p, 1.0, &generated, rng);
                cand[ci]
            } else {
                cand[0]
            };
            generated.push(nxt);
            if let Some(tk) = verbose_tk {
                let txt = tk.decode(&[nxt]);
                println!("[STEP {step}] id={nxt} text={txt:?}");
            }
            if eos.contains(&nxt) {
                break;
            }
            if step + 1 < max_tokens {
                let tp = std::time::Instant::now();
                x = self.forward(nxt, ids.len() + step)?;
                logits = self.compute_logits(&x)?;
                forwards += 1;
                if verbose_tk.is_some() {
                    eprintln!("[run270] decode step {step} in {:.2}s", tp.elapsed().as_secs_f64());
                }
            }
        }
        let decode_s = t1.elapsed().as_secs_f64();
        Ok((first_logits, generated, prefill_s, decode_s, forwards))
    }

    fn head_topk(&self, logits: &[f32], k: usize) -> (Vec<u32>, Vec<f64>) {
        let k = k.min(logits.len()).max(1);
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        if k < idx.len() {
            idx.select_nth_unstable_by(k - 1, |&a, &b| {
                logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
            });
            idx.truncate(k);
        }
        idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
        (idx.iter().map(|&t| t as u32).collect(), idx.iter().map(|&t| logits[t] as f64).collect())
    }
}

fn load_embed_from_safetensors(sf: &SafeTensors, mode: EmbedMode, threads: usize) -> Result<EmbedBackend, String> {
    match mode {
        EmbedMode::F32 => Ok(EmbedBackend::ExactF32(sf.read_tensor_f32("model.embed_tokens.weight")?)),
        EmbedMode::Bf16 => Ok(EmbedBackend::PackedBf16(sf.read_tensor_bf16_u16("model.embed_tokens.weight")?)),
        EmbedMode::Quant { scheme, use_h8 } => {
            let meta = sf
                .tensors
                .get("model.embed_tokens.weight")
                .ok_or_else(|| "missing model.embed_tokens.weight".to_string())?;
            if meta.shape.len() != 2 {
                return Err(format!("embed_tokens expected rank-2, got {:?}", meta.shape));
            }
            let d_out = meta.shape[0];
            let d_in = meta.shape[1];
            let mut pm = PackedMatrix::new_empty(d_out, d_in, scheme, use_h8)?;
            let mut r0 = 0usize;
            while r0 < d_out {
                let take = (d_out - r0).min(4096);
                let chunk = sf.read_tensor_rows_f32("model.embed_tokens.weight", r0, take)?;
                pm.pack_rows_range_f32(r0, take, &chunk, threads)?;
                r0 += take;
            }
            Ok(EmbedBackend::Packed(pm))
        }
    }
}

fn load_safetensors_artifacts(
    model_path: &Path,
    proj_quant: Option<(QuantScheme, bool)>,
    embed_mode: EmbedMode,
    threads: usize,
) -> Result<Art, String> {
    let sf = SafeTensors::open(model_path)?;
    let mut keys: Vec<String> = sf.tensors.keys().cloned().collect();
    keys.sort();
    let mut projs = HashMap::new();
    let mut norms = HashMap::new();
    for name in &keys {
        let meta = &sf.tensors[name];
        if meta.shape.len() == 2 && name != "model.embed_tokens.weight" {
            let w = sf.read_tensor_f32(name)?;
            let pb = match proj_quant {
                Some((scheme, use_h8)) => {
                    ProjBackend::Packed(PackedMatrix::pack_from_f32_mt(meta.shape[0], meta.shape[1], &w, scheme, use_h8, threads)?)
                }
                None => ProjBackend::ExactF32 {
                    d_out: meta.shape[0],
                    d_in: meta.shape[1],
                    w,
                },
            };
            projs.insert(name.clone(), pb);
        } else if meta.shape.len() == 1 {
            norms.insert(name.clone(), sf.read_tensor_f32(name)?);
        }
    }
    let embed = load_embed_from_safetensors(&sf, embed_mode, threads)?;
    let q_label = match proj_quant {
        Some((s, h8)) => format!("PACKED_{}{}", s.name().to_ascii_uppercase(), if h8 { "+H8" } else { "" }),
        None => "EXACT_SAFETENSORS".to_string(),
    };
    eprintln!(
        "[run270] weight_source={q_label} projections={} norms={} embed_mode={} embed_bytes={}",
        projs.len(),
        norms.len(),
        embed_mode.label(),
        embed.byte_len()
    );
    Ok(Art { projs, embed, norms, threads })
}

fn load_t16c_artifacts(
    t16c_path: &Path,
    native_lib: Option<&Path>,
    disable_native: bool,
    fallback_model: Option<&Path>,
    fallback_embed_mode: EmbedMode,
    threads: usize,
) -> Result<Art, String> {
    let mut t16c = run270_rust::t16c::T16CModel::load(t16c_path, native_lib, disable_native)?;
    let native = t16c.native.take().map(std::sync::Arc::new);
    let embed = if let Some(mat) = t16c.mats.remove("model.embed_tokens.weight") {
        EmbedBackend::T16C { mat, native: native.clone() }
    } else if let Some(mp) = fallback_model {
        let sf = SafeTensors::open(mp)?;
        load_embed_from_safetensors(&sf, fallback_embed_mode, threads)?
    } else {
        return Err("T16C missing model.embed_tokens.weight and no --model fallback provided".into());
    };
    let mut projs = HashMap::with_capacity(t16c.mats.len());
    for (k, mat) in t16c.mats {
        projs.insert(k, ProjBackend::T16C { mat, native: native.clone() });
    }
    eprintln!(
        "[run270] weight_source=T16C native_ffi={} projections={} norms={} embed_bytes={}",
        native.as_ref().map_or("disabled", |n| n.path.as_str()),
        projs.len(),
        t16c.norms.len(),
        embed.byte_len()
    );
    Ok(Art {
        projs,
        embed,
        norms: t16c.norms,
        threads,
    })
}

fn load_glyph_artifacts(
    stream_path: &Path,
    layout_path: &Path,
    cb_path: &Path,
    embed_path: &Path,
    norms_bin_path: &Path,
    norms_json_path: &Path,
    threads: usize,
) -> Result<Art, String> {
    let stream_raw = fs::read(stream_path).map_err(|e| format!("read stream {}: {e}", stream_path.display()))?;
    if stream_raw.len() < STREAM_HEADER_BYTES || &stream_raw[0..4] != LANU_MAGIC {
        return Err("invalid LANU glyph stream header".into());
    }
    let cb_raw = fs::read(cb_path).map_err(|e| format!("read codebook {}: {e}", cb_path.display()))?;
    if cb_raw.len() != CB_CLASSES * CB_DIM * 4 {
        return Err(format!("invalid codebook size {}", cb_raw.len()));
    }
    let codebook: Vec<f32> = cb_raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let lj = json::parse(&fs::read(layout_path).map_err(|e| e.to_string())?)?;
    let arr = lj.get("projections").and_then(|v| v.as_arr()).ok_or("layout missing projections")?;
    let mut projs = HashMap::with_capacity(arr.len());
    for p in arr {
        let fk = p.get("full_key").and_then(|v| v.as_str()).ok_or("projection missing full_key")?.to_string();
        let d_out = p.get("d_out").and_then(|v| v.as_usize()).ok_or("projection missing d_out")?;
        let d_in = p.get("d_in").and_then(|v| v.as_usize()).ok_or("projection missing d_in")?;
        let off = p.get("stream_byte_offset").and_then(|v| v.as_usize()).ok_or("projection missing stream_byte_offset")?;
        let nglyphs = (d_out * d_in).div_ceil(GLYPH_WEIGHTS);
        let need_end = off.checked_add(nglyphs * GLYPH_BYTES).ok_or("glyph stream offset overflow")?;
        if need_end > stream_raw.len() {
            return Err(format!("{fk}: glyph stream truncated"));
        }
        let mut w = vec![0f32; d_out * d_in];
        for g in 0..nglyphs {
            let goff = off + g * GLYPH_BYTES;
            let mut gw = [0f32; GLYPH_WEIGHTS];
            decode_opt4_glyph(&stream_raw[goff..goff + GLYPH_BYTES], &codebook, &mut gw);
            let base = g * GLYPH_WEIGHTS;
            let n = (d_out * d_in - base).min(GLYPH_WEIGHTS);
            w[base..base + n].copy_from_slice(&gw[..n]);
        }
        projs.insert(fk, ProjBackend::ExactF32 { d_out, d_in, w });
    }

    let embed_raw = fs::read(embed_path).map_err(|e| format!("read embed {}: {e}", embed_path.display()))?;
    if embed_raw.len() % 2 != 0 {
        return Err("embed_tokens.bin byte count is not even BF16".into());
    }
    let embed = EmbedBackend::ExactF32(
        embed_raw
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
    );

    let nb = fs::read(norms_bin_path).map_err(|e| format!("read norms {}: {e}", norms_bin_path.display()))?;
    let nj = json::parse(&fs::read(norms_json_path).map_err(|e| format!("read norms json: {e}"))?)?;
    let mut norms = HashMap::new();
    if let json::Json::Obj(map) = &nj {
        for (k, v) in map {
            let off = v.get("offset").and_then(|x| x.as_usize()).ok_or("norm missing offset")?;
            let len = v.get("len").and_then(|x| x.as_usize()).ok_or("norm missing len")?;
            let end = off.checked_add(len * 4).ok_or("norm offset overflow")?;
            if end > nb.len() {
                return Err(format!("norm {k} truncated"));
            }
            norms.insert(
                k.clone(),
                nb[off..end]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            );
        }
    }
    eprintln!(
        "[run270] weight_source=OPT4_GLYPH_STREAM_RAM_DECODED stream_bytes={} projections={} norms={} embed_bytes={}",
        stream_raw.len(),
        projs.len(),
        norms.len(),
        embed.byte_len()
    );
    Ok(Art { projs, embed, norms, threads })
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

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn normalize_eval_text(text: &str) -> String {
    let cleaned = text.replace("<end_of_turn>", "").replace("<eos>", "");
    let collapsed: Vec<&str> = cleaned.split_whitespace().collect();
    collapsed
        .join(" ")
        .trim_matches(|c: char| matches!(c, ' ' | '.' | '!' | '?' | ','))
        .to_lowercase()
}

fn run_bench_5(
    m: &mut Model270,
    tk: &Tokenizer,
    out_dir: &Path,
    max_tokens: usize,
    mode_tag: &str,
) -> Result<(), String> {
    fs::create_dir_all(out_dir).map_err(|e| format!("mkdir {}: {e}", out_dir.display()))?;
    const HF5: &[(&str, &[u32])] = &[
        ("arithmetic", &[2,105,2364,107,3689,563,236743,236778,900,236743,236778,236881,106,107,105,4368,107]),
        ("capital", &[2,105,2364,107,3689,563,506,5279,529,7001,236881,106,107,105,4368,107]),
        ("hamlet", &[2,105,2364,107,15938,7971,124600,236881,106,107,105,4368,107]),
        ("multistep", &[2,105,2364,107,50702,236743,236770,236778,2782,236743,236828,236764,30579,236743,236778,236832,236764,1299,1138,236743,236770,236810,236761,24181,607,1186,506,3890,236761,106,107,105,4368,107]),
        ("minimum", &[2,105,2364,107,3689,563,506,7100,529,236743,236810,236812,236832,532,236743,236770,236819,236800,236881,106,107,105,4368,107]),
    ];
    let mut rng = Rng::new(0x9E3779B97F4A7C15);
    let mut total_s = 0f64;
    let mut total_fwds = 0usize;
    let mut entries = Vec::new();

    for &(name, ids) in HF5 {
        let (first_logits, generated, prefill_s, decode_s, fwds) =
            m.generate_sequence(ids, max_tokens, None, &mut rng, None)?;
        let dp = out_dir.join(format!("{name}.f32"));
        let bytes: Vec<u8> = first_logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        fs::write(&dp, bytes).map_err(|e| format!("write {}: {e}", dp.display()))?;
        let argmax = first_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i);
        let txt = tk.decode(&generated);
        let dt = prefill_s + decode_s;
        total_s += dt;
        total_fwds += fwds;
        println!(
            "BENCH5 name={name} argmax={argmax} fwds={fwds} s={dt:.2} s_per_fwd={:.3} gen_ids={generated:?} text={txt:?}",
            dt / fwds.max(1) as f64
        );
        entries.push(format!(
            "    {{\"name\": {}, \"argmax\": {argmax}, \"fwds\": {fwds}, \"seconds\": {dt:.4}, \"generated_ids\": {generated:?}, \"text\": {}}}",
            json_escape(name),
            json_escape(&txt)
        ));
    }

    let hamlet_play_ids = tk.encode("<start_of_turn>user\nWho wrote the play Hamlet?<end_of_turn>\n<start_of_turn>model\n", true);
    let (_, hgen, hps, hds, hfwds) = m.generate_sequence(&hamlet_play_ids, max_tokens, None, &mut rng, None)?;
    let htxt = tk.decode(&hgen);
    println!("HAMLET_PLAY text={htxt:?} ids={hgen:?} fwds={hfwds} s={:.2}", hps + hds);

    let (proj_bytes, proj_weights, embed_bytes, norm_bytes) = m.art.memory_bytes();
    let total_weight_bytes = proj_bytes + embed_bytes + norm_bytes;
    let proj_bpw = (proj_bytes as f64 * 8.0) / proj_weights.max(1) as f64;
    let total_weights = proj_weights + m.cfg.vocab * m.cfg.d;
    let eff_bpw = (total_weight_bytes as f64 * 8.0) / total_weights.max(1) as f64;
    let (vm_hwm, vm_rss) = proc_status_kb();
    println!(
        "MEM mode={mode_tag} proj_bytes={proj_bytes} proj_bpw={proj_bpw:.4} embed_bytes={embed_bytes} norm_bytes={norm_bytes} total_weight_bytes={total_weight_bytes} eff_bpw={eff_bpw:.4} vm_hwm_kb={vm_hwm:?} vm_rss_kb={vm_rss:?} avg_s_per_fwd={:.4}",
        total_s / total_fwds.max(1) as f64
    );

    let summary_json = format!(
        "{{\n  \"mode\": {},\n  \"proj_bytes\": {proj_bytes},\n  \"proj_weights\": {proj_weights},\n  \"proj_bpw\": {proj_bpw:.6},\n  \"embed_bytes\": {embed_bytes},\n  \"norm_bytes\": {norm_bytes},\n  \"total_weight_bytes\": {total_weight_bytes},\n  \"eff_bpw\": {eff_bpw:.6},\n  \"vm_hwm_kb\": {},\n  \"vm_rss_kb\": {},\n  \"total_forwards\": {total_fwds},\n  \"total_seconds\": {total_s:.4},\n  \"avg_s_per_fwd\": {:.6},\n  \"hamlet_play_text\": {},\n  \"prompts\": [\n{}\n  ]\n}}\n",
        json_escape(mode_tag),
        vm_hwm.map_or("null".to_string(), |v| v.to_string()),
        vm_rss.map_or("null".to_string(), |v| v.to_string()),
        total_s / total_fwds.max(1) as f64,
        json_escape(&htxt),
        entries.join(",\n")
    );
    fs::write(out_dir.join("bench5_summary.json"), summary_json)
        .map_err(|e| format!("write bench5_summary.json: {e}"))?;
    Ok(())
}

fn run_eval_suite(
    m: &mut Model270,
    tk: &Tokenizer,
    suite_path: &Path,
    out_path: Option<&Path>,
    max_tokens: usize,
    mode_tag: &str,
) -> Result<(), String> {
    let raw = fs::read(suite_path).map_err(|e| format!("read suite {}: {e}", suite_path.display()))?;
    let parsed = json::parse(&raw)?;
    let tasks = parsed.get("tasks").and_then(|v| v.as_arr()).ok_or("suite missing 'tasks' array")?;
    let mut rng = Rng::new(0x9E3779B97F4A7C15);
    let mut correct = 0usize;
    let mut total_s = 0f64;
    let mut total_fwds = 0usize;
    let mut domain_counts: HashMap<String, (usize, usize)> = HashMap::new();
    let mut results_json = Vec::with_capacity(tasks.len());

    for (idx, item) in tasks.iter().enumerate() {
        let id = item.get("id").and_then(|v| v.as_str()).ok_or("task missing id")?;
        let domain = item.get("domain").and_then(|v| v.as_str()).unwrap_or("unknown");
        let q = item.get("question").and_then(|v| v.as_str()).ok_or("task missing question")?;
        let gold = item.get("gold").and_then(|v| v.as_str()).ok_or("task missing gold")?;

        let prompt = format!("<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n");
        let ids = tk.encode(&prompt, true);
        let (_, generated, ps, ds, fwds) = m.generate_sequence(&ids, max_tokens, None, &mut rng, None)?;
        let completion = tk.decode(&generated);
        let norm_pred = normalize_eval_text(&completion);
        let norm_gold = normalize_eval_text(gold);
        let ok = norm_pred == norm_gold;
        if ok {
            correct += 1;
        }
        let ent = domain_counts.entry(domain.to_string()).or_insert((0, 0));
        if ok {
            ent.0 += 1;
        }
        ent.1 += 1;
        let dt = ps + ds;
        total_s += dt;
        total_fwds += fwds;
        println!(
            "TASK {}/{} id={id} domain={domain} ok={ok} gold={gold:?} pred={completion:?} dt={dt:.2}s",
            idx + 1,
            tasks.len()
        );
        results_json.push(format!(
            "    {{\"id\": {}, \"domain\": {}, \"gold\": {}, \"completion\": {}, \"norm_pred\": {}, \"norm_gold\": {}, \"correct\": {ok}, \"seconds\": {dt:.4}}}",
            json_escape(id),
            json_escape(domain),
            json_escape(gold),
            json_escape(&completion),
            json_escape(&norm_pred),
            json_escape(&norm_gold)
        ));
    }

    let (proj_bytes, proj_weights, embed_bytes, norm_bytes) = m.art.memory_bytes();
    let total_weight_bytes = proj_bytes + embed_bytes + norm_bytes;
    let proj_bpw = (proj_bytes as f64 * 8.0) / proj_weights.max(1) as f64;
    let (vm_hwm, vm_rss) = proc_status_kb();
    println!(
        "SUITE_SCORE mode={mode_tag} correct={correct} total={} pct={:.2}% proj_bpw={proj_bpw:.4} total_weight_bytes={total_weight_bytes} vm_hwm_kb={vm_hwm:?} s_per_fwd={:.4} total_s={total_s:.2}",
        tasks.len(),
        100.0 * correct as f64 / tasks.len().max(1) as f64,
        total_s / total_fwds.max(1) as f64
    );

    if let Some(op) = out_path {
        if let Some(par) = op.parent() {
            if !par.as_os_str().is_empty() {
                fs::create_dir_all(par).map_err(|e| format!("mkdir {}: {e}", par.display()))?;
            }
        }
        let mut dom_keys: Vec<String> = domain_counts.keys().cloned().collect();
        dom_keys.sort();
        let dom_json: Vec<String> = dom_keys
            .iter()
            .map(|k| {
                let (c, t) = domain_counts[k];
                format!("    {}: {{\"correct\": {c}, \"total\": {t}}}", json_escape(k))
            })
            .collect();
        let out_txt = format!(
            "{{\n  \"mode\": {},\n  \"suite_path\": {},\n  \"correct\": {correct},\n  \"total\": {},\n  \"proj_bytes\": {proj_bytes},\n  \"proj_bpw\": {proj_bpw:.6},\n  \"embed_bytes\": {embed_bytes},\n  \"norm_bytes\": {norm_bytes},\n  \"total_weight_bytes\": {total_weight_bytes},\n  \"vm_hwm_kb\": {},\n  \"vm_rss_kb\": {},\n  \"total_forwards\": {total_fwds},\n  \"total_seconds\": {total_s:.4},\n  \"by_domain\": {{\n{}\n  }},\n  \"results\": [\n{}\n  ]\n}}\n",
            json_escape(mode_tag),
            json_escape(&suite_path.display().to_string()),
            tasks.len(),
            vm_hwm.map_or("null".to_string(), |v| v.to_string()),
            vm_rss.map_or("null".to_string(), |v| v.to_string()),
            dom_json.join(",\n"),
            results_json.join(",\n")
        );
        fs::write(op, out_txt).map_err(|e| format!("write {}: {e}", op.display()))?;
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let get = |name: &str| -> Option<String> {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
    };
    let exact = args.iter().any(|a| a == "--exact");
    let t16c_arg = get("--t16c");
    let so_arg = get("--so");
    let disable_native = args.iter().any(|a| a == "--no-native-ffi");
    let quant_arg = get("--quantize");
    let use_h8 = args.iter().any(|a| a == "--h8");
    let embed_h8 = args.iter().any(|a| a == "--embed-h8");
    let cfgp = get("--config").ok_or("missing --config")?;
    let tkp = get("--tokenizer").ok_or("missing --tokenizer")?;
    let threads: usize = match get("--threads") {
        Some(s) => s.parse().map_err(|_| format!("invalid --threads '{s}'"))?,
        None => std::thread::available_parallelism().map(|n| n.get().clamp(1, 8)).unwrap_or(4),
    };
    let prompt_text = get("--prompt-text").unwrap_or_else(|| "2+2=?".to_string());
    let bench5_dir = get("--bench-5");
    let eval_suite = get("--eval-suite");
    let suite_out = get("--suite-out");
    let max_tokens_opt: Option<usize> = match get("--max-tokens") {
        Some(s) => Some(s.parse().map_err(|_| format!("invalid --max-tokens '{s}'"))?),
        None => None,
    };
    let max_tokens = max_tokens_opt.unwrap_or(16);
    let sample = args.iter().any(|a| a == "--sample");
    let top_k: usize = match get("--top-k") {
        Some(s) => s.parse().map_err(|_| format!("invalid --top-k '{s}'"))?,
        None => 64,
    };
    let top_p: f64 = match get("--top-p") {
        Some(s) => s.parse().map_err(|_| format!("invalid --top-p '{s}'"))?,
        None => 0.95,
    };
    let ids_arg = get("--ids");

    let proj_quant = match &quant_arg {
        Some(q) => Some((QuantScheme::parse(q)?, use_h8)),
        None => {
            if use_h8 {
                return Err("--h8 requires --quantize <scheme>".into());
            }
            None
        }
    };
    if exact && proj_quant.is_some() {
        return Err("--exact and --quantize cannot be used together".into());
    }
    if t16c_arg.is_some() && (exact || proj_quant.is_some()) {
        return Err("--t16c cannot be used together with --exact or --quantize".into());
    }
    let embed_mode = match get("--embed-mode") {
        Some(s) => EmbedMode::parse(&s, embed_h8)?,
        None => {
            if embed_h8 {
                return Err("--embed-h8 requires --embed-mode <scheme>".into());
            }
            if proj_quant.is_some() || t16c_arg.is_some() {
                EmbedMode::Bf16
            } else {
                EmbedMode::F32
            }
        }
    };
    let mode_tag = if t16c_arg.is_some() {
        format!("t16c_emb-{}", embed_mode.label())
    } else {
        match proj_quant {
            Some((s, h8)) => format!(
                "{}{}_emb-{}",
                s.name(),
                if h8 { "+h8" } else { "" },
                embed_mode.label()
            ),
            None if exact => format!("exact_emb-{}", embed_mode.label()),
            None => format!("opt4_emb-{}", embed_mode.label()),
        }
    };

    let cfg = load_cfg(Path::new(&cfgp))?;
    eprintln!(
        "[run270] cfg d={} layers={} ff={} heads={}/{} hd={} window={} vocab={}",
        cfg.d, cfg.n_layers, cfg.ff, cfg.n_heads, cfg.n_kv, cfg.hd, cfg.window, cfg.vocab
    );

    if t16c_arg.is_none() && so_arg.is_some() {
        eprintln!("[run270] note: ignoring deprecated --so; non-T16C Gemma3-270M uses pure-Rust FP64-accumulated {threads}-thread matvec");
    }

    let art = if let Some(tp) = t16c_arg.as_deref() {
        let model_fb = get("--model");
        load_t16c_artifacts(
            Path::new(tp),
            so_arg.as_deref().map(Path::new),
            disable_native,
            model_fb.as_deref().map(Path::new),
            embed_mode,
            threads,
        )?
    } else if exact || proj_quant.is_some() {
        let model = get("--model").ok_or("missing --model for --exact/--quantize mode")?;
        load_safetensors_artifacts(Path::new(&model), proj_quant, embed_mode, threads)?
    } else {
        let stream = get("--stream").ok_or("missing --stream")?;
        let layout = get("--layout").ok_or("missing --layout")?;
        let cb = get("--codebook").ok_or("missing --codebook")?;
        let embed = get("--embed").ok_or("missing --embed")?;
        let norms = get("--norms").ok_or("missing --norms")?;
        let norms_json = get("--norms-json").ok_or("missing --norms-json")?;
        load_glyph_artifacts(
            Path::new(&stream),
            Path::new(&layout),
            Path::new(&cb),
            Path::new(&embed),
            Path::new(&norms),
            Path::new(&norms_json),
            threads,
        )?
    };
    eprintln!("[run270] compute=rust_std_{threads}t_matvec mode={mode_tag}");

    let mut m = Model270 {
        art,
        kv: (0..cfg.n_layers).map(|_| Kv { k: Vec::new(), v: Vec::new() }).collect(),
        cfg,
    };
    let tk = Tokenizer::load(Path::new(&tkp))?;

    if let Some(out_dir) = bench5_dir {
        run_bench_5(&mut m, &tk, Path::new(&out_dir), max_tokens_opt.unwrap_or(24), &mode_tag)?;
        if eval_suite.is_none() {
            return Ok(());
        }
    }

    if let Some(suite) = eval_suite {
        run_eval_suite(
            &mut m,
            &tk,
            Path::new(&suite),
            suite_out.as_deref().map(Path::new),
            max_tokens_opt.unwrap_or(24),
            &mode_tag,
        )?;
        return Ok(());
    }

    let prompt = format!("<start_of_turn>user\n{prompt_text}<end_of_turn>\n<start_of_turn>model\n");
    let ids: Vec<u32> = match ids_arg {
        Some(s) => {
            let parsed: Result<Vec<u32>, _> = s.split(',').map(|x| x.trim().parse::<u32>()).collect();
            let v = parsed.map_err(|_| format!("invalid integer in --ids '{s}'"))?;
            if v.is_empty() {
                return Err("--ids must not be empty".into());
            }
            v
        }
        None => tk.encode(&prompt, true),
    };
    eprintln!("[run270] prompt {} tokens", ids.len());

    let mut rng = Rng::new(0x9E3779B97F4A7C15);
    let sample_cfg = if sample { Some((top_k, top_p)) } else { None };
    let (logits, generated, prefill_s, decode_s, _) = m.generate_sequence(&ids, max_tokens, sample_cfg, &mut rng, Some(&tk))?;
    eprintln!("[run270] prefill {} tokens in {prefill_s:.2}s", ids.len());

    if let Some(dp) = get("--dump-logits") {
        let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        fs::write(&dp, bytes).map_err(|e| format!("write logits {dp}: {e}"))?;
        eprintln!("[run270] dumped {} logits to {dp}", logits.len());
    }

    println!("GEN ids={generated:?}");
    println!("GEN text={:?}", tk.decode(&generated));
    println!(
        "GEN prefill={prefill_s:.2}s decode={decode_s:.2}s tokens={} => {:.2}s/tok",
        generated.len(),
        decode_s / generated.len().max(1) as f64
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
    use run270_rust::t16c::{Int8LowRank, T16CMat};

    #[test]
    fn t16c_proj_and_embed_roundtrip_and_logits() {
        let (d_out, d_in, bsize, nb) = (32usize, 128usize, 128usize, 2u8);
        let sc: Vec<f32> = (0..d_out).map(|r| 0.012 + 0.003 * (r % 4) as f32).collect();
        let bi: Vec<f32> = (0..d_out).map(|r| -0.005 + 0.002 * (r % 3) as f32).collect();
        let idx: Vec<u8> = (0..d_out * d_in).map(|i| ((i * 5 + 1) & 3) as u8).collect();
        let inv_s: Vec<f32> = (0..d_in).map(|j| 0.8 + 0.4 * ((j % 7) as f32 / 6.0)).collect();
        let make_mat = || {
            let lr = Int8LowRank {
                rank: 4,
                d_out,
                d_in,
                su: (0..d_out).map(|r| 0.01 + 0.001 * r as f32).collect(),
                sv: vec![0.02, 0.015, 0.01, 0.025],
                u_i8: (0..d_out * 4).map(|i| ((i as isize * 13 % 90) - 45) as i8).collect(),
                v_i8: (0..4 * d_in).map(|i| ((i as isize * 17 % 100) - 50) as i8).collect(),
            };
            T16CMat::pack_from_components(
                "model.layers.0.self_attn.q_proj.weight".into(),
                d_out,
                d_in,
                bsize,
                nb,
                true,
                true,
                &sc,
                Some(&bi),
                &idx,
                Some(inv_s.clone()),
                Some(lr),
            )
            .unwrap()
        };
        let x: Vec<f32> = (0..d_in).map(|j| ((j as f32) * 0.17).cos()).collect();
        let proj = ProjBackend::T16C { mat: make_mat(), native: None };
        let y_proj = proj.matvec(&x, 2).unwrap();
        assert_eq!(y_proj.len(), d_out);

        let emb = EmbedBackend::T16C { mat: make_mat(), native: None };
        let escale = (d_in as f32).sqrt();
        let row3 = emb.lookup_scaled(3, d_in, d_out, escale).unwrap();
        assert_eq!(row3.len(), d_in);
        let want_dot: f32 = row3.iter().zip(x.iter()).map(|(w, xv)| (w / escale) * xv).sum();
        assert!((y_proj[3] - want_dot).abs() < 1e-3, "got {} want {}", y_proj[3], want_dot);

        let logits = emb.compute_logits(&x, d_in, d_out, 2).unwrap();
        assert_eq!(logits.len(), d_out);
        for i in 0..d_out {
            assert!((logits[i] - y_proj[i]).abs() < 1e-4);
        }
    }
}
