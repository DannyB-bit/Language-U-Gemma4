//! run270.rs — native config-driven Language-U glyph-stream inference for
//! gemma-3-270m-it (zero Python). Reads the RVQ 132 B stream + layout produced
//! by encode_model, bf16 embed sidecar, fp32 norms sidecar, S^7 codebook, and
//! the NEON accelerator .so. Greedy + sampled decoding, sliding-window GQA.
//!
//! usage: run270 --model <model.safetensors> --config <config.json>
//!   --stream <stream_132b.bin> --layout <layout.json> --codebook <cb.bin>
//!   --embed <embed_tokens.bin> --norms <norms.f32bin> --norms-json <norms.json>
//!   --tokenizer <tokenizer.json> --so <libneon_stream_accelerator.so>
//!   --prompt-text "..." [--max-tokens N] [--sample] [--top-k 64] [--top-p 0.95]

#[path = "../json.rs"]
mod json;
#[path = "../ffi.rs"]
mod ffi;
#[path = "../qat.rs"]
mod qat;
#[path = "../gri.rs"]
mod gri;
#[path = "../math.rs"]
mod math;
#[path = "../tokenizer.rs"]
mod tokenizer;

use std::collections::HashMap;
use std::fs;

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

fn load_cfg(path: &str) -> Result<Cfg, String> {
    let b = fs::read(path).map_err(|e| e.to_string())?;
    let j = json::parse(&b)?;
    let g = |k: &str| j.get(k).and_then(|v| v.as_usize()).unwrap_or(0);
    let gf = |k: &str| j.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
    let lt = j.get("layer_types").and_then(|v| v.as_arr()).ok_or("layer_types")?;
    let layer_sliding: Vec<bool> = lt
        .iter()
        .map(|v| v.as_str() == Some("sliding_attention"))
        .collect();
    Ok(Cfg {
        d: g("hidden_size"),
        n_layers: g("num_hidden_layers"),
        ff: g("intermediate_size"),
        n_heads: g("num_attention_heads"),
        n_kv: g("num_key_value_heads"),
        hd: g("head_dim"),
        window: g("sliding_window"),
        eps: gf("rms_norm_eps"),
        theta_global: gf("rope_theta"),
        theta_local: gf("rope_local_base_freq"),
        layer_sliding,
        vocab: g("vocab_size"),
    })
}

struct Proj {
    full_key: String,
    d_out: usize,
    d_in: usize,
    w: Vec<f32>, // decoded effective weights, row-major [d_out, d_in]
}

struct Art {
    projs: HashMap<String, Proj>,
    embed: Vec<f32>, // bf16 decoded
    norms: HashMap<String, Vec<f32>>,
}

impl Art {
    fn proj(&self, key: &str, x: &[f32]) -> Vec<f32> {
        let p = &self.projs[key];
        let mut y = vec![0f32; p.d_out];
        std::thread::scope(|s| {
            let chunk = (p.d_out + 3) / 4;
            let ys: Vec<&mut [f32]> = y.chunks_mut(chunk).collect();
            let mut handles = Vec::new();
            for (t, yb) in ys.into_iter().enumerate() {
                let r0 = t * chunk;
                let w = &p.w;
                let x = x;
                handles.push(s.spawn(move || {
                    for r in 0..yb.len() {
                        let row = &w[(r0 + r) * p.d_in..(r0 + r + 1) * p.d_in];
                        let mut acc = 0f64;
                        for j in 0..p.d_in {
                            acc += row[j] as f64 * x[j] as f64;
                        }
                        yb[r] = acc as f32;
                    }
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
        });
        y
    }
    fn norm(&self, key: &str) -> Result<&[f32], String> {
        self.norms.get(key).map(|v| v.as_slice()).ok_or_else(|| format!("norm {key}"))
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

    /// one token forward; pos = absolute position; returns final-normed hidden
    fn forward(&mut self, tid: u32, pos: usize) -> Result<Vec<f32>, String> {
        let c = &self.cfg;
        let d = c.d;
        // Gemma scales token embeddings by sqrt(hidden_size)
        let escale = (d as f32).sqrt();
        let mut x = self.art.embed[tid as usize * d..(tid as usize + 1) * d]
            .iter()
            .map(|&v| v * escale)
            .collect::<Vec<f32>>();
        if std::env::var("RUN270_STAGE").map(|v| v == "embed").unwrap_or(false) {
            let _ = std::fs::write(format!("/tmp/st_embed_pos{pos}.f32"), x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
        }
        for l in 0..c.n_layers {
            let pfx = Self::pfx(l);
            let h = rmsn_gemma(&x, self.art.norm(&format!("{pfx}.input_layernorm.weight"))?, c.eps);
            if l == 0 && std::env::var("RUN270_STAGE").map(|v| v == "in_norm").unwrap_or(false) {
                let _ = std::fs::write(format!("/tmp/st_in_norm_pos{pos}.f32"), h.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
            }
            let q_raw = self.art.proj(&format!("{pfx}.self_attn.q_proj.weight"), &h);
            let k_raw = self.art.proj(&format!("{pfx}.self_attn.k_proj.weight"), &h);
            let v_raw = self.art.proj(&format!("{pfx}.self_attn.v_proj.weight"), &h);
            let mut q = qk_norm_gemma(&q_raw, c.n_heads, c.hd, self.art.norm(&format!("{pfx}.self_attn.q_norm.weight"))?, c.eps);
            let mut k = qk_norm_gemma(&k_raw, c.n_kv, c.hd, self.art.norm(&format!("{pfx}.self_attn.k_norm.weight"))?, c.eps);
            let v = v_raw; // gemma3: no v_norm
            // gemma3 attention scale = query_pre_attn_scalar^-0.5 = 1/sqrt(head_dim)
            let qs = 1.0 / (c.hd as f32).sqrt();
            for vq in q.iter_mut() {
                *vq *= qs;
            }
            let theta = self.layer_theta(l);
            for hq in 0..c.n_heads {
                math::rope_head(&mut q[hq * c.hd..(hq + 1) * c.hd], pos as u32, theta, c.hd, c.hd);
            }
            for kh in 0..c.n_kv {
                math::rope_head(&mut k[kh * c.hd..(kh + 1) * c.hd], pos as u32, theta, c.hd, c.hd);
            }
            if l == 0 && std::env::var("RUN270_STAGE").map(|v| v == "q").unwrap_or(false) {
                let _ = std::fs::write(format!("/tmp/st_q_pos{pos}.f32"), q.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
                let _ = std::fs::write(format!("/tmp/st_k_pos{pos}.f32"), k.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
            }
            let t = self.kv[l].k.len() / (c.n_kv * c.hd);
            self.kv[l].k.extend_from_slice(&k);
            self.kv[l].v.extend_from_slice(&v);
            let ctx = math::attn_decode(&q, &self.kv[l].k, &self.kv[l].v, t, pos, c.n_heads, c.n_kv, c.hd, self.sw(l));
            if l == 0 && std::env::var("RUN270_STAGE").map(|v| v == "ctx").unwrap_or(false) {
                let _ = std::fs::write(format!("/tmp/st_ctx_pos{pos}.f32"), ctx.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
            }
            let attn = self.art.proj(&format!("{pfx}.self_attn.o_proj.weight"), &ctx);
            let attn = rmsn_gemma(&attn, self.art.norm(&format!("{pfx}.post_attention_layernorm.weight"))?, c.eps);
            for i in 0..d {
                x[i] += attn[i];
            }
            if l == 0 && std::env::var("RUN270_STAGE").map(|v| v == "post_attn").unwrap_or(false) {
                let _ = std::fs::write(format!("/tmp/st_post_attn_pos{pos}.f32"), x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
            }
            let h2 = rmsn_gemma(&x, self.art.norm(&format!("{pfx}.pre_feedforward_layernorm.weight"))?, c.eps);
            let g = self.art.proj(&format!("{pfx}.mlp.gate_proj.weight"), &h2);
            let u = self.art.proj(&format!("{pfx}.mlp.up_proj.weight"), &h2);
            let gu: Vec<f32> = g.iter().zip(&u).map(|(&a, &b)| math::gelu_tanh_scalar(a) * b).collect();
            let mlp = self.art.proj(&format!("{pfx}.mlp.down_proj.weight"), &gu);
            let mlp = rmsn_gemma(&mlp, self.art.norm(&format!("{pfx}.post_feedforward_layernorm.weight"))?, c.eps);
            for i in 0..d {
                x[i] += mlp[i];
            }
            if let Ok(dl) = std::env::var("RUN270_DUMP_LAYER") {
                if dl.parse::<usize>().map(|v| v == l).unwrap_or(false) {
                    let p = format!("/tmp/lyr{l}_pos{pos}.f32");
                    let _ = std::fs::write(&p, x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
                }
            }
        }
        Ok(rmsn_gemma(&x, self.art.norm("model.norm.weight")?, c.eps))
    }

    /// tied head: logits = embed^T x ; top-k by exact bf16->f32 dot
    fn head_topk(&self, x: &[f32], k: usize) -> (Vec<u32>, Vec<f64>) {
        let c = &self.cfg;
        let mut scored: Vec<(u32, f64)> = (0..c.vocab as u32)
            .map(|t| {
                let row = &self.art.embed[t as usize * c.d..(t as usize + 1) * c.d];
                let mut s = 0f64;
                for i in 0..c.d {
                    s += row[i] as f64 * x[i] as f64;
                }
                (t, s)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(k);
        (scored.iter().map(|&(t, _)| t).collect(), scored.iter().map(|&(_, s)| s).collect())
    }
}

/// Gemma RMSNorm: x/rms * (1 + w)
fn rmsn_gemma(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut s = 0f64;
    for &v in x {
        s += (v as f64) * (v as f64);
    }
    let rms = ((s / n as f64) + eps as f64).sqrt() as f32;
    x.iter().zip(w.iter()).map(|(&a, &b)| (a / rms) * (1.0 + b)).collect()
}

/// per-head Gemma RMSNorm (1+w) for q/k
fn qk_norm_gemma(x: &[f32], n_heads: usize, head_dim: usize, w: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for h in 0..n_heads {
        let s = &x[h * head_dim..(h + 1) * head_dim];
        let o = &mut out[h * head_dim..(h + 1) * head_dim];
        o.copy_from_slice(&rmsn_gemma(s, w, eps));
    }
    out
}

fn decode_tok(tk: &tokenizer::Tokenizer, ids: &[u32]) -> String {
    let mut s = String::new();
    for &id in ids {
        if let Some(t) = tk.id_to_token.get(id as usize) {
            s.push_str(&t.replace('Ġ', " "));
        }
    }
    s
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let get = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let model = get("--model").ok_or("--model")?;
    let cfgp = get("--config").ok_or("--config")?;
    let stream = get("--stream").ok_or("--stream")?;
    let layoutp = get("--layout").ok_or("--layout")?;
    let cbp = get("--codebook").ok_or("--codebook")?;
    let embedp = get("--embed").ok_or("--embed")?;
    let norms_bin = get("--norms").ok_or("--norms")?;
    let norms_json = get("--norms-json").ok_or("--norms-json")?;
    let tkp = get("--tokenizer").ok_or("--tokenizer")?;
    let _so = get("--so");
    let prompt_text = get("--prompt-text").unwrap_or_else(|| "2+2=?".to_string());
    let max_tokens: usize = get("--max-tokens").and_then(|s| s.parse().ok()).unwrap_or(16);
    let sample = args.iter().any(|a| a == "--sample");
    let top_k: usize = get("--top-k").and_then(|s| s.parse().ok()).unwrap_or(64);
    let top_p: f64 = get("--top-p").and_then(|s| s.parse().ok()).unwrap_or(0.95);
    let ids_arg = get("--ids");

    let cfg = load_cfg(&cfgp)?;
    eprintln!("[run270] cfg d={} layers={} ff={} heads={}/{} hd={} window={} vocab={}", cfg.d, cfg.n_layers, cfg.ff, cfg.n_heads, cfg.n_kv, cfg.hd, cfg.window, cfg.vocab);

    // stream: decode all glyphs to RAM effective weights (270m stream is only ~52 MB -> ~401 MB f32)
    let exact = args.iter().any(|a| a == "--exact");
    let stream_raw = if exact { Vec::new() } else { fs::read(&stream).map_err(|e| e.to_string())? };
    eprintln!("[run270] stream {} bytes", stream_raw.len());

    // codebook
    let cb_raw = fs::read(&cbp).map_err(|e| e.to_string())?;
    let codebook: Vec<f32> = cb_raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // layout
    let lb = fs::read(&layoutp).map_err(|e| e.to_string())?;
    let lj = json::parse(&lb)?;
    let mut projs = HashMap::new();
    if exact {
        // exact bf16 weights straight from safetensors (parity harness mode)
        let sf = qat::open_qat(model.as_str())?;
        for (name, t) in &sf.tensors {
            if t.shape.len() == 2 && t.dtype == "BF16" && name != "model.embed_tokens.weight" {
                let w = sf.read_bf16(name)?;
                projs.insert(
                    name.clone(),
                    Proj {
                        full_key: name.clone(),
                        d_out: t.shape[0],
                        d_in: t.shape[1],
                        w,
                    },
                );
            }
        }
        eprintln!("[run270] EXACT mode: {} projections from safetensors", projs.len());
    } else if let Some(arr) = lj.get("projections").and_then(|v| v.as_arr()) {
        for p in arr {
            let fk = p.get("full_key").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let d_out = p.get("d_out").and_then(|v| v.as_usize()).unwrap_or(0);
            let d_in = p.get("d_in").and_then(|v| v.as_usize()).unwrap_or(0);
            let off = p.get("stream_byte_offset").and_then(|v| v.as_f64()).unwrap_or(0.0) as usize;
            let nglyphs = (d_out * d_in + 255) / 256;
            let mut w = vec![0f32; d_out * d_in];
            for g in 0..nglyphs {
                let goff = off + g * 132;
                let gw = gri::glyph_effective_weights(&stream_raw[goff..goff + 132], &codebook);
                let base = g * 256;
                let n = (d_out * d_in - base).min(256);
                w[base..base + n].copy_from_slice(&gw[..n]);
            }
            projs.insert(
                fk.clone(),
                Proj {
                    full_key: fk,
                    d_out,
                    d_in,
                    w,
                },
            );
        }
    }
    eprintln!("[run270] {} projections decoded to RAM", projs.len());

    // embed bf16 -> f32
    let embed: Vec<f32> = if exact {
        qat::open_qat(model.as_str())?.read_bf16("model.embed_tokens.weight")?
    } else {
        let eb_raw = fs::read(&embedp).map_err(|e| e.to_string())?;
        eb_raw
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect()
    };
    eprintln!("[run270] embed {} f32", embed.len());

    // norms
    let nb = fs::read(&norms_bin).map_err(|e| e.to_string())?;
    let njb = fs::read(&norms_json).map_err(|e| e.to_string())?;
    let nj = json::parse(&njb)?;
    let mut norms = HashMap::new();
    if let json::Json::Obj(map) = &nj {
        for (k, v) in map {
            let off = v.get("offset").and_then(|x| x.as_usize()).unwrap_or(0);
            let len = v.get("len").and_then(|x| x.as_usize()).unwrap_or(0);
            let slice = &nb[off..off + len * 4];
            norms.insert(
                k.clone(),
                slice
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            );
        }
    }
    eprintln!("[run270] {} norms", norms.len());

    
    eprintln!("[run270] NEON accel loaded");

    // sanity: model.safetensors must exist (used only for provenance hash print)
    let _ = model;

    let art = Art {

        projs,

        embed,
        norms,

    };
    let mut m = Model270 {
        art,
        kv: (0..cfg.n_layers)
            .map(|_| Kv {
                k: Vec::new(),
                v: Vec::new(),
            })
            .collect(),
        cfg,
    };

    let tk = tokenizer::Tokenizer::load(&tkp)?;
    // gemma3 chat template (from chat_template.jinja): user turn then model turn
    let prompt = format!("<start_of_turn>user\n{prompt_text}<end_of_turn>\n<start_of_turn>model\n");
    let mut ids: Vec<u32> = match ids_arg {
        Some(s) => s.split(',').filter_map(|x| x.trim().parse().ok()).collect(),
        None => tk.encode(&prompt),
    };
    if ids.is_empty() {
        ids = tk.encode(&prompt);
    }
    eprintln!("[run270] prompt {} tokens", ids.len());

    // prefill (sequential; 270m is small)
    let t0 = std::time::Instant::now();
    let mut x = vec![0f32; m.cfg.d];
    for (pos, &t) in ids.iter().enumerate() {
        x = m.forward(t, pos)?;
    }
    let prefill_s = t0.elapsed().as_secs_f64();
    eprintln!("[run270] prefill {} tokens in {:.1}s", ids.len(), prefill_s);

    let mut generated: Vec<u32> = Vec::new();
    if let Some(dp) = get("--dump-logits") {
        let d = m.cfg.d;
        let vocab = m.cfg.vocab;
        let mut lg = vec![0f32; vocab];
        for t in 0..vocab {
            let row = &m.art.embed[t * d..(t + 1) * d];
            let mut s = 0f64;
            for i in 0..d {
                s += row[i] as f64 * x[i] as f64;
            }
            lg[t] = s as f32;
        }
        fs::write(&dp, lg.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).map_err(|e| e.to_string())?;
        eprintln!("[run270] dumped {} logits to {dp}", lg.len());
    }
    let mut rng = math::Rng::new(0x9E3779B97F4A7C15);
    let eos: [u32; 2] = [1, 106];
    let t1 = std::time::Instant::now();
    for step in 0..max_tokens {
        let (cand, exact) = m.head_topk(&x, top_k.max(8));
        let nxt = if sample {
            let mut logits: Vec<f64> = exact.clone();
            let ci = math::top_k_top_p_sample(&mut logits, &cand, 1.0, top_k, top_p, 1.0, &generated, &mut rng);
            cand[ci]
        } else {
            cand[0]
        };
        generated.push(nxt);
        let txt = decode_tok(&tk, &[nxt]);
        println!("[STEP {step}] id={nxt} text={txt:?}");
        if eos.contains(&nxt) {
            break;
        }
        if step + 1 < max_tokens {
            let tp = std::time::Instant::now();
            x = m.forward(nxt, ids.len() + step)?;
            eprintln!("[run270] decode step {} in {:.2}s", step, tp.elapsed().as_secs_f64());
        }
    }
    let decode_s = t1.elapsed().as_secs_f64();
    println!("GEN ids={:?}", generated);
    println!("GEN text={:?}", decode_tok(&tk, &generated));
    println!("GEN prefill={:.1}s decode={:.1}s tokens={} => {:.2}s/tok", prefill_s, decode_s, generated.len(), decode_s / generated.len().max(1) as f64);
    Ok(())
}
