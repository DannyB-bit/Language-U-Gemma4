//! run270 — standalone native Rust inference runtime for Gemma 3 270M (zero Python).
//!
//! Supports two weight modes:
//!   1. `--exact`: reads BF16/F32 2D weights, embeddings, and 1D norms directly from `model.safetensors`
//!      (only `--model`, `--config`, `--tokenizer` required).
//!   2. Glyph-stream mode (default): decodes the 132 B OPT4 stream + sidecars into RAM
//!      using `codec::decode_opt4_glyph`, executed with 4-thread FP64-accumulated matvec.

use run270_rust::codec::{bf16_to_f32, decode_opt4_glyph, CB_CLASSES, CB_DIM, GLYPH_BYTES, GLYPH_WEIGHTS, LANU_MAGIC, STREAM_HEADER_BYTES};
use run270_rust::json;
use run270_rust::math::{attn_decode, gelu_tanh_scalar, qk_norm_gemma, rmsn_gemma, rope_head, top_k_top_p_sample, Rng};
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

struct Proj {
    d_out: usize,
    d_in: usize,
    w: Vec<f32>,
}

struct Art {
    projs: HashMap<String, Proj>,
    embed: Vec<f32>,
    norms: HashMap<String, Vec<f32>>,
    threads: usize,
}

impl Art {
    fn proj(&self, key: &str, x: &[f32]) -> Result<Vec<f32>, String> {
        let p = self.projs.get(key).ok_or_else(|| format!("missing projection {key}"))?;
        if x.len() != p.d_in {
            return Err(format!("{key}: x len {} != d_in {}", x.len(), p.d_in));
        }
        let mut y = vec![0f32; p.d_out];
        let nthreads = self.threads.max(1);
        std::thread::scope(|s| {
            let chunk = p.d_out.div_ceil(nthreads).max(1);
            let mut handles = Vec::new();
            for (t, yb) in y.chunks_mut(chunk).enumerate() {
                let r0 = t * chunk;
                let w = &p.w;
                let d_in = p.d_in;
                handles.push(s.spawn(move || {
                    for (r, slot) in yb.iter_mut().enumerate() {
                        let row = &w[(r0 + r) * d_in..(r0 + r + 1) * d_in];
                        let mut acc = 0f64;
                        for j in 0..d_in {
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

    fn norm(&self, key: &str) -> Result<&[f32], String> {
        self.norms.get(key).map(|v| v.as_slice()).ok_or_else(|| format!("missing norm {key}"))
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

    fn forward(&mut self, tid: u32, pos: usize) -> Result<Vec<f32>, String> {
        let c = &self.cfg;
        let d = c.d;
        let ti = tid as usize;
        if ti >= c.vocab || (ti + 1) * d > self.art.embed.len() {
            return Err(format!("token id {tid} out of bounds for vocab {}", c.vocab));
        }
        let escale = (d as f32).sqrt();
        let mut x: Vec<f32> = self.art.embed[ti * d..(ti + 1) * d].iter().map(|&v| v * escale).collect();

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

    fn compute_logits(&self, x: &[f32]) -> Vec<f32> {
        let d = self.cfg.d;
        let vocab = self.cfg.vocab;
        let mut logits = vec![0f32; vocab];
        let nthreads = self.art.threads.max(1);
        let chunk = vocab.div_ceil(nthreads).max(1);
        std::thread::scope(|s| {
            let mut handles = Vec::new();
            for (tid, out_chunk) in logits.chunks_mut(chunk).enumerate() {
                let t0 = tid * chunk;
                let embed = &self.art.embed;
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
        logits
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

fn load_exact_artifacts(model_path: &Path, threads: usize) -> Result<Art, String> {
    let sf = SafeTensors::open(model_path)?;
    let mut projs = HashMap::new();
    let mut norms = HashMap::new();
    for (name, meta) in &sf.tensors {
        if meta.shape.len() == 2 && name != "model.embed_tokens.weight" {
            let w = sf.read_tensor_f32(name)?;
            projs.insert(
                name.clone(),
                Proj {
                    d_out: meta.shape[0],
                    d_in: meta.shape[1],
                    w,
                },
            );
        } else if meta.shape.len() == 1 {
            norms.insert(name.clone(), sf.read_tensor_f32(name)?);
        }
    }
    let embed = sf.read_tensor_f32("model.embed_tokens.weight")?;
    eprintln!(
        "[run270] weight_source=EXACT_SAFETENSORS projections={} norms={} embed_f32={}",
        projs.len(),
        norms.len(),
        embed.len()
    );
    Ok(Art { projs, embed, norms, threads })
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
        projs.insert(fk, Proj { d_out, d_in, w });
    }

    let embed_raw = fs::read(embed_path).map_err(|e| format!("read embed {}: {e}", embed_path.display()))?;
    if embed_raw.len() % 2 != 0 {
        return Err("embed_tokens.bin byte count is not even BF16".into());
    }
    let embed: Vec<f32> = embed_raw
        .chunks_exact(2)
        .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();

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
        "[run270] weight_source=OPT4_GLYPH_STREAM_RAM_DECODED stream_bytes={} projections={} norms={} embed_f32={}",
        stream_raw.len(),
        projs.len(),
        norms.len(),
        embed.len()
    );
    Ok(Art { projs, embed, norms, threads })
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let get = |name: &str| -> Option<String> {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
    };
    let exact = args.iter().any(|a| a == "--exact");
    let cfgp = get("--config").ok_or("missing --config")?;
    let tkp = get("--tokenizer").ok_or("missing --tokenizer")?;
    let threads: usize = match get("--threads") {
        Some(s) => s.parse().map_err(|_| format!("invalid --threads '{s}'"))?,
        None => std::thread::available_parallelism().map(|n| n.get().clamp(1, 8)).unwrap_or(4),
    };
    let prompt_text = get("--prompt-text").unwrap_or_else(|| "2+2=?".to_string());
    let max_tokens: usize = match get("--max-tokens") {
        Some(s) => s.parse().map_err(|_| format!("invalid --max-tokens '{s}'"))?,
        None => 16,
    };
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

    let cfg = load_cfg(Path::new(&cfgp))?;
    eprintln!(
        "[run270] cfg d={} layers={} ff={} heads={}/{} hd={} window={} vocab={}",
        cfg.d, cfg.n_layers, cfg.ff, cfg.n_heads, cfg.n_kv, cfg.hd, cfg.window, cfg.vocab
    );

    if get("--so").is_some() {
        eprintln!("[run270] note: ignoring deprecated --so; Gemma3-270M uses pure-Rust FP64-accumulated {threads}-thread matvec");
    }

    let art = if exact {
        let model = get("--model").ok_or("missing --model for --exact mode")?;
        load_exact_artifacts(Path::new(&model), threads)?
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
    eprintln!("[run270] compute=rust_std_{threads}t_f64_matvec");

    let mut m = Model270 {
        art,
        kv: (0..cfg.n_layers).map(|_| Kv { k: Vec::new(), v: Vec::new() }).collect(),
        cfg,
    };

    let tk = Tokenizer::load(Path::new(&tkp))?;
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

    let t0 = std::time::Instant::now();
    let mut x = vec![0f32; m.cfg.d];
    for (pos, &t) in ids.iter().enumerate() {
        x = m.forward(t, pos)?;
    }
    let prefill_s = t0.elapsed().as_secs_f64();
    eprintln!("[run270] prefill {} tokens in {prefill_s:.2}s", ids.len());

    let mut logits = m.compute_logits(&x);
    if let Some(dp) = get("--dump-logits") {
        let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        fs::write(&dp, bytes).map_err(|e| format!("write logits {dp}: {e}"))?;
        eprintln!("[run270] dumped {} logits to {dp}", logits.len());
    }

    let mut generated: Vec<u32> = Vec::new();
    let mut rng = Rng::new(0x9E3779B97F4A7C15);
    let eos: [u32; 2] = [1, 106];
    let t1 = std::time::Instant::now();
    for step in 0..max_tokens {
        let (cand, top_logits) = m.head_topk(&logits, top_k.max(8));
        let nxt = if sample {
            let mut lcopy = top_logits.clone();
            let ci = top_k_top_p_sample(&mut lcopy, &cand, 1.0, top_k, top_p, 1.0, &generated, &mut rng);
            cand[ci]
        } else {
            cand[0]
        };
        generated.push(nxt);
        let txt = tk.decode(&[nxt]);
        println!("[STEP {step}] id={nxt} text={txt:?}");
        if eos.contains(&nxt) {
            break;
        }
        if step + 1 < max_tokens {
            let tp = std::time::Instant::now();
            x = m.forward(nxt, ids.len() + step)?;
            logits = m.compute_logits(&x);
            eprintln!("[run270] decode step {step} in {:.2}s", tp.elapsed().as_secs_f64());
        }
    }
    let decode_s = t1.elapsed().as_secs_f64();
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
