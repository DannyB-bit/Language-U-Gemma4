//! encode_model.rs — native (zero-Python) Language-U stream encoder.
//! Encodes any safetensors checkpoint's 2D weight matrices into the 132 B
//! dual-stage RVQ (Opt4) glyph stream + layout JSON, 1D norms into a raw fp32
//! sidecar. Byte law identical to the published 31B v3book stream:
//!   glyph = [f16 s1][f16 s2][32 x u16 idx1][32 x u16 idx2]
//!   w[oct*8+t] = s1*cb[idx1*8+t] + s2*cb[idx2*8+t]   (oct in 0..32)
//! Stream header (16 B): "LANU", u16 ver=1, u16 codec=45, u32 nglyphs, u32 rows.
//! Usage: encode_model <model.safetensors> <codebook_2048x8.bin> <out_dir> [max_projections]

#[path = "../json.rs"]
mod json;
#[path = "../ffi.rs"]
mod ffi;
#[path = "../gri.rs"]
mod gri;
#[path = "../opt5.rs"]
mod opt5;
#[path = "../qat.rs"]
mod qat;
#[path = "../sha256.rs"]
mod sha256;

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::Path;

const CB_CLASSES: usize = 2048;
const OCTANTS: usize = 32;

fn f32_to_fp16_bits(v: f32) -> u16 {
    opt5::fp32_to_fp16(v)
}

/// RVQ-encode one 256-weight glyph -> 132 B payload; returns (payload, cos, snr_db).
fn encode_glyph(w256: &[f32; 256], cb: &[f32]) -> ([u8; 132], f64, f64) {
    let mut payload = [0u8; 132];
    let mut recon = [0f32; 256];
    let mut dot_ws = 0f64;
    let mut norm_w = 0f64;
    let mut norm_r = 0f64;
    for oct in 0..OCTANTS {
        let w = &w256[oct * 8..oct * 8 + 8];
        // stage 1
        let mut best = 0usize;
        let mut best_dot = f32::NEG_INFINITY;
        for c in 0..CB_CLASSES {
            let cc = &cb[c * 8..c * 8 + 8];
            let mut d = 0f32;
            for t in 0..8 {
                d += w[t] * cc[t];
            }
            if d > best_dot {
                best_dot = d;
                best = c;
            }
        }
        let idx1 = best;
        let c1 = &cb[idx1 * 8..idx1 * 8 + 8];
        let mut denom1 = 1e-12f32;
        let mut num1 = 0f32;
        for t in 0..8 {
            denom1 += c1[t] * c1[t];
            num1 += w[t] * c1[t];
        }
        let s1 = f32_to_fp16_bits(num1 / denom1);
        let s1f = gri::fp16_to_f32(s1);
        let mut res = [0f32; 8];
        for t in 0..8 {
            res[t] = w[t] - s1f * c1[t];
        }
        // stage 2 on residual
        let mut best2 = 0usize;
        let mut best_dot2 = f32::NEG_INFINITY;
        for c in 0..CB_CLASSES {
            let cc = &cb[c * 8..c * 8 + 8];
            let mut d = 0f32;
            for t in 0..8 {
                d += res[t] * cc[t];
            }
            if d > best_dot2 {
                best_dot2 = d;
                best2 = c;
            }
        }
        let idx2 = best2;
        let c2 = &cb[idx2 * 8..idx2 * 8 + 8];
        let mut denom2 = 1e-12f32;
        let mut num2 = 0f32;
        for t in 0..8 {
            denom2 += c2[t] * c2[t];
            num2 += res[t] * c2[t];
        }
        let s2 = f32_to_fp16_bits(num2 / denom2);
        let s2f = gri::fp16_to_f32(s2);
        for t in 0..8 {
            recon[oct * 8 + t] = s1f * c1[t] + s2f * c2[t];
        }
        // cos/snr measured in FWHT domain (orthonormal => equals weight-domain cos)
        payload[0..2].copy_from_slice(&s1.to_le_bytes());
        payload[2..4].copy_from_slice(&s2.to_le_bytes());
        payload[4 + oct * 2..6 + oct * 2].copy_from_slice(&(idx1 as u16).to_le_bytes());
        payload[68 + oct * 2..70 + oct * 2].copy_from_slice(&(idx2 as u16).to_le_bytes());
    }
    for i in 0..256 {
        dot_ws += w256[i] as f64 * recon[i] as f64;
        norm_w += (w256[i] as f64) * (w256[i] as f64);
        norm_r += (recon[i] as f64) * (recon[i] as f64);
    }
    let cos = dot_ws / (norm_w.sqrt() * norm_r.sqrt() + 1e-12);
    let mut err = 0f64;
    for i in 0..256 {
        err += (w256[i] as f64 - recon[i] as f64).powi(2);
    }
    let snr = 10.0 * (norm_w / (err + 1e-12)).log10();
    (payload, cos, snr)
}

fn read_tensor_f32(sf: &qat::QatFile, name: &str) -> Result<Vec<f32>, String> {
    let t = sf
        .tensors
        .get(name)
        .ok_or_else(|| format!("tensor {name} not found"))?
        .clone();
    match t.dtype.as_str() {
        "BF16" => sf.read_bf16(name),
        "F32" => {
            let raw = sf.read_bytes(&t)?;
            Ok(raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        other => Err(format!("{name}: unsupported dtype {other}")),
    }
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: encode_model <model.safetensors> <codebook_2048x8.bin> <out_dir> [max_projections]");
        std::process::exit(2);
    }
    let model_path = &args[1];
    let cb_path = &args[2];
    let out_dir = &args[3];
    let max_proj: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
    fs::create_dir_all(out_dir).unwrap();

    // codebook
    let cb_raw = fs::read(cb_path).expect("codebook read");
    assert_eq!(cb_raw.len(), CB_CLASSES * 8 * 4, "codebook size");
    let cb: Vec<f32> = cb_raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let sf = qat::open_qat(model_path).expect("open safetensors");

    // select 2D weight tensors, canonical order: embed, layers ascending, head
    let mut names: Vec<String> = sf
        .tensors
        .iter()
        .filter(|(k, t)| {
            t.shape.len() == 2 && k.ends_with(".weight") && !k.starts_with("model.embed_tokens")
        })
        .map(|(k, _)| k.clone())
        .collect();
    fn layer_of(k: &str) -> (i64, String) {
        if let Some(rest) = k.strip_prefix("model.layers.") {
            let n: i64 = rest
                .split('.')
                .next()
                .and_then(|s| s.parse().ok())
                .unwrap_or(-1);
            (n, rest.to_string())
        } else if k.starts_with("model.embed_tokens") {
            (-2, k.to_string())
        } else {
            (999, k.to_string())
        }
    }
    names.sort_by(|a, b| {
        let (la, ra) = layer_of(a);
        let (lb, rb) = layer_of(b);
        la.cmp(&lb).then(ra.cmp(&rb))
    });
    if max_proj > 0 {
        names.truncate(max_proj);
    }
    let n1d: Vec<String> = sf
        .tensors
        .keys()
        .filter(|k| sf.tensors[*k].shape.len() == 1)
        .cloned()
        .collect();

    eprintln!("[encode_model] {} 2D projections, {} 1D norms", names.len(), n1d.len());

    let stream_path = Path::new(out_dir).join("stream_132b.bin");
    let mut out = fs::File::create(&stream_path).expect("stream create");
    let mut layout_path = Path::new(out_dir).join("layout.json");
    let mut norms_bin = fs::File::create(Path::new(out_dir).join("norms.f32bin")).unwrap();

    // header placeholder (16 B), patched at the end
    out.write_all(b"LANU").unwrap();
    out.write_all(&1u16.to_le_bytes()).unwrap();
    out.write_all(&45u16.to_le_bytes()).unwrap();
    out.write_all(&0u32.to_le_bytes()).unwrap();
    out.write_all(&0u32.to_le_bytes()).unwrap();

    let mut layout = String::from("{\n \"model\": \"");
    layout.push_str(model_path);
    layout.push_str("\",\n \"stream_file\": \"stream_132b.bin\",\n \"codebook_file\": \"");
    layout.push_str(Path::new(cb_path).file_name().unwrap().to_str().unwrap());
    layout.push_str("\",\n \"codec_id\": 4,\n \"projections\": [\n");

    let mut norms_json = String::from("{\n");
    let mut total_glyphs: u64 = 0;
    let mut total_rows: u64 = 0;
    let mut total_weights: u64 = 0;
    let mut sum_cos = 0f64;
    let mut min_cos = 1f64;
    let mut receipts = String::from("{\n \"projections\": [\n");
    let mut first = true;
    let mut proj_idx = 0usize;

    for name in &names {
        let t = sf.tensors[name].clone();
        let d_out = t.shape[0];
        let d_in = t.shape[1];
        let w = read_tensor_f32(&sf, name)?;
        assert_eq!(w.len(), d_out * d_in);
        total_weights += w.len() as u64;
        let nglyph = (w.len() + 255) / 256;
        let byte_off = 16 + total_glyphs * 132;
        let mut cos_acc = 0f64;
        let mut snr_acc = 0f64;
        let mut buf = vec![0u8; 132 * nglyph];
        let nthreads = 4usize;
        let chunk = (nglyph + nthreads - 1) / nthreads;
        std::thread::scope(|s| {
            let mut handles = Vec::new();
            let mut segs = buf.chunks_mut(chunk * 132);
            for tid in 0..nthreads {
                let lo = tid * chunk;
                let hi = (lo + chunk).min(nglyph);
                if lo >= hi {
                    break;
                }
                let w = &w;
                let cb = &cb;
                let bseg = segs.next().unwrap();
                handles.push(s.spawn(move || {
                    let mut ca = 0f64;
                    let mut sa = 0f64;
                    for g in lo..hi {
                        let mut w256 = [0f32; 256];
                        let base = g * 256;
                        for i in 0..256 {
                            if base + i < w.len() {
                                w256[i] = w[base + i];
                            }
                        }
                        let (payload, cos, snr) = encode_glyph(&w256, cb);
                        bseg[(g - lo) * 132..(g - lo + 1) * 132].copy_from_slice(&payload);
                        ca += cos;
                        sa += snr;
                    }
                    (ca, sa)
                }));
            }
            for h in handles {
                let (ca, sa) = h.join().unwrap();
                cos_acc += ca;
                snr_acc += sa;
            }
        });
        out.write_all(&buf).unwrap();
        let cos = cos_acc / nglyph as f64;
        let snr = snr_acc / nglyph as f64;
        sum_cos += cos;
        if cos < min_cos {
            min_cos = cos;
        }
        total_glyphs += nglyph as u64;
        total_rows += d_out as u64;
        if !first {
            layout.push_str(",\n");
            receipts.push_str(",\n");
        }
        first = false;
        let layer = layer_of(name).0;
        layout.push_str(&format!(
            "  {{\"layer\": {}, \"tensor_name\": \"{}\", \"full_key\": \"{}\", \"codec\": \"OPT4\", \"codec_id\": 4, \"bytes_per_glyph\": 132, \"d_in\": {}, \"d_out\": {}, \"glyphs\": {}, \"tensor_bytes\": {}, \"stream_byte_offset\": {}, \"shape\": [{}, {}]}}",
            layer, name, name, d_in, d_out, nglyph, w.len() * 2, byte_off, d_out, d_in
        ));
        receipts.push_str(&format!(
            "  {{\"full_key\": \"{}\", \"d_out\": {}, \"d_in\": {}, \"glyphs\": {}, \"cos\": {:.6}, \"snr_db\": {:.3}, \"stream_byte_offset\": {}}}",
            name, d_out, d_in, nglyph, cos, snr, byte_off
        ));
        eprintln!(
            "[encode_model] proj {} {} [{},{}] glyphs={} cos={:.6} snr={:.2}dB",
            proj_idx, name, d_out, d_in, nglyph, cos, snr
        );
        proj_idx += 1;
    }
    layout.push_str("\n ],\n");
    receipts.push_str("\n ],\n");

    // norms sidecar: raw fp32 + json index
    let mut noff = 0u64;
    let mut nfirst = true;
    for name in &n1d {
        let w = read_tensor_f32(&sf, name)?;
        let raw: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        norms_bin.write_all(&raw).unwrap();
        if !nfirst {
            norms_json.push_str(",\n");
        }
        nfirst = false;
        norms_json.push_str(&format!(
            " \"{}\": {{\"offset\": {}, \"len\": {}, \"shape\": {:?}}}",
            name,
            noff,
            w.len(),
            sf.tensors[name].shape
        ));
        noff += raw.len() as u64;
    }
    norms_json.push_str("\n}\n");
    fs::write(Path::new(out_dir).join("norms.json"), &norms_json).unwrap();

    // embed sidecar: raw bf16 tensor bytes (runtime decodes bf16 -> f32), because
    // RVQ on the embedding matrix measures cos ~0.52 (same reason the 31B pipeline
    // keeps embed_tokens.bin as fp16/bf16, not glyphs).
    if let Some(t) = sf.tensors.get("model.embed_tokens.weight") {
        let t = t.clone();
        let raw = sf.read_bytes(&t)?;
        fs::write(Path::new(out_dir).join("embed_tokens.bin"), &raw).map_err(|e| e.to_string())?;
        eprintln!(
            "[encode_model] embed sidecar: {} bytes (dtype {})",
            raw.len(),
            t.dtype
        );
    }

    // patch header
    use std::os::unix::fs::FileExt;
    let mut f = fs::OpenOptions::new()
        .write(true)
        .open(&stream_path)
        .unwrap();
    f.write_all_at(&LANU_HDR(total_glyphs, total_rows), 0).unwrap();
    drop(f);
    out.flush().unwrap();

    let stream_size = 16 + total_glyphs * 132;
    layout.push_str(&format!(
        " \"stream_size_bytes\": {}, \"total_projections\": {}, \"total_macro_glyphs\": {}, \"total_rows\": {}\n}}\n",
        stream_size, proj_idx, total_glyphs, total_rows
    ));
    fs::write(&mut layout_path, &layout).unwrap();

    // stream sha256 (whole-file, stream is ~100-200 MB — fits RAM)
    let all = fs::read(&stream_path).unwrap();
    let stream_sha = sha256::sha256_hex(&all);
    drop(all);

    let overall_cos = sum_cos / proj_idx as f64;
    receipts.push_str(&format!(
        " \"overall_cos\": {:.6}, \"min_proj_cos\": {:.6}, \"census_R_mod_256\": {}, \"total_2d_weights\": {}, \"total_glyphs\": {}, \"stream_bytes\": {}, \"stream_sha256\": \"{}\"\n}}\n",
        overall_cos,
        min_cos,
        total_weights % 256,
        total_weights,
        total_glyphs,
        stream_size,
        stream_sha
    ));
    fs::write(Path::new(out_dir).join("receipts.json"), &receipts).unwrap();

    println!("ENCODE_DONE projections={} glyphs={} weights={} census_R={} stream_bytes={} overall_cos={:.6} min_cos={:.6} sha256={}",
        proj_idx, total_glyphs, total_weights, total_weights % 256, stream_size, overall_cos, min_cos, stream_sha);
    Ok(())
}

fn LANU_HDR(nglyphs: u64, rows: u64) -> [u8; 16] {
    let mut h = [0u8; 16];
    h[0..4].copy_from_slice(b"LANU");
    h[4..6].copy_from_slice(&1u16.to_le_bytes());
    h[6..8].copy_from_slice(&45u16.to_le_bytes());
    h[8..12].copy_from_slice(&(nglyphs as u32).to_le_bytes());
    h[12..16].copy_from_slice(&(rows as u32).to_le_bytes());
    h
}

// silence unused-import warning for BTreeMap when unused
#[allow(dead_code)]
fn _keep(_: BTreeMap<String, String>) {}
