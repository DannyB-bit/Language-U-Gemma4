//! encode_model — native Rust encoder from SafeTensors to 132 B OPT4 glyph stream + sidecars.
//!
//! Usage: encode_model <model.safetensors> <codebook_2048x8.bin> <out_dir> [max_projections]

use run270_rust::codec::{
    encode_opt4_glyph, make_stream_header, CB_CLASSES, CB_DIM, GLYPH_BYTES, GLYPH_WEIGHTS,
    STREAM_HEADER_BYTES,
};
use run270_rust::safetensors::SafeTensors;
use run270_rust::sha256::sha256_file_hex;
use std::fs::{self, File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

fn layer_order_key(k: &str) -> (i64, &str) {
    if let Some(rest) = k.strip_prefix("model.layers.") {
        let n = rest.split('.').next().and_then(|s| s.parse::<i64>().ok()).unwrap_or(-1);
        (n, rest)
    } else if k.starts_with("model.embed_tokens") {
        (-2, k)
    } else {
        (999, k)
    }
}

fn read_codebook(path: &Path) -> Result<Vec<f32>, String> {
    let raw = fs::read(path).map_err(|e| format!("read codebook {}: {e}", path.display()))?;
    let expected = CB_CLASSES * CB_DIM * 4;
    if raw.len() != expected {
        return Err(format!("codebook size {} != expected {expected}", raw.len()));
    }
    Ok(raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn write_at_start(path: &Path, header: &[u8; STREAM_HEADER_BYTES]) -> Result<(), String> {
    let mut f = OpenOptions::new()
        .write(true)
        .open(path)
        .map_err(|e| format!("reopen {}: {e}", path.display()))?;
    f.seek(SeekFrom::Start(0)).map_err(|e| format!("seek header: {e}"))?;
    f.write_all(header).map_err(|e| format!("write header: {e}"))
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        return Err("usage: encode_model <model.safetensors> <codebook_2048x8.bin> <out_dir> [max_projections]".into());
    }
    let model_path = Path::new(&args[1]);
    let cb_path = Path::new(&args[2]);
    let out_dir = Path::new(&args[3]);
    let max_proj: usize = match args.get(4) {
        Some(s) => s.parse().map_err(|_| format!("invalid max_projections '{s}'"))?,
        None => 0,
    };
    let nthreads: usize = std::thread::available_parallelism()
        .map(|n| n.get().clamp(1, 8))
        .unwrap_or(4);

    fs::create_dir_all(out_dir).map_err(|e| format!("mkdir {}: {e}", out_dir.display()))?;
    let cb = read_codebook(cb_path)?;
    let sf = SafeTensors::open(model_path)?;

    let mut names: Vec<String> = sf
        .tensors
        .iter()
        .filter(|(k, t)| t.shape.len() == 2 && k.ends_with(".weight") && !k.starts_with("model.embed_tokens"))
        .map(|(k, _)| k.clone())
        .collect();
    names.sort_by(|a, b| layer_order_key(a).cmp(&layer_order_key(b)));
    if max_proj > 0 {
        names.truncate(max_proj);
    }
    if names.is_empty() {
        return Err("no 2D weight projections found in checkpoint".into());
    }

    let mut n1d: Vec<String> = sf
        .tensors
        .iter()
        .filter(|(_, t)| t.shape.len() == 1)
        .map(|(k, _)| k.clone())
        .collect();
    n1d.sort();

    eprintln!("[encode_model] {} 2D projections, {} 1D norms (threads={nthreads})", names.len(), n1d.len());

    let stream_path = out_dir.join("stream_132b.bin");
    let mut out = File::create(&stream_path).map_err(|e| format!("create {}: {e}", stream_path.display()))?;
    let zero_hdr = [0u8; STREAM_HEADER_BYTES];
    out.write_all(&zero_hdr).map_err(|e| e.to_string())?;

    let cb_name = cb_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("codebook.bin");
    let mut layout = format!(
        "{{\n \"model\": \"{}\",\n \"stream_file\": \"stream_132b.bin\",\n \"codebook_file\": \"{cb_name}\",\n \"codec_id\": 4,\n \"projections\": [\n",
        model_path.display()
    );
    let mut receipts = String::from("{\n \"projections\": [\n");

    let mut total_glyphs: u64 = 0;
    let mut total_rows: u64 = 0;
    let mut total_weights: u64 = 0;
    let mut sum_proj_cos = 0f64;
    let mut min_proj_cos = 1f64;

    for (proj_idx, name) in names.iter().enumerate() {
        let meta = sf.tensors[name].clone();
        let d_out = meta.shape[0];
        let d_in = meta.shape[1];
        let w = sf.read_tensor_f32(name)?;
        if w.len() != d_out * d_in {
            return Err(format!("{name}: element count {} != {}*{}", w.len(), d_out, d_in));
        }
        total_weights += w.len() as u64;
        let nglyph = w.len().div_ceil(GLYPH_WEIGHTS);
        let byte_off = STREAM_HEADER_BYTES as u64 + total_glyphs * GLYPH_BYTES as u64;

        let mut buf = vec![0u8; nglyph * GLYPH_BYTES];
        let chunk = nglyph.div_ceil(nthreads).max(1);
        let mut cos_acc = 0f64;
        let mut snr_acc = 0f64;

        std::thread::scope(|s| {
            let mut handles = Vec::new();
            let mut segs = buf.chunks_mut(chunk * GLYPH_BYTES);
            for tid in 0..nthreads {
                let lo = tid * chunk;
                let hi = (lo + chunk).min(nglyph);
                if lo >= hi {
                    break;
                }
                let seg = segs.next().unwrap();
                let w_ref = &w;
                let cb_ref = &cb;
                handles.push(s.spawn(move || {
                    let mut ca = 0f64;
                    let mut sa = 0f64;
                    for g in lo..hi {
                        let mut w256 = [0f32; GLYPH_WEIGHTS];
                        let base = g * GLYPH_WEIGHTS;
                        let n = (w_ref.len() - base).min(GLYPH_WEIGHTS);
                        w256[..n].copy_from_slice(&w_ref[base..base + n]);
                        let (payload, cos, snr) = encode_opt4_glyph(&w256, cb_ref);
                        let dst = (g - lo) * GLYPH_BYTES;
                        seg[dst..dst + GLYPH_BYTES].copy_from_slice(&payload);
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

        out.write_all(&buf).map_err(|e| format!("write {name}: {e}"))?;
        let mean_glyph_cos = cos_acc / nglyph as f64;
        let mean_glyph_snr = snr_acc / nglyph as f64;
        sum_proj_cos += mean_glyph_cos;
        if mean_glyph_cos < min_proj_cos {
            min_proj_cos = mean_glyph_cos;
        }
        total_glyphs += nglyph as u64;
        total_rows += d_out as u64;

        if proj_idx > 0 {
            layout.push_str(",\n");
            receipts.push_str(",\n");
        }
        let layer = layer_order_key(name).0;
        layout.push_str(&format!(
            "  {{\"layer\": {layer}, \"tensor_name\": \"{name}\", \"full_key\": \"{name}\", \"codec\": \"OPT4\", \"codec_id\": 4, \"bytes_per_glyph\": {GLYPH_BYTES}, \"d_in\": {d_in}, \"d_out\": {d_out}, \"glyphs\": {nglyph}, \"tensor_bytes\": {}, \"stream_byte_offset\": {byte_off}, \"shape\": [{d_out}, {d_in}]}}",
            w.len() * 2
        ));
        receipts.push_str(&format!(
            "  {{\"full_key\": \"{name}\", \"d_out\": {d_out}, \"d_in\": {d_in}, \"glyphs\": {nglyph}, \"decode_cos\": {mean_glyph_cos:.6}, \"decode_snr_db\": {mean_glyph_snr:.3}, \"stream_byte_offset\": {byte_off}}}"
        ));
        eprintln!(
            "[encode_model] proj {proj_idx} {name} [{d_out},{d_in}] glyphs={nglyph} decode_cos={mean_glyph_cos:.6} decode_snr={mean_glyph_snr:.2}dB"
        );
    }
    out.flush().map_err(|e| e.to_string())?;
    drop(out);

    let hdr = make_stream_header(total_glyphs, total_rows)?;
    write_at_start(&stream_path, &hdr)?;

    // 1D norms sidecar
    let mut norms_bin = File::create(out_dir.join("norms.f32bin")).map_err(|e| e.to_string())?;
    let mut norms_json = String::from("{\n");
    let mut noff = 0u64;
    for (i, name) in n1d.iter().enumerate() {
        let w = sf.read_tensor_f32(name)?;
        let raw: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        norms_bin.write_all(&raw).map_err(|e| e.to_string())?;
        if i > 0 {
            norms_json.push_str(",\n");
        }
        norms_json.push_str(&format!(
            " \"{name}\": {{\"offset\": {noff}, \"len\": {}, \"shape\": {:?}}}",
            w.len(),
            sf.tensors[name].shape
        ));
        noff += raw.len() as u64;
    }
    norms_json.push_str("\n}\n");
    fs::write(out_dir.join("norms.json"), &norms_json).map_err(|e| e.to_string())?;

    if let Some(meta) = sf.tensors.get("model.embed_tokens.weight").cloned() {
        let raw = sf.read_raw_bytes(&meta)?;
        fs::write(out_dir.join("embed_tokens.bin"), &raw).map_err(|e| e.to_string())?;
        eprintln!("[encode_model] embed sidecar: {} bytes (dtype {})", raw.len(), meta.dtype);
    }

    let stream_size = STREAM_HEADER_BYTES as u64 + total_glyphs * GLYPH_BYTES as u64;
    layout.push_str(&format!(
        "\n ],\n \"stream_size_bytes\": {stream_size}, \"total_projections\": {}, \"total_macro_glyphs\": {total_glyphs}, \"total_rows\": {total_rows}\n}}\n",
        names.len()
    ));
    fs::write(out_dir.join("layout.json"), &layout).map_err(|e| e.to_string())?;

    let stream_sha = sha256_file_hex(&stream_path)?;
    let source_sha = sha256_file_hex(model_path)?;
    let overall_cos = sum_proj_cos / names.len() as f64;
    receipts.push_str(&format!(
        "\n ],\n \"source_model_sha256\": \"{source_sha}\", \"overall_decode_cos\": {overall_cos:.6}, \"min_proj_decode_cos\": {min_proj_cos:.6}, \"census_R_mod_256\": {}, \"total_2d_weights\": {total_weights}, \"total_glyphs\": {total_glyphs}, \"stream_bytes\": {stream_size}, \"stream_sha256\": \"{stream_sha}\"\n}}\n",
        total_weights % 256
    ));
    fs::write(out_dir.join("receipts.json"), &receipts).map_err(|e| e.to_string())?;

    println!(
        "ENCODE_DONE projections={} glyphs={} weights={} census_R={} stream_bytes={} overall_decode_cos={overall_cos:.6} min_cos={min_proj_cos:.6} sha256={stream_sha}",
        names.len(), total_glyphs, total_weights, total_weights % 256, stream_size
    );
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("FAIL: {e}");
        std::process::exit(1);
    }
}
