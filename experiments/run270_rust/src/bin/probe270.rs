//! probe270 — fail-closed verifier comparing decoded OPT4 glyph projections against raw SafeTensors weights.
//!
//! Usage: probe270 <model.safetensors> <stream_132b.bin> <layout.json> <codebook_2048x8.bin> [proj_key|all] [min_cos]

use run270_rust::codec::{
    decode_opt4_glyph, encode_opt4_glyph, CB_CLASSES, CB_DIM, GLYPH_BYTES, GLYPH_WEIGHTS,
    LANU_MAGIC, STREAM_HEADER_BYTES,
};
use run270_rust::json;
use run270_rust::math::cosine_and_rel_l2;
use run270_rust::safetensors::SafeTensors;
use std::fs;
use std::path::Path;

fn check_projection(
    sf: &SafeTensors,
    stream: &[u8],
    cb: &[f32],
    key: &str,
    soff: usize,
    d_out: usize,
    d_in: usize,
    min_cos_gate: f64,
) -> Result<(), String> {
    let raw_w = sf.read_tensor_f32(key)?;
    if raw_w.len() != d_out * d_in {
        return Err(format!("{key}: element count {} != {d_out}*{d_in}", raw_w.len()));
    }
    let nglyphs = (d_out * d_in).div_ceil(GLYPH_WEIGHTS);
    let need_end = soff
        .checked_add(nglyphs * GLYPH_BYTES)
        .ok_or_else(|| format!("{key}: stream offset overflow"))?;
    if need_end > stream.len() {
        return Err(format!("{key}: stream truncated (need {need_end}, have {})", stream.len()));
    }

    let mut decoded_w = vec![0f32; d_out * d_in];
    let mut glyph_cos = Vec::with_capacity(nglyphs);
    let mut reencode_diff_max = 0usize;

    for g in 0..nglyphs {
        let goff = soff + g * GLYPH_BYTES;
        let glyph = &stream[goff..goff + GLYPH_BYTES];
        let mut gw = [0f32; GLYPH_WEIGHTS];
        decode_opt4_glyph(glyph, cb, &mut gw);
        let base = g * GLYPH_WEIGHTS;
        let n = (d_out * d_in - base).min(GLYPH_WEIGHTS);
        for &v in &gw[..n] {
            if !v.is_finite() {
                return Err(format!("{key} glyph {g}: non-finite decoded weight"));
            }
        }
        decoded_w[base..base + n].copy_from_slice(&gw[..n]);

        let (gcos, _, _) = cosine_and_rel_l2(&raw_w[base..base + n], &gw[..n]);
        glyph_cos.push(gcos);

        if g == 0 {
            let mut w256 = [0f32; GLYPH_WEIGHTS];
            w256[..n].copy_from_slice(&raw_w[base..base + n]);
            let (re_payload, _, _) = encode_opt4_glyph(&w256, cb);
            let diffs = re_payload.iter().zip(glyph.iter()).filter(|(a, b)| a != b).count();
            reencode_diff_max = reencode_diff_max.max(diffs);
        }
    }

    glyph_cos.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let (tensor_cos, rel_l2, max_abs) = cosine_and_rel_l2(&raw_w, &decoded_w);

    // One-hot column-0 GEMV consistency check vs column 0 of source BF16 tensor
    let mut y_col0 = Vec::with_capacity(d_out);
    let mut raw_col0 = Vec::with_capacity(d_out);
    for r in 0..d_out {
        y_col0.push(decoded_w[r * d_in]);
        raw_col0.push(raw_w[r * d_in]);
    }
    let (col0_cos, _, col0_max_d) = cosine_and_rel_l2(&raw_col0, &y_col0);

    println!(
        "PROJ {key} [{d_out}x{d_in}] glyphs={nglyphs} per_glyph_cos[min={:.4} p10={:.4} med={:.4} max={:.4}] tensor_cos={tensor_cos:.6} rel_l2={rel_l2:.6} max|d|={max_abs:.4e} col0_cos={col0_cos:.6} col0_max|d|={col0_max_d:.4e} g0_reencode_byte_diffs={reencode_diff_max}/132",
        glyph_cos[0],
        glyph_cos[nglyphs / 10],
        glyph_cos[nglyphs / 2],
        glyph_cos[nglyphs - 1]
    );

    if tensor_cos < min_cos_gate {
        return Err(format!("{key}: tensor_cos {tensor_cos:.6} < min_cos_gate {min_cos_gate:.6}"));
    }
    if reencode_diff_max != 0 {
        return Err(format!("{key}: glyph 0 deterministic re-encode byte mismatch ({reencode_diff_max}/132)"));
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        return Err("usage: probe270 <model.safetensors> <stream_132b.bin> <layout.json> <codebook_2048x8.bin> [proj_key|all] [min_cos]".into());
    }
    let model_p = Path::new(&args[1]);
    let stream_p = Path::new(&args[2]);
    let layout_p = Path::new(&args[3]);
    let cb_p = Path::new(&args[4]);
    let target = args.get(5).map(|s| s.as_str()).unwrap_or("all");
    let min_cos_gate: f64 = match args.get(6) {
        Some(s) => s.parse().map_err(|_| format!("invalid min_cos '{s}'"))?,
        None => 0.90,
    };

    let sf = SafeTensors::open(model_p)?;
    let stream = fs::read(stream_p).map_err(|e| format!("read stream {}: {e}", stream_p.display()))?;
    if stream.len() < STREAM_HEADER_BYTES || &stream[0..4] != LANU_MAGIC {
        return Err("invalid or truncated LANU stream header".into());
    }
    let cb_raw = fs::read(cb_p).map_err(|e| format!("read codebook {}: {e}", cb_p.display()))?;
    if cb_raw.len() != CB_CLASSES * CB_DIM * 4 {
        return Err(format!("invalid codebook byte length {}", cb_raw.len()));
    }
    let cb: Vec<f32> = cb_raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let lj = json::parse(&fs::read(layout_p).map_err(|e| e.to_string())?)?;
    let projs = lj
        .get("projections")
        .and_then(|v| v.as_arr())
        .ok_or("layout.json missing projections array")?;

    let mut checked = 0usize;
    for ent in projs {
        let key = ent.get("full_key").and_then(|v| v.as_str()).ok_or("projection missing full_key")?;
        if target != "all" && key != target {
            continue;
        }
        let soff = ent.get("stream_byte_offset").and_then(|v| v.as_usize()).ok_or("missing stream_byte_offset")?;
        let d_out = ent.get("d_out").and_then(|v| v.as_usize()).ok_or("missing d_out")?;
        let d_in = ent.get("d_in").and_then(|v| v.as_usize()).ok_or("missing d_in")?;
        check_projection(&sf, &stream, &cb, key, soff, d_out, d_in, min_cos_gate)?;
        checked += 1;
    }
    if checked == 0 {
        return Err(format!("no projection matched target '{target}'"));
    }
    println!("PROBE270_PASS checked={checked}");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("FAIL: {e}");
        std::process::exit(1);
    }
}
