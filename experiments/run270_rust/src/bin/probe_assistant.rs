//! Minimal binary probe for `AssistantRuntime` golden-parity verification against PyTorch HF.

use run270_rust::assistant::AssistantRuntime;
use std::path::Path;

fn read_u32_le(buf: &[u8], off: usize) -> Result<u32, String> {
    let sl = buf
        .get(off..off + 4)
        .ok_or_else(|| format!("probe input truncated reading u32 at {off}"))?;
    Ok(u32::from_le_bytes([sl[0], sl[1], sl[2], sl[3]]))
}

fn read_f32_slice(buf: &[u8], off: &mut usize, count: usize) -> Result<Vec<f32>, String> {
    let bytes = count.checked_mul(4).ok_or("f32 byte count overflow")?;
    let end = off.checked_add(bytes).ok_or("probe offset overflow")?;
    let sl = buf
        .get(*off..end)
        .ok_or_else(|| format!("probe input truncated reading {count} f32s at {}", *off))?;
    let mut out = Vec::with_capacity(count);
    for ch in sl.chunks_exact(4) {
        out.push(f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
    }
    *off = end;
    Ok(out)
}

#[derive(Debug)]
struct ProbeInput {
    kv_len: usize,
    q_pos: usize,
    bh: usize,
    n_kv_slide: usize,
    hd_slide: usize,
    n_kv_full: usize,
    hd_full: usize,
    tok_emb: Vec<f32>,
    backbone_h: Vec<f32>,
    kv_sk: Vec<f32>,
    kv_sv: Vec<f32>,
    kv_fk: Vec<f32>,
    kv_fv: Vec<f32>,
}

fn parse_probe_input(raw: &[u8]) -> Result<ProbeInput, String> {
    if raw.len() < 28 {
        return Err(format!("probe input too small: {} < 28 bytes", raw.len()));
    }
    let kv_len = read_u32_le(raw, 0)? as usize;
    let q_pos = read_u32_le(raw, 4)? as usize;
    let bh = read_u32_le(raw, 8)? as usize;
    let n_kv_slide = read_u32_le(raw, 12)? as usize;
    let hd_slide = read_u32_le(raw, 16)? as usize;
    let n_kv_full = read_u32_le(raw, 20)? as usize;
    let hd_full = read_u32_le(raw, 24)? as usize;
    let mut off = 28usize;

    let tok_emb = read_f32_slice(raw, &mut off, bh)?;
    let backbone_h = read_f32_slice(raw, &mut off, bh)?;
    let slide_elems = kv_len * n_kv_slide * hd_slide;
    let full_elems = kv_len * n_kv_full * hd_full;
    let kv_sk = read_f32_slice(raw, &mut off, slide_elems)?;
    let kv_sv = read_f32_slice(raw, &mut off, slide_elems)?;
    let kv_fk = read_f32_slice(raw, &mut off, full_elems)?;
    let kv_fv = read_f32_slice(raw, &mut off, full_elems)?;
    if off != raw.len() {
        return Err(format!("probe trailing bytes: parsed {off} != len {}", raw.len()));
    }
    Ok(ProbeInput {
        kv_len,
        q_pos,
        bh,
        n_kv_slide,
        hd_slide,
        n_kv_full,
        hd_full,
        tok_emb,
        backbone_h,
        kv_sk,
        kv_sv,
        kv_fk,
        kv_fv,
    })
}

fn encode_probe_output(bh: usize, top1: u32, post_h: &[f32], logits: &[f32]) -> Result<Vec<u8>, String> {
    if post_h.len() != bh {
        return Err(format!("post_h len {} != {bh}", post_h.len()));
    }
    let vocab = logits.len();
    let mut out = Vec::with_capacity(12 + (bh + vocab) * 4);
    out.extend_from_slice(&(bh as u32).to_le_bytes());
    out.extend_from_slice(&(vocab as u32).to_le_bytes());
    out.extend_from_slice(&top1.to_le_bytes());
    for &v in post_h {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for &v in logits {
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok(out)
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str| -> Option<String> {
        args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned()
    };
    let ast_p = get("--assistant").ok_or("missing --assistant <model.t16c>")?;
    let in_p = get("--in").ok_or("missing --in <probe_in.bin>")?;
    let out_p = get("--out").ok_or("missing --out <probe_out.bin>")?;
    let so_p = get("--so");
    let disable_native = args.iter().any(|a| a == "--no-native-ffi");
    let threads: usize = match get("--threads") {
        Some(s) => s.parse().map_err(|_| format!("invalid --threads '{s}'"))?,
        None => 4,
    };

    let raw = std::fs::read(&in_p).map_err(|e| format!("read {in_p}: {e}"))?;
    let inp = parse_probe_input(&raw)?;
    let rt = AssistantRuntime::load_31b(
        Path::new(&ast_p),
        so_p.as_deref().map(Path::new),
        disable_native,
        threads,
        inp.n_kv_slide,
        inp.n_kv_full,
        1024,
    )?;
    if rt.cfg.backbone_hidden != inp.bh || rt.cfg.hd_slide != inp.hd_slide || rt.cfg.hd_full != inp.hd_full {
        return Err(format!(
            "probe input dims (bh={}, hds={}, hdf={}) != assistant ({}, {}, {})",
            inp.bh, inp.hd_slide, inp.hd_full, rt.cfg.backbone_hidden, rt.cfg.hd_slide, rt.cfg.hd_full
        ));
    }

    let t0 = std::time::Instant::now();
    let (top1, post_h, logits) = rt.forward_step_logits(
        &inp.tok_emb,
        &inp.backbone_h,
        &inp.kv_sk,
        &inp.kv_sv,
        &inp.kv_fk,
        &inp.kv_fv,
        inp.kv_len,
        inp.q_pos,
    )?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let out_bytes = encode_probe_output(inp.bh, top1, &post_h, &logits)?;
    std::fs::write(&out_p, &out_bytes).map_err(|e| format!("write {out_p}: {e}"))?;

    println!(
        "AST_PROBE top1={top1} native={} time_ms={elapsed_ms:.2} resident_mb={:.2}",
        rt.model.native.is_some(),
        rt.resident_bytes() as f64 / (1024.0 * 1024.0)
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
    fn probe_binary_roundtrip_and_rejects_truncation() {
        let mut buf = Vec::new();
        for v in [2u32, 2, 4, 1, 2, 1, 2] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let n_floats = 4 + 4 + 2 * 2 + 2 * 2 + 2 * 2 + 2 * 2;
        for i in 0..n_floats {
            buf.extend_from_slice(&((i as f32) * 0.25).to_le_bytes());
        }
        let parsed = parse_probe_input(&buf).unwrap();
        assert_eq!(parsed.kv_len, 2);
        assert_eq!(parsed.q_pos, 2);
        assert_eq!(parsed.tok_emb.len(), 4);
        assert_eq!(parsed.kv_sk.len(), 4);

        assert!(parse_probe_input(&buf[..buf.len() - 4]).is_err());
        let mut extra = buf;
        extra.push(0);
        assert!(parse_probe_input(&extra).is_err());

        let out = encode_probe_output(4, 9, &[1.0, 2.0, 3.0, 4.0], &[0.1, 0.2]).unwrap();
        assert_eq!(out.len(), 12 + (4 + 2) * 4);
    }
}
