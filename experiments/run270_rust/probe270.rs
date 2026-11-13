//! probe270.rs — verify one projection: glyph GEMV row 0 vs raw bf16 safetensors row 0
#[path = "../json.rs"]
mod json;
#[path = "../ffi.rs"]
mod ffi;
#[path = "../gri.rs"]
mod gri;
#[path = "../sha256.rs"]
mod sha256;
#[path = "../opt5.rs"]
mod opt5;
#[path = "../qat.rs"]
mod qat;

use std::fs;

fn f2h(v: f32) -> u16 {
    opt5::fp32_to_fp16(v)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // probe270 <model.safetensors> <stream> <layout.json> <codebook> <so> <proj_key> <row>
    let model = &args[1];
    let stream_p = &args[2];
    let layout_p = &args[3];
    let cb_p = &args[4];
    let so_p = &args[5];
    let key = &args[6];
    let row: usize = args[7].parse().unwrap();

    // safetensors header
    let f = fs::File::open(model).unwrap();
    use std::io::Read;
    let mut f = f;
    let mut hlen = [0u8; 8];
    f.read_exact(&mut hlen).unwrap();
    let hlen = u64::from_le_bytes(hlen) as usize;
    let mut hdr = vec![0u8; hlen];
    f.read_exact(&mut hdr).unwrap();
    let h = json::parse(&hdr).unwrap();
    let ent = h.get(key).unwrap();
    let shape = ent.get("shape").unwrap().as_arr().unwrap();
    let d_out = shape[0].as_usize().unwrap();
    let d_in = shape[1].as_usize().unwrap();
    let offs = ent.get("data_offsets").unwrap().as_arr().unwrap();
    let (b0, b1) = (offs[0].as_f64().unwrap() as u64, offs[1].as_f64().unwrap() as u64);
    let data_base = 8 + hlen;
    let mut raw = vec![0u8; (b1 - b0) as usize];
    use std::os::unix::fs::FileExt;
    f.read_exact_at(&mut raw, data_base as u64 + b0).unwrap();
    let row_raw: Vec<f32> = (0..d_in).map(|j| gri::fp16_to_f32(u16::from_le_bytes([raw[row * d_in * 2 + j * 2], raw[row * d_in * 2 + j * 2 + 1]]))).collect();

    // stream side
    let stream = fs::read(stream_p).unwrap();
    let layout = json::parse(&fs::read(layout_p).unwrap()).unwrap();
    let ent = layout.get("projections").unwrap().as_arr().unwrap().iter().find(|e| e.get("full_key").and_then(|v| v.as_str()) == Some(key.as_str())).unwrap().clone();
    let soff = ent.get("stream_byte_offset").unwrap().as_f64().unwrap() as u64;
    let cb = fs::read(cb_p).unwrap();
    let cb_f: Vec<f32> = cb.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let accel = unsafe { ffi::Accel::load(so_p).unwrap() };
    let _ = accel;

    // x = one-hot at column 0 -> y should equal column 0 of weight
    let mut x = vec![0f32; d_in];
    x[0] = 1.0;
    // RAM decode of the whole projection (same path as run270)
    let nglyphs = (d_out * d_in + 255) / 256;
    let mut w = vec![0f32; d_out * d_in];
    let qf0 = qat::open_qat(model).unwrap();
    let wq_all = qf0.read_bf16(key).unwrap();
    // per-glyph cos histogram against true weights
    let mut coss = vec![0f64; nglyphs];
    for g in 0..nglyphs {
        let goff = soff as usize + g * 132;
        let gw = gri::glyph_effective_weights(&stream[goff..goff + 132], &cb_f);
        let base = g * 256;
        let n = (d_out * d_in - base).min(256);
        w[base..base + n].copy_from_slice(&gw[..n]);
        let mut d = 0f64; let mut a = 0f64; let mut b = 0f64;
        for i in 0..n { d += gw[i] as f64 * wq_all[base + i] as f64; a += gw[i] as f64 * gw[i] as f64; b += wq_all[base + i] as f64 * wq_all[base + i] as f64; }
        coss[g] = d / (a.sqrt() * b.sqrt() + 1e-30);
    }
    coss.sort_by(|x, y| x.partial_cmp(y).unwrap());
    println!("per-glyph cos: min={:.4} p10={:.4} med={:.4} max={:.4}", coss[0], coss[nglyphs / 10], coss[nglyphs / 2], coss[nglyphs - 1]);
    let y: Vec<f32> = (0..d_out).map(|r| w[r * d_in]).collect();
    // glyph-0 diagnostics: decode vs raw first 256 weights
    {
        let g0 = &stream[soff as usize..soff as usize + 132];
        let gw = gri::glyph_effective_weights(g0, &cb_f);
        let mut dot = 0f64; let mut n1 = 0f64; let mut n2 = 0f64; let mut maxd = 0f32;
        for i in 0..256 {
            let rawv = gri::fp16_to_f32(u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]));
            dot += gw[i] as f64 * rawv as f64; n1 += gw[i] as f64 * gw[i] as f64; n2 += rawv as f64 * rawv as f64;
            let d = (gw[i] - rawv).abs(); if d > maxd { maxd = d; }
        }
        println!("glyph0 decode-vs-raw256: cos={:.6} max|d|={:.3e}", dot / (n1.sqrt() * n2.sqrt() + 1e-30), maxd);
        println!("glyph0 s1,s2 bytes: {:02x?} {:02x?}", &g0[0..2], &g0[2..4]);
        { let qf2 = qat::open_qat(model).unwrap(); let wq2 = qf2.read_bf16(key).unwrap(); let mut d=0f64; let mut a=0f64; let mut b=0f64; for i in 0..256 { d += gw[i] as f64 * wq2[i] as f64; a += gw[i] as f64 * gw[i] as f64; b += wq2[i] as f64 * wq2[i] as f64; } println!("glyph0 decode-vs-qat256: cos={:.6}", d/(a.sqrt()*b.sqrt()+1e-30)); }
        // re-encode glyph 0 from qat-read weights and compare payload bytes
        {
            let qf0 = qat::open_qat(model).unwrap();
            let wq0 = qf0.read_bf16(key).unwrap();
            let mut w256 = [0f32; 256];
            w256.copy_from_slice(&wq0[..256]);
            // replicate encode_glyph
            let mut payload = [0u8; 132];
            for oct in 0..32usize {
                let w = &w256[oct * 8..oct * 8 + 8];
                let mut best = 0usize; let mut bd = f32::NEG_INFINITY;
                for c in 0..2048usize {
                    let cc = &cb_f[c * 8..c * 8 + 8];
                    let mut d = 0f32;
                    for t in 0..8 { d += w[t] * cc[t]; }
                    if d > bd { bd = d; best = c; }
                }
                let idx1 = best;
                let c1 = &cb_f[idx1 * 8..idx1 * 8 + 8];
                let mut den = 1e-12f32; let mut num = 0f32;
                for t in 0..8 { den += c1[t] * c1[t]; num += w[t] * c1[t]; }
                let s1 = f2h(num / den);
                let s1f = gri::fp16_to_f32(s1);
                let mut res = [0f32; 8];
                for t in 0..8 { res[t] = w[t] - s1f * c1[t]; }
                let mut best2 = 0usize; let mut bd2 = f32::NEG_INFINITY;
                for c in 0..2048usize {
                    let cc = &cb_f[c * 8..c * 8 + 8];
                    let mut d = 0f32;
                    for t in 0..8 { d += res[t] * cc[t]; }
                    if d > bd2 { bd2 = d; best2 = c; }
                }
                let idx2 = best2;
                let c2 = &cb_f[idx2 * 8..idx2 * 8 + 8];
                let mut den2 = 1e-12f32; let mut num2 = 0f32;
                for t in 0..8 { den2 += c2[t] * c2[t]; num2 += res[t] * c2[t]; }
                let s2 = f2h(num2 / den2);
                payload[0..2].copy_from_slice(&s1.to_le_bytes());
                payload[2..4].copy_from_slice(&s2.to_le_bytes());
                payload[4 + oct * 2..6 + oct * 2].copy_from_slice(&(idx1 as u16).to_le_bytes());
                payload[68 + oct * 2..70 + oct * 2].copy_from_slice(&(idx2 as u16).to_le_bytes());
            }
            let mut ndiff = 0usize;
            for i in 0..132 { if payload[i] != g0[i] { ndiff += 1; } }
            println!("re-encode glyph0: byte diffs vs stream = {ndiff}/132");
        }
        // qat-side read vs gri-side read of same tensor
        {
            let qf = qat::open_qat(model).unwrap();
            let t = qf.tensors.get(key).unwrap().clone(); println!("qat start={} size={} dtype={}", t.start, t.size, t.dtype); let wq = qf.read_bf16(key).unwrap();
            let mut nd = 0usize; let mut maxd = 0f32;
            for i in 0..wq.len().min(d_out * d_in) {
                let gv = gri::fp16_to_f32(u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]));
                if (wq[i] - gv).abs() > 1e-6 { nd += 1; }
                let d = (wq[i] - gv).abs(); if d > maxd { maxd = d; }
            }
            println!("qat.read_bf16 vs gri raw: diffs={nd} max|d|={maxd:.3e} (n={})", wq.len().min(d_out * d_in));
            println!("qat w[0..4]={:?} gri raw[0..4]={:?}", &wq[..4], (0..4).map(|i| gri::fp16_to_f32(u16::from_le_bytes([raw[i*2], raw[i*2+1]]))).collect::<Vec<_>>());
        }
        // full-tensor cos decoded-vs-qat-read (authoritative source read)
        {
            let qf = qat::open_qat(model).unwrap();
            let wq = qf.read_bf16(key).unwrap();
            let mut dot = 0f64; let mut n1 = 0f64; let mut n2 = 0f64;
            for i in 0..d_out * d_in {
                dot += w[i] as f64 * wq[i] as f64; n1 += w[i] as f64 * w[i] as f64; n2 += wq[i] as f64 * wq[i] as f64;
            }
            println!("full-tensor decode-vs-qat: cos={:.6} (n={})", dot / (n1.sqrt() * n2.sqrt() + 1e-30), d_out * d_in);
        }
    }
    let mut maxd = 0f32;
    let mut dot_rr = 0f64;
    let mut dot_ss = 0f64;
    for i in 0..d_out.min(64) {
        let d = (y[i] - row_raw[0] * 0.0 - 0.0).abs(); // placeholder
        let _ = d;
    }
    // proper: compare y (col 0) against raw column 0
    let col0: Vec<f32> = (0..d_out).map(|r| {
        let b = r * d_in * 2;
        gri::fp16_to_f32(u16::from_le_bytes([raw[b], raw[b + 1]]))
    }).collect();
    for i in 0..d_out {
        let dd = (y[i] - col0[i]).abs();
        if dd > maxd { maxd = dd; }
        dot_rr += col0[i] as f64 * col0[i] as f64;
        dot_ss += y[i] as f64 * y[i] as f64;
    }
    let mut dot = 0f64;
    for i in 0..d_out { dot += y[i] as f64 * col0[i] as f64; }
    let cos = dot / (dot_rr.sqrt() * dot_ss.sqrt() + 1e-30);
    println!("proj {key} d_out={d_out} d_in={d_in} stream_off={soff}");
    println!("col0 gemv-vs-raw: max|d|={maxd:.6e} cos={cos:.6}");
    println!("sample raw[0..4]={:?} gemv[0..4]={:?}", &col0[..4], &y[..4]);
}
