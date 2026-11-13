//! OPT4 132 B glyph codec and IEEE binary16 / bfloat16 conversion primitives.
//!
//! Layout of each 132 B OPT4 glyph (256 weights = 32 octants x 8 dims):
//!   [0..2)    s1   fp16 global stage-1 scale shared by all 32 octants
//!   [2..4)    s2   fp16 global stage-2 scale shared by all 32 octants
//!   [4..68)   idx1 32 x u16 codebook indices (< 2048) for stage 1
//!   [68..132) idx2 32 x u16 codebook indices (< 2048) for stage 2
//!
//! Decoded weight for octant `oct` in 0..32, coord `j` in 0..8:
//!   w[oct*8 + j] = fp16(s1) * cb[idx1[oct]][j] + fp16(s2) * cb[idx2[oct]][j]

pub const CB_CLASSES: usize = 2048;
pub const CB_DIM: usize = 8;
pub const OCTANTS: usize = 32;
pub const GLYPH_WEIGHTS: usize = 256;
pub const GLYPH_BYTES: usize = 132;
pub const STREAM_HEADER_BYTES: usize = 16;
pub const LANU_MAGIC: &[u8; 4] = b"LANU";
pub const LANU_VERSION: u16 = 1;
pub const LANU_CODEC_OPT4: u16 = 45;

#[inline]
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

#[inline]
pub fn f32_to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let lsb = (bits >> 16) & 1;
    (bits.wrapping_add(0x7FFF).wrapping_add(lsb) >> 16) as u16
}

#[inline]
pub fn fp16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let v = (mant as f32) * 5.9604644775390625e-8; // 2^-24
            (sign << 31) | v.to_bits()
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

pub fn fp32_to_fp16(val: f32) -> u16 {
    let f = val.to_bits();
    let sign = ((f >> 16) & 0x8000u32) as u16;
    let exp = ((f >> 23) & 0xFFu32) as i32 - 127 + 15;
    let mut man = f & 0x007FFFFFu32;
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        man |= 0x00800000u32;
        let shift = (14 - exp) as u32;
        let half_bit = 1u32 << (shift - 1);
        let rem_mask = half_bit - 1u32;
        let mut h_man = man >> shift;
        if (man & half_bit) != 0 && ((man & rem_mask) != 0 || (h_man & 1u32) != 0) {
            h_man += 1;
        }
        sign | (h_man as u16)
    } else if exp >= 31 {
        if ((f >> 23) & 0xFFu32) == 0xFFu32 && man != 0 {
            sign | 0x7E00u16
        } else {
            sign | 0x7C00u16
        }
    } else {
        if (man & 0x00001000u32) != 0 && ((man & 0x00000FFFu32) != 0 || ((man >> 13) & 1u32) != 0) {
            man += 0x00002000u32;
            if (man & 0x00800000u32) != 0 {
                let next_exp = exp + 1;
                if next_exp >= 31 {
                    return sign | 0x7C00u16;
                }
                return sign | ((next_exp as u16) << 10);
            }
        }
        sign | (((exp as u16) << 10) | ((man >> 13) as u16))
    }
}

#[inline]
fn rd_u16_le(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

pub fn make_stream_header(nglyphs: u64, rows: u64) -> Result<[u8; STREAM_HEADER_BYTES], String> {
    let ng = u32::try_from(nglyphs).map_err(|_| format!("glyph count {nglyphs} overflows u32 header"))?;
    let nr = u32::try_from(rows).map_err(|_| format!("row count {rows} overflows u32 header"))?;
    let mut h = [0u8; STREAM_HEADER_BYTES];
    h[0..4].copy_from_slice(LANU_MAGIC);
    h[4..6].copy_from_slice(&LANU_VERSION.to_le_bytes());
    h[6..8].copy_from_slice(&LANU_CODEC_OPT4.to_le_bytes());
    h[8..12].copy_from_slice(&ng.to_le_bytes());
    h[12..16].copy_from_slice(&nr.to_le_bytes());
    Ok(h)
}

pub fn decode_opt4_glyph(glyph: &[u8], cb: &[f32], out: &mut [f32; GLYPH_WEIGHTS]) {
    debug_assert!(glyph.len() >= GLYPH_BYTES);
    debug_assert!(cb.len() >= CB_CLASSES * CB_DIM);
    let s1 = fp16_to_f32(rd_u16_le(glyph, 0));
    let s2 = fp16_to_f32(rd_u16_le(glyph, 2));
    for oct in 0..OCTANTS {
        let i1 = (rd_u16_le(glyph, 4 + oct * 2) as usize) & (CB_CLASSES - 1);
        let i2 = (rd_u16_le(glyph, 68 + oct * 2) as usize) & (CB_CLASSES - 1);
        let c1 = &cb[i1 * CB_DIM..i1 * CB_DIM + CB_DIM];
        let c2 = &cb[i2 * CB_DIM..i2 * CB_DIM + CB_DIM];
        let o = oct * CB_DIM;
        for j in 0..CB_DIM {
            out[o + j] = s1 * c1[j] + s2 * c2[j];
        }
    }
}

#[inline]
fn argmax_dot8(v: &[f32; 8], cb: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best_dot = f32::NEG_INFINITY;
    for c in 0..CB_CLASSES {
        let cc = &cb[c * CB_DIM..c * CB_DIM + CB_DIM];
        let d = v[0] * cc[0]
            + v[1] * cc[1]
            + v[2] * cc[2]
            + v[3] * cc[3]
            + v[4] * cc[4]
            + v[5] * cc[5]
            + v[6] * cc[6]
            + v[7] * cc[7];
        if d > best_dot {
            best_dot = d;
            best_idx = c;
        }
    }
    best_idx
}

/// Encode 256 weights into a 132 B OPT4 glyph using a coherent glyph-wide `(s1, s2)` solve.
///
/// Stage 1 selects 32 codebook centroids maximizing `+dot(w_oct, c)` and solves one global `s1`.
/// Stage 2 selects 32 residual centroids against the true global Stage-1 residual `w - s1 * C1`.
/// A joint 2x2 least-squares solve then yields the exact global `(s1, s2)` quantized to binary16,
/// and cosine/SNR are measured strictly from `decode_opt4_glyph(payload, cb)` so encoding
/// receipts always match decoded stream weights bit-for-bit.
pub fn encode_opt4_glyph(w256: &[f32; GLYPH_WEIGHTS], cb: &[f32]) -> ([u8; GLYPH_BYTES], f64, f64) {
    let mut idx1 = [0u16; OCTANTS];
    let mut num1 = 0f64;
    let mut a11 = 0f64;

    for oct in 0..OCTANTS {
        let w: &[f32; 8] = w256[oct * 8..oct * 8 + 8].try_into().unwrap();
        let best = argmax_dot8(w, cb);
        idx1[oct] = best as u16;
        let c1 = &cb[best * 8..best * 8 + 8];
        for t in 0..8 {
            let wf = w[t] as f64;
            let cf = c1[t] as f64;
            num1 += wf * cf;
            a11 += cf * cf;
        }
    }

    let s1_est = if a11 > 1e-18 { (num1 / a11) as f32 } else { 0.0f32 };

    let mut idx2 = [0u16; OCTANTS];
    let mut num2 = 0f64;
    let mut a12 = 0f64;
    let mut a22 = 0f64;

    for oct in 0..OCTANTS {
        let w = &w256[oct * 8..oct * 8 + 8];
        let i1 = idx1[oct] as usize;
        let c1 = &cb[i1 * 8..i1 * 8 + 8];
        let mut res = [0f32; 8];
        for t in 0..8 {
            res[t] = w[t] - s1_est * c1[t];
        }
        let best2 = argmax_dot8(&res, cb);
        idx2[oct] = best2 as u16;
        let c2 = &cb[best2 * 8..best2 * 8 + 8];
        for t in 0..8 {
            let wf = w[t] as f64;
            let f1 = c1[t] as f64;
            let f2 = c2[t] as f64;
            num2 += wf * f2;
            a12 += f1 * f2;
            a22 += f2 * f2;
        }
    }

    // Joint 2x2 least squares for global (s1, s2):
    //   [a11, a12] [s1] = [num1]
    //   [a12, a22] [s2]   [num2]
    let det = a11 * a22 - a12 * a12;
    let (s1_opt, s2_opt) = if det.abs() > 1e-12 * a11 * a22 {
        (((num1 * a22 - num2 * a12) / det) as f32, ((a11 * num2 - a12 * num1) / det) as f32)
    } else {
        let res2 = if a22 > 1e-18 { ((num2 - s1_est as f64 * a12) / a22) as f32 } else { 0.0f32 };
        (s1_est, res2)
    };

    let s1_h = fp32_to_fp16(s1_opt);
    let s2_h = fp32_to_fp16(s2_opt);

    let mut payload = [0u8; GLYPH_BYTES];
    payload[0..2].copy_from_slice(&s1_h.to_le_bytes());
    payload[2..4].copy_from_slice(&s2_h.to_le_bytes());
    for oct in 0..OCTANTS {
        payload[4 + oct * 2..6 + oct * 2].copy_from_slice(&idx1[oct].to_le_bytes());
        payload[68 + oct * 2..70 + oct * 2].copy_from_slice(&idx2[oct].to_le_bytes());
    }

    // Measure reconstruction fidelity strictly from decoded glyph bytes.
    let mut recon = [0f32; GLYPH_WEIGHTS];
    decode_opt4_glyph(&payload, cb, &mut recon);
    let mut dot = 0f64;
    let mut nw = 0f64;
    let mut nr = 0f64;
    let mut err = 0f64;
    for i in 0..GLYPH_WEIGHTS {
        let wi = w256[i] as f64;
        let ri = recon[i] as f64;
        dot += wi * ri;
        nw += wi * wi;
        nr += ri * ri;
        let d = wi - ri;
        err += d * d;
    }
    let cos = if nw > 0.0 && nr > 0.0 { dot / (nw.sqrt() * nr.sqrt()) } else { 0.0 };
    let snr = 10.0 * ((nw + 1e-30) / (err + 1e-30)).log10();
    (payload, cos, snr)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_codebook() -> Vec<f32> {
        let mut cb = vec![0f32; CB_CLASSES * CB_DIM];
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next_f32 = || -> f32 {
            // Sum of 4 uniforms approximates N(0, 1/3) so vectors spread isotropically in R^8.
            let mut acc = 0.0f32;
            for _ in 0..4 {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                let u = ((z >> 40) as f32) * (1.0 / 16_777_216.0) - 0.5;
                acc += u;
            }
            acc
        };
        for c in 0..CB_CLASSES {
            let mut s = 0f32;
            for j in 0..CB_DIM {
                let val = if c & 1 == 1 { -cb[(c - 1) * CB_DIM + j] } else { next_f32() };
                cb[c * CB_DIM + j] = val;
                s += val * val;
            }
            let inv = 1.0 / s.sqrt().max(1e-6);
            for j in 0..CB_DIM {
                cb[c * CB_DIM + j] *= inv;
            }
        }
        cb
    }

    #[test]
    fn bf16_vs_fp16_distinction() {
        // 1.0f32 in BF16 is 0x3F80; in FP16 0x3F80 is 1.875.
        assert_eq!(bf16_to_f32(0x3F80), 1.0f32);
        assert_eq!(bf16_to_f32(0xBF80), -1.0f32);
        assert_eq!(f32_to_bf16(1.0f32), 0x3F80);
        assert_eq!(f32_to_bf16(-1.0f32), 0xBF80);
        assert_eq!(f32_to_bf16(f32::INFINITY), 0x7F80);
        assert_eq!(fp16_to_f32(0x3C00), 1.0f32);
        assert_eq!(fp16_to_f32(0x3F80), 1.875f32);
    }

    #[test]
    fn fp16_subnormals_and_overflow_rounding() {
        for mant in 1u32..1024 {
            let h = mant as u16;
            let got = fp16_to_f32(h);
            let want = (mant as f32) * 5.9604644775390625e-8;
            assert_eq!(got.to_bits(), want.to_bits());
            assert_eq!(fp16_to_f32(0x8000 | h), -want);
        }
        assert_eq!(fp32_to_fp16(65535.0), 0x7C00); // saturates to +inf cleanly
    }

    #[test]
    fn encode_and_decode_opt4_metrics_match_exact() {
        let cb = test_codebook();
        let mut w = [0f32; 256];
        for i in 0..256 {
            w[i] = ((i as f32 * 0.37).sin() + 0.3 * (i as f32 * 1.13).cos()) * 0.05;
        }
        let (payload, enc_cos, enc_snr) = encode_opt4_glyph(&w, &cb);
        let mut decoded = [0f32; 256];
        decode_opt4_glyph(&payload, &cb, &mut decoded);
        let mut dot = 0f64;
        let mut nw = 0f64;
        let mut nd = 0f64;
        let mut err = 0f64;
        for i in 0..256 {
            dot += w[i] as f64 * decoded[i] as f64;
            nw += (w[i] as f64).powi(2);
            nd += (decoded[i] as f64).powi(2);
            err += (w[i] as f64 - decoded[i] as f64).powi(2);
        }
        let dec_cos = dot / (nw.sqrt() * nd.sqrt());
        let dec_snr = 10.0 * ((nw + 1e-30) / (err + 1e-30)).log10();
        assert!((enc_cos - dec_cos).abs() < 1e-12, "enc={enc_cos} vs dec={dec_cos}");
        assert!((enc_snr - dec_snr).abs() < 1e-10);
        assert!(dec_cos > 0.85, "expected isotropic OPT4 dec_cos > 0.85, got {dec_cos}");
    }
}
