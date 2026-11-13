//! Genuine sub-byte packed quantization containers and on-the-fly dequant-in-matvec kernels.
//!
//! Zero unpacked FP32/BF16 weight arrays are materialized at inference time: projection matrices and
//! optional packed embeddings reside strictly as compact byte buffers in RAM and unpack tile-by-tile
//! on the stack during dot-product accumulation.

use crate::codec::{fp16_to_f32, fp32_to_fp16};

/// Normalized 8x8 Sylvester-Hadamard orthogonal transform (`H8^T = H8`, `H8 * H8 = I8`).
#[inline]
pub fn fwht8(v: &[f32; 8]) -> [f32; 8] {
    const INV_SQRT8: f32 = 0.3535533905932737622;
    let a0 = v[0] + v[1];
    let a1 = v[0] - v[1];
    let a2 = v[2] + v[3];
    let a3 = v[2] - v[3];
    let a4 = v[4] + v[5];
    let a5 = v[4] - v[5];
    let a6 = v[6] + v[7];
    let a7 = v[6] - v[7];

    let b0 = a0 + a2;
    let b2 = a0 - a2;
    let b1 = a1 + a3;
    let b3 = a1 - a3;
    let b4 = a4 + a6;
    let b6 = a4 - a6;
    let b5 = a5 + a7;
    let b7 = a5 - a7;

    [
        (b0 + b4) * INV_SQRT8,
        (b1 + b5) * INV_SQRT8,
        (b2 + b6) * INV_SQRT8,
        (b3 + b7) * INV_SQRT8,
        (b0 - b4) * INV_SQRT8,
        (b1 - b5) * INV_SQRT8,
        (b2 - b6) * INV_SQRT8,
        (b3 - b7) * INV_SQRT8,
    ]
}

/// Apply `fwht8` across every 8-element octonion block of `src` (padding trailing tail with 0 if any).
pub fn fwht8_vec(src: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(src.len());
    let mut off = 0usize;
    while off < src.len() {
        let mut blk = [0f32; 8];
        let take = (src.len() - off).min(8);
        blk[..take].copy_from_slice(&src[off..off + take]);
        let r = fwht8(&blk);
        out.extend_from_slice(&r[..take]);
        off += take;
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantScheme {
    /// PR #21 36 B 2-means binary container per 256 weights (1.125 bits/weight).
    B1_2Means,
    /// Group-32 mid-riser symmetric 2-bit: 4 uniform levels {-1.5, -0.5, +0.5, +1.5} + f16 scale (2.50 bpw).
    Q2G32,
    /// Group-32 symmetric 3-bit signed [-3, 3] + f16 scale (3.50 bpw).
    Q3G32,
    /// Group-32 symmetric 4-bit signed [-7, 7] + f16 scale (4.50 bpw).
    Q4G32,
    /// Group-32 symmetric 5-bit signed [-15, 15] + f16 scale (5.50 bpw).
    Q5G32,
    /// Group-32 symmetric 6-bit signed [-31, 31] + f16 scale (6.50 bpw).
    Q6G32,
    /// Group-32 symmetric 8-bit signed [-127, 127] + f16 scale (8.50 bpw).
    Q8G32,
    /// Group-32 symmetric 10-bit signed [-511, 511] + f32 scale (11.00 bpw).
    Q10G32,
    /// Group-32 symmetric 12-bit signed [-2047, 2047] + f32 scale (13.00 bpw).
    Q12G32,
}

impl QuantScheme {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "b1" | "b1_2means" | "36b" | "1bit" => Ok(Self::B1_2Means),
            "q2" | "q2_g32" => Ok(Self::Q2G32),
            "q3" | "q3_g32" => Ok(Self::Q3G32),
            "q4" | "q4_g32" => Ok(Self::Q4G32),
            "q5" | "q5_g32" => Ok(Self::Q5G32),
            "q6" | "q6_g32" => Ok(Self::Q6G32),
            "q8" | "q8_g32" => Ok(Self::Q8G32),
            "q10" | "q10_g32" => Ok(Self::Q10G32),
            "q12" | "q12_g32" => Ok(Self::Q12G32),
            _ => Err(format!("unknown quant scheme '{s}' (expected b1, q2, q3, q4, q5, q6, q8, q10, q12)")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::B1_2Means => "b1_2means",
            Self::Q2G32 => "q2_g32",
            Self::Q3G32 => "q3_g32",
            Self::Q4G32 => "q4_g32",
            Self::Q5G32 => "q5_g32",
            Self::Q6G32 => "q6_g32",
            Self::Q8G32 => "q8_g32",
            Self::Q10G32 => "q10_g32",
            Self::Q12G32 => "q12_g32",
        }
    }

    pub const fn group_size(self) -> usize {
        match self {
            Self::B1_2Means => 256,
            _ => 32,
        }
    }

    pub const fn bits(self) -> u8 {
        match self {
            Self::B1_2Means => 1,
            Self::Q2G32 => 2,
            Self::Q3G32 => 3,
            Self::Q4G32 => 4,
            Self::Q5G32 => 5,
            Self::Q6G32 => 6,
            Self::Q8G32 => 8,
            Self::Q10G32 => 10,
            Self::Q12G32 => 12,
        }
    }

    pub const fn scale_is_f32(self) -> bool {
        matches!(self, Self::Q10G32 | Self::Q12G32)
    }

    pub const fn block_bytes(self) -> usize {
        match self {
            Self::B1_2Means => 36,
            _ => {
                let sc = if self.scale_is_f32() { 4 } else { 2 };
                sc + (self.bits() as usize) * 4
            }
        }
    }

    pub fn effective_bpw(self) -> f64 {
        (self.block_bytes() * 8) as f64 / self.group_size() as f64
    }
}

fn encode_2means_256(w: &[f32; 256], dst: &mut Vec<u8>) {
    let mut c0 = w[0];
    let mut c1 = w[0];
    for &v in &w[1..] {
        if v < c0 {
            c0 = v;
        }
        if v > c1 {
            c1 = v;
        }
    }
    for _ in 0..8 {
        let mut s0 = 0f64;
        let mut s1 = 0f64;
        let mut n0 = 0usize;
        let mut n1 = 0usize;
        for &v in w {
            if (v - c0).abs() <= (v - c1).abs() {
                s0 += v as f64;
                n0 += 1;
            } else {
                s1 += v as f64;
                n1 += 1;
            }
        }
        let nc0 = if n0 > 0 { (s0 / n0 as f64) as f32 } else { c0 };
        let nc1 = if n1 > 0 { (s1 / n1 as f64) as f32 } else { c1 };
        if nc0 == c0 && nc1 == c1 {
            break;
        }
        c0 = nc0;
        c1 = nc1;
    }
    if c1 < c0 {
        std::mem::swap(&mut c0, &mut c1);
    }
    let bias = 0.5 * (c0 + c1);
    let scale = 0.5 * (c1 - c0).abs();
    let hs = fp32_to_fp16(scale);
    let hb = fp32_to_fp16(bias);
    let ds = fp16_to_f32(hs);
    let db = fp16_to_f32(hb);
    let r0 = db - ds;
    let r1 = db + ds;
    dst.extend_from_slice(&hs.to_le_bytes());
    dst.extend_from_slice(&hb.to_le_bytes());
    let mut mask = [0u8; 32];
    for i in 0..256 {
        if (w[i] - r1).abs() < (w[i] - r0).abs() {
            mask[i >> 3] |= 0x80u8 >> (i & 7);
        }
    }
    dst.extend_from_slice(&mask);
}

fn encode_group32(w: &[f32; 32], scheme: QuantScheme, dst: &mut Vec<u8>) {
    let bits = scheme.bits();
    let max_abs = w.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let qmax: f32 = if bits == 2 {
        1.5
    } else {
        ((1u32 << (bits - 1)) - 1) as f32
    };

    // For low-bit regimes (<= 5 bits), sweep clipping factors cleanly to guard against outliers.
    const CLIP_LOW: [f32; 4] = [1.0, 0.88, 0.78, 0.68];
    const CLIP_HIGH: [f32; 1] = [1.0];
    let cands: &[f32] = if bits <= 5 { &CLIP_LOW } else { &CLIP_HIGH };

    let mut best_err = f64::INFINITY;
    let mut best_scale_bytes = [0u8; 4];
    let mut best_codes = [0u32; 32];

    for &cf in cands {
        let raw_delta = if max_abs > 1e-30 { (max_abs * cf) / qmax } else { 0.0 };
        let (delta, sbytes) = if scheme.scale_is_f32() {
            (raw_delta, raw_delta.to_le_bytes())
        } else {
            let h = fp32_to_fp16(raw_delta);
            let b = h.to_le_bytes();
            (fp16_to_f32(h), [b[0], b[1], 0, 0])
        };

        let mut codes = [0u32; 32];
        let mut err = 0f64;
        if delta > 0.0 {
            let inv = 1.0 / delta;
            if bits == 2 {
                for j in 0..32 {
                    let c = (w[j] * inv + 1.5).round().clamp(0.0, 3.0) as u32;
                    codes[j] = c;
                    let rw = ((c as f32) - 1.5) * delta;
                    let d = (w[j] - rw) as f64;
                    err += d * d;
                }
            } else {
                let qmax_i = ((1i32 << (bits - 1)) - 1) as i32;
                for j in 0..32 {
                    let q = (w[j] * inv).round().clamp(-qmax_i as f32, qmax_i as f32) as i32;
                    codes[j] = (q + qmax_i) as u32;
                    let rw = (q as f32) * delta;
                    let d = (w[j] - rw) as f64;
                    err += d * d;
                }
            }
        } else if bits == 2 {
            // Exact zero group maps evenly near zero when delta == 0 (all levels reconstruct to 0).
            codes = [1; 32];
            err = 0.0;
        } else {
            let qmax_u = (1u32 << (bits - 1)) - 1;
            codes = [qmax_u; 32];
            err = 0.0;
        }

        if err < best_err {
            best_err = err;
            best_scale_bytes = sbytes;
            best_codes = codes;
        }
    }

    if scheme.scale_is_f32() {
        dst.extend_from_slice(&best_scale_bytes);
    } else {
        dst.extend_from_slice(&best_scale_bytes[..2]);
    }

    // Pack 32 x `bits` unsigned codes tightly LSB-first into `4 * bits` bytes.
    let mut acc: u64 = 0;
    let mut nbits: u32 = 0;
    for &c in &best_codes {
        acc |= (c as u64) << nbits;
        nbits += bits as u32;
        while nbits >= 8 {
            dst.push((acc & 0xff) as u8);
            acc >>= 8;
            nbits -= 8;
        }
    }
    debug_assert_eq!(nbits, 0);
}

#[inline]
fn dot_2means_256(block: &[u8], x: &[f32]) -> f64 {
    let scale = fp16_to_f32(u16::from_le_bytes([block[0], block[1]])) as f64;
    let bias = fp16_to_f32(u16::from_le_bytes([block[2], block[3]])) as f64;
    let mut sum_x = 0f64;
    let mut signed_x = 0f64;
    let n = x.len().min(256);
    for i in 0..n {
        let xv = x[i] as f64;
        sum_x += xv;
        let bit = (block[4 + (i >> 3)] >> (7 - (i & 7))) & 1;
        signed_x += if bit != 0 { xv } else { -xv };
    }
    bias * sum_x + scale * signed_x
}

#[inline]
fn dot_group32(block: &[u8], x: &[f32], scheme: QuantScheme) -> f64 {
    let (delta, pbytes) = if scheme.scale_is_f32() {
        (f32::from_le_bytes([block[0], block[1], block[2], block[3]]) as f64, &block[4..])
    } else {
        (fp16_to_f32(u16::from_le_bytes([block[0], block[1]])) as f64, &block[2..])
    };
    if delta == 0.0 {
        return 0.0;
    }
    let bits = scheme.bits();
    let n = x.len().min(32);

    if bits == 4 {
        let mut sub = 0f32;
        if n == 32 {
            for pair in 0..16 {
                let b = pbytes[pair];
                let q0 = ((b & 0x0F) as i32 - 7) as f32;
                let q1 = ((b >> 4) as i32 - 7) as f32;
                sub += q0 * x[pair * 2] + q1 * x[pair * 2 + 1];
            }
        } else {
            for pair in 0..16 {
                let b = pbytes[pair];
                let j0 = pair * 2;
                if j0 < n {
                    sub += ((b & 0x0F) as i32 - 7) as f32 * x[j0];
                }
                let j1 = j0 + 1;
                if j1 < n {
                    sub += ((b >> 4) as i32 - 7) as f32 * x[j1];
                }
            }
        }
        return delta * (sub as f64);
    }

    if bits == 8 {
        let mut sub = 0f32;
        if n == 32 {
            for j in 0..32 {
                let q = (pbytes[j] as i32 - 127) as f32;
                sub += q * x[j];
            }
        } else {
            for j in 0..n {
                let q = (pbytes[j] as i32 - 127) as f32;
                sub += q * x[j];
            }
        }
        return delta * (sub as f64);
    }

    let mask = (1u64 << bits) - 1;
    let mut acc: u64 = 0;
    let mut have: u32 = 0;
    let mut bidx: usize = 0;
    let mut sub = 0f32;
    if bits == 2 {
        for j in 0..n {
            while have < 2 {
                acc |= (pbytes[bidx] as u64) << have;
                bidx += 1;
                have += 8;
            }
            let c = (acc & 0x3) as i32;
            acc >>= 2;
            have -= 2;
            sub += (c as f32 - 1.5) * x[j];
        }
        return delta * (sub as f64);
    }

    let qmax = (1i32 << (bits - 1)) - 1;
    for j in 0..n {
        while have < bits as u32 {
            acc |= (pbytes[bidx] as u64) << have;
            bidx += 1;
            have += 8;
        }
        let c = (acc & mask) as i32;
        acc >>= bits as u32;
        have -= bits as u32;
        sub += (c - qmax) as f32 * x[j];
    }
    delta * (sub as f64)
}

#[inline]
fn decode_group_into(block: &[u8], scheme: QuantScheme, out: &mut [f32]) {
    if scheme == QuantScheme::B1_2Means {
        let scale = fp16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let bias = fp16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        for (i, slot) in out.iter_mut().enumerate().take(256) {
            let bit = (block[4 + (i >> 3)] >> (7 - (i & 7))) & 1;
            *slot = bias + if bit != 0 { scale } else { -scale };
        }
        return;
    }
    let bits = scheme.bits();
    let (delta, pbytes) = if scheme.scale_is_f32() {
        (f32::from_le_bytes([block[0], block[1], block[2], block[3]]), &block[4..])
    } else {
        (fp16_to_f32(u16::from_le_bytes([block[0], block[1]])), &block[2..])
    };
    let mask = (1u64 << bits) - 1;
    let mut acc: u64 = 0;
    let mut have: u32 = 0;
    let mut bidx: usize = 0;
    let qmax = (1i32 << (bits - 1)) - 1;
    for slot in out.iter_mut().take(32) {
        while have < bits as u32 {
            acc |= (pbytes[bidx] as u64) << have;
            bidx += 1;
            have += 8;
        }
        let c = (acc & mask) as i32;
        acc >>= bits as u32;
        have -= bits as u32;
        *slot = if bits == 2 {
            (c as f32 - 1.5) * delta
        } else {
            (c - qmax) as f32 * delta
        };
    }
}

/// Row-major packed sub-byte matrix in RAM; never expands weights into FP32/BF16 heap storage.
pub struct PackedMatrix {
    pub d_out: usize,
    pub d_in: usize,
    pub scheme: QuantScheme,
    pub use_h8: bool,
    pub row_bytes: usize,
    pub bytes: Vec<u8>,
}

impl PackedMatrix {
    pub fn pack_from_f32(d_out: usize, d_in: usize, w_f32: &[f32], scheme: QuantScheme, use_h8: bool) -> Result<Self, String> {
        Self::pack_from_f32_mt(d_out, d_in, w_f32, scheme, use_h8, 1)
    }

    pub fn pack_from_f32_mt(
        d_out: usize,
        d_in: usize,
        w_f32: &[f32],
        scheme: QuantScheme,
        use_h8: bool,
        threads: usize,
    ) -> Result<Self, String> {
        let mut mat = Self::new_empty(d_out, d_in, scheme, use_h8)?;
        mat.pack_rows_range_f32(0, d_out, w_f32, threads)?;
        Ok(mat)
    }

    pub fn new_empty(d_out: usize, d_in: usize, scheme: QuantScheme, use_h8: bool) -> Result<Self, String> {
        if use_h8 && d_in % 8 != 0 {
            return Err(format!("H8 rotation requires d_in ({d_in}) divisible by 8"));
        }
        let gs = scheme.group_size();
        let gb = scheme.block_bytes();
        let groups_per_row = d_in.div_ceil(gs);
        let row_bytes = groups_per_row * gb;
        Ok(Self {
            d_out,
            d_in,
            scheme,
            use_h8,
            row_bytes,
            bytes: vec![0u8; d_out * row_bytes],
        })
    }

    pub fn pack_rows_range_f32(
        &mut self,
        row0: usize,
        n_rows: usize,
        w_slice: &[f32],
        threads: usize,
    ) -> Result<(), String> {
        if row0 + n_rows > self.d_out {
            return Err(format!("pack rows [{}..{}) exceeds d_out {}", row0, row0 + n_rows, self.d_out));
        }
        if w_slice.len() != n_rows * self.d_in {
            return Err(format!("pack slice len {} != {}*{}", w_slice.len(), n_rows, self.d_in));
        }
        if n_rows == 0 {
            return Ok(());
        }
        let gs = self.scheme.group_size();
        let groups_per_row = self.d_in.div_ceil(gs);
        let row_bytes = self.row_bytes;
        let d_in = self.d_in;
        let scheme = self.scheme;
        let use_h8 = self.use_h8;
        let nthreads = threads.max(1).min(n_rows);
        let chunk = n_rows.div_ceil(nthreads);

        let dst_slice = &mut self.bytes[row0 * row_bytes..(row0 + n_rows) * row_bytes];
        std::thread::scope(|s| {
            let mut handles = Vec::new();
            for (t, dst_chunk) in dst_slice.chunks_mut(chunk * row_bytes).enumerate() {
                let local_r0 = t * chunk;
                let local_rows = dst_chunk.len() / row_bytes;
                let src_chunk = &w_slice[local_r0 * d_in..(local_r0 + local_rows) * d_in];
                handles.push(s.spawn(move || {
                    let mut row_buf = vec![0f32; groups_per_row * gs];
                    let mut tmp = Vec::with_capacity(row_bytes);
                    for r in 0..local_rows {
                        row_buf.fill(0.0);
                        let src_row = &src_chunk[r * d_in..(r + 1) * d_in];
                        if use_h8 {
                            let mut off = 0usize;
                            while off < d_in {
                                let blk: &[f32; 8] = src_row[off..off + 8].try_into().unwrap();
                                let rot = fwht8(blk);
                                row_buf[off..off + 8].copy_from_slice(&rot);
                                off += 8;
                            }
                        } else {
                            row_buf[..d_in].copy_from_slice(src_row);
                        }
                        tmp.clear();
                        if scheme == QuantScheme::B1_2Means {
                            for g in 0..groups_per_row {
                                let blk: &[f32; 256] = row_buf[g * 256..(g + 1) * 256].try_into().unwrap();
                                encode_2means_256(blk, &mut tmp);
                            }
                        } else {
                            for g in 0..groups_per_row {
                                let blk: &[f32; 32] = row_buf[g * 32..(g + 1) * 32].try_into().unwrap();
                                encode_group32(blk, scheme, &mut tmp);
                            }
                        }
                        dst_chunk[r * row_bytes..(r + 1) * row_bytes].copy_from_slice(&tmp);
                    }
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
        });
        Ok(())
    }

    /// Multithreaded direct dequant-in-matvec (`y = W * x`).
    pub fn matvec(&self, x: &[f32], threads: usize) -> Result<Vec<f32>, String> {
        if x.len() != self.d_in {
            return Err(format!("x len {} != d_in {}", x.len(), self.d_in));
        }
        let x_rot;
        let xv: &[f32] = if self.use_h8 {
            x_rot = fwht8_vec(x);
            &x_rot
        } else {
            x
        };
        let mut y = vec![0f32; self.d_out];
        let nthreads = threads.max(1);
        let chunk = self.d_out.div_ceil(nthreads).max(1);
        let gs = self.scheme.group_size();
        let gb = self.scheme.block_bytes();
        let groups_per_row = self.d_in.div_ceil(gs);

        std::thread::scope(|s| {
            let mut handles = Vec::new();
            for (t, yb) in y.chunks_mut(chunk).enumerate() {
                let r0 = t * chunk;
                let row_bytes = self.row_bytes;
                let bytes = &self.bytes;
                let d_in = self.d_in;
                let scheme = self.scheme;
                handles.push(s.spawn(move || {
                    for (local_r, slot) in yb.iter_mut().enumerate() {
                        let r = r0 + local_r;
                        let rslice = &bytes[r * row_bytes..(r + 1) * row_bytes];
                        let mut acc = 0f64;
                        if scheme == QuantScheme::B1_2Means {
                            for g in 0..groups_per_row {
                                let off = g * 256;
                                acc += dot_2means_256(&rslice[g * 36..(g + 1) * 36], &xv[off..d_in.min(off + 256)]);
                            }
                        } else {
                            for g in 0..groups_per_row {
                                let off = g * 32;
                                acc += dot_group32(&rslice[g * gb..(g + 1) * gb], &xv[off..d_in.min(off + 32)], scheme);
                            }
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

    /// Decode a single packed row into `out[0..self.d_in]` (e.g. tied embedding lookup).
    pub fn decode_row(&self, r: usize) -> Result<Vec<f32>, String> {
        if r >= self.d_out {
            return Err(format!("row {r} out of bounds for d_out {}", self.d_out));
        }
        let gs = self.scheme.group_size();
        let gb = self.scheme.block_bytes();
        let groups_per_row = self.d_in.div_ceil(gs);
        let mut buf = vec![0f32; groups_per_row * gs];
        let rslice = &self.bytes[r * self.row_bytes..(r + 1) * self.row_bytes];
        for g in 0..groups_per_row {
            decode_group_into(&rslice[g * gb..(g + 1) * gb], self.scheme, &mut buf[g * gs..(g + 1) * gs]);
        }
        buf.truncate(self.d_in);
        if self.use_h8 {
            Ok(fwht8_vec(&buf))
        } else {
            Ok(buf)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fwht8_is_self_inverse_and_preserves_dot() {
        let x = [0.5f32, -1.2, 3.1, 0.0, -0.7, 2.2, 1.4, -0.3];
        let w = [-0.1f32, 0.8, -1.5, 0.4, 2.1, -0.9, 0.2, 0.6];
        let hx = fwht8(&x);
        let hw = fwht8(&w);
        let hhx = fwht8(&hx);
        for i in 0..8 {
            assert!((x[i] - hhx[i]).abs() < 1e-5);
        }
        let dot_orig: f32 = w.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let dot_rot: f32 = hw.iter().zip(hx.iter()).map(|(a, b)| a * b).sum();
        assert!((dot_orig - dot_rot).abs() < 1e-5);
    }

    #[test]
    fn packed_schemes_roundtrip_and_matvec_parity() {
        let d_out = 4usize;
        let d_in = 256usize;
        let mut w = vec![0f32; d_out * d_in];
        for (idx, slot) in w.iter_mut().enumerate() {
            let f = idx as f32;
            *slot = ((f * 0.13).sin() + 0.25 * (f * 0.47).cos()) * 0.2;
        }
        // Inject one strong outlier inside row 0 to check H8 attenuation on low bits.
        w[11] = 2.4;

        let mut x = vec![0f32; d_in];
        for (i, slot) in x.iter_mut().enumerate() {
            *slot = ((i as f32 * 0.31).cos()) * 0.5;
        }
        let mut y_exact = vec![0f32; d_out];
        for r in 0..d_out {
            y_exact[r] = w[r * d_in..(r + 1) * d_in]
                .iter()
                .zip(x.iter())
                .map(|(&a, &b)| a * b)
                .sum();
        }

        let schemes = [
            QuantScheme::B1_2Means,
            QuantScheme::Q2G32,
            QuantScheme::Q3G32,
            QuantScheme::Q4G32,
            QuantScheme::Q5G32,
            QuantScheme::Q6G32,
            QuantScheme::Q8G32,
            QuantScheme::Q10G32,
            QuantScheme::Q12G32,
        ];
        for &scheme in &schemes {
            for &use_h8 in &[false, true] {
                let pm = PackedMatrix::pack_from_f32(d_out, d_in, &w, scheme, use_h8).unwrap();
                let y = pm.matvec(&x, 2).unwrap();
                let mut dot_direct = 0f64;
                let mut norm_e = 0f64;
                let mut norm_q = 0f64;
                for r in 0..d_out {
                    let dec_row = pm.decode_row(r).unwrap();
                    let y_row: f32 = dec_row.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
                    assert!((y_row - y[r]).abs() < 1e-4, "{scheme:?} h8={use_h8} matvec mismatch: {y_row} vs {}", y[r]);
                    dot_direct += y_exact[r] as f64 * y[r] as f64;
                    norm_e += (y_exact[r] as f64).powi(2);
                    norm_q += (y[r] as f64).powi(2);
                }
                let cos = dot_direct / (norm_e.sqrt() * norm_q.sqrt());
                if scheme.bits() >= 4 {
                    assert!(cos > 0.985, "{scheme:?} h8={use_h8} expected cos > 0.985 got {cos}");
                }
            }
        }
    }
}
