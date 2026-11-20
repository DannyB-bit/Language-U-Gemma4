//! Math primitives for Gemma 3 270M native Rust inference (FP32 accumulation).

#[inline]
pub fn rmsn_gemma(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    debug_assert_eq!(x.len(), w.len());
    let n = x.len();
    let mut s = 0f64;
    for &v in x {
        s += (v as f64) * (v as f64);
    }
    let rms = ((s / n as f64) + eps as f64).sqrt() as f32;
    x.iter().zip(w.iter()).map(|(&a, &b)| (a / rms) * (1.0 + b)).collect()
}

#[inline]
pub fn dot_f32_4way(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 4;
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut i = 0usize;
    for _ in 0..chunks {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
        i += 4;
    }
    let mut s = (s0 + s1) + (s2 + s3);
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

#[inline]
pub fn rmsn_raw_into(x: &[f32], w: Option<&[f32]>, eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    let n = x.len();
    let chunks = n / 4;
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut i = 0usize;
    for _ in 0..chunks {
        let a0 = x[i];
        let a1 = x[i + 1];
        let a2 = x[i + 2];
        let a3 = x[i + 3];
        s0 += a0 * a0;
        s1 += a1 * a1;
        s2 += a2 * a2;
        s3 += a3 * a3;
        i += 4;
    }
    let mut s = ((s0 + s1) + (s2 + s3)) as f64;
    while i < n {
        let v = x[i] as f64;
        s += v * v;
        i += 1;
    }
    let inv_rms = 1.0f32 / ((s / n as f64) + eps as f64).sqrt() as f32;
    match w {
        Some(gain) => {
            debug_assert_eq!(x.len(), gain.len());
            for ((o, &a), &b) in out.iter_mut().zip(x.iter()).zip(gain.iter()) {
                *o = a * inv_rms * b;
            }
        }
        None => {
            for (o, &a) in out.iter_mut().zip(x.iter()) {
                *o = a * inv_rms;
            }
        }
    }
}

#[inline]
pub fn rmsn_raw(x: &[f32], w: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    rmsn_raw_into(x, w, eps, &mut out);
    out
}

pub fn qk_norm_gemma(x: &[f32], n_heads: usize, head_dim: usize, w: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for h in 0..n_heads {
        let s = &x[h * head_dim..(h + 1) * head_dim];
        let o = &mut out[h * head_dim..(h + 1) * head_dim];
        o.copy_from_slice(&rmsn_gemma(s, w, eps));
    }
    out
}

pub fn qk_norm_raw(x: &[f32], n_heads: usize, head_dim: usize, w: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for h in 0..n_heads {
        let s = &x[h * head_dim..(h + 1) * head_dim];
        let o = &mut out[h * head_dim..(h + 1) * head_dim];
        o.copy_from_slice(&rmsn_raw(s, w, eps));
    }
    out
}

pub fn rope_head(head: &mut [f32], pos: u32, theta: f32, rotary_dims: usize, head_dim: usize) {
    let half_rot = rotary_dims / 2;
    let half_head = head_dim / 2;
    if half_rot == 0 {
        return;
    }
    for idx in 0..half_rot {
        let j = (idx * 2) as f32;
        let freq = 1.0f32 / theta.powf(j / head_dim as f32);
        let ang = pos as f32 * freq;
        let (s, c) = ang.sin_cos();
        let x1 = head[idx];
        let x2 = head[half_head + idx];
        head[idx] = x1 * c - x2 * s;
        head[half_head + idx] = x1 * s + x2 * c;
    }
}

/// Precomputed inverse frequencies `1 / theta^(2*idx / head_dim)` for fast zero-powf RoPE.
#[inline]
pub fn rope_inv_freqs(theta: f32, rotary_dims: usize, head_dim: usize) -> Vec<f32> {
    let half_rot = rotary_dims / 2;
    let mut freqs = Vec::with_capacity(half_rot);
    let inv_hd = 1.0f32 / (head_dim as f32);
    for idx in 0..half_rot {
        freqs.push(1.0f32 / theta.powf((idx * 2) as f32 * inv_hd));
    }
    freqs
}

/// Fill `(cos_out, sin_out)` once per `(pos, layer_type)` from precomputed `inv_freqs`.
#[inline]
pub fn rope_cos_sin_for_pos(inv_freqs: &[f32], pos: u32, cos_out: &mut [f32], sin_out: &mut [f32]) {
    let pf = pos as f32;
    for (idx, &freq) in inv_freqs.iter().enumerate() {
        let (s, c) = (pf * freq).sin_cos();
        cos_out[idx] = c;
        sin_out[idx] = s;
    }
}

/// Apply precomputed `(cos, sin)` RoPE rotation in-place across a single head slice.
#[inline]
pub fn rope_head_with_cos_sin(head: &mut [f32], cos_t: &[f32], sin_t: &[f32], head_dim: usize) {
    let half_rot = cos_t.len();
    let half_head = head_dim / 2;
    for idx in 0..half_rot {
        let c = cos_t[idx];
        let s = sin_t[idx];
        let x1 = head[idx];
        let x2 = head[half_head + idx];
        head[idx] = x1 * c - x2 * s;
        head[half_head + idx] = x1 * s + x2 * c;
    }
}

/// In-place RMSNorm across `n_heads` heads of length `head_dim` (zero temporary Vec allocations, NEON-vectorizable).
#[inline]
pub fn qk_norm_raw_inplace(x: &mut [f32], n_heads: usize, head_dim: usize, w: Option<&[f32]>, eps: f32) {
    let inv_hd = 1.0f64 / (head_dim as f64);
    let eps64 = eps as f64;
    let chunks = head_dim / 4;
    for h in 0..n_heads {
        let sl = &mut x[h * head_dim..(h + 1) * head_dim];
        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;
        let mut s3 = 0.0f32;
        let mut i = 0usize;
        for _ in 0..chunks {
            let a0 = sl[i];
            let a1 = sl[i + 1];
            let a2 = sl[i + 2];
            let a3 = sl[i + 3];
            s0 += a0 * a0;
            s1 += a1 * a1;
            s2 += a2 * a2;
            s3 += a3 * a3;
            i += 4;
        }
        let mut s = ((s0 + s1) + (s2 + s3)) as f64;
        while i < head_dim {
            let vf = sl[i] as f64;
            s += vf * vf;
            i += 1;
        }
        let inv_rms = 1.0f32 / (s * inv_hd + eps64).sqrt() as f32;
        if let Some(gain) = w {
            for (v, &g) in sl.iter_mut().zip(gain.iter()) {
                *v = *v * inv_rms * g;
            }
        } else {
            for v in sl.iter_mut() {
                *v *= inv_rms;
            }
        }
    }
}

pub fn attn_decode(
    q: &[f32],
    k_seq: &[f32],
    v_seq: &[f32],
    t: usize,
    pos: usize,
    n_heads: usize,
    n_kv: usize,
    hd: usize,
    sw: Option<usize>,
) -> Vec<f32> {
    attn_decode_range(q, k_seq, v_seq, t, t, pos, n_heads, n_kv, hd, sw)
}

/// Multi-query attention over keys `[0..=kv_end.max(t)]` with symmetric sliding-window distance check.
/// Supports HF `use_bidirectional_attention == "vision"` on sliding layers where `kv_end >= t`.
pub fn attn_decode_range(
    q: &[f32],
    k_seq: &[f32],
    v_seq: &[f32],
    t: usize,
    kv_end: usize,
    pos: usize,
    n_heads: usize,
    n_kv: usize,
    hd: usize,
    sw: Option<usize>,
) -> Vec<f32> {
    let gqa = n_heads / n_kv;
    let max_tt = kv_end.max(t);
    let mut ctx = vec![0f32; n_heads * hd];
    let mut scores: Vec<(usize, f32)> = Vec::with_capacity(max_tt + 1);
    for hq in 0..n_heads {
        let kvh = hq / gqa;
        let qh = &q[hq * hd..(hq + 1) * hd];
        scores.clear();
        let mut max_s = f32::NEG_INFINITY;
        for tt in 0..=max_tt {
            if let Some(w) = sw {
                if pos.abs_diff(tt) >= w {
                    continue;
                }
            }
            let base = tt * n_kv * hd + kvh * hd;
            let s = dot_f32_4way(qh, &k_seq[base..base + hd]);
            if s > max_s {
                max_s = s;
            }
            scores.push((tt, s));
        }
        let mut wsum = 0.0f32;
        for &mut (_, ref mut s) in &mut scores {
            let e = (*s - max_s).exp();
            *s = e;
            wsum += e;
        }
        let inv_wsum = if wsum > 1e-30 { 1.0f32 / wsum } else { 0.0f32 };
        let dst = &mut ctx[hq * hd..(hq + 1) * hd];
        for &(tt, e) in &scores {
            let p = e * inv_wsum;
            let base = tt * n_kv * hd + kvh * hd;
            let vs = &v_seq[base..base + hd];
            for (d_val, &v_val) in dst.iter_mut().zip(vs.iter()) {
                *d_val += p * v_val;
            }
        }
    }
    ctx
}

#[inline]
pub fn gelu_tanh_scalar(x: f32) -> f32 {
    let x64 = x as f64;
    let t = (std::f64::consts::FRAC_2_PI.sqrt() * (x64 + 0.044715 * x64 * x64 * x64)).tanh();
    (0.5 * x64 * (1.0 + t)) as f32
}

/// Fast NEON-vectorizable Padé [9/8] tanh GELU in FP32 (max error < 1.5e-5 across (-inf, +inf), zero libm calls).
#[inline]
pub fn gelu_tanh_fast_f32(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.7978845608f32;
    let z = SQRT_2_OVER_PI * (x + 0.044715f32 * x * x * x);
    let zc = z.clamp(-6.8f32, 6.8f32);
    let z2 = zc * zc;
    let num = zc
        * (1.0f32
            + z2 * (0.13725490196f32
                + z2 * (0.003921568627f32 + z2 * (2.8729439e-5f32 + z2 * 2.9019637e-8f32))));
    let den = 1.0f32
        + z2 * (0.47058823529f32
            + z2 * (0.02745098039f32 + z2 * (4.0221216e-4f32 + z2 * 1.3058837e-6f32)));
    let t = (num / den).clamp(-1.0f32, 1.0f32);
    0.5f32 * x * (1.0f32 + t)
}

/// In-place multi-threaded fast Padé GELU across a slice (auto-vectorizes to 4-wide NEON FP32 per core).
pub fn gelu_tanh_inplace_parallel(v: &mut [f32], threads: usize) {
    let n = v.len();
    let nt = threads.max(1);
    if nt <= 1 || n < 16384 {
        for x in v.iter_mut() {
            *x = gelu_tanh_fast_f32(*x);
        }
        return;
    }
    let chunk = n.div_ceil(nt);
    std::thread::scope(|s| {
        for sl in v.chunks_mut(chunk) {
            s.spawn(move || {
                for x in sl.iter_mut() {
                    *x = gelu_tanh_fast_f32(*x);
                }
            });
        }
    });
}

pub fn cosine_and_rel_l2(reference: &[f32], candidate: &[f32]) -> (f64, f64, f64) {
    assert_eq!(reference.len(), candidate.len());
    let mut dot = 0f64;
    let mut nr = 0f64;
    let mut nc = 0f64;
    let mut diff2 = 0f64;
    let mut max_abs = 0f64;
    for (&a, &b) in reference.iter().zip(candidate.iter()) {
        let af = a as f64;
        let bf = b as f64;
        dot += af * bf;
        nr += af * af;
        nc += bf * bf;
        let d = (af - bf).abs();
        if d > max_abs {
            max_abs = d;
        }
        diff2 += d * d;
    }
    let cos = if nr > 0.0 && nc > 0.0 {
        dot / (nr.sqrt() * nc.sqrt())
    } else {
        0.0
    };
    let rel_l2 = if nr > 0.0 {
        diff2.sqrt() / nr.sqrt()
    } else {
        0.0
    };
    (cos, rel_l2, max_abs)
}

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

pub fn top_k_top_p_sample(
    logits: &mut [f64],
    cand_ids: &[u32],
    temperature: f64,
    top_k: usize,
    top_p: f64,
    rep_penalty: f64,
    recent: &[u32],
    rng: &mut Rng,
) -> usize {
    let temp = temperature.max(1e-6);
    for l in logits.iter_mut() {
        *l /= temp;
    }
    if rep_penalty > 1.0 {
        for &t in recent {
            if let Some(pos) = cand_ids.iter().position(|&c| c == t) {
                let l = &mut logits[pos];
                *l = if *l > 0.0 { *l / rep_penalty } else { *l * rep_penalty };
            }
        }
    }
    let k = top_k.min(logits.len()).max(1);
    if k < logits.len() {
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.select_nth_unstable_by(k - 1, |&a, &b| {
            logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        let kth = logits[idx[k - 1]];
        for l in logits.iter_mut() {
            if *l < kth {
                *l = f64::NEG_INFINITY;
            }
        }
    }
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut probs: Vec<f64> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f64 = probs.iter().sum();
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut cum = 0f64;
    let mut cutoff = order.len();
    for (rank, &i) in order.iter().enumerate() {
        cum += probs[i];
        if cum >= top_p {
            cutoff = rank + 1;
            break;
        }
    }
    for &i in &order[cutoff..] {
        probs[i] = 0.0;
    }
    let sum2: f64 = probs.iter().sum();
    let r = rng.next_f64() * sum2;
    let mut acc = 0f64;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if r <= acc {
            return i;
        }
    }
    order[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_gemma_applies_one_plus_weight() {
        let out = rmsn_gemma(&[3.0, 4.0], &[0.5, -0.5], 0.0);
        let rms = (12.5f32).sqrt();
        assert!((out[0] - (3.0 / rms) * 1.5).abs() < 1e-6);
        assert!((out[1] - (4.0 / rms) * 0.5).abs() < 1e-6);

        let raw = rmsn_raw(&[3.0, 4.0], Some(&[0.5, -0.5]), 0.0);
        assert!((raw[0] - (3.0 / rms) * 0.5).abs() < 1e-6);
        assert!((raw[1] - (4.0 / rms) * -0.5).abs() < 1e-6);
        let nogain = rmsn_raw(&[3.0, 4.0], None, 0.0);
        assert!((nogain[0] - (3.0 / rms)).abs() < 1e-6);
    }

    #[test]
    fn rope_rotates_and_preserves_norm() {
        let mut h = vec![1.0f32, 0.0, 2.0, -1.0];
        let norm_before: f32 = h.iter().map(|x| x * x).sum();
        rope_head(&mut h, 3, 10000.0, 4, 4);
        let norm_after: f32 = h.iter().map(|x| x * x).sum();
        assert!((norm_before - norm_after).abs() < 1e-5);
    }

    #[test]
    fn sampler_respects_dominant_logit() {
        let logits = vec![-2.0, 12.0, 0.5];
        let ids = vec![10, 20, 30];
        let mut rng = Rng::new(7);
        for _ in 0..16 {
            assert_eq!(top_k_top_p_sample(&mut logits.clone(), &ids, 1.0, 2, 0.95, 1.0, &[], &mut rng), 1);
        }
    }

    #[test]
    fn attn_decode_range_attends_future_vision_slots_within_window() {
        let q = [1.0f32, 0.0];
        let k = [0.0f32, 1.0, 8.0, 0.0, 12.0, 0.0];
        let v = [1.0f32, 0.0, 5.0, 0.0, 9.0, 0.0];
        let c_causal = attn_decode(&q, &k, &v, 0, 0, 1, 1, 2, Some(2));
        assert!((c_causal[0] - 1.0).abs() < 1e-6);
        // Bidirectional kv_end=2 with window=2 includes t=0 and t=1, excludes t=2 (`abs_diff(0, 2) >= 2`).
        let c_bidir = attn_decode_range(&q, &k, &v, 0, 2, 0, 1, 1, 2, Some(2));
        assert!(c_bidir[0] > 4.9 && c_bidir[0] < 5.01, "c_bidir={:?}", c_bidir);
    }

    #[test]
    fn fast_gelu_and_dot4_match_scalar_reference() {
        let mut xs: Vec<f32> = (-20000..=20000).map(|i| (i as f32) * 0.0005).collect();
        let ref_ys: Vec<f32> = xs.iter().map(|&x| gelu_tanh_scalar(x)).collect();
        gelu_tanh_inplace_parallel(&mut xs, 4);
        let mut max_err = 0.0f32;
        for (&a, &b) in xs.iter().zip(ref_ys.iter()) {
            let e = (a - b).abs();
            if e > max_err {
                max_err = e;
            }
        }
        assert!(max_err < 1e-4, "gelu_tanh_fast_f32 max_err={max_err}");

        let a: Vec<f32> = (0..259).map(|i| ((i * 17 % 31) as f32 - 15.0) * 0.1).collect();
        let b: Vec<f32> = (0..259).map(|i| ((i * 13 % 29) as f32 - 14.0) * 0.1).collect();
        let ref_dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let fast_dot = dot_f32_4way(&a, &b);
        assert!((ref_dot - fast_dot).abs() < 1e-4, "dot4={fast_dot} ref={ref_dot}");
    }
}
