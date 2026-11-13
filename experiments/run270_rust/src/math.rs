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

pub fn qk_norm_gemma(x: &[f32], n_heads: usize, head_dim: usize, w: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for h in 0..n_heads {
        let s = &x[h * head_dim..(h + 1) * head_dim];
        let o = &mut out[h * head_dim..(h + 1) * head_dim];
        o.copy_from_slice(&rmsn_gemma(s, w, eps));
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
    let gqa = n_heads / n_kv;
    let mut ctx = vec![0f32; n_heads * hd];
    for hq in 0..n_heads {
        let kvh = hq / gqa;
        let qh = &q[hq * hd..(hq + 1) * hd];
        let mut scores: Vec<(usize, f64)> = Vec::with_capacity(t + 1);
        for tt in 0..=t {
            if let Some(w) = sw {
                if pos - tt >= w {
                    continue;
                }
            }
            let base = tt * n_kv * hd + kvh * hd;
            let mut s = 0f64;
            for i in 0..hd {
                s += qh[i] as f64 * k_seq[base + i] as f64;
            }
            scores.push((tt, s));
        }
        let max = scores.iter().map(|&(_, s)| s).fold(f64::NEG_INFINITY, f64::max);
        let mut wsum = 0f64;
        let mut w: Vec<(usize, f64)> = Vec::with_capacity(scores.len());
        for &(tt, s) in &scores {
            let e = (s - max).exp();
            w.push((tt, e));
            wsum += e;
        }
        for &(tt, e) in &w {
            let p = (e / wsum) as f32;
            let base = tt * n_kv * hd + kvh * hd;
            for i in 0..hd {
                ctx[hq * hd + i] += p * v_seq[base + i];
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
}
