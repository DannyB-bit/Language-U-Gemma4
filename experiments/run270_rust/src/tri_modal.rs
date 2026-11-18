//! Tri-Modal "Read + Hear + See" machine-code accuracy driver for T16C + C8DB + Vision.
//!
//! Extracts three orthogonal health signals directly from layer states and logits:
//!   - **HEAR**: Walsh-Hadamard (`FWHT`) 4-band sequency energy spectrum (`hf_ratio`, entropy).
//!   - **SEE**: Fast spatial tile statistics (`crest_factor`, `tile_cv`, `edge_ratio`) +
//!     on-demand `672x672` Optical Rescue Canvas rendering for `VisionWeights::encode_image`.
//!   - **READ**: Discrete top-1/top-2 logit margin, softmax confidence, and checksum integrity.
//!
//! Fuses all three through an asymmetric consensus gate to adapt `--lmhead-cs`, `--cs-skip`,
//! speculative draft length `k`, `C8DB` rescue gating, and turn-level `--optical-self-analyze`
//! without penalizing clean tokens.

use crate::math::Rng;
use crate::t16c::fwht_block_inplace;
use crate::vision::{render_optical_va_canvas, RgbImage};

/// Sample an approximate standard normal `N(0, 1)` scalar from `Rng` (Box-Muller).
#[inline]
pub fn sample_normal_f32(rng: &mut Rng) -> f32 {
    let u1 = rng.next_f64().clamp(1e-12, 1.0 - 1e-12);
    let u2 = rng.next_f64();
    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
}

/// Four-band Walsh-Hadamard sequency spectrum (`HEAR` signal).
#[derive(Clone, Debug, PartialEq)]
pub struct HearSpectrum {
    /// Normalized sequency energy across `[low, mid1, mid2, high]` (sums to 1.0 when non-zero).
    pub bands: [f32; 4],
    /// Ratio of upper-half sequency energy `(mid2 + high)` to lower-half `(low + mid1)`.
    pub hf_ratio: f32,
    /// Ratio of highest quartile sequency energy `high` to lowest quartile `low`.
    pub high_to_low: f32,
    /// Normalized energy-weighted sequency centroid in `[0.0, 1.0]`.
    pub sequency_centroid: f32,
    /// Normalized Shannon entropy across the 4 sequency bands in `[0.0, 1.0]`.
    pub spectral_entropy: f32,
    /// Total L2 energy across all elements.
    pub total_energy: f32,
    /// Spectral health score in `[0.0, 1.0]` (1.0 = clean low-sequency structure).
    pub health: f32,
}

/// Fast spatial tile geometry fingerprint (`SEE` micro-signal).
#[derive(Clone, Debug, PartialEq)]
pub struct SeeSpatial {
    pub n_tiles: usize,
    pub rms: f32,
    pub peak: f32,
    /// Global `peak / rms` crest factor (spiky outliers push this high).
    pub crest_factor: f32,
    /// Coefficient of variation of per-tile RMS across spatial tiles (collapse / sinks raise this).
    pub tile_cv: f32,
    /// Mean adjacent-tile RMS jump normalized by mean tile RMS.
    pub edge_ratio: f32,
    /// Fraction of spatial tiles whose RMS falls below `5%` of the mean tile RMS.
    pub dead_tile_frac: f32,
    /// Spatial health score in `[0.0, 1.0]`.
    pub health: f32,
}

/// Discrete logit & integrity fingerprint (`READ` signal).
#[derive(Clone, Debug, PartialEq)]
pub struct ReadSignal {
    pub top1_id: u32,
    pub top1_val: f32,
    pub top2_val: f32,
    pub margin: f32,
    pub top1_prob: f32,
    pub crc_ok: bool,
    pub health: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConsensusMode {
    Clean,
    Caution,
    Rescue,
}

impl ConsensusMode {
    pub fn as_str(self) -> &'static str {
        match self {
            ConsensusMode::Clean => "CLEAN",
            ConsensusMode::Caution => "CAUTION",
            ConsensusMode::Rescue => "RESCUE",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ConsensusConfig {
    pub base_sieve_frac: f32,
    pub base_cs_skip: f32,
    pub base_draft_k: usize,
    pub caution_thresh: f32,
    pub rescue_thresh: f32,
    pub rescue_streak_for_optical: usize,
    pub ema_alpha: f32,
    /// Use baseline-relative HEAR/SEE drift scoring alongside READ confidence and
    /// structural collapse checks (for live LLM hidden states with natural outlier channels).
    pub transformer_relative: bool,
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            base_sieve_frac: 0.25,
            base_cs_skip: 0.25,
            base_draft_k: 4,
            caution_thresh: 0.72,
            rescue_thresh: 0.48,
            rescue_streak_for_optical: 2,
            ema_alpha: 0.15,
            transformer_relative: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TriModalDecision {
    pub mode: ConsensusMode,
    pub consensus_score: f32,
    pub read_score: f32,
    pub hear_score: f32,
    pub see_score: f32,
    pub hf_drift_ratio: f32,
    pub crest_drift_ratio: f32,
    pub lmhead_cs_sieve: Option<f32>,
    pub cs_skip_thresh: f32,
    pub draft_k: usize,
    pub engage_c8db_rescue: bool,
    pub trigger_optical_self_analyze: bool,
    pub rescue_streak: usize,
    /// Compact 8D `[read, margin_norm, band0..band3, crest_norm, tile_cv_norm]` vector aligned to `H8 -> S^7`.
    pub fingerprint8: [f32; 8],
}

/// Exact zero-crossing sequency index (`0..2^log2_n - 1`) of natural-order Hadamard row `idx`.
///
/// Bit-reversing `idx` yields the Gray-coded sequency; Gray-decoding (`val ^= val >> 1, 2, 4, ...`)
/// yields the exact number of sign changes across the row.
#[inline]
pub fn walsh_sequency(idx: usize, log2_n: u32) -> usize {
    if log2_n == 0 {
        return 0;
    }
    let mask = (1usize << log2_n) - 1;
    let mut val = (idx & mask).reverse_bits() >> (usize::BITS - log2_n);
    let mut shift = 1u32;
    while shift < log2_n {
        val ^= val >> shift;
        shift <<= 1;
    }
    val
}

fn validate_finite_slice(x: &[f32], label: &str) -> Result<(), String> {
    if x.is_empty() {
        return Err(format!("{label}: input slice must be non-empty"));
    }
    if let Some((i, &v)) = x.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(format!("{label}: non-finite value {v} at index {i}"));
    }
    Ok(())
}

/// Compute 4-band Walsh-Hadamard sequency spectrum (`HEAR`) on `x` using block width `bsize`.
pub fn analyze_hear_fwht(x: &[f32], bsize: usize, already_fwht: bool) -> Result<HearSpectrum, String> {
    validate_finite_slice(x, "analyze_hear_fwht")?;
    if !bsize.is_power_of_two() || !(8..=256).contains(&bsize) || x.len() % bsize != 0 {
        return Err(format!(
            "analyze_hear_fwht invalid bsize={bsize} for len={} (require power-of-2 in 8..=256 dividing len)",
            x.len()
        ));
    }

    let mut scratch;
    let spec: &[f32] = if already_fwht {
        x
    } else {
        scratch = x.to_vec();
        fwht_block_inplace(&mut scratch, bsize)?;
        &scratch
    };

    let log2_b = bsize.trailing_zeros();
    let mut seq_lut = [0u16; 256];
    for (i, slot) in seq_lut[..bsize].iter_mut().enumerate() {
        *slot = walsh_sequency(i, log2_b) as u16;
    }

    let mut band_e = [0.0f64; 4];
    let mut total_e = 0.0f64;
    let mut weighted_seq = 0.0f64;
    let norm_denom = (bsize - 1).max(1) as f64;

    for blk in spec.chunks_exact(bsize) {
        for (i, &v) in blk.iter().enumerate() {
            let e = (v as f64) * (v as f64);
            let seq = seq_lut[i] as usize;
            let q = (seq * 4) / bsize;
            band_e[q.min(3)] += e;
            total_e += e;
            weighted_seq += (seq as f64 / norm_denom) * e;
        }
    }

    if total_e <= 1e-24 {
        return Ok(HearSpectrum {
            bands: [0.25; 4],
            hf_ratio: 1.0,
            high_to_low: 1.0,
            sequency_centroid: 0.5,
            spectral_entropy: 1.0,
            total_energy: 0.0,
            health: 0.0,
        });
    }

    let mut bands = [0.0f32; 4];
    let mut entropy = 0.0f64;
    for (k, &eb) in band_e.iter().enumerate() {
        let p = eb / total_e;
        bands[k] = p as f32;
        if p > 1e-12 {
            entropy -= p * p.ln();
        }
    }
    let spectral_entropy = (entropy / 4.0f64.ln()).clamp(0.0, 1.0) as f32;
    let hf_ratio = (bands[2] + bands[3]) / (bands[0] + bands[1] + 1e-6);
    let high_to_low = bands[3] / (bands[0] + 1e-6);
    let sequency_centroid = (weighted_seq / total_e).clamp(0.0, 1.0) as f32;

    // Healthy structured activations concentrate energy in lower sequencies; quantization noise pushes hf_ratio -> >=1.0.
    let hf_pen = ((hf_ratio - 0.65) / 1.15).clamp(0.0, 1.0);
    let ent_pen = ((spectral_entropy - 0.96) / 0.04).clamp(0.0, 1.0);
    let health = (1.0 - 0.78 * hf_pen - 0.22 * ent_pen).clamp(0.0, 1.0);

    Ok(HearSpectrum {
        bands,
        hf_ratio,
        high_to_low,
        sequency_centroid,
        spectral_entropy,
        total_energy: total_e as f32,
        health,
    })
}

/// Compute spatial tile geometry fingerprint (`SEE` fast eye) across `tile_w`-wide tiles.
pub fn analyze_see_spatial(x: &[f32], tile_w: usize) -> Result<SeeSpatial, String> {
    validate_finite_slice(x, "analyze_see_spatial")?;
    if tile_w < 8 || x.len() % tile_w != 0 {
        return Err(format!(
            "analyze_see_spatial invalid tile_w={tile_w} for len={}",
            x.len()
        ));
    }

    let n_tiles = x.len() / tile_w;
    let mut tile_rms = Vec::with_capacity(n_tiles);
    let mut sum_sq_all = 0.0f64;
    let mut peak = 0.0f32;

    for tile in x.chunks_exact(tile_w) {
        let mut s2 = 0.0f64;
        for &v in tile {
            let av = v.abs();
            if av > peak {
                peak = av;
            }
            s2 += (v as f64) * (v as f64);
        }
        sum_sq_all += s2;
        tile_rms.push((s2 / tile_w as f64).sqrt() as f32);
    }

    let rms = (sum_sq_all / x.len() as f64).sqrt() as f32;
    if rms <= 1e-12 {
        return Ok(SeeSpatial {
            n_tiles,
            rms: 0.0,
            peak: 0.0,
            crest_factor: 0.0,
            tile_cv: 0.0,
            edge_ratio: 0.0,
            dead_tile_frac: 1.0,
            health: 0.0,
        });
    }

    let mean_t = (tile_rms.iter().map(|&r| r as f64).sum::<f64>() / n_tiles as f64) as f32;
    let var_t = tile_rms
        .iter()
        .map(|&r| {
            let d = (r - mean_t) as f64;
            d * d
        })
        .sum::<f64>()
        / n_tiles as f64;
    let tile_cv = (var_t.sqrt() as f32) / (mean_t + 1e-8);

    let edge_ratio = if n_tiles > 1 {
        let diff_sum: f64 = tile_rms.windows(2).map(|w| (w[1] - w[0]).abs() as f64).sum();
        ((diff_sum / (n_tiles - 1) as f64) as f32) / (mean_t + 1e-8)
    } else {
        0.0
    };

    let dead_cnt = tile_rms.iter().filter(|&&r| r < 0.05 * mean_t).count();
    let dead_tile_frac = dead_cnt as f32 / n_tiles as f32;
    let crest_factor = peak / (rms + 1e-8);

    // Gaussian/clean high-dim tiles stay around crest ~3..4.2, low CV (<0.35), zero dead tiles.
    let crest_pen = ((crest_factor - 4.4) / 4.0).clamp(0.0, 1.0);
    let cv_pen = ((tile_cv - 0.35) / 0.65).clamp(0.0, 1.0);
    let dead_pen = (dead_tile_frac * 2.0).clamp(0.0, 1.0);
    let health = (1.0 - 0.45 * crest_pen - 0.35 * cv_pen - 0.20 * dead_pen).clamp(0.0, 1.0);

    Ok(SeeSpatial {
        n_tiles,
        rms,
        peak,
        crest_factor,
        tile_cv,
        edge_ratio,
        dead_tile_frac,
        health,
    })
}

/// Extract discrete logit confidence and integrity check (`READ` signal).
pub fn analyze_read_logits(logits: &[f32], crc_ok: bool) -> Result<ReadSignal, String> {
    if logits.len() < 2 {
        return Err(format!("analyze_read_logits requires >= 2 logits, got {}", logits.len()));
    }
    validate_finite_slice(logits, "analyze_read_logits")?;

    let mut top1_id = 0u32;
    let mut top1_val = f32::NEG_INFINITY;
    let mut top2_val = f32::NEG_INFINITY;
    for (idx, &v) in logits.iter().enumerate() {
        if v > top1_val {
            top2_val = top1_val;
            top1_val = v;
            top1_id = idx as u32;
        } else if v > top2_val {
            top2_val = v;
        }
    }
    let margin = top1_val - top2_val;
    let mut z_sum = 0.0f64;
    for &v in logits {
        z_sum += ((v - top1_val) as f64).exp();
    }
    let top1_prob = if z_sum > 0.0 && z_sum.is_finite() {
        (1.0 / z_sum).clamp(0.0, 1.0) as f32
    } else {
        0.0
    };

    let marg_score = (margin / 1.5).clamp(0.0, 1.0);
    let raw_health = (0.65 * marg_score + 0.35 * top1_prob).clamp(0.0, 1.0);
    let health = if crc_ok { raw_health } else { 0.0 };

    Ok(ReadSignal {
        top1_id,
        top1_val,
        top2_val,
        margin,
        top1_prob,
        crc_ok,
        health,
    })
}

/// Stateful asymmetric consensus driver fusing READ + HEAR + SEE into runtime accuracy controls.
#[derive(Clone, Debug)]
pub struct TriModalConsensusGate {
    pub cfg: ConsensusConfig,
    pub baseline_hf_ratio: Option<f32>,
    pub baseline_crest: Option<f32>,
    pub rescue_streak: usize,
    pub caution_streak: usize,
}

impl TriModalConsensusGate {
    pub fn new(cfg: ConsensusConfig) -> Self {
        Self {
            cfg,
            baseline_hf_ratio: None,
            baseline_crest: None,
            rescue_streak: 0,
            caution_streak: 0,
        }
    }

    pub fn evaluate(
        &mut self,
        read: &ReadSignal,
        hear: &HearSpectrum,
        see: &SeeSpatial,
        update_baseline: bool,
    ) -> TriModalDecision {
        let base_hf = self.baseline_hf_ratio.unwrap_or(hear.hf_ratio.max(0.25));
        let base_cr = self.baseline_crest.unwrap_or(see.crest_factor.max(2.0));
        let hf_drift_ratio = hear.hf_ratio / (base_hf + 1e-6);
        let crest_drift_ratio = see.crest_factor / (base_cr + 1e-6);

        let read_score = read.health.clamp(0.0, 1.0);
        let (hear_score, see_score, consensus_score, mode) = if self.cfg.transformer_relative {
            let hf_rel_pen = ((hf_drift_ratio - 1.15) / 0.75).clamp(0.0, 1.0);
            let h_sc = (1.0 - hf_rel_pen).clamp(0.0, 1.0);
            let cr_rel_pen = ((crest_drift_ratio - 1.25) / 0.85).clamp(0.0, 1.0);
            let cv_pen = ((see.tile_cv - 0.55) / 0.65).clamp(0.0, 1.0);
            let dead_pen = (see.dead_tile_frac * 2.0).clamp(0.0, 1.0);
            let s_sc = (1.0 - 0.45 * cr_rel_pen - 0.35 * cv_pen - 0.20 * dead_pen).clamp(0.0, 1.0);
            let c_sc = if !read.crc_ok {
                0.0
            } else {
                (0.45 * read_score + 0.275 * h_sc + 0.275 * s_sc).clamp(0.0, 1.0)
            };
            let m = if !read.crc_ok
                || read_score < 0.45
                || c_sc < self.cfg.rescue_thresh
                || h_sc < 0.30
                || s_sc < 0.30
                || (hf_drift_ratio >= 2.2 && crest_drift_ratio >= 1.35)
            {
                ConsensusMode::Rescue
            } else if read_score < 0.78
                || c_sc < self.cfg.caution_thresh
                || h_sc < 0.55
                || s_sc < 0.55
                || hf_drift_ratio >= 1.65
                || crest_drift_ratio >= 1.50
            {
                ConsensusMode::Caution
            } else {
                ConsensusMode::Clean
            };
            (h_sc, s_sc, c_sc, m)
        } else {
            // Penalize relative drift spikes above observed clean baseline in addition to absolute health.
            let hf_rel_pen = ((hf_drift_ratio - 1.25) / 0.90).clamp(0.0, 1.0);
            let cr_rel_pen = ((crest_drift_ratio - 1.25) / 0.90).clamp(0.0, 1.0);
            let h_sc = (hear.health * (1.0 - 0.55 * hf_rel_pen)).clamp(0.0, 1.0);
            let s_sc = (see.health * (1.0 - 0.55 * cr_rel_pen)).clamp(0.0, 1.0);
            let c_sc = if !read.crc_ok {
                0.0
            } else {
                (0.40 * read_score + 0.30 * h_sc + 0.30 * s_sc).clamp(0.0, 1.0)
            };
            let m = if !read.crc_ok
                || c_sc < self.cfg.rescue_thresh
                || h_sc < 0.30
                || s_sc < 0.30
                || (h_sc < 0.65 && s_sc < 0.72)
                || (hf_drift_ratio >= 3.0 && crest_drift_ratio >= 1.35)
            {
                ConsensusMode::Rescue
            } else if c_sc < self.cfg.caution_thresh
                || h_sc < 0.55
                || s_sc < 0.55
                || hf_drift_ratio >= 2.0
                || crest_drift_ratio >= 1.50
            {
                ConsensusMode::Caution
            } else {
                ConsensusMode::Clean
            };
            (h_sc, s_sc, c_sc, m)
        };

        if update_baseline
            && self.cfg.transformer_relative
            && self.baseline_hf_ratio.is_none()
            && see.dead_tile_frac < 0.15
        {
            self.baseline_hf_ratio = Some(hear.hf_ratio.max(0.25));
            self.baseline_crest = Some(see.crest_factor.max(2.0));
        }

        match mode {
            ConsensusMode::Clean => {
                self.rescue_streak = 0;
                self.caution_streak = 0;
                if update_baseline {
                    let a = self.cfg.ema_alpha;
                    self.baseline_hf_ratio = Some(match self.baseline_hf_ratio {
                        Some(b) => (1.0 - a) * b + a * hear.hf_ratio,
                        None => hear.hf_ratio.max(0.25),
                    });
                    self.baseline_crest = Some(match self.baseline_crest {
                        Some(b) => (1.0 - a) * b + a * see.crest_factor,
                        None => see.crest_factor.max(2.0),
                    });
                }
            }
            ConsensusMode::Caution => {
                self.rescue_streak = 0;
                self.caution_streak += 1;
            }
            ConsensusMode::Rescue => {
                self.rescue_streak += 1;
                self.caution_streak += 1;
            }
        }

        let (lmhead_cs_sieve, cs_skip_thresh, draft_k, engage_c8db_rescue, trigger_optical_self_analyze) =
            match mode {
                ConsensusMode::Clean => (
                    Some(self.cfg.base_sieve_frac),
                    self.cfg.base_cs_skip,
                    self.cfg.base_draft_k.max(1),
                    false,
                    false,
                ),
                ConsensusMode::Caution => (
                    Some(0.0),
                    self.cfg.base_cs_skip * 0.5,
                    (self.cfg.base_draft_k / 2).max(1),
                    true,
                    false,
                ),
                ConsensusMode::Rescue => (
                    Some(0.0),
                    0.0,
                    1,
                    true,
                    !read.crc_ok || self.rescue_streak >= self.cfg.rescue_streak_for_optical,
                ),
            };

        let fingerprint8 = [
            read_score,
            (read.margin / 3.0).clamp(0.0, 1.0),
            hear.bands[0],
            hear.bands[1],
            hear.bands[2],
            hear.bands[3],
            (see.crest_factor / 8.0).clamp(0.0, 1.0),
            see.tile_cv.clamp(0.0, 1.0),
        ];

        TriModalDecision {
            mode,
            consensus_score,
            read_score,
            hear_score,
            see_score,
            hf_drift_ratio,
            crest_drift_ratio,
            lmhead_cs_sieve,
            cs_skip_thresh,
            draft_k,
            engage_c8db_rescue,
            trigger_optical_self_analyze,
            rescue_streak: self.rescue_streak,
            fingerprint8,
        }
    }
}

/// Render a `672x672` Optical Rescue Canvas encoding live READ/HEAR/SEE drift telemetry and CRC32.
pub fn render_tri_modal_rescue_canvas(
    step: usize,
    decision: &TriModalDecision,
    read: &ReadSignal,
    hear: &HearSpectrum,
    see: &SeeSpatial,
    layer_health_scores: &[f32],
) -> (RgbImage, u32) {
    let l1 = format!(
        "TRI-MODAL {} STEP={} SCORE={:.2}",
        decision.mode.as_str(),
        step,
        decision.consensus_score
    );
    let l2 = format!(
        "READ={:.2} MARG={:.2} HEAR={:.2} HF={:.2}",
        decision.read_score, read.margin, decision.hear_score, hear.hf_ratio
    );
    let l3 = format!(
        "SEE={:.2} CREST={:.2} CV={:.2} K={}",
        decision.see_score, see.crest_factor, see.tile_cv, decision.draft_k
    );
    // Map health in [0,1] into 1..5 bars (1=red critical drift, 5=green healthy).
    let layer_bars: Vec<u8> = layer_health_scores
        .iter()
        .map(|&h| {
            let hc = h.clamp(0.0, 1.0);
            if hc < 0.35 {
                1
            } else if hc < 0.55 {
                2
            } else if hc < 0.80 {
                3
            } else {
                5
            }
        })
        .collect();
    render_optical_va_canvas(&[&l1, &l2, &l3], &layer_bars)
}

/// Compute Pearson correlation coefficient `r` for Step-1 telemetry correlation validation.
pub fn pearson_r(xs: &[f64], ys: &[f64]) -> Result<f64, String> {
    if xs.len() != ys.len() || xs.len() < 2 {
        return Err(format!("pearson_r requires equal slices of len >= 2, got {} and {}", xs.len(), ys.len()));
    }
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut num = 0.0f64;
    let mut dx2 = 0.0f64;
    let mut dy2 = 0.0f64;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        if !x.is_finite() || !y.is_finite() {
            return Err("pearson_r encountered non-finite input".into());
        }
        let dx = x - mx;
        let dy = y - my;
        num += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
    }
    let den = (dx2 * dy2).sqrt();
    Ok(if den > 1e-18 { num / den } else { 0.0 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::{crc32_ieee, decode_pnm, encode_ppm_p6};

    #[test]
    fn walsh_sequency_matches_direct_sign_change_count() {
        for &n in &[8usize, 16, 64, 256] {
            let log2_n = n.trailing_zeros();
            let mut seen = vec![false; n];
            for row in 0..n {
                let seq = walsh_sequency(row, log2_n);
                assert!(seq < n && !seen[seq], "duplicate or OOB sequency {seq} for n={n}");
                seen[seq] = true;

                let mut prev = 1i32;
                let mut changes = 0usize;
                for col in 1..n {
                    let sgn = if ((row & col).count_ones() & 1) == 0 { 1 } else { -1 };
                    if sgn != prev {
                        changes += 1;
                    }
                    prev = sgn;
                }
                assert_eq!(seq, changes, "sequency mismatch for n={n} row={row}");
            }
        }
    }

    #[test]
    fn hear_separates_low_sequency_signal_from_high_sequency_noise() {
        let mut clean = vec![0.0f32; 512];
        for (i, v) in clean.iter_mut().enumerate() {
            // Smooth low-frequency sine across blocks (concentrated in low Walsh sequencies).
            *v = ((i as f32) * 0.025).sin() + 0.5 * ((i as f32) * 0.05).cos();
        }
        let sp_clean = analyze_hear_fwht(&clean, 256, false).unwrap();

        let mut noisy = clean.clone();
        let mut rng = Rng::new(1337);
        for (i, v) in noisy.iter_mut().enumerate() {
            // Alternating sign high-sequency quantization chatter.
            let alt = if i % 2 == 0 { 1.4 } else { -1.4 };
            *v += alt + 0.6 * sample_normal_f32(&mut rng);
        }
        let sp_noisy = analyze_hear_fwht(&noisy, 256, false).unwrap();

        assert!((sp_clean.bands.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        assert!(sp_clean.hf_ratio < 0.25, "clean hf_ratio={}", sp_clean.hf_ratio);
        assert!(sp_noisy.hf_ratio > 1.5, "noisy hf_ratio={}", sp_noisy.hf_ratio);
        assert!(sp_clean.health > sp_noisy.health + 0.35);
    }

    #[test]
    fn see_detects_outlier_crest_and_tile_rank_collapse() {
        let mut rng = Rng::new(2026);
        let healthy: Vec<f32> = (0..1024).map(|_| sample_normal_f32(&mut rng) * 0.8).collect();
        let see_ok = analyze_see_spatial(&healthy, 256).unwrap();

        let mut collapsed = healthy.clone();
        // Zero out half the tiles (dead channels) and inject a huge spike in tile 0.
        for v in collapsed[512..].iter_mut() {
            *v = 0.0;
        }
        collapsed[17] = 28.0;
        let see_bad = analyze_see_spatial(&collapsed, 256).unwrap();

        assert!(see_ok.crest_factor < 4.5 && see_ok.dead_tile_frac == 0.0);
        assert!(see_bad.crest_factor > 10.0 && see_bad.dead_tile_frac >= 0.45);
        assert!(see_ok.health > 0.85 && see_bad.health < 0.40);
    }

    #[test]
    fn consensus_gate_adapts_knobs_and_triggers_rescue_on_streak() {
        let mut gate = TriModalConsensusGate::new(ConsensusConfig::default());
        let mut rng = Rng::new(9);
        let clean_x: Vec<f32> = (0..512).map(|i| ((i as f32) * 0.03).sin() + 0.05 * sample_normal_f32(&mut rng)).collect();
        let hear_ok = analyze_hear_fwht(&clean_x, 256, false).unwrap();
        let see_ok = analyze_see_spatial(&clean_x, 256).unwrap();
        let read_ok = analyze_read_logits(&[3.5, 0.2, -0.5, -1.0], true).unwrap();

        let d0 = gate.evaluate(&read_ok, &hear_ok, &see_ok, true);
        assert_eq!(d0.mode, ConsensusMode::Clean);
        assert_eq!(d0.lmhead_cs_sieve, Some(0.25));
        assert!(!d0.engage_c8db_rescue);
        assert!(!d0.trigger_optical_self_analyze);

        // Severe drift: weak margin + alternating high-sequency noise + spatial outlier spike.
        let mut bad_x = clean_x;
        for (i, v) in bad_x.iter_mut().enumerate() {
            *v += if i % 2 == 0 { 2.5 } else { -2.5 };
        }
        bad_x[3] = 35.0;
        let hear_bad = analyze_hear_fwht(&bad_x, 256, false).unwrap();
        let see_bad = analyze_see_spatial(&bad_x, 256).unwrap();
        let read_bad = analyze_read_logits(&[1.05, 1.00, 0.98, 0.95], true).unwrap();

        let d1 = gate.evaluate(&read_bad, &hear_bad, &see_bad, true);
        assert_eq!(d1.mode, ConsensusMode::Rescue);
        assert_eq!(d1.lmhead_cs_sieve, Some(0.0));
        assert_eq!(d1.cs_skip_thresh, 0.0);
        assert_eq!(d1.draft_k, 1);
        assert!(d1.engage_c8db_rescue);
        assert!(!d1.trigger_optical_self_analyze);

        // Second consecutive rescue step triggers turn-level Optical Self-Analyze.
        let d2 = gate.evaluate(&read_bad, &hear_bad, &see_bad, true);
        assert_eq!(d2.rescue_streak, 2);
        assert!(d2.trigger_optical_self_analyze);
    }

    #[test]
    fn rescue_canvas_renders_valid_crc_and_nonuniform_telemetry() {
        let mut gate = TriModalConsensusGate::new(ConsensusConfig::default());
        let x: Vec<f32> = (0..512).map(|i| if i % 2 == 0 { 2.0 } else { -2.0 }).collect();
        let hear = analyze_hear_fwht(&x, 256, false).unwrap();
        let see = analyze_see_spatial(&x, 256).unwrap();
        let read = analyze_read_logits(&[0.5, 0.45, 0.1], true).unwrap();
        let dec = gate.evaluate(&read, &hear, &see, false);
        let (img, crc) = render_tri_modal_rescue_canvas(7, &dec, &read, &hear, &see, &[0.95, 0.25, 0.90]);
        assert_eq!((img.width, img.height), (672, 672));
        assert_ne!(crc, 0);
        assert_ne!(crc32_ieee(&img.data), 0);
        let dec_ppm = decode_pnm(&encode_ppm_p6(&img)).unwrap();
        assert_eq!(dec_ppm.data, img.data);
    }

    #[test]
    fn tri_modal_rejects_invalid_and_nonfinite_inputs_fail_closed() {
        assert!(analyze_hear_fwht(&[], 256, false).is_err());
        assert!(analyze_hear_fwht(&[1.0; 256], 200, false).is_err());
        assert!(analyze_hear_fwht(&[f32::NAN; 256], 256, false).is_err());
        assert!(analyze_see_spatial(&[1.0; 250], 64).is_err());
        assert!(analyze_read_logits(&[1.0], true).is_err());
        assert!(analyze_read_logits(&[1.0, f32::INFINITY], true).is_err());
        assert!(pearson_r(&[1.0], &[2.0]).is_err());
    }

    #[test]
    fn transformer_relative_mode_handles_llm_outliers_and_low_margin_rescue() {
        let cfg = ConsensusConfig {
            transformer_relative: true,
            ..ConsensusConfig::default()
        };
        let mut gate = TriModalConsensusGate::new(cfg);
        let mut rng = Rng::new(77);
        let mut x_llm: Vec<f32> = (0..5376).map(|_| sample_normal_f32(&mut rng)).collect();
        x_llm[11] = 12.5;
        x_llm[777] = -11.0;

        let hear = analyze_hear_fwht(&x_llm, 64, false).unwrap();
        let see = analyze_see_spatial(&x_llm, 64).unwrap();
        assert!(see.crest_factor > 8.0);

        let read_uncertain = analyze_read_logits(&[2.3, 2.0, 1.0, 0.0], true).unwrap();
        let d0 = gate.evaluate(&read_uncertain, &hear, &see, true);
        assert_eq!(d0.mode, ConsensusMode::Rescue);
        assert_eq!(d0.lmhead_cs_sieve, Some(0.0));

        let read_confident = analyze_read_logits(&[8.5, 2.0, 0.5, -1.0], true).unwrap();
        let d1 = gate.evaluate(&read_confident, &hear, &see, true);
        assert_eq!(d1.mode, ConsensusMode::Clean);
        assert_eq!(d1.lmhead_cs_sieve, Some(0.25));
        assert_eq!(d1.draft_k, 4);

        let mut x_spiky = x_llm;
        x_spiky[11] = 45.0;
        let see_spike = analyze_see_spatial(&x_spiky, 64).unwrap();
        let d2 = gate.evaluate(&read_confident, &hear, &see_spike, true);
        assert_ne!(d2.mode, ConsensusMode::Clean);
    }
}
