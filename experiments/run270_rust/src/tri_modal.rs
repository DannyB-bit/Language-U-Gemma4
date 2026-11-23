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

/// Dimension of the V8.0 6-Sense Player Point-of-Perspective (POV) sensory cortex vector.
pub const SIX_SENSE_DIM: usize = 32;

/// 32-dimensional V8.0 6-Sense Player Point-of-Perspective (POV) sensory feature vector:
///   - `0..4`:   `SEE` (`[rms_log, crest_norm, tile_cv, edge_ratio]`)
///   - `4..10`:  `HEAR` (`[band_low, band_mid1, band_mid2, band_high, hf_ratio_norm, spectral_entropy]`)
///   - `10..14`: `FEEL` (`[token_velocity, kinematic_jerk, anisotropy, surface_roughness]`)
///   - `14..18`: `SMELL` (`[kurtosis_trace, signed_skew, top1pct_concentration, rare_trace_ratio]`)
///   - `18..24`: `TASTE` (`[sweet_pos, bitter_neg, sour_sharp, salty_dense, umami_coherent, valence_balance]`)
///   - `24..32`: `CUNEIFORM_S7` (`g_hat = normalize(u_bar @ H8)` in `S^7`)
#[derive(Clone, Debug, PartialEq)]
pub struct SixSensePovFeatures {
    pub features: [f32; SIX_SENSE_DIM],
}

impl SixSensePovFeatures {
    pub fn see_4d(&self) -> &[f32] {
        &self.features[0..4]
    }
    pub fn hear_6d(&self) -> &[f32] {
        &self.features[4..10]
    }
    pub fn feel_4d(&self) -> &[f32] {
        &self.features[10..14]
    }
    pub fn smell_4d(&self) -> &[f32] {
        &self.features[14..18]
    }
    pub fn taste_6d(&self) -> &[f32] {
        &self.features[18..24]
    }
    pub fn cuneiform_8d(&self) -> &[f32] {
        &self.features[24..32]
    }
}

/// Per-layer / per-sequence V8.0 6-Sense Player POV consensus gate decision.
#[derive(Clone, Debug, PartialEq)]
pub struct SixSenseConsensusDecision {
    pub mode: ConsensusMode,
    pub consensus_score: f32,
    pub see_health: f32,
    pub hear_health: f32,
    pub feel_health: f32,
    pub smell_health: f32,
    pub taste_health: f32,
    pub cuneiform_s7_health: f32,
    pub dyn_top_k: usize,
    pub dyn_draft_k: usize,
    pub adaptive_crtr_top_p: f32,
    pub adaptive_mod_q: f32,
}

/// Extract the 32D 6-Sense Player POV sensory vector from `[seq_len, d_model]` layer hidden states,
/// matching `run_v8_0_player_pov_spatial_a100.py::extract_6sense_pov_features`.
pub fn extract_6sense_pov_features(
    x_seq: &[f32],
    seq_len: usize,
    d_model: usize,
    tile_w: usize,
) -> Result<SixSensePovFeatures, String> {
    if seq_len == 0 || d_model < 8 || d_model % 8 != 0 {
        return Err(format!(
            "extract_6sense_pov_features invalid dims seq_len={seq_len} d_model={d_model}"
        ));
    }
    if x_seq.len() != seq_len * d_model {
        return Err(format!(
            "extract_6sense_pov_features x_seq len {} != {seq_len}*{d_model}",
            x_seq.len()
        ));
    }
    validate_finite_slice(x_seq, "extract_6sense_pov_features")?;

    // Prompt mean POV position xm[0..d_model]
    let inv_t = 1.0f32 / (seq_len as f32);
    let mut xm = vec![0.0f32; d_model];
    for t in 0..seq_len {
        let row = &x_seq[t * d_model..(t + 1) * d_model];
        for (dst, &v) in xm.iter_mut().zip(row.iter()) {
            *dst += v * inv_t;
        }
    }

    let mut tw = tile_w.min(d_model).max(8);
    while d_model % tw != 0 && tw > 8 {
        tw /= 2;
    }
    let n_tiles = d_model / tw;

    // 1. SEE (4D Visual / Spatial Tile Geometry)
    let mut sq_global = 0.0f64;
    let mut peak_global = 0.0f32;
    let mut l1_tot = 0.0f64;
    let mut sum_x = 0.0f64;
    for &v in &xm {
        let av = v.abs();
        if av > peak_global {
            peak_global = av;
        }
        sq_global += (v as f64) * (v as f64);
        l1_tot += av as f64;
        sum_x += v as f64;
    }
    let rms_global = ((sq_global / d_model as f64) + 1e-8).sqrt() as f32;
    let mut tile_rms = Vec::with_capacity(n_tiles);
    let mut max_tr = 0.0f32;
    for tile in xm.chunks_exact(tw) {
        let s2: f64 = tile.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let tr = ((s2 / tw as f64) + 1e-8).sqrt() as f32;
        if tr > max_tr {
            max_tr = tr;
        }
        tile_rms.push(tr);
    }
    let mean_tr = (tile_rms.iter().map(|&v| v as f64).sum::<f64>() / n_tiles as f64).max(1e-6) as f32;
    let var_tr = tile_rms
        .iter()
        .map(|&v| {
            let d = (v - mean_tr) as f64;
            d * d
        })
        .sum::<f64>()
        / n_tiles as f64;
    let rms_log = rms_global.ln_1p().tanh();
    let crest_norm = ((peak_global / rms_global.max(1e-6)) / 8.0).clamp(0.0, 1.5);
    let tile_cv = ((var_tr.sqrt() as f32) / mean_tr).clamp(0.0, 1.5);
    let edge_ratio = if n_tiles > 1 {
        let diff_s: f64 = tile_rms.windows(2).map(|w| (w[1] - w[0]).abs() as f64).sum();
        (((diff_s / (n_tiles - 1) as f64) as f32) / mean_tr).clamp(0.0, 1.5)
    } else {
        0.0
    };

    // 2. HEAR (6D Auditory / Walsh-Hadamard 4-Band Sequency Spectrum)
    let bsize_h = if d_model >= 64 { 64usize } else { 8usize };
    let d_h = (d_model / bsize_h) * bsize_h;
    let n_blks_h = (d_h / bsize_h).max(1);
    let mut spec_buf = xm[..d_h].to_vec();
    fwht_block_inplace(&mut spec_buf, bsize_h)?;
    let log2_b = bsize_h.trailing_zeros();
    let mut bands = [0.0f64; 4];
    for blk in spec_buf.chunks_exact(bsize_h) {
        for (i, &v) in blk.iter().enumerate() {
            let seq = walsh_sequency(i, log2_b);
            let q = (seq * 4) / bsize_h;
            bands[q.min(3)] += ((v as f64) * (v as f64)) / (n_blks_h as f64);
        }
    }
    let bands_sum = (bands[0] + bands[1] + bands[2] + bands[3]).max(1e-8);
    let mut bands_prob = [0.0f32; 4];
    let mut ent = 0.0f64;
    for k in 0..4 {
        let p = (bands[k] / bands_sum) as f32;
        bands_prob[k] = p;
        let pc = p.max(1e-8) as f64;
        ent -= (p as f64) * pc.ln();
    }
    let hf_ratio_norm = ((bands_prob[2] + bands_prob[3]) / (bands_prob[0] + bands_prob[1] + 1e-5)).tanh();
    let spec_ent = (ent / 4.0f64.ln()).clamp(0.0, 1.0) as f32;

    // 3. FEEL (4D Tactile / Proprioceptive Trajectory Mechanics & Surface Roughness)
    let (vel, acc, coherence) = if seq_len >= 3 {
        let mut vel_sum = 0.0f64;
        let mut coh_sum = 0.0f64;
        for t in 0..(seq_len - 1) {
            let r0 = &x_seq[t * d_model..(t + 1) * d_model];
            let r1 = &x_seq[(t + 1) * d_model..(t + 2) * d_model];
            let mut d2 = 0.0f64;
            let mut dot = 0.0f64;
            let mut n0 = 0.0f64;
            let mut n1 = 0.0f64;
            for j in 0..d_model {
                let df = (r1[j] - r0[j]) as f64;
                d2 += df * df;
                dot += (r1[j] as f64) * (r0[j] as f64);
                n0 += (r0[j] as f64) * (r0[j] as f64);
                n1 += (r1[j] as f64) * (r1[j] as f64);
            }
            vel_sum += ((d2 / d_model as f64) + 1e-8).sqrt();
            coh_sum += dot / (n0.sqrt().max(1e-12) * n1.sqrt().max(1e-12));
        }
        let mut acc_sum = 0.0f64;
        for t in 0..(seq_len - 2) {
            let r0 = &x_seq[t * d_model..(t + 1) * d_model];
            let r1 = &x_seq[(t + 1) * d_model..(t + 2) * d_model];
            let r2 = &x_seq[(t + 2) * d_model..(t + 3) * d_model];
            let mut a2 = 0.0f64;
            for j in 0..d_model {
                let aj = (r2[j] - 2.0 * r1[j] + r0[j]) as f64;
                a2 += aj * aj;
            }
            acc_sum += ((a2 / d_model as f64) + 1e-8).sqrt();
        }
        let v_m = (vel_sum / (seq_len - 1) as f64) as f32 / rms_global.max(1e-6);
        let a_m = (acc_sum / (seq_len - 2) as f64) as f32 / rms_global.max(1e-6);
        let c_m = ((coh_sum / (seq_len - 1) as f64) as f32).clamp(-1.0, 1.0);
        (v_m, a_m, c_m)
    } else if seq_len == 2 {
        let r0 = &x_seq[0..d_model];
        let r1 = &x_seq[d_model..2 * d_model];
        let mut d2 = 0.0f64;
        let mut dot = 0.0f64;
        let mut n0 = 0.0f64;
        let mut n1 = 0.0f64;
        for j in 0..d_model {
            let df = (r1[j] - r0[j]) as f64;
            d2 += df * df;
            dot += (r1[j] as f64) * (r0[j] as f64);
            n0 += (r0[j] as f64) * (r0[j] as f64);
            n1 += (r1[j] as f64) * (r1[j] as f64);
        }
        let v_m = (((d2 / d_model as f64) + 1e-8).sqrt() as f32) / rms_global.max(1e-6);
        let c_m = ((dot / (n0.sqrt().max(1e-12) * n1.sqrt().max(1e-12))) as f32).clamp(-1.0, 1.0);
        (v_m, v_m * 0.5, c_m)
    } else {
        (edge_ratio, tile_cv, 0.5f32)
    };
    let vel_norm = vel.tanh();
    let jerk_norm = acc.tanh();
    let anisotropy = ((max_tr / mean_tr) / 4.0).clamp(0.0, 1.5);
    let diff_xm: f64 = xm.windows(2).map(|w| (w[1] - w[0]).abs() as f64).sum();
    let roughness = (((diff_xm / (d_model - 1).max(1) as f64) as f32) / rms_global.max(1e-6)).clamp(0.0, 1.5);

    // 4. SMELL (4D Olfactory / Heavy-Tail Outlier & Volatility Trace)
    let mean_x = (sum_x / d_model as f64) as f32;
    let var_x = xm
        .iter()
        .map(|&v| {
            let d = (v - mean_x) as f64;
            d * d
        })
        .sum::<f64>()
        / d_model as f64;
    let std_x = (var_x + 1e-8).sqrt() as f32;
    let mut z_vec = Vec::with_capacity(d_model);
    let mut z4_sum = 0.0f64;
    let mut z3_sum = 0.0f64;
    let mut rare_cnt = 0usize;
    let mut salty_cnt = 0usize;
    let mut pos_sum = 0.0f64;
    let mut neg_sum = 0.0f64;
    let mut pos_sq = 0.0f64;
    let mut neg_sq = 0.0f64;
    for &v in &xm {
        let z = (v - mean_x) / std_x;
        let z64 = z as f64;
        let z2 = z64 * z64;
        z3_sum += z2 * z64;
        z4_sum += z2 * z2;
        if z.abs() > 2.2 {
            rare_cnt += 1;
        }
        if z.abs() > 0.75 {
            salty_cnt += 1;
        }
        if z > 0.0 {
            pos_sum += z64;
            pos_sq += z2;
        } else if z < 0.0 {
            let nz = -z64;
            neg_sum += nz;
            neg_sq += z2;
        }
        z_vec.push(z);
    }
    let kurt_trace = (((z4_sum / d_model as f64) / 12.0) as f32).tanh();
    let signed_skew = (((z3_sum / d_model as f64) / 3.0) as f32).tanh();
    let k_top = (d_model / 64).max(1);
    let mut abs_vals: Vec<f32> = xm.iter().map(|v| v.abs()).collect();
    abs_vals.sort_unstable_by(|a, b| b.total_cmp(a));
    let top_sum: f64 = abs_vals[..k_top].iter().map(|&v| v as f64).sum();
    let top1pct_conc = (((top_sum / l1_tot.max(1e-6)) * 8.0) as f32).clamp(0.0, 1.5);
    let rare_trace = (((rare_cnt as f32) / (d_model as f32)) * 10.0).clamp(0.0, 1.5);

    // 5. TASTE (6D Gustatory / 5-Receptor Channel Valence Profile)
    let sweet_pos = (((pos_sum / d_model as f64) * 2.0) as f32).tanh();
    let bitter_neg = (((neg_sum / d_model as f64) * 2.0) as f32).tanh();
    let sign_flips = z_vec.windows(2).filter(|w| w[1] * w[0] < 0.0).count();
    let sour_sharp = (((sign_flips as f32) / ((d_model - 1).max(1) as f32)) * 1.5).tanh();
    let salty_dense = ((salty_cnt as f32) / (d_model as f32)).clamp(0.0, 1.0);
    let umami_coherent = (0.5 * (coherence + 1.0)).clamp(0.0, 1.0);
    let valence_bal = ((((pos_sq - neg_sq) / d_model as f64) * 2.0) as f32).tanh();

    // 6. CUNEIFORM_S7 (8D Walsh-Hadamard S^7 Unit Radical Signature)
    let kb8 = d_model / 8;
    let mut u_bar = [0.0f32; 8];
    for blk8 in xm[..kb8 * 8].chunks_exact(8) {
        for j in 0..8 {
            u_bar[j] += blk8[j] / (kb8 as f32);
        }
    }
    fwht_block_inplace(&mut u_bar, 8)?;
    let norm_u = u_bar
        .iter()
        .map(|&v| (v as f64) * (v as f64))
        .sum::<f64>()
        .sqrt()
        .max(1e-8) as f32;
    for v in &mut u_bar {
        *v /= norm_u;
    }

    let mut features = [0.0f32; SIX_SENSE_DIM];
    features[0..4].copy_from_slice(&[rms_log, crest_norm, tile_cv, edge_ratio]);
    features[4..8].copy_from_slice(&bands_prob);
    features[8] = hf_ratio_norm;
    features[9] = spec_ent;
    features[10..14].copy_from_slice(&[vel_norm, jerk_norm, anisotropy, roughness]);
    features[14..18].copy_from_slice(&[kurt_trace, signed_skew, top1pct_conc, rare_trace]);
    features[18..24].copy_from_slice(&[
        sweet_pos,
        bitter_neg,
        sour_sharp,
        salty_dense,
        umami_coherent,
        valence_bal,
    ]);
    features[24..32].copy_from_slice(&u_bar);
    for v in &mut features {
        if v.is_nan() {
            *v = 0.0;
        } else {
            *v = v.clamp(-1.5, 1.5);
        }
    }

    Ok(SixSensePovFeatures { features })
}

/// Evaluate V8.0 6-Sense Player POV consensus gate on a 32D sensory feature vector,
/// matching `run_v8_0_player_pov_spatial_a100.py::evaluate_6sense_consensus_gate_batch`.
pub fn evaluate_6sense_consensus_gate(
    s32: &SixSensePovFeatures,
    caution_thresh: f32,
    rescue_thresh: f32,
    base_top_k: usize,
    base_top_p: f32,
    base_mod_q: f32,
    base_draft_k: usize,
) -> SixSenseConsensusDecision {
    let f = &s32.features;
    let (see_crest, see_cv, see_edge) = (f[1], f[2], f[3]);
    let (hear_b0, hear_b1, hear_hf, hear_ent) = (f[4], f[5], f[8], f[9]);
    let (feel_vel, feel_jerk, feel_rough) = (f[10], f[11], f[13]);
    let (smell_kurt, smell_top1, smell_rare) = (f[14], f[16], f[17]);
    let (taste_sour, taste_umami, taste_bal) = (f[20], f[22], f[23]);
    let c8d_s7 = &f[24..32];

    let s_see = (1.0
        - 0.45 * ((see_crest - 0.55) / 0.75).clamp(0.0, 1.0)
        - 0.35 * ((see_cv - 0.45) / 0.75).clamp(0.0, 1.0)
        - 0.20 * (see_edge / 1.2).clamp(0.0, 1.0))
    .clamp(0.0, 1.0);
    let s_hear = (0.65 * (hear_b0 + hear_b1).clamp(0.0, 1.0)
        + 0.35 * (1.0 - 0.75 * hear_hf.clamp(0.0, 1.0) - 0.25 * hear_ent.clamp(0.0, 1.0)))
    .clamp(0.0, 1.0);
    let s_feel = (1.0
        - 0.45 * (feel_jerk / 1.1).clamp(0.0, 1.0)
        - 0.30 * (feel_rough / 1.2).clamp(0.0, 1.0)
        - 0.25 * (feel_vel - 0.85).max(0.0))
    .clamp(0.0, 1.0);
    let s_smell = (1.0
        - 0.45 * ((smell_kurt - 0.35) / 0.85).clamp(0.0, 1.0)
        - 0.30 * ((smell_rare - 0.35) / 0.90).clamp(0.0, 1.0)
        - 0.25 * ((smell_top1 - 0.65) / 0.85).clamp(0.0, 1.0))
    .clamp(0.0, 1.0);
    let s_taste = (0.55 * taste_umami.clamp(0.0, 1.0)
        + 0.25 * (1.0 - taste_sour.clamp(0.0, 1.0))
        + 0.20 * (1.0 - taste_bal.abs().clamp(0.0, 1.0)))
    .clamp(0.0, 1.0);
    let max_c8d = c8d_s7.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let s_c8d = (max_c8d / 0.85).clamp(0.25, 1.0);

    let consensus = (0.22 * s_see
        + 0.22 * s_hear
        + 0.16 * s_feel
        + 0.14 * s_smell
        + 0.14 * s_taste
        + 0.12 * s_c8d)
        .clamp(0.0, 1.0);

    let is_rescue = consensus < rescue_thresh || s_see < 0.30 || s_hear < 0.30;
    let is_caution = !is_rescue
        && (consensus < caution_thresh
            || s_see < 0.52
            || s_hear < 0.52
            || s_feel < 0.45
            || s_smell < 0.42);
    let (mode, dyn_top_k, dyn_draft_k, caut_f, resc_f) = if is_rescue {
        (ConsensusMode::Rescue, base_top_k + 2, 1, 0.0f32, 1.0f32)
    } else if is_caution {
        (
            ConsensusMode::Caution,
            base_top_k + 1,
            (base_draft_k / 2).max(1),
            1.0f32,
            0.0f32,
        )
    } else {
        (ConsensusMode::Clean, base_top_k, base_draft_k.max(1), 0.0f32, 0.0f32)
    };
    let adaptive_crtr_top_p = (base_top_p + 0.15 * caut_f + 0.30 * resc_f).clamp(0.05, 1.0);
    let adaptive_mod_q = (base_mod_q * (1.0 - 0.40 * caut_f - 0.75 * resc_f)).max(0.0);

    SixSenseConsensusDecision {
        mode,
        consensus_score: consensus,
        see_health: s_see,
        hear_health: s_hear,
        feel_health: s_feel,
        smell_health: s_smell,
        taste_health: s_taste,
        cuneiform_s7_health: s_c8d,
        dyn_top_k,
        dyn_draft_k,
        adaptive_crtr_top_p,
        adaptive_mod_q,
    }
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

    #[test]
    fn six_sense_pov_features_and_consensus_gate_separate_clean_from_noisy() {
        // Smooth low-sequency 3-token trajectory across d_model=128
        let seq_len = 3usize;
        let d_model = 128usize;
        let mut clean = vec![0.0f32; seq_len * d_model];
        for t in 0..seq_len {
            for j in 0..d_model {
                clean[t * d_model + j] =
                    ((j as f32) * 0.05 + (t as f32) * 0.02).sin() + 0.5 * ((j as f32) * 0.025).cos();
            }
        }
        let s_clean = extract_6sense_pov_features(&clean, seq_len, d_model, 64).unwrap();
        assert!(s_clean.features.iter().all(|v| v.is_finite()));
        let n_s7: f32 = s_clean.cuneiform_8d().iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((n_s7 - 1.0).abs() < 1e-4, "expected unit S^7 norm, got {n_s7}");
        let dec_clean = evaluate_6sense_consensus_gate(&s_clean, 0.72, 0.48, 2, 0.30, 0.20, 4);
        assert_eq!(dec_clean.mode, ConsensusMode::Clean);
        assert_eq!(dec_clean.dyn_top_k, 2);
        assert_eq!(dec_clean.dyn_draft_k, 4);

        // High-sequency alternating static + heavy-tail outlier spike + erratic jerk
        let mut noisy = clean.clone();
        for t in 0..seq_len {
            let sgn_t = if t % 2 == 0 { 1.0f32 } else { -1.5f32 };
            for j in 0..d_model {
                let alt = if j % 2 == 0 { 3.5f32 } else { -3.5f32 };
                noisy[t * d_model + j] += sgn_t * alt;
            }
        }
        noisy[7] = 65.0;
        let s_noisy = extract_6sense_pov_features(&noisy, seq_len, d_model, 64).unwrap();
        let dec_noisy = evaluate_6sense_consensus_gate(&s_noisy, 0.72, 0.48, 2, 0.30, 0.20, 4);
        assert!(
            dec_noisy.consensus_score + 0.15 < dec_clean.consensus_score,
            "noisy consensus {} not lower than clean {}",
            dec_noisy.consensus_score,
            dec_clean.consensus_score
        );
        assert_ne!(dec_noisy.mode, ConsensusMode::Clean);
        assert!(dec_noisy.dyn_top_k >= 3);
        assert!(dec_noisy.adaptive_crtr_top_p > 0.30);
    }
}
