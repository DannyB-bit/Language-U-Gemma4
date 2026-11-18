//! Standalone Offline Machine-Code Validation Probe for the Tri-Modal
//! ("Read + Hear + See") Accuracy Driver (`tri_modal.rs`).
//!
//! Runs completely separately from the live Pi 4 31B V4.2 run on real
//! quantized `T16CMat` projections + real `C8DB` INT8 residual banks +
//! `672x672` Optical Rescue Canvas rendering:
//!   - Step 1: Log & correlate FWHT 4-band sequency (`HEAR`) + tile crest/CV (`SEE`)
//!             against hidden-state cosine drift and logit margin collapse.
//!   - Step 2: Dynamic consensus gating (`--lmhead-cs` sieve <-> exact, `--cs-skip`,
//!             draft `k`) vs. always-pruned and always-exact baselines.
//!   - Step 3: Targeted `C8DB` residual recovery + turn-level `Optical Rescue Canvas`.

use run270_rust::cuneiform_tbi::{
    Cuneiform6DRadicals, CuneiformBank, CuneiformBankEntryInput, CuneiformExpertInput,
};
use run270_rust::math::{cosine_and_rel_l2, rmsn_raw, Rng};
use run270_rust::t16c::{NativeTile16Kernel, T16CMat};
use run270_rust::tri_modal::{
    analyze_hear_fwht, analyze_read_logits, analyze_see_spatial, pearson_r,
    render_tri_modal_rescue_canvas, sample_normal_f32, ConsensusConfig, ConsensusMode,
    TriModalConsensusGate,
};
use std::path::Path;
use std::time::Instant;

fn repo_native_lib() -> Option<std::path::PathBuf> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("neon_runner")
        .join(if cfg!(windows) {
            "neon_stream_accelerator.dll"
        } else {
            "libneon_stream_accelerator.so"
        });
    p.is_file().then_some(p)
}

fn matmul_f32_rowmajor(w: &[f32], x: &[f32], d_out: usize, d_in: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; d_out];
    for r in 0..d_out {
        let row = &w[r * d_in..(r + 1) * d_in];
        let mut acc = 0.0f64;
        for c in 0..d_in {
            acc += (row[c] as f64) * (x[c] as f64);
        }
        y[r] = acc as f32;
    }
    y
}

/// Pack a dense FP32 matrix into real `T16CMat` (`n_bits`, `has_fwht`, `block_size`).
/// Uses optimal `mean(|w|)` scale for `1-bit` and symmetric/asymmetric block range for `2..4-bit`.
fn pack_real_t16c_mat(
    name: &str,
    w: &[f32],
    d_out: usize,
    d_in: usize,
    bsize: usize,
    n_bits: u8,
    is_asym: bool,
    has_fwht: bool,
) -> Result<T16CMat, String> {
    let mut w_rot = w.to_vec();
    if has_fwht {
        run270_rust::t16c::fwht_block_inplace(&mut w_rot, bsize)?;
    }

    let n_blks = d_in / bsize;
    let eff_bits = n_bits & 0x7F;
    let levels = (1u8 << eff_bits) - 1;
    let half_span = levels as f32 * 0.5;
    let mut sc = vec![0.0f32; d_out * n_blks];
    let mut bi = if is_asym {
        Some(vec![0.0f32; d_out * n_blks])
    } else {
        None
    };
    let mut idx_u8 = vec![0u8; d_out * d_in];

    for r in 0..d_out {
        for blk in 0..n_blks {
            let off = r * d_in + blk * bsize;
            let slice = &w_rot[off..off + bsize];
            if is_asym {
                let min_v = slice.iter().copied().fold(f32::INFINITY, f32::min);
                let max_v = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mid = 0.5 * (max_v + min_v);
                let half = (0.5 * (max_v - min_v)).max(1e-6);
                let step_sc = half / half_span.max(1.0);
                sc[r * n_blks + blk] = step_sc;
                bi.as_mut().unwrap()[r * n_blks + blk] = mid;
                for j in 0..bsize {
                    let q_signed = ((slice[j] - mid) / half * half_span).round();
                    let lv = (q_signed + half_span).clamp(0.0, levels as f32).round() as u8;
                    idx_u8[off + j] = lv;
                }
            } else if eff_bits == 1 {
                let mean_abs = (slice.iter().map(|v| v.abs()).sum::<f32>() / bsize as f32).max(1e-6);
                sc[r * n_blks + blk] = mean_abs;
                for j in 0..bsize {
                    idx_u8[off + j] = if slice[j] >= 0.0 { 1 } else { 0 };
                }
            } else {
                let max_abs = slice.iter().copied().map(f32::abs).fold(0.0f32, f32::max).max(1e-6);
                let step_sc = max_abs / half_span.max(1.0);
                sc[r * n_blks + blk] = step_sc;
                for j in 0..bsize {
                    let lv = ((slice[j] / max_abs + 1.0) * half_span)
                        .round()
                        .clamp(0.0, levels as f32) as u8;
                    idx_u8[off + j] = lv;
                }
            }
        }
    }

    T16CMat::pack_from_components(
        name.to_string(),
        d_out,
        d_in,
        bsize,
        n_bits,
        is_asym,
        has_fwht,
        &sc,
        bi.as_deref(),
        &idx_u8,
        None,
        None,
    )
}

/// Build an authentic rank-`r` INT8 `C8DB` residual adapter via power iteration on `(W_ref - W_q_deq)`.
fn build_residual_c8db_bank(
    tmp_path: &Path,
    mat_name: &str,
    w_ref: &[f32],
    mat_q: &T16CMat,
    d_out: usize,
    d_in: usize,
    rank: usize,
    native_lib: Option<&Path>,
) -> Result<CuneiformBank, String> {
    let mut err_mat = vec![0.0f32; d_out * d_in];
    for r in 0..d_out {
        let q_row = mat_q.decode_row(r)?;
        for c in 0..d_in {
            err_mat[r * d_in + c] = w_ref[r * d_in + c] - q_row[c];
        }
    }

    // Extract top-`rank` factors of `err_mat` by orthogonalized power iterations (pure FP32, no stubs).
    let mut u_f = vec![0.0f32; d_out * rank];
    let mut v_f = vec![0.0f32; rank * d_in];
    let mut rem = err_mat;
    let mut rng = Rng::new(777);

    for k in 0..rank {
        let mut v: Vec<f32> = (0..d_in).map(|_| sample_normal_f32(&mut rng)).collect();
        let mut u = vec![0.0f32; d_out];
        for _ in 0..8 {
            for r in 0..d_out {
                let mut acc = 0.0f64;
                for c in 0..d_in {
                    acc += (rem[r * d_in + c] as f64) * (v[c] as f64);
                }
                u[r] = acc as f32;
            }
            let un = u.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt().max(1e-12) as f32;
            for x in &mut u {
                *x /= un;
            }
            v.fill(0.0);
            for r in 0..d_out {
                let ur = u[r] as f64;
                for c in 0..d_in {
                    v[c] += (ur * (rem[r * d_in + c] as f64)) as f32;
                }
            }
            let vn = v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt().max(1e-12) as f32;
            for x in &mut v {
                *x /= vn;
            }
        }
        // Sigma = u^T rem v
        let mut sig = 0.0f64;
        for r in 0..d_out {
            let ur = u[r] as f64;
            for c in 0..d_in {
                sig += ur * (rem[r * d_in + c] as f64) * (v[c] as f64);
            }
        }
        let s_rt = sig.abs().sqrt() as f32 * sig.signum() as f32;
        let s_abs = sig.abs().sqrt() as f32;
        for r in 0..d_out {
            u_f[r * rank + k] = u[r] * s_rt;
            for c in 0..d_in {
                rem[r * d_in + c] -= (sig as f32) * u[r] * v[c];
            }
        }
        for c in 0..d_in {
            v_f[k * d_in + c] = v[c] * s_abs;
        }
    }

    // Quantize u_f -> (su, u_i8) and v_f -> (sv, v_i8) cleanly.
    let mut su = vec![0.0f32; d_out];
    let mut sv = vec![0.0f32; rank];
    let mut u_i8 = vec![0i8; d_out * rank];
    let mut v_i8 = vec![0i8; rank * d_in];
    for r in 0..d_out {
        let row = &u_f[r * rank..(r + 1) * rank];
        let mx = row.iter().copied().map(f32::abs).fold(0.0f32, f32::max).max(1e-8);
        su[r] = mx / 127.0;
        for k in 0..rank {
            u_i8[r * rank + k] = (row[k] / su[r]).round().clamp(-127.0, 127.0) as i8;
        }
    }
    for k in 0..rank {
        let row = &v_f[k * d_in..(k + 1) * d_in];
        let mx = row.iter().copied().map(f32::abs).fold(0.0f32, f32::max).max(1e-8);
        sv[k] = mx / 127.0;
        for c in 0..d_in {
            v_i8[k * d_in + c] = (row[c] / sv[k]).round().clamp(-127.0, 127.0) as i8;
        }
    }

    // Configure 4 experts aligned to canonical Clifford octants so C8DB routes smoothly.
    let n_exp = 4usize;
    let mut centroids = vec![0.0f32; n_exp * 8];
    for e in 0..n_exp {
        centroids[e * 8 + e] = 1.0;
    }
    let rads: Vec<Cuneiform6DRadicals> = (0..n_exp)
        .map(|e| Cuneiform6DRadicals::from_axes(10, 14, 6, 7, 10, 0, e as u8))
        .collect();
    let experts: Vec<CuneiformExpertInput<'_>> = (0..n_exp)
        .map(|e| CuneiformExpertInput {
            radicals: rads[e],
            su: &su,
            sv: &sv,
            u_i8: &u_i8,
            v_i8: &v_i8,
        })
        .collect();
    let entry = CuneiformBankEntryInput {
        name: mat_name.to_string(),
        d_out,
        d_in,
        rank,
        top_k: 2,
        c6d: rads[0],
        temperature: 0.5,
        gate_scale: 1.0,
        centroids: &centroids,
        experts,
    };
    CuneiformBank::write_file(tmp_path, &[], &[entry])?;
    CuneiformBank::load(tmp_path, native_lib, false, true)
}

fn run_probe() -> Result<(), String> {
    let t_start = Instant::now();
    let d = 256usize;
    let bsize = 64usize;
    let vocab = 512usize;
    let n_tokens = 24usize;

    let native_path = repo_native_lib();
    let native_t16c = native_path
        .as_deref()
        .and_then(|p| NativeTile16Kernel::try_load(p).ok());

    println!("=== Tri-Modal Read+Hear+See Offline Machine-Code Probe ===");
    println!(
        "Scope: Separate micro-kernel & quantization-cliff validation (d={d}, bsize={bsize}, vocab={vocab}, native_ffi={})",
        native_t16c.is_some()
    );

    let mut rng = Rng::new(42);
    // Construct well-conditioned smooth residual projection W_clean + an explicit cliff layer W_cliff + LM head W_lm.
    let make_smooth_w = |rng: &mut Rng, rows: usize, cols: usize, scale: f32| -> Vec<f32> {
        let mut w = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let phase = (r as f32) * 0.08;
            for c in 0..cols {
                let smooth = ((c as f32) * 0.05 + phase).cos() * 0.35;
                w[r * cols + c] = (smooth + 0.15 * sample_normal_f32(rng)) * scale;
            }
        }
        w
    };

    let w_clean_ref = make_smooth_w(&mut rng, d, d, 0.22);
    let w_cliff_ref = make_smooth_w(&mut rng, d, d, 0.38);
    let mut w_lm_ref = vec![0.0f32; vocab * d];
    for r in 0..vocab {
        let f1 = 0.02 + 0.0005 * (r as f32);
        let f2 = 0.05 + 0.0011 * ((r % 37) as f32);
        let ph = (r as f32) * 0.43;
        for c in 0..d {
            w_lm_ref[r * d + c] =
                (((c as f32) * f1 + ph).sin() + ((c as f32) * f2 - ph).cos()) * 0.32
                    + 0.05 * sample_normal_f32(&mut rng);
        }
    }

    // Real quantized T16C matrices with FWHT rotation: 3-bit clean layer, 1-bit cliff layer, 3-bit LM head with CS norms.
    let mat_clean = pack_real_t16c_mat("layer.clean.down", &w_clean_ref, d, d, bsize, 3, false, true)?;
    let mat_cliff = pack_real_t16c_mat("layer.cliff.down", &w_cliff_ref, d, d, bsize, 1, false, true)?;
    let mat_lm = pack_real_t16c_mat("lm_head.weight", &w_lm_ref, vocab, d, bsize, 3, false, true)?;

    let clean_norms = mat_clean.compute_tile_norms_fp16();
    let cliff_norms = mat_cliff.compute_tile_norms_fp16();
    let lm_norms = mat_lm.compute_tile_norms_fp16();

    // Real INT8 C8DB rescue bank fit on (W_cliff_ref - dequant(mat_cliff)).
    let tmp_c8d = std::env::temp_dir().join("tri_modal_probe_rescue.c8d");
    let c8d_bank = build_residual_c8db_bank(
        &tmp_c8d,
        "layer.cliff.down",
        &w_cliff_ref,
        &mat_cliff,
        d,
        d,
        16,
        native_path.as_deref(),
    )?;
    let _ = std::fs::remove_file(&tmp_c8d);
    let rescue_entry = c8d_bank.entries.get("layer.cliff.down").ok_or("missing rescue C8DB entry")?;

    // Build 12 clean inputs (smooth low-sequency) and 12 cliff-stressed inputs (quantization chatter + spike).
    let mut tokens_x: Vec<Vec<f32>> = Vec::with_capacity(n_tokens);
    let mut is_cliff_tok: Vec<bool> = Vec::with_capacity(n_tokens);
    for t in 0..n_tokens {
        let cliff = t >= n_tokens / 2;
        is_cliff_tok.push(cliff);
        let mut x = vec![0.0f32; d];
        let f_base = 0.04 + 0.01 * ((t % 6) as f32);
        for c in 0..d {
            x[c] = ((c as f32) * f_base + (t as f32) * 0.3).sin() * 1.1 + 0.06 * sample_normal_f32(&mut rng);
        }
        if cliff {
            // Simulate upstream cliff noise: high-sequency alternating chatter + spatial spike in tile 0.
            for (c, v) in x.iter_mut().enumerate() {
                *v += if c % 2 == 0 { 2.0 } else { -2.0 };
            }
            x[5] += 24.0;
        }
        tokens_x.push(x);
    }

    // Step 1 & 2 & 3 execution across all 24 tokens.
    let mut gate = TriModalConsensusGate::new(ConsensusConfig {
        base_sieve_frac: 0.25,
        base_cs_skip: 0.55,
        base_draft_k: 4,
        caution_thresh: 0.72,
        rescue_thresh: 0.50,
        rescue_streak_for_optical: 2,
        ema_alpha: 0.20,
        ..ConsensusConfig::default()
    });

    let mut telemetry_signal: Vec<f64> = Vec::with_capacity(n_tokens);
    let mut cosine_drift: Vec<f64> = Vec::with_capacity(n_tokens);

    let mut pruned_hits = 0usize;
    let mut pruned_margin_sum = 0.0f64;
    let mut pruned_skipped_total = 0i64;

    let mut tri_hits = 0usize;
    let mut tri_margin_sum = 0.0f64;
    let mut tri_skipped_total = 0i64;
    let mut tri_c8db_calls = 0usize;
    let mut tri_optical_triggers = 0usize;
    let mut tri_rescue_cos_before_sum = 0.0f64;
    let mut tri_rescue_cos_after_sum = 0.0f64;
    let mut rescue_cnt = 0usize;

    let mut exact_hits = 0usize;
    let mut exact_margin_sum = 0.0f64;

    let mut clean_modes = 0usize;
    let mut cliff_nonclean_modes = 0usize;

    for (t, x_in) in tokens_x.iter().enumerate() {
        let cliff = is_cliff_tok[t];
        // FP32 teacher reference path:
        let h_ref_raw = if cliff {
            matmul_f32_rowmajor(&w_cliff_ref, x_in, d, d)
        } else {
            matmul_f32_rowmajor(&w_clean_ref, x_in, d, d)
        };
        let h_ref: Vec<f32> = x_in.iter().zip(h_ref_raw.iter()).map(|(a, b)| a + b).collect();
        let h_ref_n = rmsn_raw(&h_ref, None, 1e-6);
        let ref_logits = matmul_f32_rowmajor(&w_lm_ref, &h_ref_n, vocab, d);
        let ref_read = analyze_read_logits(&ref_logits, true)?;
        let target_id = ref_read.top1_id;

        // Baseline 1: Always-Pruned (aggressive cs_skip=0.55 + sieve=0.25, no C8DB).
        let (q_pruned_raw, sk_p) = if cliff {
            mat_cliff.project_batch_cs(x_in, 1, native_t16c.as_ref(), 2, Some(&cliff_norms), 0.55)?
        } else {
            mat_clean.project_batch_cs(x_in, 1, native_t16c.as_ref(), 2, Some(&clean_norms), 0.55)?
        };
        let h_p: Vec<f32> = x_in.iter().zip(q_pruned_raw.iter()).map(|(a, b)| a + b).collect();
        let h_p_n = rmsn_raw(&h_p, None, 1e-6);
        let (logits_p, sk_lm_p) = mat_lm.project_lmhead_cs(
            &h_p_n,
            1,
            native_t16c.as_ref(),
            2,
            Some(&lm_norms),
            32,
            0.25,
        )?;
        let read_p = analyze_read_logits(&logits_p, true)?;
        if read_p.top1_id == target_id {
            pruned_hits += 1;
        }
        pruned_margin_sum += read_p.margin as f64;
        pruned_skipped_total += sk_p + sk_lm_p;

        // Step 1 Telemetry on the quantized pass: Hear (FWHT) + See (Spatial) + Read (pruned logits).
        let hear = analyze_hear_fwht(&h_p_n, bsize, false)?;
        let see = analyze_see_spatial(&h_p_n, bsize)?;
        let (cos_before, _, _) = cosine_and_rel_l2(&h_ref_n, &h_p_n);
        telemetry_signal.push((hear.hf_ratio as f64) + 0.25 * (see.crest_factor as f64));
        cosine_drift.push(1.0 - cos_before);

        // Baseline 2: Always-Exact without C8DB.
        let (q_exact_raw, _) = if cliff {
            mat_cliff.project_batch_cs(x_in, 1, native_t16c.as_ref(), 2, None, 0.0)?
        } else {
            mat_clean.project_batch_cs(x_in, 1, native_t16c.as_ref(), 2, None, 0.0)?
        };
        let h_e: Vec<f32> = x_in.iter().zip(q_exact_raw.iter()).map(|(a, b)| a + b).collect();
        let h_e_n = rmsn_raw(&h_e, None, 1e-6);
        let (logits_e, _) = mat_lm.project_lmhead_cs(
            &h_e_n,
            1,
            native_t16c.as_ref(),
            2,
            Some(&lm_norms),
            32,
            0.0,
        )?;
        let read_e = analyze_read_logits(&logits_e, true)?;
        if read_e.top1_id == target_id {
            exact_hits += 1;
        }
        exact_margin_sum += read_e.margin as f64;

        // Step 2 & 3: Tri-Modal Dynamic Consensus Driver.
        let decision = gate.evaluate(&read_p, &hear, &see, !cliff);
        if !cliff && decision.mode == ConsensusMode::Clean {
            clean_modes += 1;
        }
        if cliff && decision.mode != ConsensusMode::Clean {
            cliff_nonclean_modes += 1;
        }

        let (tri_logits, sk_tri) = if decision.mode == ConsensusMode::Clean {
            (logits_p.clone(), sk_p + sk_lm_p)
        } else {
            // Re-run layer projection with consensus-adapted `cs_skip_thresh`, then C8DB rescue, then exact LM-head.
            let (mut q_dyn, sk_d) = if cliff {
                mat_cliff.project_batch_cs(
                    x_in,
                    1,
                    native_t16c.as_ref(),
                    2,
                    Some(&cliff_norms),
                    decision.cs_skip_thresh,
                )?
            } else {
                mat_clean.project_batch_cs(
                    x_in,
                    1,
                    native_t16c.as_ref(),
                    2,
                    Some(&clean_norms),
                    decision.cs_skip_thresh,
                )?
            };
            if decision.engage_c8db_rescue {
                rescue_entry.apply_batch_into(x_in, 1, &mut q_dyn, c8d_bank.native.as_ref(), 2, None)?;
                tri_c8db_calls += 1;
            }
            let h_dyn: Vec<f32> = x_in.iter().zip(q_dyn.iter()).map(|(a, b)| a + b).collect();
            let h_dyn_n = rmsn_raw(&h_dyn, None, 1e-6);
            let (cos_after, _, _) = cosine_and_rel_l2(&h_ref_n, &h_dyn_n);
            tri_rescue_cos_before_sum += cos_before;
            tri_rescue_cos_after_sum += cos_after;
            rescue_cnt += 1;

            let sieve = decision.lmhead_cs_sieve.unwrap_or(0.0);
            let (lg, sk_lm_d) = mat_lm.project_lmhead_cs(
                &h_dyn_n,
                1,
                native_t16c.as_ref(),
                2,
                Some(&lm_norms),
                32,
                sieve,
            )?;
            if decision.trigger_optical_self_analyze {
                let layer_health = [1.0, 0.96, hear.health, see.health, 0.92, decision.consensus_score];
                let (img, crc) = render_tri_modal_rescue_canvas(t, &decision, &read_p, &hear, &see, &layer_health);
                if img.width != 672 || crc == 0 {
                    return Err("invalid rescue optical canvas".into());
                }
                tri_optical_triggers += 1;
            }
            (lg, sk_d + sk_lm_d)
        };
        let read_tri = analyze_read_logits(&tri_logits, true)?;
        if read_tri.top1_id == target_id {
            tri_hits += 1;
        }
        tri_margin_sum += read_tri.margin as f64;
        tri_skipped_total += sk_tri;
    }

    let r_corr = pearson_r(&telemetry_signal, &cosine_drift)?;
    let acc_p = (pruned_hits as f64 / n_tokens as f64) * 100.0;
    let acc_e = (exact_hits as f64 / n_tokens as f64) * 100.0;
    let acc_t = (tri_hits as f64 / n_tokens as f64) * 100.0;
    let cos_b_avg = if rescue_cnt > 0 { tri_rescue_cos_before_sum / rescue_cnt as f64 } else { 0.0 };
    let cos_a_avg = if rescue_cnt > 0 { tri_rescue_cos_after_sum / rescue_cnt as f64 } else { 0.0 };

    println!(
        "[STEP 1 Log & Correlate] Pearson r(hear_hf+0.25*see_crest, 1-cos_drift) = {r_corr:.4} | clean_Clean={clean_modes}/12 cliff_Gated={cliff_nonclean_modes}/12"
    );
    println!(
        "[STEP 2 Dynamic Gating] Always-Pruned: top1={pruned_hits}/{n_tokens} ({acc_p:.1}%) avg_marg={:.3} skipped={pruned_skipped_total}",
        pruned_margin_sum / n_tokens as f64
    );
    println!(
        "[STEP 2 Dynamic Gating] Always-Exact : top1={exact_hits}/{n_tokens} ({acc_e:.1}%) avg_marg={:.3} skipped=0",
        exact_margin_sum / n_tokens as f64
    );
    println!(
        "[STEP 2+3 Tri-Modal   ] Tri-Modal Dyn: top1={tri_hits}/{n_tokens} ({acc_t:.1}%) avg_marg={:.3} skipped={tri_skipped_total} c8db_calls={tri_c8db_calls} optical_rescues={tri_optical_triggers}",
        tri_margin_sum / n_tokens as f64
    );
    println!(
        "[STEP 3 C8DB Rescue   ] Drifted hidden cosine: {cos_b_avg:.5} -> {cos_a_avg:.5} (elapsed={:.2}s)",
        t_start.elapsed().as_secs_f64()
    );

    if r_corr < 0.60 {
        return Err(format!("Step 1 fail: Pearson correlation {r_corr:.4} < 0.60"));
    }
    if cliff_nonclean_modes < 10 {
        return Err(format!("Step 2 fail: detected only {cliff_nonclean_modes}/12 cliff tokens"));
    }
    if tri_hits < pruned_hits || cos_a_avg <= cos_b_avg {
        return Err(format!(
            "Step 3 fail: Tri-Modal did not improve/preserve accuracy (tri={tri_hits} pruned={pruned_hits} cos {cos_b_avg:.5}->{cos_a_avg:.5})"
        ));
    }
    if tri_optical_triggers == 0 {
        return Err("Step 3 fail: expected turn-level Optical Rescue Canvas trigger on sustained cliff streak".into());
    }

    println!("[PASS] tri_modal_probe verified all 3 Read+Hear+See steps end-to-end.");
    Ok(())
}

fn main() {
    if let Err(e) = run_probe() {
        eprintln!("FAIL: {e}");
        std::process::exit(1);
    }
}
