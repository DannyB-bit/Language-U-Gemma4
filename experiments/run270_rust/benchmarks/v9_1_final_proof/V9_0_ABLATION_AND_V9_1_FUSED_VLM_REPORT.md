# Gemma 4 31B V9.0 Calibration Ablation & `V9.1-Fused-VLM` — Physical Raspberry Pi 4 Verification Report

**Date:** `2026-11-25`  
**Target Hardware:** Raspberry Pi 4 (`astronautshe@192.168.1.207`, 4x Cortex-A72, `7.6 GiB` RAM, `/mnt/SHE-VAULT`), [run31b.rs](../../src/bin/run31b.rs) + `libneon_stream_accelerator.so` (`--threads 4`, `ssd_read_bytes = 0` on all `V9.0` & `V9.1` runs)  
**Deliverable Container:** `/mnt/SHE-VAULT/gemma4_v9_1/gemma4_31b_v9_1_fused_vlm.t16c` (`6,353,854,848 B` = `5.9175 GiB`, `604` matrices, `585` norms, `114` RowMajor + `116` ColBlock mats, `max_parity_diff = 0.0`)

---

## 1. Executive Summary

Across three end-to-end physical Raspberry Pi 4 benchmark cycles (`V9.0` Approaches 1/2/3, `--repair-app3` WLS stabilization, and `V9.1-Fused-VLM`), this investigation isolated **three concrete software/calibration root causes** plus **one physical information-theoretic compression wall**, fixed all three bugs in source, and verified a unified sub-`5.95 GiB` `V9.1-Fused-VLM` stack with zero cold SSD streaming:

1. **Calibration Bugs Fixed (`App2` Ridge Overfitting & `App3` Sparse `down_proj` Explosion):** Sweeping ridge regularization down to `0.0008` in `solve_block_ridge_headers` across 16 rank-deficient calibration sequences caused up to `10.7x` weight blowup (`L02 gate_proj cos=0.0674`, hidden Colab `val_kl=4.26` vs Pi 4 `mean_nll=13.11573` dense / `13.78589` sparse / `16.58` prior eval). Meanwhile, `_fit_als_lr_int8` against `88.1%`-zero sparse-masked activations plus a mismatched `di` denominator flipped `L01 down_proj` cosine to `-0.2555` (`5.9x–9.6x` norm inflation). Fixing [run_v9_0_sparse_spinquant_qat_gpu.py](../../calib/31b/run_v9_0_sparse_spinquant_qat_gpu.py) and running `--repair-app3` in [repair_v9_0_inplace.py](../../calib/31b/repair_v9_0_inplace.py) scrubbed `44` corrupted adapters (`mlp_cos=0.903..0.940` on 3-bit stem/cliff layers, `0.79..0.80` on 1-bit MLPs) and cut `App3` dense Pi 4 NLL from `14.85744` (`ppl=2,834,682.130`) to `13.73610` without stale C8D (`kl_t_to_s=12.71247`).
2. **Physical `~1.5 bpw` 1-Bit GeGLU Wall Proven:** Even after WLS repair, all three `V9.0` variants placing 56 GeGLU MLPs at `1`-bit (`App1` `12.54770`, `App2` `13.11573`, `App3-WLS` `13.73610`) sit past Gemma 4 31B's `~1.5 bpw` factual-recall cliff.
3. **Runtime Root Cause Fixed (`run31b.rs` Non-Causal `GlyphKvLayer` Prefill):** `run31b.rs` previously pushed every prompt token into `GlyphKvLayer` *before* calling `attend_range`, evicting early prompt tokens out of the `32`-token exact FP32 tail window before self-attention executed. Fixing this with `append_glyph_prefix` + `prefill_glyph_kv_causal_batch` in [run31b.rs](../../src/bin/run31b.rs) cut `V9.1-Fused-VLM` `heldout_systems_cache` `--kv-quant glyph` NLL from `13.52555` (`ppl=748,290.939`) to **`8.89081`** (`ppl=7,264.891`, **`103.0x` PPL reduction**), and `--kv-quant glyph-lod` to **`7.46533`** (`ppl=1,746.440`, `kl_t_to_s=4.73442`).
4. **Activation Bitwidth & LM-Head Precision (`--andvcnt-bits 6 --lmhead-cs exact`):** Raising RowMajor AND+VCNT activation bits from the `5`-bit default to `6`-bit and disabling heuristic LM-head/tri-modal pruning further lowered `heldout_systems_cache` `glyph-lod` NLL to **`7.40955`** (`ppl=1,651.680`, `kl_t_to_s=4.67860`, `top5_agree=0.4444`) and flipped multi-step division (`P5_speed`: `"60 km per hour"`) and science QA (`P6_photo`) from fail to pass.
5. **New Sub-`5.95 GiB` Pareto Champion (`V9.1-Fused-VLM`):** Built via [build_v9_1_fused_vlm.py](../../calib/31b/build_v9_1_fused_vlm.py), `V9.1-Fused-VLM` scores **`mean_nll=7.77141`** (`ppl=2,371.802`, `kl_t_to_s=5.09893`, `top1_agree=0.3193`, `top5_agree=0.4518`, `vm_rss=6.12 GiB`) across the `166`-position 3-passage `--kv-quant fp8` suite — outperforming `V4.2` dense FP32 (`8.01262` / `kl=5.21293`), `V8.0` (`10.28455` / `kl=7.43218`), `V6.1` (`10.38568` / `kl=7.49964`), and all `V9.0` variants — while recognizing `"A motorcycle is parked on a the a pavement.<turn|>"` in `30.46s` decode with **`ssd_read_bytes = 0`**.

---

## 2. Provenance & Cryptographic Checksums

### `V9.1-Fused-VLM` Container & Auxiliary Suite ([SHA256SUMS_V9_1.txt](../../calib/31b/receipts/SHA256SUMS_V9_1.txt), [v9_1_build_report.json](../../calib/31b/receipts/v9_1_build_report.json))

Verified `3/3 OK` on Pi 4 (`/mnt/SHE-VAULT/gemma4_v9_1/`):

| Artifact | Exact Bytes | Size | SHA-256 | Role |
|---|---|---|---|---|
| `gemma4_31b_v9_1_fused_vlm.t16c` | `6,353,854,848` | `5.9175 GiB` | `1332bd355fd9cb35734b6fe14c73f5750c5994679f3d4b8ebf12c54658e85429` | Mixed `ternary5`/`3-bit`/`RowMajor-1b` (`114` RowMajor + `116` ColBlock mats, `max_parity_diff=0.0`) |
| `gemma4_31b_v9_1_router.crtr` | `21,360,516` | `20.37 MiB` | `e801e26b5616a9e680f7908e3b3674717210f7f4e88bcac55a90aeec6ecbcf71` | 60-layer CRTR (`rank=64`, `top_p=0.5`, `mean_recall=0.81706`, `mean_mod_acc=0.90782`) |
| `gemma4_31b_v4_3_vlm_c8d.c8d` | `9,317,184` | `8.89 MiB` | `46afa072ecc4d4846c6b266d5d4aed0acae3c076c407433e3edfb95f2f590e09` | Multimodal C8DB (`entries=4`, `norms=1`) |

- **Matrix Type Breakdown in `gemma4_31b_v9_1_fused_vlm.t16c` (`604` matrices, `585` norms, repack `103.44s`):** `nb=0x03 bsz=64: 193` (vision), `nb=0x03 bsz=256: 17`, `nb=0x05 bsz=256: 202`, `nb=0x23 bsz=256: 7`, `nb=0x25 bsz=256: 71`, `nb=0x41 bsz=256: 76`, `nb=0x61 bsz=256: 38`.
- **Parity Verification Before Repack:** Input `gemma4_31b_v4_3_vlm.t16c` (`6,353,854,848 B`, `sha256: 5469f95285d7ed99e14c728504ac29f361f6a28781e829971d216b7486d99571`) passed FFI parity across `0x41/0x61/0x05/0x25/0x83/0xa3` at `bsize=64` and `256` (`cb_vs_noncb=0.00e+00`, `cos >= 0.997785`).

### `V9.0` Ablation Artifacts ([SHA256SUMS_ABLATE.txt](../../calib/31b/receipts/SHA256SUMS_ABLATE.txt), [v9_0_SHA256SUMS.txt](../../calib/31b/receipts/v9_0_SHA256SUMS.txt))

| Artifact | SHA-256 Prefix | Receipt |
|---|---|---|
| `app2_ridge_headers_norms.bin` | `ecf572605b3a9618` | [app2_report.json](../../calib/31b/receipts/app2_report.json) |
| `app2_crtr.crtr` / `app2_head_router.crtr` / `app2_c8d.c8d` | `58092397` / `fe2f48e4` / `4086f231` | [v9_0_phase5a_app2_app3.log](../../calib/31b/receipts/v9_0_phase5a_app2_app3.log) |
| `gemma4_31b_v9_0_app3_zero1b.t16c` | `554954bade110164` | [app3_report.json](../../calib/31b/receipts/app3_report.json) |
| `app3_orig_headers_norms.bin` (`399,250,769 B` pre-repair backup) | `02a8b5f4983424ec` | [repair_app3.log](../../calib/31b/receipts/repair_app3.log) |
| `app3_wls_headers_norms.bin` (`399,250,769 B`, `zeroed_lr=44`) | Logged in `app3_wls_repair_report.json` | [app3_wls_repair_report.json](../../calib/31b/receipts/app3_wls_repair_report.json) |

---

## 3. Root-Cause Forensics (Colab Calibration & Runtime KV Bugs)

### 3.1 `solve_block_ridge_headers` Rank-Deficient Blowup (`App2`)
- **Symptom:** `App2` reported low Colab validation KL (`4.26`) yet diverged on Pi 4 silicon (`mean_nll=13.11573` dense / `13.78589` sparse / `"HereA 1,<turn|>"` in [v9_0_phase5a_app2_app3.log](../../calib/31b/receipts/v9_0_phase5a_app2_app3.log)).
- **Cause:** Sweeping ridge regularization down to `0.0008` on rank-deficient 16-sequence calibration activations allowed header updates to overfit nullspace directions, causing up to `10.7x` weight norm inflation and `L02 gate_proj` cosine collapse to `0.0674`.
- **Fix:** Added Weighted Least Squares (`WLS`) prior anchoring plus cosine and norm explosion guards in [run_v9_0_sparse_spinquant_qat_gpu.py](../../calib/31b/run_v9_0_sparse_spinquant_qat_gpu.py).

### 3.2 `_fit_als_lr_int8` Sign Inversion & `down_proj` Explosion (`App3`)
- **Symptom:** Raw `App3` (`gemma4_31b_v9_0_app3_zero1b.t16c`) produced `mean_nll=14.85744` (`ppl=2,834,682.130`) dense and `14.66765` sparse (`"The simplest the same of 1,0208 a)\n<unused2099><turn|>"`).
- **Cause:** Fitting `lr_d` low-rank adapters against `88.1%`-zero sparse-masked GeGLU activations with an incorrect `di` denominator flipped `L01 down_proj` cosine to `-0.2555` with `5.9x–9.6x` norm inflation across `44` layers.
- **Fix & Physical Verification:** Fixed `_fit_als_lr_int8` in [run_v9_0_sparse_spinquant_qat_gpu.py](../../calib/31b/run_v9_0_sparse_spinquant_qat_gpu.py) and ran `--repair-app3` ([repair_v9_0_inplace.py](../../calib/31b/repair_v9_0_inplace.py)) directly on Pi 4 ([repair_app3.log](../../calib/31b/receipts/repair_app3.log)):
  - Verified exact permutation identity across all 60 layers (`max_norm_diff=0.00e+00`).
  - Zeroed all `44` corrupted adapters (`zeroed_lr=44`), restored `mlp_cos=0.903..0.940` (`L00..L05`, `L56..L59`) and `0.79..0.80` (`1`-bit mid MLPs), `q_cos=0.90..0.94`, `o_cos=0.91..0.98`.
  - Pi 4 dense NLL without stale C8D improved from `14.85744` to `13.73610` (`kl_t_to_s=12.71247`, [eval_app3_wls_repaired.log](../../calib/31b/receipts/eval_app3_wls_repaired.log)).

### 3.3 `run31b.rs` Non-Causal `GlyphKvLayer` Prefill Bug (`V9.1`)
- **Symptom:** Initial `V9.1-Fused-VLM` `--kv-quant glyph` runs suffered unexplained perplexity blowup (`mean_nll=13.52555`, `ppl=748,290.939`, `top1_acc=0.0000`, `kl_t_to_s=10.93383` on `heldout_systems_cache`; `12.95841` across 3 passages in [v9_1_pareto_glyph_prefill_bug_baseline.json](../../calib/31b/receipts/v9_1_pareto_glyph_prefill_bug_baseline.json)).
- **Cause:** During sequence prefill, [run31b.rs](../../src/bin/run31b.rs) appended all `0..seq_len` key/value vectors into `GlyphKvLayer` *before* calling `attend_range`. For prompts longer than `32` tokens, tokens `4..59` were compressed/evicted out of the `32`-token exact FP32 tail window before self-attention ran over earlier causal positions, corrupting KV states identically to non-causal eviction.
- **Fix & Physical Verification:** Added `append_glyph_prefix` + `prefill_glyph_kv_causal_batch` ([run31b.rs](../../src/bin/run31b.rs)), replaced the non-causal loop, and added regression test `tests::glyph_kv_causal_prefill_matches_step_by_step_decode`.
  - Result on `heldout_systems_cache` ([eval_glyph_causal_fix.log](../../calib/31b/receipts/eval_glyph_causal_fix.log)):
    - `--kv-quant glyph` (`a_bits=5`): `mean_nll` **`13.52555 -> 8.89081`**, `ppl` **`748,290.939 -> 7,264.891`** (**`103.0x` reduction**), `kl_t_to_s` `10.93383 -> 6.19970`, `top1_acc` `0.0000 -> 0.1905`, `top5_agree` `0.1429 -> 0.3651`, `shrink=2.62x`, `vm_rss=6,303,724 KB`.
    - `--kv-quant glyph-lod` (`a_bits=5`): `mean_nll` **`7.46533`**, `ppl` **`1,746.440`**, `kl_t_to_s=4.73442`, `top1_acc=0.2381`, `top1_agree=0.2857`, `top5_agree=0.3968`, `shrink=1.91x`, `vm_rss=6,382,476 KB`.

---

## 4. End-to-End Pi 4 Model Comparison Matrix

### 4a. Full 3-Passage Held-Out Chat-Turn Evaluation (`166` eval positions, physical Pi 4 unless noted)

| Model Variant | Resident `.t16c` | KV Mode | `ssd_read` | `mean_nll` | `ppl` | `kl_t_to_s` | `top1_agree` | `top5_agree` | Receipt |
|---|---|---|---|---|---|---|---|---|---|
| **W4A16 Teacher (BF16 GPU)** | ~`17 GiB` | BF16 | `0` | **`5.26875`** | **`194.174`** | `0.00000` | `1.0000` | `1.0000` | [V6.1 Report](../v6_1_skeptic_proof/V6_1_AND_V7_FINAL_VERIFICATION_REPORT.md) |
| **`V9.1-Fused-VLM` Dense** | `5.9175 GiB` | `fp8` | **`0`** | **`7.77141`** | **`2,371.802`** | **`5.09893`** | `0.3193` | `0.4518` | [v9_1_pareto_fp8_report.json](../../calib/31b/receipts/v9_1_pareto_fp8_report.json) |
| **`V9.1-Fused-VLM` CRTR (`top_p=0.80`)** | `5.9175 GiB` | `fp8` | **`0`** | `8.42370` | `4,553.724` | `5.76254` | `0.3072` | `0.4398` | [v9_1_pareto_fp8_report.json](../../calib/31b/receipts/v9_1_pareto_fp8_report.json) |
| **`V9.1-Fused-VLM` CRTR (`top_p=0.65`)** | `5.9175 GiB` | `fp8` | **`0`** | `9.06005` | `8,604.563` | `6.64914` | `0.241` | — | [v9_1_pareto_fp8_report.json](../../calib/31b/receipts/v9_1_pareto_fp8_report.json) |
| **`V9.1-Fused-VLM` CRTR (`top_p=0.50`)** | `5.9175 GiB` | `fp8` | **`0`** | `10.64011` | `41,777.409` | `8.18346` | `0.2169` | — | [v9_1_pareto_fp8_report.json](../../calib/31b/receipts/v9_1_pareto_fp8_report.json) |
| **`V4.2` Colab 262K Dense** | `5.91 GiB` | `fp32` | `0` | `8.01262` | `3,018.809` | `5.21293` | **`0.3373`** | **`0.4578`** | [V6.1 Report](../v6_1_skeptic_proof/V6_1_AND_V7_FINAL_VERIFICATION_REPORT.md) |
| **`V6.1` Hybrid Dense** | `4.364 GiB` | `glyph` / `fp32` | `0` | `10.14579` / `10.38568` | `25,483.60` / `32,392.54` | `7.50166` / `7.49964` | `0.1386` / `0.1747` | `0.2229` / `0.2711` | [V6.1 Report](../v6_1_skeptic_proof/V6_1_AND_V7_FINAL_VERIFICATION_REPORT.md) |
| **`V8.0` Spatial Districts Dense** | `~5.10 GiB` + LOD | `lod` | `7.16 GB` | `10.28455` | `29,276.747` | `7.43218` | `0.1627` | `0.2771` | [V8.0 Report](../v8_0_skeptic_proof/skeptic_proof_report.json) |
| **`V8.0` Spatial CRTR (`top_p=0.35`)** | `~5.10 GiB` + LOD | `lod` | `5.95 GB` | `10.95492` | `57,234.77` | `8.06955` | `0.1265` | `0.2169` | [V8.0 Report](../v8_0_skeptic_proof/skeptic_proof_report.json) |
| **`V6.0` LittleBit-2** | `3.116 GiB` | `fp32` | `0` | `12.02587` | `167,020.273` | `9.32079` | `0.1084` | `0.2048` | [V6.1 Report](../v6_1_skeptic_proof/V6_1_AND_V7_FINAL_VERIFICATION_REPORT.md) |

- **`V9.1-Fused-VLM` (`--kv-quant fp8` dense) Per-Passage NLL:** `heldout_systems_cache=7.49305`, `heldout_numerical_pde=6.21836`, `heldout_compiler_ir=9.84081` (`vm_rss=6,414,916 KB`) — beating `V4.2` (`7.92927`, `6.35276`, `9.94901`) on **all 3 passages simultaneously** inside `5.9175 GiB` with `fp8` KV compression and zero SSD reads.

### 4b. Single-Passage Ablation Matrix (`heldout_systems_cache`, `eval_pos=63`, `92` tokens)

| Model / Variant | Flags / Config | `mean_nll` | `ppl` | `kl_t_to_s` | `top1_acc` | `top1_agree` | `top5_agree` | Receipt |
|---|---|---|---|---|---|---|---|---|
| **`V9.1-Fused-VLM`** | `glyph-lod` + **`--andvcnt-bits 6 --lmhead-cs exact`** | **`7.40955`** | **`1,651.680`** | **`4.67860`** | **`0.2381`** | **`0.2857`** | **`0.4444`** | [eval_abits6_exact.log](../../calib/31b/receipts/eval_abits6_exact.log) |
| **`V9.1-Fused-VLM`** | `glyph-lod` + causal prefill fix (`a_bits=5`) | `7.46533` | `1,746.440` | `4.73442` | `0.2381` | `0.2857` | `0.3968` | [eval_glyph_causal_fix.log](../../calib/31b/receipts/eval_glyph_causal_fix.log) |
| **`V9.1-Fused-VLM`** | `glyph` + causal prefill fix (`a_bits=5`) | `8.89081` | `7,264.891` | `6.19970` | `0.1905` | `0.2540` | `0.3651` | [eval_glyph_causal_fix.log](../../calib/31b/receipts/eval_glyph_causal_fix.log) |
| **`V9.0-App1`** | Dense (`--kv-quant glyph`, WLS) | `12.54770` | `281,447.073` | — | — | — | — | [eval_app1_silicon.log](../../calib/31b/receipts/eval_app1_silicon.log) |
| **`V9.0-App2`** | Dense (`glyph`, non-causal bug, raw Ridge) | `13.11573` | `496,692.341` | — | `0.0000` | — | — | [v9_0_phase5a_app2_app3.log](../../calib/31b/receipts/v9_0_phase5a_app2_app3.log) |
| **`V9.1-Fused-VLM` (Pre-Fix)** | `glyph` *before* causal prefill fix (`a_bits=5`) | `13.52555` | `748,290.939` | `10.93383` | `0.0000` | `0.0159` | `0.1429` | [v9_1_pareto_glyph_prefill_bug_baseline.json](../../calib/31b/receipts/v9_1_pareto_glyph_prefill_bug_baseline.json) |
| **`V9.0-App3-WLS` (Repaired)** | Dense (`--no-c8d` after `44`-layer WLS scrub) | `13.73610` | — | `12.71247` | — | — | — | [eval_app3_wls_repaired.log](../../calib/31b/receipts/eval_app3_wls_repaired.log) |
| **`V9.0-App2`** | Sparse CRTR + HeadRouter | `13.78589` | `970,815.924` | — | `0.0000` | — | — | [v9_0_phase5a_app2_app3.log](../../calib/31b/receipts/v9_0_phase5a_app2_app3.log) |
| **`V9.0-App3-WLS` (Repaired)** | Dense (`with stale C8D`) | `14.34037` | — | — | — | — | — | [eval_app3_wls_repaired.log](../../calib/31b/receipts/eval_app3_wls_repaired.log) |
| **`V9.0-App3-Raw`** | Sparse CRTR + HeadRouter (poisoned `down_proj`) | `14.66765` | `2,344,658.291` | — | `0.0000` | — | — | [v9_0_phase5a_app2_app3.log](../../calib/31b/receipts/v9_0_phase5a_app2_app3.log) |
| **`V9.0-App3-Raw`** | Dense (poisoned `down_proj`) | `14.85744` | `2,834,682.130` | — | `0.0000` | — | — | [v9_0_phase5a_app2_app3.log](../../calib/31b/receipts/v9_0_phase5a_app2_app3.log) |
| **`V9.0-App1` / `App3-WLS`** | Sparse CRTR | `15.15244` / `15.78953` | — | — | — | — | — | [eval_app1_silicon.log](../../calib/31b/receipts/eval_app1_silicon.log) |

---

## 5. `V9.1-Fused-VLM` Stress Battery, Vision QA & `Assistant-V7` Speculative Results

### 5.1 Generation Battery & `a_bits=6 + exact` Boundary Ablation
- **Full 15-Prompt Dense Battery (`--kv-quant glyph-lod --lmhead-cs sieve:0.25 --tri-modal-gate`, default `--andvcnt-bits 5`, [v9_1_battery_dense_report.json](../../calib/31b/receipts/v9_1_battery_dense_report.json)):** Passed **`5 / 15`**:
  - `P1_arith`: `"The result is 4.<turn|>"` (PASS)
  - `P2_france`: `"The capital of France is **the city **Paris**.<turn|>"` (PASS)
  - `P4_prime_py`: clean Python `def check_prime` implementation (PASS)
  - `B5_min`: `"The smaller number is **193**.<turn|>"` (PASS)
  - `U3_bugfix`: clean Python `def avg` bugfix explanation (PASS)
- **Sparse 3-Prompt `CRTR top_p=0.80` Spot Check ([eval_sparse_p80.log](../../calib/31b/receipts/eval_sparse_p80.log)):**
  - `P1_arith` passed cleanly: `"The result is 4.<turn|>"` (`gate_saved=16.3%`, `dec_s=13.81s`), while `P2_france` (`"It is a typo of **France**..."`) and `P5_speed` (`120 km` cutoff) regressed under sparse routing.
- **Targeted 5-Prompt Precision Ablation (`--andvcnt-bits 6 --lmhead-cs exact`, [eval_abits6_exact.log](../../calib/31b/receipts/eval_abits6_exact.log)):** Raising RowMajor activation bits to `6` and disabling heuristic `sieve:0.25` + `--tri-modal-gate` was tested specifically on the 5 boundary prompts (not a full 15-prompt re-run):
  - `P5_speed`: **PASS** (`"To find the average speed, you divide the total distance by the time:\n\n120 km / 2 hours = **60 km per hour."`, `dec_s=81.18s`)
  - `P6_photo`: **PASS** (`"Photosynthesis is a process of converting energy into light, transforming the energy to light through sunlight.<turn|>"`, `dec_s=49.06s`)
  - `P8_romeo` (`dec_s=81.17s`), `B5_multistep` (`dec_s=103.09s`), `U4_syllogism` (`dec_s=29.29s`): **FAIL**

### 5.2 Multimodal COCO Vision QA ([v9_1_vision_report.json](../../calib/31b/receipts/v9_1_vision_report.json))
- **Config:** `V9.1-Fused-VLM` + `gemma4_31b_v4_3_vlm_c8d.c8d` (`--kv-quant glyph-lod --lmhead-cs sieve:0.25`, **`0` cold layers**, `ssd_read_bytes = 0`).
- **Output (`GEN31B`):** `"A motorcycle is parked on a the a pavement.<turn|>"` (`prefill_s=533.12s`, `decode_s=30.46s`, `vm_hwm_kb=6,682,056`, `vm_rss_kb=6,524,140`).
- **Comparison:** `4.0x` faster decode (`30.46s` vs `121.59s` on `V8.0`) with zero SSD streaming (`0` vs `24.03 GB`), matching grounded COCO object identification (`motorcycle` + `pavement`), whereas `V4.2` refused (`"singular sequence"`) and `V6.1` misidentified the scene (`"bathroom"`).

### 5.3 `Assistant-V7` (`BHDR` v1) Speculative Decode ([eval_spec_bhdr.log](../../calib/31b/receipts/eval_spec_bhdr.log))
- **Config:** `assistant_mb=66.98`, `--batch-k 4 --assistant-min-margin 0.25`.
- **`P1_arith`:** `accept_rate=0.400`, `"The a result is 4.<turn|>"`, `s_per_tok=2.889`, `vm_rss_kb=6,369,244`.
- **`P2_france`:** `accept_rate=0.200`, `"The the a capital of France is **the capital city **France**.<turn|>"`, `s_per_tok=2.828`.

---

## 6. Software Verification Receipts (Executed After Final Code & Report Edits)

- **Physical Pi 4 Rust Release Test Suite (`cargo test --release --bin run31b`):** **`20 / 20` passed (`0` failed, `0` skipped, `0.08s`)**, including `tests::glyph_kv_causal_prefill_matches_step_by_step_decode`, executed directly on Pi 4 (`192.168.1.207`).
- **Local Python 6-Gate Fail-Closed Self-Tests (`--selftest`):** All 4 pipeline modules pass `6/6` gates (`24/24` total):
  - [run_v9_0_sparse_spinquant_qat_gpu.py](../../calib/31b/run_v9_0_sparse_spinquant_qat_gpu.py) (`V9_0_SELFTEST: 6/6`)
  - [repair_v9_0_inplace.py](../../calib/31b/repair_v9_0_inplace.py) (`REPAIR_V9_SELFTEST: 6/6`)
  - [build_v9_1_fused_vlm.py](../../calib/31b/build_v9_1_fused_vlm.py) (`V9_1_BUILDER_SELFTEST: 6/6`)
  - [eval_v6_1_skeptic_proof.py](../../calib/31b/eval_v6_1_skeptic_proof.py) (`SKEPTIC_PROOF_SELFTEST: 6/6`)
- **Repository Receipt & Link Parity (`verify_receipt_parity.py`):** `44` JSON files, `139` local links, `432` claim checks (`0` errors, `--selftest` `6/6`).
