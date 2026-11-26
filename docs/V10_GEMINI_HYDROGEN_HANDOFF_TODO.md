# Language-U V10 Cortical Cascade — Gemini Hydrogen Handoff & Execution Checklist

> **Purpose:** **100% of V10 training, runtime, evaluation, and retrieval code is ALREADY built, unit-tested, and compiled on the Raspberry Pi 4.** Any lightweight model (such as **Gemini Hydrogen**) can execute the mechanical checklist below step-by-step without writing or redesigning any algorithms.

---

## 1. Live System Coordinates & Active Jobs (`2026-11-26`)

| Resource | Exact Coordinate / Status | Safety Invariants |
|---|---|---|
| **Live Colab L4 GPU Run** | `https://colab.research.google.com/drive/134g4r3fmLJH_frviutlBnD6C2L62b2yF` (`Untitled1.ipynb`, account `dannybouldiez2@gmail.com`) | Connected to **NVIDIA L4 (`22.0 GiB` VRAM + `53.0 GB` System RAM)** running `run_v10_cortical_cascade_gpu.py --run --e2e-steps 200 --micro-bs 8 --grad-accum 2` with `COLAB_AUTO_UNASSIGN=1`. **DO NOT** click `Change runtime type` or `Disconnect` while running! Once finished, confirm `0` active sessions so `0` CUs burn idle. |
| **Shared Google Drive Output Folder** | Folder ID: `1Yp_bp3KZSospNVtRTnjIulv8ev14_dpn` (`/content/drive/MyDrive/gemma4_v10_cortical_cascade`) | Already shared (`reader`) with `rivera1984feb@gmail.com`. Query via bare **`gdrive readonly ls 1Yp_bp3KZSospNVtRTnjIulv8ev14_dpn`** (never call `gdrive.cmd` by full path — only bare `gdrive` carries OAuth). |
| **Physical Raspberry Pi 4** | `astronautshe@192.168.1.207` (SSH key: `C:\Users\freed\.ssh\rpi_key`) | `7.6 GiB` LPDDR4, `/mnt/SHE-VAULT/` (`37 GiB` free). Target V10 asset dir: `/mnt/SHE-VAULT/gemma4_v10/` (`gemma4_v10_seed.cngm` already pre-built there). |
| **Windows Control Plane (`Rule 13`)** | `c:\Gemma-4-LanguageU-EXP\Language-U-Gemma4` (branch `v5-qat-dev`) | `6 GB` RAM laptop (`<= 1 GB` free). Never run heavy compute locally or concurrent multi-GB transfers; `finalize_v10_handoff.py` transfers one file at a time and deletes local staging immediately. |

---

## 2. Inventory of Pre-Built & Pre-Verified V10 Code Logic (Nothing Left to Write!)

1. **L4 High-RAM GPU Trainer ([run_v10_cortical_cascade_gpu.py](../experiments/run270_rust/calib/31b/run_v10_cortical_cascade_gpu.py)):**
   - Sub-`6.30 GiB` (`6.2675 GiB` total / `6.0465 GiB` text) Zero-Tern5 layout (`12` 2b-gauss MLPs `L0..1, L48..57`, `46` 1b-rm COLBLOCK MLPs `L2..47`, `L58..59` 3b, all Attention `>= 2b_gauss COLBLOCK`).
   - Dual-Regime 60-layer E2E Logit-KL + `early_exit_l29_norm` QAT (`50%` Dense + `50%` `12/84` Sparse), plus 3-step self-feeding `google/gemma-4-31B-it-assistant` distillation optimized for `0 MB` Target-Embed Sharing.
2. **Rust 3-Tier Cortical Cascade & General ALU Runtime ([run31b.rs](../experiments/run270_rust/src/bin/run31b.rs), [assistant.rs](../experiments/run270_rust/src/assistant.rs), [speculative.rs](../experiments/run270_rust/src/speculative.rs), [alu.rs](../experiments/run270_rust/src/alu.rs)):**
   - **Tier 0:** Zero-DRAM `prompt_suffix_lookup` + multi-branch `CNGM` v1 (`--spec-seed /mnt/SHE-VAULT/gemma4_v10/gemma4_v10_seed.cngm`).
   - **Tier 1:** 4-layer EAGLE-3 `AssistantRuntime` with **`0 MB` Target-Embed Sharing** (`score_shared_lm_head_top_k` reusing host 31B `embed_tokens` + `sieve=0.02` Cauchy-Schwarz).
   - **Tier 2:** Amortized `COLBLOCK` NEON Sparse Tree Verifier (`--cortical-cascade <tau>` swapping `--cascade-mlp-top-p 0.143`, `--cascade-head-top-p 0.25`, `--cascade-mod-thresh 0.55`, `--cascade-mod-zero-kv`).
   - **Tier 3:** Exact `60/60` Dense Pivot Rescue (`margin < tau` rolls KV back 1 slot via `truncate_kv` and runs 100% Dense 31B on the ambiguous fork token).
   - **General First-Principles Math ALU (`--general-alu`):** Pure AST arithmetic + 1-variable linear equation evaluator (`ALU31B`), reported strictly separately from pure neural `GEN31B` (Rule 1 & Rule 2 Law 4).
3. **Pi 4 Pre-Test & Seed Tools ([build_v10_cngm_seed.py](../experiments/run270_rust/calib/31b/build_v10_cngm_seed.py), [pack_official_assistant_pi4.py](../experiments/run270_rust/calib/31b/pack_official_assistant_pi4.py)):** `6/6` `--selftest` PASS.
4. **Pi 4 World-Proof Suite ([eval_v10_world_proof.py](../experiments/run270_rust/calib/31b/eval_v10_world_proof.py)):** `6/6` `--selftest` PASS; checks `VmHWM <= 6.50 GiB`, Teacher NLL/KL, 15-prompt battery (`dense`, `tree_lossless`, `cascade`, `cascade_alu`), `tok/s` speed benchmark, and COCO Vision QA.
5. **Turnkey Handoff Orchestrator ([finalize_v10_handoff.py](../experiments/run270_rust/calib/31b/finalize_v10_handoff.py)):** `6/6` `--selftest` PASS; automates Drive polling, sequential Rule-13 download->Pi 4 SCP->SHA-256 check, and final Markdown report rendering.

---

## 3. Step-by-Step Todo List for Gemini Hydrogen

### [ ] TODO 1 (Optional While Colab Finishes): Run Live Pi 4 Cascade Speed Pre-Test on `V9.2`
While waiting for Colab L4 to finish `V10`, you can already test `--cortical-cascade` on Pi 4 using `V9.2` + the official Google assistant permuted to `V9.2`'s `perm_pi`:

```powershell
ssh -i C:\Users\freed\.ssh\rpi_key -o StrictHostKeyChecking=no astronautshe@192.168.1.207 '
  python3 /home/astronautshe/gemma4_verify/experiments/run270_rust/calib/31b/pack_official_assistant_pi4.py \
    --src /mnt/SHE-VAULT/gemma-4-31B-it-assistant/model.safetensors \
    --target-t16c /mnt/SHE-VAULT/gemma4_v9_2/gemma4_31b_v9_2_cliff_fix_vlm.t16c \
    --orig-safetensors /mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/model.safetensors \
    --out /mnt/SHE-VAULT/gemma4_v9_2/gemma4_31b_official_ast_q2_v92perm.t16c --body-bits 2 &&
  /home/astronautshe/gemma4_verify/experiments/run270_rust/target/release/run31b \
    --t16c /mnt/SHE-VAULT/gemma4_v9_2/gemma4_31b_v9_2_cliff_fix_vlm.t16c \
    --config /mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/config.json \
    --tokenizer /mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/tokenizer.json \
    --so /home/astronautshe/gemma4_verify/experiments/run270_rust/libneon_stream_accelerator.so \
    --cortical-router /mnt/SHE-VAULT/gemma4_v9_2/gemma4_31b_v9_2_router.crtr \
    --head-router /mnt/SHE-VAULT/gemma4_v9_2/gemma4_31b_v9_2_head_router.crtr \
    --kv-quant glyph-lod --threads 4 --andvcnt-bits 6 \
    --speculative --cortical-cascade 3.5 --spec-tree 16 --spec-tree-depth 10 --spec-ngram 6 \
    --spec-seed /mnt/SHE-VAULT/gemma4_v10/gemma4_v10_seed.cngm \
    --spec-assistant /mnt/SHE-VAULT/gemma4_v9_2/gemma4_31b_official_ast_q2_v92perm.t16c \
    --prompt-text "Write a concise Python function to compute the greatest common divisor." --max-tokens 48
'
```
Inspect `CASCADE31B ... draft_ms=... tier2_ms=... tier3_ms=... tok_per_s=...` and `MEM31B ... vm_hwm_kb=...`!

---

### [ ] TODO 2: Poll Google Drive Until All 9 `V10` Artifacts Land & Verify `0` Colab CU Burn
Run this lightweight check periodically (use the `schedule` tool e.g. `DurationSeconds=600`, never `sleep`!):

```powershell
python c:\Gemma-4-LanguageU-EXP\Language-U-Gemma4\experiments\run270_rust\calib\31b\finalize_v10_handoff.py --mode check-drive
```
- While `ready_for_download` is `false` (exit code `2`), Colab is still running.
- As soon as `ready_for_download` is **`true`** (exit code `0`, all 9 files including `v10_report.json` and `SHA256SUMS_V10.txt` present):
  1. Check `https://colab.research.google.com/drive/134g4r3fmLJH_frviutlBnD6C2L62b2yF` via `browser` subagent to confirm `COLAB_AUTO_UNASSIGN=1` disconnected the L4 runtime. **If it is still connected, immediately click `Runtime -> Disconnect and delete runtime` so `0` CUs burn idle!**

---

### [ ] TODO 3: Pull All `V10` Artifacts Sequentially to Pi 4 & Verify SHA-256
Run the turnkey sequential puller (downloads 1 file at a time, `scp`s to `/mnt/SHE-VAULT/gemma4_v10/`, deletes temp staging on `C:`, saves `v10_report.json` + `SHA256SUMS_V10.txt` into `calib/31b/receipts/`, and runs `sha256sum -c SHA256SUMS_V10.txt` on Pi 4):

```powershell
python c:\Gemma-4-LanguageU-EXP\Language-U-Gemma4\experiments\run270_rust\calib\31b\finalize_v10_handoff.py --mode pull-to-pi4
```
*(Set `NotificationTimeoutSeconds: 600` on `run_command` since transferring `6.27 GiB` over LAN takes ~4–8 minutes.)*

---

### [ ] TODO 4: Run Full Physical Pi 4 World-Proof Suite (`eval_v10_world_proof.py`)
Sync `finalize_v10_handoff.py` if updated, then execute the complete 5-gate hardware proof on Pi 4:

```powershell
ssh -i C:\Users\freed\.ssh\rpi_key -o StrictHostKeyChecking=no astronautshe@192.168.1.207 '
  python3 -u /home/astronautshe/gemma4_verify/experiments/run270_rust/calib/31b/eval_v10_world_proof.py \
    --mode all \
    --out-dir /mnt/SHE-VAULT/gemma4_v10/eval_out
'
```
**Tuning Knobs (if `speed_bench` `tok_per_s < 20.0` or battery accuracy needs adjustment):**
- Look at `CASCADE31B` `draft_ms`, `tier2_ms`, `tier3_ms`:
  - If `tier3_ms` is high (too many dense rescues): lower `--cascade-tau` from `3.5` -> `2.0` or `1.5`.
  - If `tier2_ms` is high: raise `--cascade-mod-thresh` from `0.55` -> `0.70` and lower `--cascade-mlp-top-p` from `0.143` -> `0.119` (`10/84` blocks).
  - If `draft_ms` is high: pass `--use-byte-head` (uses `1.5 MB` `BHDR-512` instead of `project_lmhead_cs`) or `--spec-tree 12 --spec-tree-depth 8`.
- Verify all True-North gates:
  1. `VmHWM <= 6,815,744 kB` (`<= 6.50 GiB`) and `ssd_read_bytes == 0`.
  2. Held-out 3-passage `mean_nll < 7.68050` and `kl_t_to_s < 4.80107`.
  3. 15-prompt unseen battery (`dense`, `tree_lossless`, `cascade` Pure Neural, `cascade_alu`).
  4. `tok_per_s` in `20–50 tok/s` range on `speed_bench`.
  5. Clean grounded COCO Vision QA on `frame_01_coco.png`.

---

### [ ] TODO 5: Pull Pi 4 World-Proof JSON & Render Final Report
Copy `/mnt/SHE-VAULT/gemma4_v10/eval_out/v10_all_report.json` into `experiments/run270_rust/calib/31b/receipts/v10_pi4_world_proof.json` and render the final benchmark report:

```powershell
scp -i C:\Users\freed\.ssh\rpi_key -o StrictHostKeyChecking=no astronautshe@192.168.1.207:/mnt/SHE-VAULT/gemma4_v10/eval_out/v10_all_report.json c:\Gemma-4-LanguageU-EXP\Language-U-Gemma4\experiments\run270_rust\calib\31b\receipts\v10_pi4_world_proof.json
python c:\Gemma-4-LanguageU-EXP\Language-U-Gemma4\experiments\run270_rust\calib\31b\finalize_v10_handoff.py --mode render-report
```

---

### [ ] TODO 6: Check Kaggle `gemma-4-developer-agent` Submission `58662812`
Check `https://www.kaggle.com/competitions/gemma-4-developer-agent/submissions` via `browser` subagent (if logged in) or `kaggle competitions submissions -c gemma-4-developer-agent` to record the final status/LB score of `58662812` (`v11-clean-reasoning`, deadline `2026-12-02`).

---

### [ ] TODO 7: Seal Receipt Parity, Update Docs, Merge `v5-qat-dev -> main` & Tag `v10.0.0`
1. Update `README.md`, `experiments/run270_rust/FRONTIER_RECEIPTS.md`, and `c:\Gemma-4-LanguageU-EXP\PROJECT_STATE_AND_TODO.md` with the verified `V10` numbers.
2. Run full verification battery:
   ```powershell
   Set-Location c:\Gemma-4-LanguageU-EXP\Language-U-Gemma4
   python scripts/verify_receipt_parity.py --selftest
   python scripts/verify_receipt_parity.py
   python -m unittest discover -s tests
   ```
3. Commit, merge cleanly into `main`, tag `v10.0.0`, and push:
   ```powershell
   Set-Location c:\Gemma-4-LanguageU-EXP\Language-U-Gemma4
   git add experiments/run270_rust/calib/31b/receipts/ experiments/run270_rust/benchmarks/v10_final_proof/ README.md experiments/run270_rust/FRONTIER_RECEIPTS.md docs/V10_GEMINI_HYDROGEN_HANDOFF_TODO.md
   git commit -m "Complete Language-U V10 Cortical Cascade Pi 4 World-Proof and seal receipts"
   git push origin v5-qat-dev
   git checkout main
   git merge --no-ff v5-qat-dev -m "Merge Language-U V10 Cortical Cascade release (v10.0.0)"
   git tag -a v10.0.0 -m "Language-U Gemma 4 31B V10 Cortical Cascade"
   git push origin main v10.0.0
   ```
