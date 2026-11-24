# Gemini Neon Handoff Blueprint: Exact Code-Logic & Execution Instructions for `20-50 tok/s` Gemma 4 31B on Raspberry Pi 4

> **Operator Handoff Note for Gemini Neon:** Read all rules in `C:/Users/freed/.gemini/config/rules/*.md` first (Rule 2: **100% Generalized Intelligence Only -- zero prompt/answer hardcoding**, Rule 4: **Fail-Closed Real Silicon Verification**, Rule 13: **Local Windows PC has 6 GB RAM -- run all C/Rust builds & 31B inference on Raspberry Pi 4 `192.168.1.207` via SSH, and GPU training on Colab Pro**).

---

## 0. Live Environment, Paths & Active Tasks

- **Raspberry Pi 4 (`192.168.1.207`):** `ssh -i "C:\Users\freed\.ssh\rpi_key" astronautshe@192.168.1.207`
  - Compiled release binary (`95/95` tests passing): `/home/astronautshe/gemma4_verify/experiments/run270_rust/target/release/run31b`
  - Compiled C NEON accelerator (`self_test` passing): `/home/astronautshe/gemma4_verify/experiments/run270_rust/libneon_stream_accelerator.so`
  - Patriot P210 256 GB SSD: `/mnt/SHE-VAULT/` (`gemma4_v7_0/` 100% verified; `gemma4_v8_0/` finishing file 24/24 in task `task-10584`, followed by `gemma4_v7_5/`).
- **Colab Cleanup:** Page 1 (`rivera1984feb@gmail.com`) is already disconnected (`0` sessions). As soon as `task-10584` prints `ALL_BUNDLES_VERIFIED_ON_PI4_SSD`, message browser subagent `9c642582-167b-4ee5-8522-2021fb7ac63d` to disconnect & delete Page 4 (`dannybouldiez2@gmail.com`).

---

## 1. Six-Step Code-Logic Execution Sequence

1. **Step 1 -- Complete SSD Verification & Run V8.0 / V7.0 Silicon Proof:**
   - Run `sha256sum -c SHA256SUMS.txt` in `/mnt/SHE-VAULT/gemma4_v8_0` and `/mnt/SHE-VAULT/gemma4_v7_0`, disconnect Colab Page 4, and run `run31b` (`--spec-tree 16` + `Assistant-V8 Q2` + `BHDR-384`) + `eval_v6_1_skeptic_proof.py` (`nll-pareto`, `battery`, `vision` with `/mnt/SHE-VAULT/frame_01_coco.png`).
2. **Step 2 -- Single-Pass `kc_step = 24` + Draft-Seeded Exact Cauchy-Schwarz LM-Head (`neon_stream_accelerator_v2.c:5995`, `t16c.rs:1654`, `run31b.rs:4856`):**
   - Add `lmhead_cs_seeded_tile16_batch(..., const int32_t* seed_row_ids)` with `kc_step = dual ? 12 : 24` on ARM64 NEON.
   - Evaluate exact dot product of each tree node's drafted child token first (`8.7 KB`/node) to seed `thresh_k[kk]`, shrinking `p1_blks` to `3` and pruning `~94%` of the `539 MB` LM-head (`~10 ms` total for 16 nodes) with **`0 ULP` greedy argmax loss**.
3. **Step 3 -- Depth-Discounted Tree-Budgeted Cortical Routing (`cortical_router.rs`):**
   - Add `CorticalRouterLayer::predict_tree_masks` keeping `0..8` shared blocks (`2,048` dims) + `6..8` routed blocks (`14..16 / 84` blocks of `256` = `16.7-19.0%` active MLP), protecting `t = 0` (`depth = 0` root) first and weighting deeper tree nodes by `0.7^depth` so 16-node tree passes never exceed `~500 MB` DRAM.
4. **Step 4 -- Language-U V9.0 True Sparse-Active 31B + Global SpinQuant + Reasoning QAT (`calib/31b/run_v9_0_sparse_spinquant_qat_gpu.py` on Colab A100):**
   - **4A (Lossless Neuron Co-Activation Permutation):** Permute `ff = 21,504 = 84 x 256` so top `2,048` general neurons occupy `Blocks 0..8` (Always-On Shared Trunk) and `19,456` specialist neurons cluster into `76` co-activating `256`-wide `COLBLOCK` cortical columns (`Blocks 8..84`).
   - **4B (Zero-Cost Global SpinQuant $R_1 \in O(5376)$):** Fold RMSNorm gains into linear inputs and fuse orthogonal $R_1$ into `embed`, `q/k/v/o`, `gate/up/down`, and `lm_head` (`0` runtime FLOPs on Pi 4; `+3.5-5.0 dB` sub-2-bit SNR).
   - **4C (End-to-End Sparse-Route QAT + `C8DB`):** Train all 60 layers end-to-end under the exact `8 Shared + Top-6 Routed` (`14/84`) mask (`T16C_COLBLOCK_BIT` on `down_proj`, `TileMajor` on `gate/up`) with Teacher-KL (`T=1,2`) + Compass-Magnetism cosine/log-norm loss + load-balancing loss on strictly decontaminated VLM + STEM/math/code reasoning trajectories.
   - **4D (4-Step Rollout EAGLE-3 `Assistant-V9`):** Train `assistant_v9_body_q2.t16c` (`~58 MB`) + `assistant_v9.bhdr` (`2.56 MB`) with 4-step self-feeding rollout on V9.0's rotated `L59` hidden states to hit `6.0-8.5` accepted 31B tokens per `~165 ms` tree pass (**`36-48 tok/s`**).
5. **Step 5 -- `6-Sense` / `TriModal` Dynamic Rescue:** Widen `routed_top_k` (`6 -> 16`) + engage `C8DB` + full exact LM-head automatically on low-margin hard reasoning steps.
6. **Step 6 -- Full Silicon Proof & Release:** Verify bit-exact tree vs `k=1` parity, `20-50 tok/s`, `VmRSS <= 6.5 GiB`, and `>= 100%` Gemma 4 31B accuracy on unseen STEM/math/code/VLM benchmarks (`eval_v9_0_world_proof.py`), then tag `v9.0.0-pi4-31b` and push to `v5-qat-dev` and `main`.
