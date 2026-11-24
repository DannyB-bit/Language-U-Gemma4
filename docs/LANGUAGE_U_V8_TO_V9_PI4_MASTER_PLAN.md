# Language-U Gemma 4 31B on Raspberry Pi 4: Master Execution Plan (V7.0 / V8.0 -> V9.0)

> **True-North Mission:** Prove empirically on physical **Raspberry Pi 4 Model B (`4x Cortex-A72`, `7.6 GiB` LPDDR4 + Patriot P210 256 GB SSD)** that Language-U shrinks **Gemma 4 31B (`30.7B` dense multimodal VLM)** to run within `<= 6.5 GiB` RAM at **`20-50 tok/s`** with **`>= 100%` of stock Gemma 4 31B generalized intelligence** -- 100% generalized, zero hardcoded benchmark lookups, zero synthetic stubs, and reproducible bit-for-bit from GitHub.

---

## 1. Ground-Truth Milestone Snapshot (`2026-11-23`)

- **Colab Pro GPU Pipelines (`V7.0`, `V7.5`, `V8.0`):** 100% completed, SHA-256 sealed on Drive, `0` GPU compute units active.
  - **V8.0 (`gemma4_v8_0_player_pov`):** `l59_vlm_state_cos = 0.88712`, `post_cell60_val_kl = 1.8267` (`Top-1 = 63.64%`, `Agree = 71.47%`), `Assistant-V8` `Top1Agree = 68.68%` (`BHDR` beam `70.31%`), Held-Out COCO `val[0..3]` mean `cos = 0.8735`, `Top-1 = 61.7%` (`Top-5 = 88.1%`).
  - **V7.0 (`gemma4_v7_0`):** `L59 vlm_cos = 0.81708` -> `0.85052` post-`C8DB`, `crtr_rec = 0.922`, `mod_acc = 0.958`, `Assistant-V7` `Top1Agree = 66.6%` (`6/6` files transferred & SHA-256 verified on `/mnt/SHE-VAULT/gemma4_v7_0/`).
  - **V7.5 (`gemma4_v7_5_spatial_districts`):** All 20 spatial district shards sealed and queued onto `/mnt/SHE-VAULT/gemma4_v7_5/`.
- **Phase I Engine (`C NEON + Rust EAGLE-3 Speculative Tree Runtime`):**
  - Zero-copy `T16C_COLBLOCK_BIT (0x20)` execution in `neon_stream_accelerator_v2.c` and `t16c.rs`.
  - Native ARM64 NEON `gemm_ternary5_tile16_sparse_batch` (`eff_bits == 5`) + `CRTR` sparse mask unlock in `run31b.rs`.
  - Single-pass `kc_step = 24` (single-plane) / `12` (dual-plane) L1-tiled NEON batching across `tile16`, `rowmajor_andvcnt`, and `tern5`.
  - `ByteHeadDecoder::decode_top_k_tokens`, `attn_decode_tree`, `GlyphKvLayer::attend_tree_ephemeral`, zero-copy Assistant `GlyphKV` cross-attention (`draft_tree_with_attend`), `SpecTreeBuilder` + `verify_greedy_tree_batch`, `CRTR` sub-slice district masking, and `--spec-tree <2..24>` in `run31b`.

---

## 2. Six-Stage Execution Roadmap to `20-50 tok/s` & `>=100%` 31B Intelligence

1. **Stage 1 (Done):** Phase I C NEON (`kc_step = 24`, zero-copy `COLBLOCK`, sparse `tern5`) + Rust EAGLE-3 + `BHDR` + `CNGM` speculative prefix-tree runtime (`--spec-tree`).
2. **Stage 2 (In Progress):** Finish SHA-256 verified transfer of `V8.0` & `V7.5` onto `/mnt/SHE-VAULT/`, disconnect final CPU tunnel (`0.00 CU/hr`), and run head-to-head physical Pi 4 benchmarks (`V4.2-Colab` vs `V6.1` vs `V7.0` vs `V7.5` vs `V8.0` + `--spec-tree 16` + Vision QA `frame_01_coco.png`).
3. **Stage 3:** Draft-Seeded Exact Cauchy-Schwarz LM-Head Pruning (`seed_token_ids` in `T16CMat::project_lmhead_cs`), shrinking `262,144 x 5,376` (`539 MB`) tree logit verification to `~35 MB` (`~12 ms`) with `0 ULP` greedy argmax loss.
4. **Stage 4:** **Language-U V9.0 True Sparse-Active 31B Cortical Architecture** -- all `30.7B` parameters resident in RAM (`~5.7 GiB` `mmap` at `~1.5 bpw`), decomposed into always-active SpinQuant attention + `d_shared = 2,048` MLP trunk + **Top-6 of 64 fine-grained routed `COLBLOCK` cortical expert columns** (`~2.2B` active params = `~460-495 MB` active DRAM per tree pass -> **`~120-135 ms` per 16-node verification pass**).
5. **Stage 5:** **Global SpinQuant Orthogonal Rotation + Compass-Magnetism + End-to-End On-Policy Reasoning QAT + Multi-Step EAGLE-3 `Assistant-V9` (`6.0-8.5` accepted tokens/pass = `32-50 tok/s`)** -- zero-cost fused Cayley/Hadamard rotations ($W' = R_1^\top W R_2$), end-to-end Teacher-KL + strictly decontaminated open-domain math/code/science reasoning distillation, 60-layer `C8DB` $S^7$ residual bank, and `6-Sense` / `TriModal` exact consensus rescue (`>= 100%` Gemma 4 31B intelligence).
6. **Stage 6:** Unvarnished physical Raspberry Pi 4 proof suite (`eval_v9_0_world_proof.py`), cryptographic receipt parity audit, cross-document `-V9` versioning audit, and GitHub/HuggingFace release.
