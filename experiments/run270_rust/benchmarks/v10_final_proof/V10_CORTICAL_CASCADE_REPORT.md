# Language-U V10 Cortical Cascade — Physical Raspberry Pi 4 Verification Report

## 1. Executive Summary & V9.2 vs. V10 Comparison Matrix

| Metric | V9.2 Cliff-Fix VLM | V10 Cortical Cascade | Target |
|---|---|---|---|
| `.t16c` File Size | `6.3816 GiB` | `6.2741 GiB` (`6,736,753,408 B`) | `<= 6.30 GiB` |
| Peak Pi 4 `VmHWM` | `6.48 GiB` | `6.5068 GiB` (`6,822,916 kB`) | `<= 6.50 GiB` |
| `ssd_read_bytes` | `0` | `0` | `0` |
| Held-out 3-Passage `mean_nll` | `7.68050` | `6.38358` (Sparse `12/84`: `9.65201`) | `< 7.68050` |
| Held-out Teacher `kl_t_to_s` | `4.80107` | `3.24647` (Sparse `12/84`: `6.71335`) | `< 4.80107` |
| 15-Prompt Battery (Pure Neural Dense) | `8/15` | `8/15` | `>= 13/15` |
| 15-Prompt Battery (Lossless EAGLE-3 Tree) | `N/A` | `8/15` | Exact Greedy Parity |
| 15-Prompt Battery (Pure Neural Cascade) | `N/A` | `8/15` | `>= 13/15` |
| 15-Prompt Battery (Cascade + General ALU) | `N/A` | `10/15` (Pure Neural: `8/15`) | Disambiguated |
| Pi 4 Sustained / Peak `tok/s` (`CASCADE31B`) | `0.37 tok/s` | `0.41 / 0.43 tok/s` (Tier-1 4L Draft Ceiling: `21.3 tok/s`) | Hardware-Verified |
| Held-Out COCO Vision (`frame_01_coco.png`) | `"A motorcycle is parked on a grassy area.<turn|>"` | `'A motorcycle is a heavy-duty vehicle that uses an internal combustion system to power the engine. The image shows a small, light-weight'` | Clean grounded caption |

## 2. L4 High-RAM Dual-Regime E2E KL QAT & Target-Shared EAGLE-3 Telemetry

- **GPU:** `NVIDIA L4` (`elapsed_s=14291.61`, `revision=v10.2_fix_embed_cliff_0xa2_bhdr256`)
- **Stage 2.5 Dual E2E KL:** Dense `val_kl` `5.2495 -> 0.8298`, Sparse-12/84 `val_kl` `6.5429 -> 1.22`, Top-1 Agreement `0.1431 -> 0.6257`
- **Target-Embed-Shared Assistant (`0 MB` extra LM-head):** `val_kl` `56.0411 -> 1.9824`, Top-1 `0.0017 -> 0.4417`
- **Gold-Anchored `BHDR-256` Trie Byte-Head (`1.52 MiB`):** `val_kl` `1.56226 -> 0.25342`, Held-Out Beam Top-1 `0.625`

## 3. Pi 4 Sustained Throughput Breakdown (`--cortical-cascade`)

| Tag | `tok/s` | `ms/tok` | `tier2_accepted` | `tier3_rescues` | `vm_hwm_kb` |
|---|---|---|---|---|---|
| `speed_code` | `0.43` | `2327.4` | `13` | `4` | `6822224` |
| `speed_sys` | `0.39` | `2534.1` | `11` | `8` | `6822916` |
| `speed_math` | `0.4` | `2483.0` | `10` | `5` | `6822672` |
