# Gemma 4 31B `V9.2-CliffFix-VLM` & NEON `COLBLOCK` Strip-Mining Proof Report

**Timestamp:** `2026-11-25 EST`  
**Target Silicon:** Physical Raspberry Pi 4 Model B (`4x Cortex-A72 @ 1.80 GHz`, Debian `aarch64`, `7.6 GiB` RAM, `/mnt/SHE-VAULT`) + Windows x86_64 AVX2 Verification  
**Compute Budget:** **`0` Colab Compute Units** (built, calibrated, and verified locally on Raspberry Pi 4)  
**Settings:** `--andvcnt-bits 6 --lmhead-cs exact --kv-quant glyph-lod` (and `fp8` Pareto)

---

## 1. Root Causes Diagnosed & Fixed

1. **NEON `COLBLOCK` (`0x20`) DRAM-Stride Stall (`neon_runner/neon_stream_accelerator_v2.c`):**
   - Replaced single-group `acc_strip[32][4][16]` inside `gemm_bitserial_tile16_sparse_batch` when `is_colblock && n_active_tiles >= 4` with two L1D strip-mining regimes (`COL_STRIP=32`, `acc_strip[32][4][16]` for `kc <= kc_inner`; `COL_STRIP_LG=16`, `acc_strip[16][24][16]` across `s0 = 0..slots` step 4 for `kc > kc_inner`, preserving exact `dual=0/1` and `need_sum`/`need_norm` parity).
   - **Measured Pi 4 Microbenchmark (`abits=6`, 4 threads):**
     - `21504x5376 2b_gauss`: `k=1` **`8.49 ms`** (vs prior `11.67 ms`, **`-27.2%`**), `k=4` **`14.70 ms`** (`0.881x` vs `TileMajor`), `k=16` **`53.91 ms`** (vs prior `91.96 ms`, **`-41.4%`**, `0.959x` vs `TileMajor`).
     - `21504x5376 3b`: `k=1` **`13.08 ms`** (`-24.5%`), `k=16` **`76.94 ms`** (`0.966x`).
     - `5376x21504 2b_gauss`: `k=1` **`9.52 ms`** (`-27.4%`), `k=16` **`73.40 ms`** (`0.945x`).
     - `5376x21504 3b`: `k=1` **`13.11 ms`** (`-34.2%`), `k=16` **`99.91 ms`** (`0.992x`).
   - Rebuilt Windows `neon_runner/neon_stream_accelerator.dll` (`177,152 B`) and verified `100/100` Windows `--release` Rust tests alongside Pi 4 `neon_test_v2` Gates `A..P`.

2. **Empirically Pinpointed 1-Bit GeGLU & `L53` Cliff Layers (`build_v9_2_cliff_fix_vlm.py`):**
   - A full 60-layer causal `V9.1` scan (`abits=6` NEON vs W4A16 Teacher) exposed the exact worst-deficit layers: `L49` (`mlp=0.71541`), `L51` (`0.74035`), `L53` (`0x05 tern5 rk=0/0`, `d_p=0.87988`, `mlp=0.78031`), `L50` (`0.85878`), `L34` (`0.86867`), `L43` (`0.87016`), `L46` (`0.88506`), `L44` (`0.88984`), `L45` (`0.89545`), `L40` (`0.89874`), `L48` (`0.89965`), `L38` (`0.91401`), `L09` (`0.92253`, `nr=0.7493`), while keeping high-fidelity `1b_rm` layers (`L12..L27`: `0.98626..0.99911`) intact.
   - Upgraded `12` full MLPs (`9, 34, 38, 40, 43, 44, 45, 46, 48, 49, 50, 51`) -> `2b_gauss` (`hblk=256` chunked `libgauss_gptq_v2.so`, AR row scaling, held-out `rk=16` ActALS, lossless `pi_l` GeGLU channel clustering) + `L53.down_proj` (`2b_gauss`, preserving `tern5` `0.940/0.916` on `g/u`), plus calibrated Wiener norms, fresh 60-layer `CRTR` (`r=64 + MoD`) & `HeadRouter` (`r=32 + MoD`, uniform `hd_s=256`), and `V9.2` `C8DB` (preserving `V4.3-VLM` `embed_vision` + accepting `L39.down_proj` & `L53.down_proj`).

---

## 2. Sealed Artifacts & SHA-256 Digest ([SHA256SUMS_V9_2.txt](../../calib/31b/receipts/SHA256SUMS_V9_2.txt))

| Artifact | Size | SHA-256 | Notes |
|---|---:|---|---|
| `gemma4_31b_v9_2_cliff_fix_vlm.t16c` | `6,852,147,072 B` (`6.3816 GiB`) | `0ea2816878ef5f3187330c30bbe3677e6744052eac74ef40920656fe167e2971` | `12` full + `L53.down_proj` `2b_gauss`, causal `L59 cos_t2=0.92792` |
| `gemma4_31b_v9_2_c8d.c8d` | `9,093,504 B` | `3fde4f756e6c473e76e985a71accbe167812fcaca9d22d224048c9c1947c1340` | `embed_vision` (`r=32`) + `L39.down_proj` + `L53.down_proj` (`r=16`) |
| `gemma4_31b_v9_2_router.crtr` | `21,360,516 B` | `338a7adcf2e06a41ac4f5f070df1543d276e0db7aa984e43393160683fd6c127` | `60`-layer MLP CRTR (`crtr_mean_recall=0.69978`) |
| `gemma4_31b_v9_2_head_router.crtr` | `10,737,156 B` | `b5f4e0bfe1867848269cfb5e7199262cd7842fe6ec5d30f48ae025d2f1062d29` | `60`-layer 32-head Attention router (`head_crtr_mean_recall=0.75849`) |

---

## 3. Physical Pi 4 Measurements (`--andvcnt-bits 6 --lmhead-cs exact`, `ssd_read_bytes = 0`)

### A. Held-Out 3-Passage Teacher NLL / KL Pareto ([v9_2_pareto_report.json](../../calib/31b/receipts/v9_2_pareto_report.json))

| Model / Mode | Mean NLL | PPL | `kl_t_to_s` | Top-1 Agree vs Teacher | `gate_saved_pct` | Peak `VmHWM` (kB) |
|---|---:|---:|---:|---:|---:|---:|
| **W4A16 GPU Teacher (`teacher_lgt1`)** | `5.26875` | `194.17` | `0.00000` | `1.0000` | — | — |
| `V9.1-Fused-VLM dense_fp8` *(baseline)* | `7.77141` | `2371.80` | `5.09893` | `0.3193` | — | `6,414,916` |
| `V9.1-Fused-VLM crtr_p0.80_fp8` | `8.42370` | `4553.72` | `5.76254` | `0.3072` | `0.0%` | `6,431,940` |
| **`V9.2 dense_fp8`** | **`7.68050`** | **`2165.70`** | **`4.80107`** | `0.30724` | — | `6,900,412` |
| **`V9.2 dense_glyph-lod`** | **`7.69618`** | **`2199.93`** | **`4.82309`** | **`0.31325`** | — | `6,876,548` |
| **`V9.2 crtr_p80_glyph-lod`** | **`7.74860`** | **`2318.32`** | `5.13520` | `0.30122` | `0.0%` *(92-tok union)* | `6,897,592` |
| `V9.2 crtr_p80_fp8` | `7.75317` | `2328.94` | `5.14877` | `0.30122` | `0.0%` | `6,924,940` |
| `V9.2 crtr_p75_hr85_glyph-lod` | `7.97925` | `2919.74` | `5.39888` | `0.30725` | `0.1%` | `6,908,284` |
| `V9.2 crtr_p75_hr85_fp8` | `7.98560` | `2938.34` | `5.41806` | `0.30123` | `0.1%` | `6,935,676` |

### B. 15-Prompt Deterministic Battery & Decode Speed ([v9_2_battery_dense_report.json](../../calib/31b/receipts/v9_2_battery_dense_report.json), [v9_2_battery_sparse_report.json](../../calib/31b/receipts/v9_2_battery_sparse_report.json))

- **`V9.2` Dense (`glyph-lod`):** **`8 / 15` passed** (up from **`5 / 15`** on `V9.1` and **`7 / 15`** on `V4.2`!), mean decode **`2.768 s/tok`**, `vm_hwm_kb = 6,866,980`, `ssd_read_bytes = 0`.
  - Passed (`8`): `P1_arith` (`"The answer is 4.<turn|>"`), `P2_france` (`"...Paris.<turn|>"`), `P4_prime_py`, `P5_speed` (`"**120 km / 2 hours = 60 km/h"`), `P6_photo` (`"Photosynthesis is the process of using light to create a chemical reaction.<turn|>"`), `B5_min` (`"**193**.<turn|>"`), `U3_bugfix`, `U4_syllogism` (`"Based on the provided text, **No**.<turn|>"`).
  - Remaining edge limits (`7`): `P3_hamlet_quote`, `P7_bsearch`, `P8_romeo`, `B5_hamlet`, `B5_multistep` (`"12 * 8 = 96\n96 - 27 = **53**<turn|>"`), `U1_algebra`, `U2_units`.
- **`PROFILE31B` Dense Breakdown (`1,709.67s` across 15 runs):** `attn_s=32.80%`, `gate_s=18.07%`, `up_s=18.22%`, `down_s=22.12%`, `norm_kv_s=4.54%`, `lmhead_s=4.25%`.
- **`V9.2` Sparse (`crtr80_hr85`, `--shared-mlp-blocks 8`):** **`5 / 15` passed** (`P1`, `P4`, `P5`, `B5_min`, `U4_syllogism`), mean decode **`2.541 s/tok`** (**`-8.2%`** faster vs Dense, `18.12%` mean MLP gate blocks skipped `[16.3%..18.7%]`), `vm_hwm_kb = 6,898,852`, `ssd_read_bytes = 0`.

### C. Multimodal COCO Vision QA (`frame_01_coco.png`, [v9_2_vision_report.json](../../calib/31b/receipts/v9_2_vision_report.json))

- Output: **`"A motorcycle is parked on a grassy area.<turn|>"`** (zero stutter vs `V9.1`'s `"a the a pavement"`).
- Metrics: `ViT=932.38s`, `prefill_s=581.38s`, `decode_s=29.75s` (`3.306 s/tok`, `10` tokens), `KV31B GlyphLod shrink=3.47x` (`156.8 MB` vs `544.3 MB` raw), `vm_hwm_kb=6,953,508` (`6.631 GiB`), `vm_rss_kb=6,893,284`, `ssd_read_bytes=0`.

### D. `Assistant-V7 Q2 + BHDR` Speculative A/B ([v9_2_spec_ab_report.json](../../calib/31b/receipts/v9_2_spec_ab_report.json))

- Tested across `P1_arith`, `P2_france`, `P5_speed`, `B5_multistep` (`abits=6`, `--lmhead-cs exact`, `glyph-lod`):
  - `dense_k1`: `2.695–2.784 s/tok`
  - `bhdr_spec_k4`: **100% bit-exact output parity vs `dense_k1`** on all `4/4` prompts, `accept_rate=0.158–0.286`, `3.219–3.587 s/tok`, `vm_hwm_kb=6,866,936–6,901,296`.
  - `bhdr_spec_k4_tmg`: `accept_rate=0.167–0.333`, `3.014–3.429 s/tok`, `vm_hwm_kb=6,866,956–6,901,408`.
  - Honest throughput finding: on Pi 4 LPDDR4, verifying `k=4` with the V6.1-distilled Assistant at `16–33%` acceptance takes ~`1.18–1.33x` `dense_k1` decode time.
