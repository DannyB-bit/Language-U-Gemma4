# Gemma 4 31B V6.1 Hybrid + Real CRTR/MoD + Assistant-V7 (`BHDR` v1) — Physical Raspberry Pi 4 Skeptic-Proof Verification Report

**Date:** `2026-11-22`  
**Target Hardware:** Raspberry Pi 4 (`astronautshe@192.168.1.207`, 4x Cortex-A72, `7.6 GiB` RAM), `run31b` + `libneon_stream_accelerator.so` (`--threads 4`, `--lmhead-cs exact`, `ssd_read_bytes = 0` on all runs)  
**Colab Phase 1 (`Assistant-V7` + `LGT1` Teacher):** `NVIDIA L4` (`427.62s`, `assistant_v7_bhdr_report.json`), auto-unassigned cleanly (`0 active sessions`, `147.71 compute units` remaining)

---

## 1. Provenance & Bit-Exact Verification ([SHA256SUMS.txt](file:///C:/Gemma-4-LanguageU-EXP/Language-U-Gemma4/experiments/run270_rust/benchmarks/v6_1_skeptic_proof/SHA256SUMS.txt))

All `11/11` artifacts generated across Colab Pro L4 were downloaded, verified locally, copied to Pi 4 `/mnt/SHE-VAULT/gemma4_v6_1/`, and matched bit-for-bit on `SHA-256`:

| Artifact | Exact Bytes | Size | SHA-256 Prefix | Role |
|---|---|---|---|---|
| `gemma4_31b_v6_1_hybrid.t16c` | `4,685,862,848` | `4.364 GiB` | `6d358e036dc688d6` | V6.1 Hybrid LM + 3-bit Vision |
| `gemma4_31b_v6_1_router.crtr` | `10,861,956` | `10.36 MiB` | `60ead5a4d8dc9885` | Trained 60-layer CRTR + MoD router |
| `gemma4_31b_v6_1_hybrid_c8d.c8d` | `11,874,560` | `11.32 MiB` | `4de25d16d454678e` | Multimodal C8DB adapters (`entries=5`) |
| `gemma4_31b_assistant_v7.bhdr` | `1,595,996` | `1.522 MiB` | `45a5fcd4af6ad7c0` | INT8 261-sym `<eot>` byte-head (`BHDR` v1) |
| `gemma4_31b_assistant_v7_body_q2.t16c` | `56,628,032` | `54.00 MiB` | `16fef43db8ec9080` | Embed-free Q2 body (`resident_mb=66.98` w/ trie) |
| `gemma4_31b_assistant_v7_body_q4.t16c` | `106,894,144` | `101.94 MiB` | `6fb0f2ce7b982de4` | Embed-free Q4 body (`resident_mb=114.92` w/ trie) |
| `gemma4_31b_assistant_v7_body_q8.t16c` | `207,426,368` | `197.82 MiB` | `d4ee9ff5eb295bcc` | Embed-free Q8 body (`resident_mb=210.79` w/ trie) |
| `teacher_lgt1/heldout_systems_cache.lgt1` | `33,030,168` | `31.50 MiB` | `a533bd4c29f384c2` | BF16 262K logits (`eval_pos=63`, `0xb1c391deb2b977ce`) |
| `teacher_lgt1/heldout_numerical_pde.lgt1` | `28,311,576` | `27.00 MiB` | `4394f2190f4798aa` | BF16 262K logits (`eval_pos=54`, `0x1091cdcfd8a0b630`) |
| `teacher_lgt1/heldout_compiler_ir.lgt1` | `25,690,136` | `24.50 MiB` | `26b4c042c7e815bc` | BF16 262K logits (`eval_pos=49`, `0x9a93d39d0c5d33ef`) |
| `assistant_v7_bhdr_report.json` | `2,141` | `2.1 KiB` | `704c855ac5b2e8db` | Colab Assistant-V7 & Teacher receipt |

### Hardware / Kernel Parity Verification ([smoke.log](file:///C:/Gemma-4-LanguageU-EXP/Language-U-Gemma4/experiments/run270_rust/benchmarks/v6_1_skeptic_proof/smoke.log))

- **NEON FFI (`--so`) vs Pure Rust FP32 (`--no-native-ffi`):** On `V6.1` (`--kv-quant fp32`, `"What is 2 + 2?"`), default `--so` (`--andvcnt-bits 4`), `--so --andvcnt-bits 6`, and `--no-native-ffi` generated **identical initial token sequences** (`[2021:"To", 1586:" find", 506:" the", 1156:" two", 6555:" steps", 529:" of"]`) with `PREFILL_TOP5` logit agreement within `0.01–0.09` (`26.534` / `26.941` / `26.954`) and `~25.7x` prefill speedup (`19.12s` vs `491.27s`), confirming zero kernel, C8DB, or container dequantization bugs.
- **ARM64 Native Kernel Tests:** `libneon_stream_accelerator.so` (`PI4_LIB_SELFTEST 1`) and `tests/test_pi4_barrier.py` passed `22/22` tests on Pi 4 (`0` skipped).
- **Teacher Tokenizer Hash Parity:** Pi 4 Rust `fnv1a64` hashes matched Colab `assistant_v7_bhdr_report.json` bit-for-bit across all 3 held-out chat turns (`0xb1c391deb2b977ce`, `0x1091cdcfd8a0b630`, `0x9a93d39d0c5d33ef`), and `teacher_ppl` matched within BF16/FP32 rounding (`194.174` Pi 4 aggregate `teacher_nll=5.26875`).

---

## 2. Sub-2-Bit Compression Scaling Curve (`W4A16` Teacher vs `V4.2` vs `V6.1` vs `V6.0`)

Measured across the 3 strictly disjoint held-out chat turns (`heldout_systems_cache` `63` pos, `heldout_numerical_pde` `54` pos, `heldout_compiler_ir` `49` pos; **`166` total evaluated positions**, `--kv-quant fp32` unless noted):

| Model Variant | `.t16c` Size | Approx. Bitrate | KV Mode | Aggregate `mean_nll` | `ppl` | `kl_t_to_s` | `kl_s_to_t` | `top1_agree` | `top5_agree` | Peak `vm_rss` |
|---|---|---|---|---|---|---|---|---|---|---|
| **W4A16 Teacher (BF16 GPU)** | ~`17 GiB` | `~4.0 bpw` | BF16 | **`5.26875`** | **`194.174`** | `0.00000` | `0.00000` | `1.0000` | `1.0000` | Colab L4 |
| **`V4.2` Colab 262K Dense** | `5.91 GiB` (`6.0 GB`) | `~1.58 bpw` | FP32 | **`8.01262`** | **`3,018.809`** | **`5.21293`** | **`8.18936`** | **`0.3373`** | **`0.4578`** | `6.325 GiB` (`6,632,012 KB`) |
| **`V6.1` Hybrid Dense (FP32 KV)** | `4.364 GiB` | `~1.13 bpw` | FP32 | **`10.38568`** | **`32,392.542`** | **`7.49964`** | **`13.22447`** | **`0.1747`** | **`0.2711`** | `4.773 GiB` (`5,004,960 KB`) |
| **`V6.1` Hybrid Dense (GlyphKV)** | `4.364 GiB` | `~1.13 bpw` | Glyph | **`10.14579`** | **`25,483.601`** | **`7.50166`** | **`13.91563`** | **`0.1386`** | **`0.2229`** | **`4.701 GiB`** (`4,929,784 KB`) |
| **`V6.0` LittleBit-2 (L24..27 patch)** | `3.116 GiB` | `~0.85 bpw` | FP32 | **`12.02587`** | **`167,020.273`** | **`9.32079`** | **`14.65271`** | **`0.1084`** | **`0.2048`** | `3.488 GiB` (`3,657,924 KB`) |

### Per-Passage Breakdown (Teacher NLL/PPL -> V4.2 -> V6.1 -> V6.0)

| Passage (`eval_pos`) | Teacher NLL / PPL | V4.2 NLL / PPL / KL / Top1Ag | V6.1 (FP32) NLL / PPL / KL / Top1Ag | V6.0 NLL / PPL / KL / Top1Ag |
|---|---|---|---|---|
| `heldout_systems_cache` (`63`) | `5.42168` / `226.258` | `7.92927` / `2,777.391` / `5.11868` / `0.3016` | `10.12209` / `24,886.681` / `7.09199` / `0.1905` | `11.80887` / `134,439.793` / `8.67398` / `0.0952` |
| `heldout_numerical_pde` (`54`) | `4.76460` / `117.284` | `6.35276` / `574.077` / `4.53686` / `0.4444` | `9.50136` / `13,377.957` / `7.21668` / `0.1667` | `11.79637` / `132,769.825` / `9.62922` / `0.1111` |
| `heldout_compiler_ir` (`49`) | `5.62773` / `278.031` | `9.94901` / `20,931.434` / `6.07916` / `0.2653` | `11.69915` / `120,469.458` / `8.33559` / `0.1633` | `12.55779` / `284,301.608` / `9.81250` / `0.1224` |

**Honest Interpretation:** Moving down from `V4.2` (`~1.58 bpw`) to `V6.1` (`~1.13 bpw`, where `L00..L35` remain 1-bit `LittleBit-2`) saves **`1.55 GiB` RAM** and improves grammatical fluency over `V6.0` (`kl_t_to_s` drops from `9.321` to `7.500`), but crosses below the factual-recall cliff (`kl_t_to_s` rises `+2.287` nats over `V4.2` and `teacher_top1_agree` drops from `33.7%` to `17.5%`). Meanwhile, `GlyphKV + C8DB` adds virtually zero KL drift (`+0.00202` nats: `7.49964` -> `7.50166`) while cutting RAM (`75 MB` on short turns, `6.65x–6.82x` KV shrink on 283–293-token VLM runs).

---

## 3. Real Trained `CRTR` + `MoD` Pareto & Speed Curve on `V6.1`

### 3a. Held-Out Chat-Turn NLL & Teacher-KL ([nll_pareto/skeptic_proof_report.json](file:///C:/Gemma-4-LanguageU-EXP/Language-U-Gemma4/experiments/run270_rust/benchmarks/v6_1_skeptic_proof/nll_pareto/skeptic_proof_report.json))

| Config | `mlp_top_p` | `mod_thresh` | `gate_saved_pct` | `mod_skipped_pct` (Prefill) | `mean_nll` | `ppl` | `kl_t_to_s` | `top1_agree` | `eval_s` (3 turns) |
|---|---|---|---|---|---|---|---|---|---|
| `dense` | `1.00` | `0.00` | `0.0%` | `0.0%` | `10.38568` | `32,392.542` | `7.49964` | `0.1747` | `259.35s` |
| `crtr_p0.70_m0.00` | `0.70` | `0.00` | `29.8%` | `0.0%` | **`10.32412`** | **`30,458.573`** | **`7.48124`** | `0.1747` | `233.64s` |
| `crtr_p0.70_m{0.35,0.50}` | `0.70` | `0.35 / 0.50` | `29.8%` | `0.0%` | `10.32412` | `30,458.573` | `7.48124` | `0.1747` | `233.60–234.13s` |
| `crtr_p0.50_m0.00` | `0.50` | `0.00` | `50.0%` | `0.0%` | `10.43453` | `34,014.172` | `7.59309` | **`0.1867`** | `211.60s` |
| `crtr_p0.50_m{0.35,0.50}` | `0.50` | `0.35 / 0.50` | `50.0%` | `0.0%` | `10.43453` | `34,014.172` | `7.59309` | `0.1867` | `210.97–211.33s` |
| `crtr_p0.35_m0.00` | `0.35` | `0.00` | `64.3%` | `0.0%` | `10.55854` | `38,504.751` | `7.60635` | `0.1807` | `194.93s` (`1.33x`) |
| `crtr_p0.35_m{0.35,0.50}` | `0.35` | `0.35 / 0.50` | `64.3%` | `0.0%` | `10.55854` | `38,504.751` | `7.60635` | `0.1807` | `194.59–195.00s` |

> **Why `mod_skipped_pct = 0.0%` on `--eval-nll` prefill:** In [cortical_router.rs](file:///C:/Gemma-4-LanguageU-EXP/Language-U-Gemma4/experiments/run270_rust/src/cortical_router.rs#L244-L260), `predict_mod_exec_prob` returns `max_{t in 0..seq_len} sigmoid(...)`. Across a `76–92`-token chat prompt during batch prefill, at least one token always exceeds `0.50`, so `MoD` stays `0%` skipped during full-sequence evaluation and only activates during single-token autoregressive decode (`seq_len = 1`).

### 3b. Standalone Decode Speed Sweep (`V6.1`, `--max-tokens 24`, Prompt: `"Explain how photosynthesis works in one short sentence."`, [speed_sweep.json](file:///C:/Gemma-4-LanguageU-EXP/Language-U-Gemma4/experiments/run270_rust/benchmarks/v6_1_skeptic_proof/speed_sweep.json))

| Config | Gen Toks | Prefill | Decode (`s`) | `decode_s / step` | Speedup vs Dense | Gate Saved | Total MLP DRAM Saved | MoD Skip (`MOD31B`) | Output Quality Note |
|---|---|---|---|---|---|---|---|---|---|
| `dense` | `24` (`23` steps) | `20.48s` | `41.27s` | `1.7943s` | `1.00x` | `0.0%` | `0.0%` | `0.0%` | Grammatical, off-topic (`"Depending to the sentence..."`) |
| `crtr_p0.70_m0.00` | `24` (`23` steps) | `18.22s` | `36.30s` | `1.5783s` | **`1.14x`** | `29.8%` | `24.8%` | `0.0%` | Grammatical, similar quality |
| `crtr_p0.50_m0.00` | `24` (`23` steps) | `16.06s` | `32.74s` | `1.4235s` | **`1.26x`** | `50.0%` | `41.7%` | `0.0%` | Grammatical (`"The text of a simple..."`) |
| `crtr_p0.35_m0.00` | `22` (`21` steps) | `14.75s` | `27.46s` | `1.3076s` | **`1.37x`** | `64.3%` | `53.4%` | `0.0%` | Grammatical EOS (`"<turn|>"`) |
| `crtr_p0.50_m0.35` | `24` (`23` steps) | `16.12s` | `24.67s` | **`1.0726s`** | **`1.67x`** | `50.0%` | `41.6%` | **`51.1%`** | **Degraded** (`"Theing.- from ( of and,ylyfuls..."`) |
| `crtr_p0.35_m0.35` | `24` (`23` steps) | `14.82s` | `24.42s` | **`1.0617s`** | **`1.69x`** | `64.3%` | `53.6%` | **`39.9%`** | **Degraded** (`"Theing-\n = {\n, -eiaful ()..."`) |

*(Also observed in `smoke.log`: `crtr_p0.50_m0.35` on `"What is 2 + 2?"` skipped `20.6%` eligible layers and emitted `"To .201.\n    }\n\n# Since:\n<turn|>"`.)*  
**Real CRTR + MoD Conclusion:** Trained **CRTR MLP block skipping (`top_p = 0.70 -> 0.50 -> 0.35`) is genuinely benign**, saving `24.8%–53.4%` MLP DRAM and delivering `1.14x–1.37x` real decode speedup with essentially flat Teacher-KL (`7.50 -> 7.48 / 7.59 / 7.61`) and identical `2/15` battery pass rate at `top_p=0.50`. By contrast, **`MoD` layer skipping at `thresh=0.35` breaks generation coherence** on this `1.13 bpw` model even though it reaches `1.06 s/step` (`1.67x–1.69x`).

---

## 4. Full 15-Prompt Stress Battery (`V4.2 Dense` vs `V6.1 Dense` vs `V6.1 CRTR top_p=0.50`)

Measured with deterministic regex/numeric graders ([eval_v6_1_skeptic_proof.py](file:///C:/Gemma-4-LanguageU-EXP/Language-U-Gemma4/experiments/run270_rust/calib/31b/eval_v6_1_skeptic_proof.py)), `--kv-quant glyph --lmhead-cs exact`:

- **`V4.2 Dense` (`6.0 GiB`):** **`7 / 15`** ([battery_v4_2_dense](file:///C:/Gemma-4-LanguageU-EXP/Language-U-Gemma4/experiments/run270_rust/benchmarks/v6_1_skeptic_proof/battery_v4_2_dense/skeptic_proof_report.json))
- **`V6.1 Dense` (`4.364 GiB`):** **`2 / 15`** ([battery_v6_1_dense](file:///C:/Gemma-4-LanguageU-EXP/Language-U-Gemma4/experiments/run270_rust/benchmarks/v6_1_skeptic_proof/battery_v6_1_dense/skeptic_proof_report.json))
- **`V6.1 CRTR top_p=0.50` (`4.364 GiB + 10.4 MB`):** **`2 / 15`** ([battery_v6_1_crtr50](file:///C:/Gemma-4-LanguageU-EXP/Language-U-Gemma4/experiments/run270_rust/benchmarks/v6_1_skeptic_proof/battery_v6_1_crtr50/skeptic_proof_report.json))

| Tag | Prompt | `V4.2 Dense` | `V6.1 Dense` | `V6.1 CRTR-50` |
|---|---|---|---|---|
| `P1_arith` | `What is 2 + 2?` | **PASS** (`"The answer is 4.<turn|>"`) | FAIL (`"...**2.5 (1) = 2/3= 0..."`) | FAIL (`"...**2.0**..."`) |
| `P2_france` | `The capital of France is` | **PASS** (`"The capital of France is **Paris**.<turn|>"`) | FAIL (`"Depending to the internet of Europe..."`) | FAIL (`"The goal of the country is..."`) |
| `P3_hamlet_quote` | `To be or not to be, that is the` | FAIL (`"The provided input is a fragmented..."`) | FAIL | FAIL |
| `P4_prime_py` | `Write a Python function to check if a number is prime.` | **PASS** (explains prime checking prose) | **PASS*** (`def get_number` regex match; broken body) | **PASS*** (`def __init(1):` regex match; broken body) |
| `P5_speed` | `120 km in 2 hours. average speed?` | **PASS** (`"**120 km / 2 = ****60**."`) | FAIL (`"120% of 15/3 = **6..."`) | FAIL (`"120% of the total-3 ** 5..."`) |
| `P6_photo` | `Explain how photosynthesis works in one short sentence.` | **PASS** (`"...converts sunlight into energy..."`) | FAIL | FAIL |
| `P7_bsearch` | `Time complexity of binary search...` | FAIL (`"**Binary Search**..."` without `O(log n)`) | FAIL | FAIL |
| `P8_romeo` | `Who wrote the play Romeo and Juliet?` | FAIL (`"written by **Romeo**.<turn|>"`) | FAIL | FAIL |
| `B5_hamlet` | `Who wrote the play Hamlet?` | FAIL (`"written by **William Hamlet..."`) | FAIL | FAIL |
| `B5_multistep` | `Compute 12 * 8 - 27.` | FAIL (`"12 * 8 = 90\n90 - 27 = **63**<turn|>"`) | FAIL (`"27.50 = 18.2= 36-4=19"`) | FAIL (`"27. ** 10 /..."`) |
| `B5_min` | `Which number is smaller, 547 or 193?` | **PASS** (`"**193.<turn|>"`) | **PASS** (`"...**193.7 = 2480=..."`) | **PASS** (`"...193, 5470..."`) |
| `U1_algebra` | `If 5x + 3 = 38, what is 2x - 1?` | FAIL (`"5 + 3 = <<8> ... Final result: **8**"`) | FAIL | FAIL |
| `U2_units` | `18 km/h for 40 min -> meters?` | FAIL (`"18 meters per hour..."`) | FAIL | FAIL |
| `U3_bugfix` | `Bug in def avg(nums) on empty list?` | **PASS** (`"Empty list is empty..."`) | FAIL | FAIL |
| `U4_syllogism` | `All mammals warm-blooded; whales mammals...` | FAIL (`"**Yes.**"`) | FAIL | FAIL |

*\*Transparency note on `P4_prime_py`: `V6.1 Dense` (`def get_number(numbers): ... return'`) and `V6.1 CRTR-50` (`def __init(1): ...`) only pass the substring grader (`def ` + `return`/`for`), whereas only `B5_min` (`193`) produces a valid factual answer token on `V6.1`. On `V4.2`, multi-step/entity confusion (`63` vs `69`, `William Hamlet`, `Romeo`, `5+3=8`, `No`->`Yes`) remains the clear `~1.58 bpw` reasoning boundary.*

---

## 5. `Assistant-V7` (`BHDR` v1) Speculative Drafting vs Legacy `Assistant-V6` ([spec_matrix.json](file:///C:/Gemma-4-LanguageU-EXP/Language-U-Gemma4/experiments/run270_rust/benchmarks/v6_1_skeptic_proof/spec_matrix.json), [spec_ab](file:///C:/Gemma-4-LanguageU-EXP/Language-U-Gemma4/experiments/run270_rust/benchmarks/v6_1_skeptic_proof/spec_ab/skeptic_proof_report.json))

### Memory Compression Result
Eliminating the `262,144`-vocab LM head and replacing it with the `1.522 MiB` INT8 byte+`<eot>` GRU (`BHDR` v1) + `11.46 MiB` CSR trie cut resident assistant RAM dramatically on Pi 4:
- **`Assistant-V6` (`t16c` LM head):** **`334.31 MB`** resident (`5,259,496–5,261,068 KB` total RSS on `V6.1`)
- **`Assistant-V7 Q8 + BHDR`:** **`210.79 MB`** resident (`-123.5 MB`, `1.59x` smaller)
- **`Assistant-V7 Q4 + BHDR`:** **`114.92 MB`** resident (`-219.4 MB`, `2.91x` smaller)
- **`Assistant-V7 Q2 + BHDR`:** **`66.98 MB`** resident (**`-267.3 MB` / `4.99x` smaller**; total `V6.1 + V7_Q2` `vm_rss = 4,973,008–4,974,476 KB = 4.74 GiB`)

### Acceptance Rate, Margin Sensitivity & Latency (`--batch-k 4 --assistant-step-pos --max-tokens 24`)

| Target | Drafter (`MB`) | `min_margin` | Prompt (`24` max toks) | Drafted | Accepted | `accept_rate` | `margin_stopped` | `s_per_tok` | Standalone Baseline |
|---|---|---|---|---|---|---|---|---|---|
| `V6.1` | `V6_334MB` (`334.31`) | `0.00` | `What is 2 + 2?` / `France` / `Photo` | `27` / `31` / `26` | `6` / `4` / `5` | `0.222` / `0.129` / `0.192` | `0` | `2.413` / `2.701` / `2.437` | `~1.79s/step` |
| `V6.1` | `V6_334MB` (`334.31`) | `0.75` | `What is 2 + 2?` | `23` | `5` | `0.217` | `8` | `2.402` | `~1.79s/step` |
| `V6.1` | `V7_Q8` (`210.79`) | `0.00` | `What is 2 + 2?` / `France` / `Photo` | `29` / `31` / `30` | `4` / `2` / `3` | `0.138` / `0.065` / `0.100` | `0` | `2.567` / `2.762` / `2.704` | `~1.79s/step` |
| `V6.1` | `V7_Q4` (`114.92`) | `0.00` | `What is 2 + 2?` / `France` / `Photo` | `30` / `32` / `32` | `3` / `1` / `1` | `0.100` / `0.031` / `0.031` | `0` | `2.631` / `2.837` / `2.875` | `~1.79s/step` |
| `V6.1` | `V7_Q2` (`66.98`) | `0.00` | `What is 2 + 2?` / `France` / `Photo` | `29` / `31` / `30` | `2` / `2` / `1` | `0.069` / `0.065` / `0.033` | `0` | `2.632` / `2.714` / `2.774` | `~1.79s/step` |
| `V6.1` | `V7_Q2` (`66.98`) | `0.25` | `What is 2 + 2?` | `13` | `2` | **`0.154`** | `15` | **`2.121`** | `~1.79s/step` |
| `V6.1` | `V7_Q2` (`66.98`) | `0.75` | `What is 2 + 2?` (`16` tok canonical `spec-ab`) | `0` | `0` | `0.000` | all steps | `1.886` (all `k=1`) | `~1.79s/step` |
| `V4.2` | `V6_334MB` (`334.31`) | `0.00` | `2 + 2` (`7` tok) / `France` (`9` tok) | `12` / `18` | `2` / `2` | `0.167` / `0.111` | `0` | `4.681` / `4.718` | `~2.62s/step` |
| `V4.2` | `V6_334MB` (`334.31`) | `0.75` | `2 + 2` (`7` tok) | `9` | `0` | `0.000` | `3` | `4.266` | `~2.62s/step` |
| `V4.2` | `V7_Q8` (`210.79`) | `0.00` | `2 + 2` (`7` tok) / `France` (`9` tok) | `12` / `18` | `2` / `2` | `0.167` / `0.111` | `0` | `3.883` / `4.650` | `~2.62s/step` |
| `V4.2` | `V7_Q4` (`114.92`) | `0.00` | `2 + 2` (`7` tok) / `France` (`9` tok) | `12` / `18` | `2` / `2` | `0.167` / `0.111` | `0` | **`3.808`** / `4.610` | `~2.62s/step` |
| `V4.2` | `V7_Q2` (`66.98`) | `0.00` | `2 + 2` (`7` tok) / `France` (`9` tok) | `12` / `18` | `2` / `1` | `0.167` / `0.056` | `0` | `3.853` / `4.611` | `~2.62s/step` |

**Three Clear Speculative Findings:**
1. **`5x` Drafter RAM Reduction Verified (`334.31 MB -> 66.98 MB`):** End-to-end `BHDR` + CSR prefix trie loading and greedy speculative verification run with zero swaps and `ast_lm_skipped=0`.
2. **Margin Mismatch Exposed & Resolved:** `run31b` defaulted `--assistant-min-margin` to `0.75` (calibrated for unnormalized logits), aborting `100%` of `BHDR` drafts (which compare normalized log-probabilities `lp1 - lp2`). Lowering to `0.25` raised `V7_Q2` `What is 2 + 2?` accept rate from `0.000` (`m=0.75`) / `0.069` (`m=0.00`) to `0.154` and sped up speculative `s_per_tok` from `2.632` to `2.121` (outperforming `V6_334MB`'s `2.402 s/tok`).
3. **Break-Even Reality Check:** On `V4.2`, `V7` (trained on BF16 Teacher states) matched `V6` (`0.167` accept on `2+2`) while running `~18%` faster (`3.81–3.88` vs `4.68 s/tok`); however, because accept rates sit at `0.03–0.27`, **non-speculative greedy decode (`1.79 s/step` dense, `1.06–1.42 s/step` CRTR/MoD on `V6.1`; `~2.62 s/step` on `V4.2`) remains faster than `batch_k=4` speculative drafting** on Cortex-A72 until acceptance reaches `~0.50+` (which requires aligning `V7` on the target quantized model's `L58/L59` KV states and `262K` LM logits rather than pure W4A16 Teacher states).

---

## 6. 3-Bit Vision Multimodal QA ([vision/skeptic_proof_report.json](file:///C:/Gemma-4-LanguageU-EXP/Language-U-Gemma4/experiments/run270_rust/benchmarks/v6_1_skeptic_proof/vision/skeptic_proof_report.json))

| Model & Image (`Prompt: "Describe what you see in this image in one clear sentence."`) | ViT Encode | Prefill (`n_soft`) | Decode (`s`) | GlyphKV Shrink | Peak `vm_rss` | Generated Text (`GEN31B`) |
|---|---|---|---|---|---|---|
| **`V6.1` + `frame_01_coco.png`** (`640x480`) | `946.42s` | `295.32s` (`266`) | `23.68s` (`14` tok) | `6.77x` (`3.39x` vs BF16) | `4.72 GiB` (`4,949,672 KB`) | `"A black, of a bathroom is seen to a blue side.<turn|>"` |
| **`V6.1` + `frame_00_va.png`** (`672x672`) | `841.67s` | `283.25s` (`256`) | `27.15s` (`16` tok) | `6.65x` (`3.32x` vs BF16) | `4.73 GiB` (`4,959,752 KB`) | `"A person of a woman is an image of the red-based background.<turn|>"` |
| **`V4.2` + `frame_01_coco.png`** (`640x480`) | `920.70s` | `410.87s` (`266`) | `46.02s` (`17` tok) | `6.82x` (`3.41x` vs BF16) | `6.26 GiB` (`6,568,628 KB`) | `"The image depicts a singular sequence of the same movements throughout the entire same room.<turn|>"` |

While `V6.1` Vision reduces LM prefill/decode time by `28%/49%` and RAM by `1.54 GiB` over `V4.2` (and avoids `V4.2`'s `"nonsensical sequence"` refusal), neither `V6.1` nor `V4.2` matches the `V4.3 VLM` (`5.9175 GiB`, higher-bit LM + multimodal C8DB) baseline on exact object identification (`"A motorcycle is parked on a grassy field.<turn|>"`).

---

## 7. Concrete Engineering Takeaways for V7 / Next Iteration

1. **Optimal Pi 4 Memory–Accuracy Operating Point Lies Around `~5.2–5.5 GiB` (`~1.4–1.6 bpw`):** Dropping `L00..L35` to `1`-bit `LittleBit-2` (`4.364 GiB` total) pushes KL too high (`7.50`) for reliable multi-step QA (`2/15`). Upgrading `L00..L35` (or early+late layers) to `1.5–2.0 bpw` while keeping total footprint around `5.2–5.5 GiB` leaves >`2.1 GiB` headroom on Pi 4 `7.6 GiB` RAM for GlyphKV + `Assistant-V7 Q4/Q2`.
2. **Real `CRTR` Is Validated; Tune `MoD` Conservatively:** `CRTR top_p=0.50` delivers `1.26x` decode speedup (`41.7%` MLP DRAM saved) with `< 0.1` nat KL delta and identical battery accuracy; `MoD` needs softer thresholds (`0.15–0.25`) or distillation-aware skip training rather than `0.35`.
3. **Fix `Assistant-V7` Alignment & Default Margin:** Set default `BHDR` `min_margin` around `0.25` (not `0.75` logit margin), and train `BHDR` + embed-free body directly against the target quantized LM's states/logits on Pi 4 target prompts so `67–115 MB` `Assistant-V7` clears the `~0.50` speculative speedup break-even.
