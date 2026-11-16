<div align="center">
  <a href="docs/assets/language_u_logo.png">
    <img src="docs/assets/language_u_logo.png" alt="Language-U — The Glyph: One Symbol, Three Meanings" width="920" />
  </a>
  <h1>Language-U · Gemma 4 on the Edge</h1>
  <p><strong>Running 31-Billion-Parameter Gemma 4 Inside 8 GB of Edge RAM — Zero Swap, Zero Stubs, Pure Silicon Receipts</strong></p>
  <p><em>Sub-Byte Gyroscopic Glyphs · Native Rust + ARMv8-A NEON &amp; AVX2 Kernels · 21.21x 128K KV Compression (Kernel Self-Test) · L1-Cache Trinity Retrieval + Lean 4 Trust Gates</em></p>
</div>

---

## The Impossible Engineering Problem

Frontier 31-billion-parameter models hit **three hard physical walls** on an 8 GB edge board like the Raspberry Pi 4:

1. **Weight Footprint Wall:** Standard BF16 weights (`~62 GB`) and even QAT INT4 weights (`~17+ GB`) blow past 8 GB RAM, forcing catastrophic SSD swap thrashing.
2. **Compute & Bandwidth Wall:** Dequantizing sub-byte weights in pure scalar loops starves ARM Cortex-A72 memory bandwidth and leaves prefill taking tens of minutes.
3. **Long-Context & Code Retrieval Wall:** 128K KV caches swallow gigabytes (`1,024 MB` on global layers alone), while naive vector search floods fragile small context windows with false positives.

**Language-U-Gemma4** attacks all three bottlenecks in one unified, zero-dependency stack — and refuses every form of synthetic simulation. Every single metric below comes from real execution on **physical Raspberry Pi 4 Model B (`4x Cortex-A72 @ 1.80 GHz`, Debian 13 `aarch64`, 8 GB RAM)** or **Windows x86_64 MSVC AVX2**, cross-checked against raw kernel, `/proc/<pid>/status`, and `/proc/<pid>/io` logs.

<div align="center">
  <img src="docs/Language-U_Architecture_Infographic.png" alt="Language-U-Gemma4 System Architecture" width="920" />
</div>

---

## Physical Silicon Scoreboard

| Bottleneck / Component | Target Hardware | Measured Proof on Physical Silicon | Raw Telemetry / Evidence |
| :--- | :--- | :--- | :--- |
| **1. 31B Full RAM Residency (`T16C v1`, `5.71 GiB`)** | **Pi 4 (8 GB)** | **`60/60` layers pinned**, **`VmSwap: 0 kB`**, **`ssd_read_bytes = 0`**, `VmHWM = 6,238,800 kB`, **`1.291 s/tok`** (`k=4` batch), **`26.32x` NEON FFI prefill speedup** (`33.78s` vs `889.23s` pure Rust; logit `cos = 0.999299`, top-1 `(715, 715)`) | [run31b_cal_v2_2plus2.log](experiments/run270_rust/evidence/run31b/run31b_cal_v2_2plus2.log)<br>[run31b_cal_v2_2plus2_proc.log](experiments/run270_rust/evidence/run31b/run31b_cal_v2_2plus2_proc.log)<br>[run31b_cal_v2_ffi_parity.txt](experiments/run270_rust/evidence/run31b/run31b_cal_v2_ffi_parity.txt) |
| **1b. 31B V3 + Cell 60 (`C8DB`) + Adaptive `Q8` + Speculative `k=4`** | **Pi 4 (8 GB)** | `gemma4_31b_v3_c8d.c8d` (`91.68 MB`, `63` entries): clean `271`-tok `post_cell60` (`val_cos 0.68875 -> 0.72910`, `0/8` turn-start `"_"`/`"s"` glitches); adaptive single-plane `Q8` + OLS sub-scale (`dual=0, kc_step=4`): FFI parity **`cos = 0.9997202955`** (**`29.04x`** prefill `37.83s` vs `1098.74s`, `top1=818`, `10/10` top-10); `8`-prompt `k=1` **`5.850 s/tok`** ILT (`0.1709 tok/s`, `42.34s` mean prefill, `VmSwap=0`, `ssd_read=0`); Speculative `k=4` **`7/7` modes bit-exact vs `k=1`** (`S2_P4` seed-only `5.868 s/tok`, `S0` Assistant `10.454 s/tok`, `S4` `--kv-quant fp8` **`3.94x`** KV shrink at `6.616 s/tok`, `--hybrid-qat-layers 59` `286.45 MiB` overlay `VmSwap=0`); Contiguous CS LM-head **`8/8` exact top-1** (`static_16band` `1.14x`/`1.05x`, `subspace_sieve` **`4.04x`** `182.51 ms` on 31B / **`2.87x`** `48.44 ms` on Assistant, `84.22%` MACs saved) | [run31b_v3_verified.json](experiments/run270_rust/evidence/run31b/run31b_v3_verified.json)<br>[run31b_v3_ffi_parity.txt](experiments/run270_rust/evidence/run31b/run31b_v3_ffi_parity.txt)<br>[run31b_v3_prompt_battery_k1.log](experiments/run270_rust/evidence/run31b/run31b_v3_prompt_battery_k1.log)<br>[run31b_v3_speculative_k4.log](experiments/run270_rust/evidence/run31b/run31b_v3_speculative_k4.log)<br>[run31b_v3_hybrid_qat_l59.log](experiments/run270_rust/evidence/run31b/run31b_v3_hybrid_qat_l59.log)<br>[run31b_v3_cs_lmhead_bench.log](experiments/run270_rust/evidence/run31b/run31b_v3_cs_lmhead_bench.log) |
| **31B Coherent Generation (`QAT pin10` + NEON)** | **Pi 4 (8 GB)** | **Barrier broken (`<4.6 GiB` coherent 31B, slow `76.2-78.9 s/tok` streamed):** emits **`"William Shakespeare wrote the play *Hamlet*.<turn|>"`**, **`cos = 0.999995`**, `rel_l2 = 0.003402`, exact top-1 & top-5 match vs FP32 reference, `VmHWM = 4,725,968 kB` (`4.51 GiB`) | [a_hamlet_q8emb_pin10_neon.log](experiments/run270_rust/evidence/run31b/a_hamlet_q8emb_pin10_neon.log)<br>[run31b_parity_2plus2.json](experiments/run270_rust/evidence/run31b/run31b_parity_2plus2.json) |
| **31B Minimal-RAM Streaming (`QAT pin0`)** | **Pi 4 (8 GB)** | Emits exact **`"4"`** on `2 + 2` in **`485,348 kB` (`474.0 MiB`) peak HWM** / `372,696 kB` RSS via zero-copy `mmap` | [b1_stream_2plus2.log](experiments/run270_rust/evidence/run31b/b1_stream_2plus2.log) |
| **31B Coherent `packed92` (`2.875 bpw`)** | **Pi 4 (8 GB)** | **`"2 + 2 = 4<turn|>"`** from `10.52 GB` (`9.80 GiB`) mmap container (`prefill 426.85s`, `8` tok decode `56.6-61.4s/step`) | [c_packed92_chat_2plus2.log](experiments/run270_rust/evidence/run31b/c_packed92_chat_2plus2.log) |
| **2. `T16C` (`v6_e2e_s2`) + `LUJ1` Joint Glyphs AVX2** | **Windows x86_64** | `T16C` (`19.35 MB` proj, `1.54 bpw`): **`0.0905 s/fwd`** AVX2 DLL vs `0.5055 s/fwd` pure Rust (**`5.59x`**); `OPT5 36B` *(isolated AVX2 GEMV microbenchmark, not LLM `tok/s`)*: **`10.86M glyphs/s`** (`0.2475 ms`, **`221.5M tok/s`**) across `54,834` tokens (`20.4` mean tok/glyph) | [v6_e2e_s2_avx2_bench5.json](experiments/run270_rust/evidence/g270_calib/v6_e2e_s2_avx2_bench5.json)<br>[v6_e2e_s2_rust_bench5.json](experiments/run270_rust/evidence/g270_calib/v6_e2e_s2_rust_bench5.json)<br>[windows_joint_fast_benchmark.json](experiments/joint_glyphs/evidence/windows_joint_fast_benchmark.json) |
| **Gemma 3 270M Exact & `Q8_eq8` (`run270`)** | **x86_64 & Pi 4** | **`5/5` prompts `cos >= 0.999999999975`** vs `HF 4.55.0 FP32`; `q8_eq8` (`285.02 MB`, `591,660 KB` HWM) preserves **`10/28` frozen / `6/30` fresh**; `q5+h8` hits `8/28` & `10/30` | [RECEIPTS.md](experiments/run270_rust/RECEIPTS.md)<br>[FRONTIER_RECEIPTS.md](experiments/run270_rust/FRONTIER_RECEIPTS.md) |
| **3. `GlyphAttentionV2` 128K KV *(Kernel Self-Test)*** | **Pi 4 (ARM64)** | *Isolated kernel self-test (not end-to-end 31B `tok/s`):* **`21.21x` memory shrink** (`48.27 MB` vs `1,024 MB`), **`1.60x` speedup** (`713.98 ms` vs `1,140.71 ms`); `W=1024` ring `4.36 ms` (`1.76x`, `o_proj_cos = 0.99597`) | [glyph_attention_v2_pi4_benchmark_report.json](docs/glyph_attention_v2_pi4_benchmark_report.json)<br>[NEON_RUNNER_INTEGRATION_V2.md](docs/NEON_RUNNER_INTEGRATION_V2.md) |
| **Trinity Retrieval + Lean 4 Trust Gate** | **x86_64 & Pi 4** | **`96.0x` shrink** (`6,027,264 B -> 62,784 B`, `H64` `1 bit/dim`), **`8/10` in `0.0478 ms`** vs `3/10` BM25 (`6.77 ms`) & `7/10` Dense768 (`11.55 ms`), OOD `MISS` abstention (**`0.0%` OOD FP** vs `100%` baseline FP), **`95.0%+` context savings** | [swe_navigation_benchmark_results.json](trinity_retrieval/swe_navigation_benchmark_results.json)<br>[TrinityGates.lean](trinity_retrieval/formal_verification/TrinityGates.lean) |
| **`OpticalCarrierV1` Context DMA *(Kernel Self-Test)*** | **Pi 4 (ARM64)** | *Isolated kernel self-test DMA rate (not end-to-end `tok/s`):* **`6,522.2 FPS`**, **`626,129.2 tok/s`**, `0.244-0.477 ms` in-memory DMA, **`100.0%` bit parity** | [optical_context_gemma4_benchmark_report.json](docs/optical_context_gemma4_benchmark_report.json) |

---

## Core Architectural Breakthroughs

### 1. Zero-Swap 31B Residency (`T16C` + Pure Rust Runtime + NEON / AVX2 FFI)
All **410 linear projections (`5.41 GB`)**, **421 norms (`5.33 MB`)**, and **4-bit tied embeddings (`715.7 MB`)** pack cleanly into a single `64B`-aligned `5.71 GiB` binary container ([t16c.rs](experiments/run270_rust/src/t16c.rs)). Each weight block pairs Fast Walsh-Hadamard incoherence rotations (`FWHT-64/128/256`), SmoothQuant input-channel equalization, and INT8 low-rank residuals, driven by C SIMD kernels ([neon_stream_accelerator_v2.c](neon_runner/neon_stream_accelerator_v2.c)) with bit-verified pure-Rust fallbacks (`--no-native-ffi`). Linux `/proc/<pid>/status` and `/proc/<pid>/io` telemetry prove **`60/60` layers stay locked in physical RAM (`VmSwap: 0 kB`, `ssd_read_bytes: 0`)** while ARMv8-A NEON accelerates prefill **`26.32x`** on `T16C v1` and **`29.04x`** (`37.83s` vs `1098.74s`, `cos = 0.9997202955`) under adaptive single-plane `Q8 + OLS` (`dual=0, kc_step=4` on FWHT activations). Scope honesty: `"Barrier broken"` refers to coherent 31B execution inside `<4.6 GiB` (`QAT pin10`, `76.2-78.9 s/tok`) and `60/60` RAM-resident `T16C V3` (`6.31 GiB` peak HWM; `6.29 GiB` with 1-layer `L59` QAT overlay), not interactive human reading speed (`5.850 s/tok` weighted `k=1` ILT; `56.6-61.4 s/step` on `packed92`).

### 2. `GlyphAttentionV2` + `OpticalCarrierV1` Zero-Disk Context Engine *(Isolated Kernel Self-Tests)*
In isolated Pi 4 component/kernel self-tests *(never end-to-end 31B `tok/s`)*, global attention KV caches at 128K context drop from `1,024 MB` to **`48.27 MB`** (**`21.21x` shrink**, **`1.60x` speedup**: `713.98 ms` vs `1,140.71 ms` on Cortex-A72), while sliding-window `W=1024` ring buffers execute in **`4.36 ms`** (`1.76x` speedup, `o_proj_cos = 0.99597`). Paired with `OpticalCarrierV1` in-memory frame DMA self-tests (`6,522.2 FPS`, **`626,129.2 tok/s`**, `100.0%` bit parity in `0.244-0.477 ms` DMA), context staging itself is not bottlenecked by flash I/O.

### 3. L1-Resident Trinity Retrieval Governed by Formal Lean 4 Proofs
An `H64` binary SimHash index (`61.31 KiB`, `1 bit/dim` across 64 hash bits / ~`32 B` per doc, compressing a `1,962`-document `embeddinggemma-2` 768-D MRL codebase index **`96.0x`** into CPU L1/L2 cache) fuses with BM25 and dense vectors under formally mechanized Lean 4 safety thresholds ([TrinityGates.lean](trinity_retrieval/formal_verification/TrinityGates.lean): `STRONG_GATE = 0.302`, `WEAK_GATE = 0.280`, `LEX_STRONG_FLOOR = 0.080`), enforcing explicit OOD **`MISS` abstention (`0.0%` OOD FP)** alongside `zero_confabulation_guarantee`, `noise_refusal_guarantee`, and `weak_requires_opt_in` with thread-safe singleton caching (`get_engine()`, `store_revision()`, `clear_engine_cache()`) and offline hash-verified self-tests.

### 4. KOFI v11 Competition SWE Agent & Native C++17 Semantic Compiler
A clean, self-contained four-file Kaggle ADK submission ([agent/submission.zip](agent/submission.zip)) paired with KOFI v11 system policies ([agent/prompts/system.md](agent/prompts/system.md), [agent/build.py](agent/build.py)) and a closed-grammar native C++17 / Python semantic glyph compiler (`compiler/`) + C++17 neural joint-glyph planner (`experiments/joint_glyphs/`).

---

## Honest Precision & Scaling Boundaries (No Hype)

Science requires publishing **where compression breaks** right alongside where it wins:

- **Sub-2-Bit PTQ vs Calibrated V3 Boundaries & Open Codec Gap:**
  - While `gemma4_31b_cal.t16c` (`~1.54 bpw` projections) achieves full `60/60` Pi 4 RAM residency (`VmSwap=0`, `1.291 s/tok k=4`), uniform post-training calibration without mixed-precision high-bit allocation collapses greedy decoding on tested prompts to `"////////////////"` ([run31b_cal_v2_2plus2.log](experiments/run270_rust/evidence/run31b/run31b_cal_v2_2plus2.log), [run31b_cal_v2_hamlet.log](experiments/run270_rust/evidence/run31b/run31b_cal_v2_hamlet.log)). Meanwhile, QAT `pin10` (`4.51 GiB` HWM, `76.2-78.9 s/tok`) and `packed92` (`2.875 bpw`, `56.6-61.4 s/step`) generate exact coherent responses (`"William Shakespeare wrote the play *Hamlet*."`, `"2 + 2 = 4"`).
  - Under `V3 + C8DB Cell 60` (`gemma4_31b_v3.t16c` + clean `271`-tok `post_cell60`) with adaptive single-plane `Q8 + OLS` (`dual=0, kc_step=4` on FWHT activations; `dual=1` automatic when block top-2 energy ratio `> 0.90`), step-0 `"_"`/`"s"` turn-start glitches stay eliminated (`8/8` prompts) while weighted `k=1` decode ILT drops from `10.352 s/tok` (`dual=1`) to **`5.850 s/tok`** (`0.1709 tok/s`, P1 = `5.400 s/tok`), FFI parity hits `cos = 0.9997202955` (`10/10` top-10, `gen31b_exact=True`), `7/7` `k=4` speculative runs stay bit-exact with `k=1` (`S2_P4` seed-only `5.868 s/tok`; `S4` `--kv-quant fp8` `3.94x` KV shrink at `6.616 s/tok`; `286.45 MiB` `L59` QAT overlay `VmSwap=0`), while open-ended sub-2-bit factual drift remains visible on trivia prompts (`P2`, `P8`) and `P4` shows minor 7-bit step-3 variation vs `dual=1`.
  - No lossless codec `<= 4.8 GB` for 31B (below `packed92`'s `2.875 bpw` / `10.52 GB`) is proven; `T16C` (`~1.54-1.62 bpw`) is lossy calibrated compression, not lossless.
  - Pre-permuted contiguous 16-band Cauchy-Schwarz LM-head pruning (`bench_cs_lmhead.c` `w_perm`) achieves `8/8` exact top-1 and flips exact multi-band CS to **`1.14x`** (`644.74 ms`, `20.66%` MACs saved on 31B) and **`1.05x`** (`131.69 ms`, `27.34%` MACs saved on Assistant), while 12.5% heavy `subspace_sieve` reaches **`4.04x`** (`182.51 ms` on 31B) and **`2.87x`** (`48.44 ms` on Assistant, `84.22%` MACs saved) at `8/8` exact top-1; holding both `w_raw` mmap and `w_perm` side-by-side inside the 31B benchmark binary pushes RSS to `5.53 GB` (`vm_swap_kb=0`, `4,866,048 B` file-backed mmap re-read vs `0 B` on Assistant) ([run31b_v3_cs_lmhead_bench.log](experiments/run270_rust/evidence/run31b/run31b_v3_cs_lmhead_bench.log)).
- **270M Sub-Byte Cliff (23-Config Pareto Sweep):**
  - *Lossless / High Fidelity:* `exact`, `q8_eq8` (`285.02 MB`), and `Q14-norms` match baseline `10/28` frozen & `6/30` fresh exact-match scores; `q10+h8` passes the `5/5` strict parity gate.
  - *Sweet Spot vs Floor:* `q5+h8` achieves `8/28` frozen & `10/30` fresh (`q4+h8` improves via FWHT over unrotated `q4`), whereas uncalibrated `OPT4` (`cos ~ 0.9718` matrix-level, `cos=0.9219` logits), `b1+h8` (`1.27 bpw`), and Windows `v6_e2e_s2.t16c` (`1.54 bpw` PTQ) establish the exact `0/28` / `0/30` strict-match floor ([FRONTIER_RECEIPTS.md](experiments/run270_rust/FRONTIER_RECEIPTS.md)).
- **Scoped Residual Study (`experiments/packed_corrections/`):**
  The rank-15 INT8 stored residual (`83,196 B`, `cos = 0.96593`, `2.67e-7` relative error vs independent `float64` oracle) is explicitly scoped as a verified excerpt study, not a full-model claim ([experiments/packed_corrections/README.md](experiments/packed_corrections/README.md)).

---

## Repository Map

```text
Language-U-Gemma4/
├── experiments/
│   ├── run270_rust/                  # Zero-dep Rust runtimes (run270, run31b, t16c, encode_model, probe270)
│   │   ├── src/                      # codec.rs, packed.rs, t16c.rs, math.rs, tokenizer.rs, bin/run31b.rs
│   │   ├── calib/                    # Parameterized T16C calibration scripts (270M + 31B)
│   │   ├── evidence/                 # Raw Pi 4 & Windows logs, /proc IO proofs, 23-config Pareto JSONs
│   │   ├── RECEIPTS.md               # Exactness & parity ledger (run270 + OPT4 boundary)
│   │   └── FRONTIER_RECEIPTS.md      # 23-config Pareto, run31b QAT/packed92, T16C 60/60 RAM receipts
│   ├── gemma270_luj/                 # Q14-norms LUJ storage & HF Transformers parity gate
│   ├── joint_glyphs/                 # C++17 native neural glyph planner + Windows AVX2 benchmarks
│   └── packed_corrections/           # RCF1 rank-k stored residuals + float64 oracle verification
├── neon_runner/                      # ARM64 NEON & Windows MSVC AVX2 kernels (neon_stream_accelerator_v2.c)
├── trinity_retrieval/                # H64 + BM25 + embeddinggemma-2 MRL RRF + formal_verification/TrinityGates.lean
├── compiler/                         # 6D Cuneiform & native C++17 closed-grammar glyph compiler
├── compression/                      # Language-U glyph & embeddinggemma codec modules
├── gemma4_e4b/ & model/              # Gemma 4 E4B census checks & verified sample glyph containers
├── agent/ & fleet-consensus-v1/      # KOFI v11 & consensus v1 competition bundles + agent/build.py --check
├── scripts/                          # Pi 4 prep verification, 31B weight lineage, optical & KV benchmarks
├── docs/                             # Hero emblem, architecture infographic, empirical writeups & reviews
└── tests/                            # Offline unit test suites + tests/retrieval/
```

---

## Reproducible Quickstart

### 1. Build & Test Native Rust Runtimes + Calibration Tools (ARM64 Linux / Windows x86_64)
```bash
# Release binaries (run270, run31b, encode_model, probe270) + pure-Rust test suite
RUSTFLAGS="-D warnings" cargo test --manifest-path experiments/run270_rust/Cargo.toml --release
RUSTFLAGS="-D warnings" cargo build --manifest-path experiments/run270_rust/Cargo.toml --release --bins

# Verify Python calibration scripts compile & parse cleanly
python -m compileall -q experiments/run270_rust/calib
python experiments/run270_rust/calib/31b/cal31b_streaming.py --help
```

### 2. Compile Native SIMD Accelerator (NEON on ARM64 / AVX2 on Windows)
```bash
# Raspberry Pi 4 (ARMv8-A Linux):
gcc -O3 -Wall -Wextra -Werror -DNEON_SHARED_LIB_ONLY -shared -fPIC \
    neon_runner/neon_stream_accelerator_v2.c -lm -o neon_runner/libneon_stream_accelerator.so

# Or cross-platform helper (builds .so on Linux/ARM64 or .dll on Windows MSVC):
python neon_runner/build_native.py
```

### 3. Run Gemma 4 31B Inference (`run31b`)
```bash
# V3 + Cell 60 C8DB + N-gram seed speculative k=4 on Pi 4 (0 MB assistant overhead, VmSwap=0):
./experiments/run270_rust/target/release/run31b \
  --t16c /mnt/SHE-VAULT/gemma4_31b_v3.t16c \
  --config /mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/config.json \
  --tokenizer /mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/tokenizer.json \
  --cuneiform-bank /mnt/SHE-VAULT/gemma4_31b_v3_c8d.c8d --cuneiform-cold-va \
  --so /home/astronautshe/gemma4_verify/neon_runner/libneon_stream_accelerator.so \
  --speculative --batch-k 4 --spec-seed /mnt/SHE-VAULT/gemma4_31b_math_seed.cngm \
  --threads 4 --max-tokens 8

# Full-RAM T16C 31B inference on Pi 4 (pass --no-native-ffi to benchmark pure-Rust fallback):
./experiments/run270_rust/target/release/run31b \
  --t16c /mnt/SHE-VAULT/gemma4_31b_cal.t16c \
  --config /mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/config.json \
  --tokenizer /mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/tokenizer.json \
  --threads 4 --batch-k 4 --max-tokens 8

# Bounded-RAM QAT streaming with Q8 tied embeddings and 10 pinned layers (~4.51 GiB HWM):
./experiments/run270_rust/target/release/run31b \
  --model /mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/model.safetensors \
  --config /mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/config.json \
  --tokenizer /mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/tokenizer.json \
  --embed-mode q8 --pin-layers 10 --threads 4 --max-tokens 9
```

### 4. Run Full Verification Suites
```bash
python -m unittest discover -s tests -v
python -m unittest discover -s tests/retrieval -v
python trinity_retrieval/edge_self_test.py
python agent/build.py --check
python scripts/verify_pi4_preparation.py
```

---

## Attributions & License

- **Repository Code License:** Open Source **Apache License, Version 2.0** ([LICENSE](LICENSE), [NOTICE](NOTICE)) — compliant with the Kaggle *Google - The Gemma 4 Developer Agent Competition* Apache-2.0 / unrestricted commercial-use rules.
- **Competition Model Scope:** The `agent/` submission bundle targets Google's Apache-2.0 **Gemma 4** model (`gemma-4-31b-it-qat-w4a16-ct`, alongside local `gemma-4-E4B-it` experiments). Optional local/ancillary validation scripts that download Google **Gemma 3** (`270M`) or **EmbeddingGemma** (`embeddinggemma-2`) depend on external upstream weights governed separately by Google's Gemma Terms of Use and are excluded from `agent/submission.zip`.
- **KOFI v11 Agent Bundle (`agent/`):** Licensed under **Apache-2.0** ([agent/LICENSE.txt](agent/LICENSE.txt)); incorporates prior competition baselines and empirical findings from Wilmer E. Henao (`verracodeguacas`), Igor Zharov (`flexonafft`, V10), and `cubres` (`prvsiyan`). Full attribution ledger in [agent/NOTICE.txt](agent/NOTICE.txt).
- **NanoQuant Initialization Component (`experiments/packed_corrections/`):** Adapted from `SamsungLabs/NanoQuant` under Apache-2.0 (scoped strictly to initialization; full attribution in [experiments/packed_corrections/README.md](experiments/packed_corrections/README.md) and [third_party/NanoQuant/NOTICE](experiments/packed_corrections/third_party/NanoQuant/NOTICE)).
