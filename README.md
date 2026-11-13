<p align="center">
  <img src="docs/assets/language_u_logo.png" alt="Language-U: The Glyph — One Symbol, Three Meanings (Google Gemma 4 SWE Agents • S⁷ Octonion)" width="760">
</p>

# Language-U × Gemma 4

**One discrete $S^7$ macro-glyph architecture unifying 256-weight neural storage, 20.4-token propositional packing (`LUJ1`), fail-closed $H_8$/$H_{64}$ codebase retrieval, $18.1\times\text{–}21.2\times$ Flash-Glyph KV attention (`GlyphAttentionV2`), high-speed 2D optical DMA (`OpticalCarrierV1`), and bounded-RAM ARM64 NEON / x86-64 AVX2 / Rust streaming execution of Gemma 4 31B inside an 8 GB Raspberry Pi 4.**

---

## Quick Navigation

- [What Is Proven on Physical Silicon (Windows AVX2 + Raspberry Pi 4 ARM64 NEON)](#what-is-proven-on-physical-silicon)
- [Empirical Precision Frontier & Honest Boundaries](#empirical-precision-frontier--honest-boundaries)
- [Architecture Overview — The Unified Stack](#architecture-overview)
- [Windows AVX2+OpenMP Benchmark: 256 Weights & 20.4 Tokens/Glyph (`LUJ1`)](#windows-avx2-joint-glyph-benchmark)
- [Raspberry Pi 4 (8 GB) Gemma 4 31B & Gemma 3 270M Hardware Receipts](#raspberry-pi-4-hardware-receipts)
- [Subsystem Deep-Dives (`Neon-Runner V2`, `GlyphAttentionV2`, `OpticalCarrierV1`, `Trinity Retrieval`, `Compiler`, `Agent`)](#subsystem-deep-dives)
- [Reproducible Verification & CLI Quickstart](#reproducible-verification--cli-quickstart)
- [Ecosystem & Attribution](#ecosystem--attribution)

---

<a id="what-is-proven-on-physical-silicon"></a>
## 1. What Is Proven on Physical Silicon

Every quantitative claim in this repository is backed by committed verification scripts, fail-closed self-test suites, and raw JSON/log telemetry measured on two physical edge hosts:
1. **Astronaut SHE (`Raspberry Pi 4 Model B 8GB`)**: ARMv8-A Cortex-A72 (`aarch64`), 4 OpenMP/Rust threads, Linux, external USB SSD (`/mnt/SHE-VAULT`).
2. **Windows x86-64 Host (`AMD64`, AVX2+FMA+OpenMP)**: Native MSVC C11/AVX2 shared library (`neon_runner/neon_stream_accelerator.dll`), freestanding MSVC QAT reference (`neon_runner/qat_reference.c`), Rust `--release`, and Python 3.14.

| Barrier Broken | Verified Result on Physical Hardware | Evidence / Receipt |
| :--- | :--- | :--- |
| **Gemma 4 31B Coherent Generation Inside < 4.6 GiB RAM (Pi 4)** | `run31b` (`--embed-mode q8 --pin-layers 10`) runs all **60 dense layers** + full **262,144-token** vocabulary head on Pi 4 inside **`4.51 GiB` peak RAM** (`vm_hwm_kb = 4,725,968`), generating `"William Shakespeare wrote the play *Hamlet*.<turn|>"`. ARM64 NEON intrinsics + single-projection async SSD prefetch cut prefill **`431.36 s -> 290.12 s` (`1.49x` faster)** and decode **`614.32 s -> 431.90 s` (`1.42x` faster)**. | [`a_hamlet_q8emb_pin10_neon.log`](experiments/run270_rust/evidence/run31b/a_hamlet_q8emb_pin10_neon.log) vs. [`hamlet_q8emb_pin10.txt`](experiments/run270_rust/evidence/run31b/hamlet_q8emb_pin10.txt) |
| **Full 262,144-Way 31B Logit Parity Gate (`rel_l2 < 0.01`, Pi 4)** | Comparing bounded-RAM `--embed-mode q8 --pin-layers 10` (`4.40 GiB` HWM) against exact BF16-stream `--pin-layers 0` (`474.0 MiB` HWM) on `"What is 2 + 2? Answer with only the number."`: **`cosine = 0.999995`**, **`rel_l2 = 0.003402`**, exact **top-1** and **top-5** token match (`236812:"4"`). | [`run31b_parity_2plus2.json`](experiments/run270_rust/evidence/run31b/run31b_parity_2plus2.json) |
| **Lossless Packed 92B (`2.875 bpw`) OPT4 Full 31B Coherence (Pi 4)** | `neon_runner/ramfit/reencode31b` losslessly packs 132B RVQ glyphs into **92 bytes** (`0.0` decoded weight difference). Streaming the `10.52 GB` 92B `packed92` stream (`114.40M` glyphs) with proper Gemma 4 chat tokens produces **`"2 + 2 = 4<turn|>"`** end-to-end (`56.6–61.4 s/step`). | [`c_packed92_chat_2plus2.log`](experiments/run270_rust/evidence/run31b/c_packed92_chat_2plus2.log) & [`selftest.txt`](experiments/run270_rust/evidence/ramfit_logs/selftest.txt) |
| **256 Weights + 20.4 Tokens/Glyph (`LUJ1`) Shrink & Fast AVX2 Processing (Windows)** | On pinned Gemma 4 31B `layers.0.self_attn.q_proj` (`128 × 5,376` = `2,688` real glyphs) + `54,834` real Gemma 4 tokens (`20.400` tok/glyph across `2,685` glyphs): **`OPT5 36B` (`14.22x` shrink)** executes at **`0.24753 ms/pass`** = **`10.86M glyphs/s` (`2.78B weights/s`, `221.52M equiv tok/s`)**; **lossless `OPT4 92B` (`5.57x` shrink)** executes at **`1.67M glyphs/s` (`427.9M weights/s`)**; bit-exact **`QAT INT4` (`3.56x` shrink)** executes at **`1.59M glyphs/s` (`406.6M weights/s`)**. | [`windows_joint_fast_benchmark.json`](experiments/joint_glyphs/evidence/windows_joint_fast_benchmark.json) & [`benchmark_joint_windows.py`](experiments/joint_glyphs/benchmark_joint_windows.py) |
| **128K Flash-Glyph KV Cache (`GlyphAttentionV2`, Pi 4)** | Stores attention keys/values directly as quantized $S^7$ macro-glyphs in L1/ring arenas: **`48.27 MB`** vs `1,024.00 MB` vLLM PagedBF16 at **`128K` context (`21.21x` compression)** at **`713.98 ms` (`1.60x` faster)**, or `132.25 MB` (`7.74x`) at `o_proj_cos = 0.97574`; sliding window (`W=1,024`) runs **`4.36 ms` (`1.76x` faster)** at `0.99597` cosine. | [`glyph_attention_v2_pi4_benchmark_report.json`](docs/glyph_attention_v2_pi4_benchmark_report.json) |
| **Lossless Optical Context DMA (`OpticalCarrierV1`)** | Projects glyph indices + token streams into CRC32-verified 2D `AIQC` grids with **`0.244–0.477 ms` in-memory DMA** (`6,522.2 FPS`, **`626,129 tok/s`**, `100%` bit parity) on Windows, and **`350.5 tok/s`** 6-gate self-test on Pi 4. | [`optical_context_gemma4_benchmark_report.json`](docs/optical_context_gemma4_benchmark_report.json) & [`GEMMA4_31B_OPTICAL_LIVE_BENCHMARK_REPORT.md`](docs/GEMMA4_31B_OPTICAL_LIVE_BENCHMARK_REPORT.md) |
| **$H_8$/$H_{64}$ Trinity Hybrid Retrieval + Fail-Closed Sieve** | Compresses `1,962` 768-D codebase embeddings **`96.0x` (`6,027,264 B -> 62,784 B`)**, achieves **`8/10` in-domain accuracy** (`0.048 ms` latency) vs `3/10` BM25 and `7/10` Dense768 while dropping **OOD false positives from `100.0%` to `0.0%`**. | [`swe_navigation_benchmark_results.json`](trinity_retrieval/swe_navigation_benchmark_results.json) & [`trinity_retrieval/README.md`](trinity_retrieval/README.md) |

---

<a id="empirical-precision-frontier--honest-boundaries"></a>
## 2. Empirical Precision Frontier & Honest Boundaries

Engineering rigor requires stating **both** what works and **exact failure thresholds** observed in hardware experiments:

1. **Why Raw ~1.3 bpw (`36B OPT5` / `b1` 2-means) Collapses Full Transformers:**
   Across three independent evaluations ([`docs/FULL_TEXT_GATE_REVIEW.md`](docs/FULL_TEXT_GATE_REVIEW.md), [`experiments/gemma270_luj/validation/README.md`](experiments/gemma270_luj/validation/README.md), and the 23-config Pareto frontier in [`sweep_bench5_all23.json`](experiments/run270_rust/evidence/g270_sweep/sweep_bench5_all23.json)), applying uncalibrated ~`1.27–1.3 bpw` 1-bit/ternary quantization across all 18–60 transformer layers degrades multi-layer representations into incoherence (`cos = -0.501`, `0/5` argmax match, `0/28` frozen / `0/30` fresh reasoning score).
2. **Where Coherence and Strict Parity Actually Cross Zero-Error Thresholds (Gemma 3 270M Pareto Law):**
   On Pi 4 via [`experiments/run270_rust`](experiments/run270_rust/FRONTIER_RECEIPTS.md):
   - **`q4+h8_eq4` (`4.50 bpw`, `151.0 MB` model, `450.0 MB` peak RAM)**: First coherent generation threshold (`"The play Hamlet is attributed to **W"`, `cos = 0.890`).
   - **`q5+h8` (`5.50 bpw`, `404.7 MB`)**: Matches/exceeds exact BF16 blind reasoning ceiling (`8/28` frozen, `10/30` fresh).
   - **`q8_eq8` (`8.50 bpw`, `285.0 MB` total, `577.8–578.0 MB` peak RAM)**: **`5/5` argmax match** (`cos = 0.99919`), identical **`10/28` frozen** and **`6/30` fresh** accuracy to exact BF16 at `0.123–0.161 s/forward`.
   - **`q10+h8` (`11.00 bpw`)**, **`q12-norms`** (`--preserve-norms`), and **`q14-norms`**: Pass the strict Hugging Face `transformers==4.55.0` FP32 logit gate on **5/5 prompts** (`rel_l2 = 0.007393`, `<= 0.001883`, and `<= 0.000393`, respectively; see [`q12_q14_native_evidence.json`](experiments/gemma270_luj/evidence/q12_q14_native_evidence.json)).
   - *Note on prior `LUJ1` norm bug fixed here*: Preserving 1D RMSNorm gain tensors bitwise (`--preserve-norms`, e.g. outliers like `1648`) is mandatory so extreme 1D gains do not corrupt adjacent group-32 scales.
3. **Sub-2-Bit Calibrated Work Remains Layer-0 Research:**
   [`experiments/packed_corrections`](experiments/packed_corrections/README.md) improves Layer-0 `q_proj` (`1.08854 bpw` BFP1 + rank-8 FP16 factor) to cosine `0.96504` (`0.06871` relative squared error), and the learned 5-glyph [`NEURAL_PLANNER.md`](experiments/joint_glyphs/NEURAL_PLANNER.md) (`1,280` weights, `1,269 B`) reaches **`91.67%` (`44/48`)** held-out operator accuracy. Neither is claimed as a full-model LLM replacement; full 31B execution in 8 GB RAM uses `QAT W4A16 + Q8_0` (`4.51 GiB` HWM) or `92B`/`132B` `OPT4` mmap streams.
4. **Honest 92B vs. 36B Tokenization Clarifier:**
   Earlier automated `langut-infer` runs on `packed92` (`10.52 GB` 92B OPT4) emitted gibberish even on the `v3book` control because the test script passed literal ASCII characters of `<|turn>` instead of special token IDs. Re-running `packed92/mixed_stream.bin` with official Gemma 4 special token IDs (`[2, 105, 2364, 107, ...]`) restores clean **`"2 + 2 = 4<turn|>"`** output on Pi 4.

---

<a id="architecture-overview"></a>
## 3. Architecture Overview

All modules map onto the unit 7-sphere $S^7 \subset \mathbb{R}^8$ via orthonormal **Sylvester-Hadamard $H_8$** (and $H_{64}$) rotations, obeying the **Zero-Remainder Census Law** ($R = (d_{\text{out}} \cdot d_{\text{in}}) \bmod 256 \equiv 0$):

```
+=======================================================================================+
|                    LANGUAGE-U x GEMMA 4 UNIFIED ARCHITECTURE (S^7)                    |
+=======================================================================================+
|  [1] STORAGE & EXECUTION          [2] MEMORY & CONTEXT BUS    [3] SEMANTIC / AGENT    |
|  -----------------------          ------------------------    ----------------------  |
|  - 256w Macro-Glyphs:             - GlyphAttentionV2 KV:      - Trinity Retrieval:    |
|    * 36B/48B OPT5 (1.125-1.5bpw)    21.2x global 128K KV        H8->S^7 (96x shrink)  |
|    * 92B Packed OPT4 (2.875bpw)     compression (48.27 MB)      H64 bitpack + Sieve   |
|    * 132B OPT4 RVQ (4.125bpw)     - OpticalCarrierV1 AIQC:    - Joint LUJ1 + Planner: |
|    * Group-32 QAT W4A16/Q4-Q14      CRC32 2D grid DMA           256w + 20.4 tok/glyph |
|  - Native Kernels:                  (0.24-0.48 ms, 6,522 FPS)   1,269 B 5-glyph bank  |
|    * AVX2+FMA+OpenMP (.dll)                                   - 6D Cuneiform Compiler |
|    * ARMv8-A NEON+OpenMP (.so)                                - KOFI v11 SWE Agent    |
|    * Rust NEON + Async Prefetch                                                       |
+=======================================================================================+
```

---

<a id="windows-avx2-joint-glyph-benchmark"></a>
## 4. Windows AVX2+OpenMP Benchmark: 256 Weights & 20.4 Tokens/Glyph (`LUJ1`)

Executed cleanly via [`experiments/joint_glyphs/benchmark_joint_windows.py`](experiments/joint_glyphs/benchmark_joint_windows.py) against real pinned bytes from `google/gemma-4-31B-it-qat-w4a16-ct` (`layers.0.self_attn.q_proj`, `128 × 5,376` = `688,128` weights = `2,688` glyphs) and Gemma 4 `tokenizer.json` (`54,834` tokens; exact `20.4000` tok/glyph over the `2,685`-glyph / `54,774`-token prefix) on Windows x86-64 (4 CPU threads):

- **Canonical 5-Glyph `LUJ1` Roundtrip (`1,280` weights, `102` tokens, `20.4` mean tok/glyph, checked DAG result `84`)**:
  - `qat_exact`: **`1,269 B`** (bit-exact FP32 source restoration, Python verify `2.36 ms`)
  - `opt5_repair_fp16`: **`2,520 B`** (bit-exact FP32 source restoration)
  - `opt5_approximate`: **`729 B`** (lossy storage control; rejects exact weight claim)
- **Native Windows AVX2 (`neon_stream_accelerator.dll`, `63,488 B`) & Freestanding MSVC+OpenMP `qat_reference.c` Throughput**:

| Kernel Path (`2,688` Real 256w Glyphs = `688,128` Weights) | Shrink vs BF16 (`1.376 MB`) | Fidelity / Parity vs Decode | Latency (`ms/pass`) | Glyph Rate (`glyphs/s`) | Weight Throughput | Equiv. Token Rate (`@ 20.4 tok/glyph`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`OPT5 36B` AVX2+OpenMP GEMV** | **`14.22x`** (`96,768 B`) | `max err 9.54e-07`, `cos 0.8016` | **`0.24753 ms`** | **`10,859,290 /s`** | **`2,779.98 M/s`** | **`221.52 M tok/s`** |
| **`OPT4 132B` FWHT-8 AVX2 GEMV** | `3.88x` (`354,816 B`) | `max err 1.01e-06`, `cos 0.9356` | **`0.28720 ms`** | **`9,359,332 /s`** | **`2,395.99 M/s`** | **`190.93 M tok/s`** |
| **Lossless `OPT4 92B` AVX2 GEMV** | **`5.57x`** (`247,296 B`) | **`0.0` bit diff vs 132B**, `err 1.31e-06` | **`1.60810 ms`** | **`1,671,538 /s`** | **`427.91 M/s`** | **`34.10 M tok/s`** |
| **`OPT5 36B` GEMV + 18-bit Token Unpack** | `14.22x` + `124 KB` tok | 100% of `54,834` tokens lossless | **`1.21704 ms`** | **`2,208,637 /s`** | **`565.41 M/s`** | **`45.06 M tok/s`** |
| **Bit-Exact `QAT4` MSVC+OpenMP DLL** | **`3.56x`** (`387,072 B`) | `max err 1.67e-06`, bit-exact | **`1.69232 ms`** | **`1,588,352 /s`** | **`406.62 M/s`** | **`32.40 M tok/s`** |

Telemetry source: [`experiments/joint_glyphs/evidence/windows_joint_fast_benchmark.json`](experiments/joint_glyphs/evidence/windows_joint_fast_benchmark.json).

---

<a id="raspberry-pi-4-hardware-receipts"></a>
## 5. Raspberry Pi 4 (8 GB) Hardware Receipts

### 5.1 Gemma 4 31B Full 60-Layer Inference on Astronaut SHE (`run31b` & `langut-infer`)

| Mode / Runtime | Disk Size | Peak RAM (`VmHWM`) | Prefill Latency | Decode Latency | Output / Gate Result | Receipt |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **`run31b` NEON + Async Prefetch (`q8` emb + `pin 10`)** | 22 GB QAT source (streamed) | **`4.51 GiB`** (`4,725,968 KB`) | **`290.12 s`** (19 tok, **`1.49x`** speedup) | **`431.90 s`** (9 steps, **`1.42x`** speedup) | `"William Shakespeare wrote the play *Hamlet*.<turn|>"` | [`a_hamlet_q8emb_pin10_neon.log`](experiments/run270_rust/evidence/run31b/a_hamlet_q8emb_pin10_neon.log) |
| **`run31b` Baseline (Scalar, pre-NEON)** | 22 GB QAT source (streamed) | `4.23 GiB` (`4,436,992 KB`) | `431.36 s` | `614.32 s` | `"William Shakespeare wrote the play *Hamlet*.<turn|>"` | [`hamlet_q8emb_pin10.txt`](experiments/run270_rust/evidence/run31b/hamlet_q8emb_pin10.txt) |
| **`run31b` BF16-Stream Exact (`pin 0`, `"2+2"`)** | 22 GB QAT source (streamed) | **`474.0 MiB`** (`485,348 KB`) | `662.21 s` (27 tok) | — (1 step) | Top-1 `236812:"4"` (`26.483`) | [`b1_stream_2plus2.log`](experiments/run270_rust/evidence/run31b/b1_stream_2plus2.log) |
| **`run31b` 31B Logit Parity Gate (`q8 pin 10` vs BF16)** | — | **`4.40 GiB`** (`4,609,548 KB`) | `368.80 s` (27 tok) | — (1 step) | **`cos = 0.999995`, `rel_l2 = 0.003402` (`PASS`)**, Top-5 exact | [`run31b_parity_2plus2.json`](experiments/run270_rust/evidence/run31b/run31b_parity_2plus2.json) |
| **`langut-infer` Lossless `92B` `packed92` (`OPT4`)** | **`10.52 GB`** (`114.4M × 92 B`) mmap | `< 8 GB` (evictable mmap + head) | `426.85 s` (21 tok) | `573.73 s` (8 steps, `56.6–61.4 s/step`) | **`"2 + 2 = 4<turn|>"`** | [`c_packed92_chat_2plus2.log`](experiments/run270_rust/evidence/run31b/c_packed92_chat_2plus2.log) |

### 5.2 Gemma 3 270M IT Sub-Byte Pareto Sweep (Pi 4, Selected Key Points from 23 Configs)

| Config (`run270`) | Proj `bpw` | Total Weights | Peak `VmHWM` | `s / fwd` | Worst `cos` (5 prompts) | Max `rel_l2` | Argmax / Gate | Sample Output (`Hamlet`) & Blind Suites |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`exact_bf16`** | `32.000` | `736.85 MB` | `1,009.0 MB` | `0.1996 s` | `1.000000` | `0.000008` | `5/5` / `5/5` | `"The play Hamlet was written by Willi..."` (`10/28` frozen, `6/30` fresh) |
| **`q12`** | `13.000` | `498.71 MB` | `781.7 MB` | `0.3924 s` | `0.999998` | `0.002328` | `5/5` / `5/5` | Strict logit gate `5/5` PASS |
| **`q10+h8`** | `11.000` | `473.64 MB` | `757.6 MB` | `0.3782 s` | `0.999973` | `0.007393` | `5/5` / `5/5` | Strict logit gate `5/5` PASS (`10/28` frozen, `6/30` fresh) |
| **`q8_eq8`** | `8.500` | `285.02 MB` | `578.0 MB` | `0.1607 s` | `0.999191` | `0.061950` | `5/5` / `0/5` | **`10/28` frozen, `6/30` fresh** (matches exact BF16 at `2.6x` smaller weights) |
| **`q5+h8`** | `5.500` | `404.70 MB` | `691.9 MB` | `0.2882 s` | `0.951566` | `0.330993` | `4/5` / `0/5` | `8/28` frozen, **`10/30` fresh** |
| **`q4+h8_eq4`** | **`4.500`** | **`151.00 MB`** | **`450.0 MB`** | `0.1565 s` | `0.890325` | `0.536594` | `2/5` / `0/5` | First coherent emergence (`"The play Hamlet is attributed to **W"`) |
| **`q3` / `b1` / `b1+h8`** | `3.500` / `1.271` | `379.6` / `351.7 MB` | `668` / `641 MB` | `0.26` / `0.16 s` | `0.208` / `-0.501` | `>= 0.986` | `0/5` / `0/5` | Coherent collapse (`0/28` frozen, `0/30` fresh) |

Full 23-config telemetry: [`experiments/run270_rust/evidence/g270_sweep/sweep_bench5_all23.json`](experiments/run270_rust/evidence/g270_sweep/sweep_bench5_all23.json).

---

<a id="subsystem-deep-dives"></a>
## 6. Subsystem Deep-Dives

1. **`Neon-Runner V2` & Cross-Platform C Runtime ([`neon_runner/`](neon_runner/), [`docs/NEON_RUNNER_INTEGRATION_V2.md`](docs/NEON_RUNNER_INTEGRATION_V2.md))**
   - Unified C11 SIMD accelerator (`neon_stream_accelerator_v2.c` / `.h`) supporting ARMv8-A NEON (`libneon_stream_accelerator.so`, `3.15M glyphs/s` = `807.7M params/s` on Pi 4 via `bridge.py --self-test`), x86-64 AVX2+FMA (`neon_stream_accelerator.dll`, `10.86M glyphs/s` = `2.78B weights/s` Opt5 AVX2 GEMV; `382.6K glyphs/s` Python `bridge.py --self-test` loop), freestanding MSVC QAT (`qat_reference.c`), 64-bit stream census (`stream_census.c`), and `ramfit` C packers (`reencode31b`, `embed_encode`, `shuffle_stream`).
2. **`GlyphAttentionV2` 128K KV-Cache Engine ([`neon_runner/glyph_attention.py`](neon_runner/glyph_attention.py))**
   - Single-pass Flash-Glyph GQA kernel (`flash_glyph_attention_gqa_c`) with $O(1)$ 19-comparator sorting network (`encode_kv_glyph_fast_c`, `46.27 us/tok` Opt5 48B, `165.19 us/tok` Opt4 132B on Pi 4), algebraic 4-bit Nibble-LUT/Histogram (`Opt5`), and Dual-Road sink+tail LSE fusion (`latency = 19.75 ms` at `18.1x` on Pi 4 self-test; `30.61 ms` at `18.1x` on Windows self-test).
3. **`OpticalCarrierV1` 2D Physical Macro-Glyph Bus ([`neon_runner/optical_carrier.py`](neon_runner/optical_carrier.py), [`docs/OPTICAL_CONTEXT_INTEGRATION_V2_2.md`](docs/OPTICAL_CONTEXT_INTEGRATION_V2_2.md))**
   - Encodes token sequences and $S^7$ macro-glyph IDs into 20-byte `AIQC` frames guarded by IEEE 802.3 CRC32 (`0.498 ms` parity check & `353.7 tok/s` prefill on Pi 4 self-test; `0.117 ms` DMA & `563.3 tok/s` on Windows self-test).
4. **Trinity Hybrid Retrieval ([`trinity_retrieval/`](trinity_retrieval/README.md))**
   - $H_8 \to S^7$ (`96.0x` index shrink) + $H_{64}$ 1-bit binary packing (`max error 0.0`) + 3-scale `NoiseStabilitySieve` (`79.1 us/query`, `0.0%` OOD false positives) + `AirTrinityLayerStreamer`, `EdgeLoRAStreamer`, and `SparseOctonionStreamer`.
5. **6D Cuneiform Compiler (`compiler/`) & Gemma 4 SWE Agent (`agent/`)**
   - [`compiler/glyphs.py`](compiler/glyphs.py) & [`compiler/glyph_lang.py`](compiler/glyph_lang.py) lower `[Action] + [Target] + [Context] + [Constraint]` propositions into 24-bit 6D Cuneiform opcodes (`RC | (RF << 8) | (RA << 16)`).
   - [`agent/build.py`](agent/build.py) deterministically builds and structurally validates `agent/submission.zip` (KOFI v11 Gemma 4 SWE Agent bundle).

---

<a id="reproducible-verification--cli-quickstart"></a>
## 7. Reproducible Verification & CLI Quickstart

### Python & Native C Verification (Windows x86-64 AVX2 or Linux ARM64 NEON)

```bash
# 1. Build native C library (AVX2+OpenMP DLL on Windows, NEON+OpenMP .so on ARM64 Linux)
python -m neon_runner.build_native --force

# 2. Run 6-gate fail-closed hardware self-test batteries
python neon_runner/glyph_attention.py --self-test
python neon_runner/bridge.py --self-test
python neon_runner/optical_carrier.py

# 3. Run unit tests (55 core tests on Linux/Pi4; 52 pass + 3 Linux-census skip on Windows)
python -m unittest discover -s tests -v
python -m unittest discover -s tests/retrieval -v
python trinity_retrieval/edge_self_test.py

# 4. Run 256-weight / 20.4-token joint LUJ1 + Windows AVX2 benchmark
python experiments/joint_glyphs/benchmark_joint_windows.py
python experiments/joint_glyphs/run_joint_proof.py --help

# 5. Validate KOFI v11 SWE agent bundle & offline Pi 4 preparation sample
python agent/build.py
python scripts/verify_pi4_preparation.py
```

### Rust NEON / AVX2 Engines (`experiments/run270_rust/`) & C++ `gemma270_luj`

```bash
# Rust tests & release binaries (add RUSTFLAGS="-C target-cpu=cortex-a72 -D warnings" on Pi 4)
cd experiments/run270_rust
cargo test --release
cargo build --release --bin run270 --bin run31b

# Example bounded-RAM Gemma 4 31B execution (< 4.6 GiB on 8 GB Pi 4)
./target/release/run31b \
  --model /mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/model.safetensors \
  --config /mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/config.json \
  --tokenizer /mnt/SHE-VAULT/gemma-4-31b-it-qat-w4a16-ct/tokenizer.json \
  --threads 4 --embed-mode q8 --pin-layers 10 \
  --prompt-text "Who wrote the play Hamlet?" --max-tokens 12

# C++ 18-layer Gemma 3 270M LUJ1 tools & strict parity checker (Linux/macOS g++)
make -C experiments/gemma270_luj -B all inference compare_logits
./experiments/gemma270_luj/luj_gemma270 selftest
bash experiments/gemma270_luj/tests/test_strict_parity.sh
```

---

<a id="ecosystem--attribution"></a>
## 8. Ecosystem & Attribution

`Language-U-Gemma4` brings together components across the **Language-U / Consider-Trinity** ecosystem (`DannyB-bit/*` & `TheAiCollectiveART/*`):

- **Execution & Streaming**: [`DannyB-bit/Neon-Runner`](https://github.com/DannyB-bit/Neon-Runner), [`DannyB-bit/Language-U-Transformer`](https://github.com/DannyB-bit/Language-U-Transformer), [`DannyB-bit/Zymatica-Engine`](https://github.com/DannyB-bit/Zymatica-Engine), [`DannyB-bit/Zymatica-Rust-Body`](https://github.com/DannyB-bit/Zymatica-Rust-Body)
- **Retrieval & Context**: [`DannyB-bit/trinity-hybrid-retrieval`](https://github.com/DannyB-bit/trinity-hybrid-retrieval), [`DannyB-bit/lossless-optical-llm-context-caching`](https://github.com/DannyB-bit/lossless-optical-llm-context-caching), [`DannyB-bit/Consider-Trinity-Coder`](https://github.com/DannyB-bit/Consider-Trinity-Coder), [`DannyB-bit/Consider-Trinity`](https://github.com/DannyB-bit/Consider-Trinity)
- **Compression, Compiler & Geometry**: [`DannyB-bit/real-language-u-compression`](https://github.com/DannyB-bit/real-language-u-compression), [`DannyB-bit/Language-u-Process`](https://github.com/DannyB-bit/Language-u-Process), [`DannyB-bit/Language-U-Compiler`](https://github.com/DannyB-bit/Language-U-Compiler), [`DannyB-bit/Cuneiform-8D-Process`](https://github.com/DannyB-bit/Cuneiform-8D-Process), [`DannyB-bit/Glyphs-9b`](https://github.com/DannyB-bit/Glyphs-9b), [`DannyB-bit/glyphs-9b-turbo`](https://github.com/DannyB-bit/glyphs-9b-turbo), [`DannyB-bit/ALIEN-CODE`](https://github.com/DannyB-bit/ALIEN-CODE), [`DannyB-bit/Compass-Magnetism-Training-Law`](https://github.com/DannyB-bit/Compass-Magnetism-Training-Law), [`DannyB-bit/ZVID`](https://github.com/DannyB-bit/ZVID)
- **Pinned Models & Third-Party Notices**:
  - [`google/gemma-4-31B-it-qat-w4a16-ct`](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct) & [`TheAiCollectiveART/Glyph-Stream-31b`](https://huggingface.co/TheAiCollectiveART/Glyph-Stream-31b) ([Hugging Face Org](https://huggingface.co/TheAiCollectiveART/models))
  - Agent & NanoQuant Apache-2.0 attributions in [`agent/NOTICE.txt`](agent/NOTICE.txt) and [`experiments/packed_corrections/third_party/NanoQuant/NOTICE`](experiments/packed_corrections/third_party/NanoQuant/NOTICE).

---

All Rights Reserved © 2026 [DannyB-bit](https://github.com/DannyB-bit).
