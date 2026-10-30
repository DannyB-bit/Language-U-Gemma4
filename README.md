# Language-U: Gemma 4 E4B & 31B Unified System (`v2.0.0`)

**The Discrete Glyph Standard for Local Gemma 4 SWE Agents & Neon-Runner Edge Execution**  
*Unifying 256-Weight $S^7$ Storage, 20.4-Token Propositions, and Self-Verifying Machine Code*

---

## Architecture Overview

Language-U breaks the storage, context, and execution bottlenecks of offline developer agents by unifying model weights, codebase semantics, and native machine execution into a single discrete data type: the **Macro-Glyph** on the octonionic unit 7-sphere ($S^7$).

```
+===================================================================================+
|                      THE MACRO-GLYPH: ONE SYMBOL, THREE NATURES                   |
+===================================================================================+
|  STAGE 1: PROCESS (Language-u-Process)                                            |
|  - 256 weights packed into 36B/48B (Opt5 Ternary) or 132B (Opt4 RVQ) on S^7.      |
|  - Zero-Remainder Census Law (R = 0 mod 256) universally across Gemma 4 E4B/31B.  |
|                                                                                   |
|  STAGE 2: COMPRESSION (real-language-u-compression)                               |
|  - Sylvester-Hadamard H_8 Orthogonal Rotation (||H_8 H_8^T - I_8||_F < 1e-12).    |
|  - 96.0x Index Compression (embeddinggemma-2: 5.28 MB -> 61.3 KB resident on S^7). |
|  - Two-Tier Trust Law (STRONG, WEAK, MISS) with 0 false positives (35/35 pass).  |
|                                                                                   |
|  STAGE 3: COMPILER & NEON-RUNNER BRIDGE (Language-U-Gemma4-NeonRunner-Bridge-V2)  |
|  - 16-Op Semantic ALU on so(8) Lie algebra + Unified ARMv8 NEON / x86-64 AVX2     |
|    stream decoder & fused GEMV (`libneon_stream_accelerator` + `libtrinity_gemv`).|
|  - Native interoperability with `DannyB-bit/Neon-Runner` (v1 & V2 ABIs).          |
+===================================================================================+
```

---

## Repository Structure

* **`neon_runner/` (`Language-U-Gemma4-NeonRunner-Bridge-V2`, `v2.0.0`)**:
  * `neon_stream_accelerator_v2.c` / `.h`: Unified C11 + ARMv8 NEON + x86-64 AVX2/FMA + OpenMP stream decoder & zero-heap fused GEMV supporting both Option 4 (132B RVQ, FWHT-8 & Direct) and Option 5 (36B & 48B Top-4 outlier), exporting 100% drop-in compatible symbols for [`DannyB-bit/Neon-Runner`](https://github.com/DannyB-bit/Neon-Runner) and `UniversalLayerOrderedCortex`.
  * `bridge.py`: `NeonRunnerBridgeV2` & `NeonStreamAccelerator` Python runtime with parameterized stream/layout paths, automatic $H_8$ input pre-rotation for unmodified `Neon-Runner` v1 `.so` binaries, and a 6-gate fail-closed `_self_test()` battery.
  * `codebooks.py` & `codebooks/*.bin`: Deterministic `CodebookRegistryV2` + physical 64 KB `s7_codebook_2048x8.bin` (True North `#1419` sentinel verified) and `trit_codebook_2048x8.bin`.
  * `build_native.py` & `Makefile`: Zero-configuration native shared library builder for Linux/ARMv8, Linux/x86-64, macOS, and Windows x64.
* **`gemma4_e4b/`**:
  * `consider_trinity_e4b_core.py`: Full Gemma 4 E4B Census Law validator, $S^7$ projection, and Semantic ALU execution gatekeeper.
  * `encode_gemma4_e4b_to_glyphs.py`: Streaming encoder converting `google/gemma-4-E4B-it` safetensors directly into `Neon-Runner`-ready `.lanu` Option 4 (`neon-opt4`) and Dual-Road (`hybrid`) macro-glyph streams + `tensor_layout.json`.
* **`compression/`**:
  * `language_u_codec.py`: Sylvester-Hadamard $H_8$ rotation and octonionic codebook quantization.
  * `compression_engine.py`: Multi-mode RVQ and ternary lattice packing engine.
  * `language_u_embeddinggemma_adapter.py`: 768-D MRL embedding adapter for `google/embeddinggemma-2`.
* **`compiler/`**:
  * `glyphs.py`: 16-Op Semantic ALU instruction definitions and register machines.
  * `glyph_lang.py`: Compiler pipeline translating glyph opcodes to constant-folded C and native machine code.
* **`model/`**:
  * `gemma4_e4b_sample_layer0_q_proj.glyph.bin`: Physical sample of converted Gemma 4 E4B Layer 0 Q-projection weights packed into 48-byte Outlier Option 5 macro-glyphs on $S^7$.
  * `gemma4_e4b_sample_layer0_q_proj.neon_opt4.lanu` & `.layout.json`: Physical Gemma 4 E4B Layer 0 Q-projection `[16, 2560]` slice encoded in 132-byte Option 4 RVQ (`compass_s7` + `fwht8`, Cosine `0.9743`, SNR `12.88 dB`) directly executable in `Neon-Runner`.
  * `gemma4_e4b_sample_neon_hybrid_v2.lanu` & `.layout.json`: Multi-layer Dual-Road sample (Layer 0 Opt4 132B + Layer 5 Opt5 48B).
* **`agent/`**:
  * `submission.zip`: The fleet's **v11 competition bundle** (KOFI, 2026-10-19) — the measured-max Gemma 4 Developer Agent submission: flexonafft Budget-Aware V10 base (public 0.22 badge) + four bounded evidence-backed prompt splices (issue contract, coverage checkpoint, final-edit verification, repo-specific verification focus). Declares `model: gemma-4-31b-it-qat-w4a16-ct`, temp 0.2, 4096 thinking, 28 calls/8 min/80 turns. The earlier E4B custom-machinery bundle (three 0.00 scores) remains in git history as the documented anti-pattern; see `docs/AGENT_V11_ANALYSIS.md` for the full score study.
  * `build.py`: deterministic offline builder + structural validator (SHA-256 fb461a4e…25a3a, timestamp-pinned since the Zymatica review PR).
* **`docs/`**:
  * `NEON_RUNNER_INTEGRATION_V2.md`: Complete V1 vs. V2 forensic comparison matrix, wire protocol specification, and `DannyB-bit/Neon-Runner` integration runbook.
  * `Language-U_Architecture_Infographic.jpg`: Verified high-resolution system architecture and mathematical geometry diagram.
  * `paper_arxiv.tex`: Full NeurIPS/IEEE LaTeX preprint manuscript.
  * `RESEARCH_WRITEUP.md`: Complete hackathon research paper writeup.

---

## Neon-Runner Integration Quickstart (`v2.0.0`)

```bash
# 1. Build native C/SIMD library (ARMv8 NEON .so on Pi 4 / Linux, AVX2 .dll/.so on x86-64)
python neon_runner/build_native.py

# 2. Run the 6-gate fail-closed physical verification battery
python neon_runner/bridge.py --self-test

# 3. Encode Gemma 4 safetensors into a Neon-Runner .lanu stream + layout JSON
python gemma4_e4b/encode_gemma4_e4b_to_glyphs.py \
  --safetensors /path/to/model.safetensors \
  --mode neon-opt4 \
  --codebook compass_s7 \
  --out-stream model/gemma4_e4b_neon_opt4.lanu \
  --out-layout model/gemma4_e4b_neon_opt4.layout.json
```

---

## Physical Hardware Verification

All benchmarks were validated on physical hardware (ARM Cortex-A72 @ 1.80 GHz Debian 13, x86-64 AVX2+OpenMP workstation, and Kaggle Tesla L4 GPU cluster):
* **Census Law:** $100\%$ $R \equiv 0 \pmod{256}$ across all $4,917,821,440$ core weights ($7,351,042,048$ total active weights / $28,715,008$ glyphs) of `google/gemma-4-E4B-it` and $114,401,280$ glyphs of `gemma-4-31b-it`.
* **Index Footprint:** 5.28 MB $\to$ 61.3 KB ($96.0\times$ compression) fitting entirely in L1/L2 cache.
* **Abstention Gate:** 35/35 Passing ($0.0\%$ False Positives) under the Two-Tier Trust Law.
* **NEON / AVX2 SIMD Component Throughput:** 2.86M - 13.13M glyphs/s on ARMv8 NEON; 3.21M glyphs/s (822.7 M FP32 params/s, 53.53 ms per 44.04M projection) on x86-64 OpenMP+AVX2 with 0.0 MB weight-matrix heap allocation.
* **Compiler & GEMV Parity:** Native ELF AArch64 == Reference VM (`max_err = 0.00e+00`); Fused `gemv_opt4_fwht_mmap` == Decoded Matrix GEMV (`max_err = 6.10e-05`).

---

## License & Privacy

Proprietary Research — Private Repository.  
All Rights Reserved © 2026.
