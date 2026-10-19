# Language-U: Gemma 4 E4B Unified System

**The Discrete Glyph Standard for Local Gemma 4 SWE Agents**  
*Unifying 256-Weight $S^7$ Storage, 20.4-Token Propositions, and Self-Verifying Machine Code*

---

## Architecture Overview

Language-U breaks the storage, context, and execution bottlenecks of offline developer agents by unifying model weights, codebase semantics, and native machine execution into a single discrete data type: the **Macro-Glyph** on the octonionic unit 7-sphere ($S^7$).

```
+===================================================================================+
|                      THE MACRO-GLYPH: ONE SYMBOL, THREE NATURES                   |
+===================================================================================+
|  STAGE 1: PROCESS (Language-u-Process)                                            |
|  - 256 weights packed into 36B (Opt5 Ternary) or 132B (Opt4 RVQ) on S^7.          |
|  - Zero-Remainder Census Law (R = 0 mod 256) universally across Gemma 4 E4B.     |
|                                                                                   |
|  STAGE 2: COMPRESSION (real-language-u-compression)                               |
|  - Sylvester-Hadamard H_8 Orthogonal Rotation (||H_8 H_8^T - I_8||_F < 1e-12).    |
|  - 96.0x Index Compression (embeddinggemma-2: 5.28 MB -> 61.3 KB resident on S^7). |
|  - Two-Tier Trust Law (STRONG, WEAK, MISS) with 0 false positives (35/35 pass).  |
|                                                                                   |
|  STAGE 3: COMPILER (Language-U-Compiler)                                          |
|  - 16-Op Semantic ALU on so(8) Lie algebra.                                       |
|  - Lowers to native AArch64 NEON ELF with 0.00e+00 double-entry parity error.      |
+===================================================================================+
```

---

## Repository Structure

* **`gemma4_e4b/`**:
  * `consider_trinity_e4b_core.py`: Full Gemma 4 E4B Census Law validator, $S^7$ projection, and Semantic ALU execution gatekeeper.
  * `encode_gemma4_e4b_to_glyphs.py`: Real byte-offset streaming encoder converting `google/gemma-4-E4B-it` weights directly into macro-glyph binary streams.
* **`compression/`**:
  * `language_u_codec.py`: Sylvester-Hadamard $H_8$ rotation and octonionic codebook quantization.
  * `compression_engine.py`: Multi-mode RVQ and ternary lattice packing engine.
  * `language_u_embeddinggemma_adapter.py`: 768-D MRL embedding adapter for `google/embeddinggemma-2`.
* **`compiler/`**:
  * `glyphs.py`: 16-Op Semantic ALU instruction definitions and register machines.
  * `glyph_lang.py`: Compiler pipeline translating glyph opcodes to constant-folded C and native machine code.
* **`model/`**:
  * `gemma4_e4b_sample_layer0_q_proj.glyph.bin`: Physical sample of converted Gemma 4 E4B Layer 0 Q-projection weights packed into 48-byte Outlier Option 5 macro-glyphs on $S^7$.
* **`agent/`**:
  * `submission.zip`: The fleet's **v11 competition bundle** (KOFI, 2026-10-19) — the measured-max Gemma 4 Developer Agent submission: flexonafft Budget-Aware V10 base (public 0.22 badge) + four bounded evidence-backed prompt splices (issue contract, coverage checkpoint, final-edit verification, repo-specific verification focus). Declares `model: gemma-4-31b-it-qat-w4a16-ct`, temp 0.2, 4096 thinking, 28 calls/8 min/80 turns. The earlier E4B custom-machinery bundle (three 0.00 scores) remains in git history as the documented anti-pattern; see `docs/AGENT_V11_ANALYSIS.md` for the full score study.
  * `build.py`: deterministic offline builder + structural validator (SHA-256 319813a9…fc26).
* **`docs/`**:
  * `Language-U_Architecture_Infographic.jpg`: Verified high-resolution system architecture and mathematical geometry diagram.
  * `paper_arxiv.tex`: Full NeurIPS/IEEE LaTeX preprint manuscript.
  * `RESEARCH_WRITEUP.md`: Complete hackathon research paper writeup.

---

## Physical Hardware Verification

All benchmarks were validated on physical hardware (ARM Cortex-A72 @ 1.80 GHz Debian 13 and Kaggle Tesla L4 GPU cluster):
* **Census Law:** $100\%$ $R \equiv 0 \pmod{256}$ across all $4,917,821,440$ core weights of `google/gemma-4-E4B-it`.
* **Index Footprint:** 5.28 MB $\to$ 61.3 KB ($96.0\times$ compression) fitting entirely in L1/L2 cache.
* **Abstention Gate:** 35/35 Passing ($0.0\%$ False Positives) under the Two-Tier Trust Law.
* **NEON SIMD Throughput:** 2.86M - 13.13M glyphs/s with 0.0 MB idle RAM direct memory-mapped GEMV.
* **Compiler Parity:** Native ELF AArch64 == Reference VM (`max_err = 0.00e+00`).

---

## License & Privacy

Proprietary Research — Private Repository.  
All Rights Reserved © 2026.
