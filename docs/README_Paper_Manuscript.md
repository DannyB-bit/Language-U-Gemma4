# The Glyph Standard for Local Gemma 4 SWE Agents: One Symbol Unifying 256-Weight $S^7$ Storage, Propositional Codebase Retrieval, and Deterministic C Kernels

**Subtitle:** Breaking the Storage, Context, and Execution Bottlenecks of Offline Software Engineering Agents via Octonionic $H_8/S^7$ Macro-Glyphs, 6D Cuneiform Radicals, and Fail-Closed Verification across `google/gemma-4-E4B-it`, `gemma-4-31b-it-qat-w4a16-ct`, and `embeddinggemma-2`

*Submitted to Google DeepMind — The Gemma 4 Developer Agent Paper Track (Overall Best Paper & Best New Resource Tracks)*

---

## Abstract

Modern artificial intelligence splits one intelligence into three incompatible currencies: raw floating-point tensors for weight storage, statistical subword fragments for language and code representation, and separate interpreted runtimes for execution. On consumer and edge hardware, this fragmentation stresses autonomous software engineering (SWE) workflows—straining RAM with gigabyte weight footprints, inflating attention KV-caches on multi-thousand-token file dumps, and inducing hallucination cascades when repository retrieval returns out-of-domain false positives.

We introduce **The Glyph Standard (Language-U & Consider-Trinity)**, a unified discrete data type on the octonionic unit 7-sphere ($S^7$) explored across three complementary formats under one shared geometry:
1. **Nature 1 — Standalone Weight Storage Container:** Packs 256 neural weights into **36 bytes** (1.125 bits/weight ternary lattice, **48 bytes** with Top-4 FP16 outlier preservation) or **132 bytes** (4.125 bits/weight dual-stage Residual Vector Quantization on a 2,048-vertex $S^7$ codebook), observing an **exact modulo-256 divisibility invariant ($R \equiv 0 \pmod{256}$)** verified against official checkpoints of **`google/gemma-4-E4B-it`** (`7,517,503,488` pure LM 2D weights $\to$ `29,365,248` glyphs; `7,523,991,552` / `29,390,592` including multimodal 2D projectors; $R = 0$), **`google/gemma-4-31b-it-qat-w4a16-ct`** (`114,401,280` glyphs $\to$ `7,080,837,136` bytes = `7.08 GB` decimal / `6.59 GiB`, `16.54x` vs FP32 / `3.29x` vs 4-bit QAT `model.safetensors`), **`google/embeddinggemma-2`** (`6,027,264 B` [`6.03 MB` / `5.75 MiB`] $\to$ `62,784 B` [`61.31 KiB`], `96.0x`), and Qwen3.5 0.8B–9B reference dimensions (`33,711,216` glyphs).
2. **Nature 2 — Semantic Proposition & Codebase Compass:** Formulates closed semantic propositions ($\text{Action} \oplus \text{Target} \oplus \text{Context} \oplus \text{Constraint}$) as an analytic sequence-collapse design target and indexes 1,962 repository code/doc chunks in `trinity_retrieval/` (integrated into `DannyB-bit/Language-U-Gemma4`) via multi-channel Hybrid Reciprocal Rank Fusion governed by a **Two-Tier Trust Law (`STRONG`, `WEAK`, `MISS`)** achieving **`0/6` OOD false positives** (`35/35` passing in `roadmap_hybrid_v5_gemma.py`; `17/23` in-domain/weak + `0/6` OOD executed live in our companion Kaggle notebook).
3. **Nature 3 — Deterministic C-Lowering Opcode & Kernels:** Packs 6D Cuneiform coordinates into a non-overlapping **24-bit glyph instruction** (`rc | (rf << 8) | (ra << 16)`) across a **16-Operation Semantic ALU**, lowering through constant-folded C verified against our Python reference ALU alongside fused zero-matrix-heap C GEMV (`trinity_gemv.c`), `GlyphAttentionV2` on Raspberry Pi 4, and standalone Python-free C text generation on ARM64 (`langu_c` on `SmolLM2-135M`).

**Competition Scope Disclosure:** On the companion *Gemma 4 Developer Agent Competition* leaderboard, all three `Consider-Trinity` submissions bundling custom codecs scored **`0.00`** because Kaggle's sealed ADK harness executes only a declarative YAML contract with 9 predefined tools over Kaggle's stock QAT weights (`docs/AUTOPSY-0.00-and-FLEET-CONSENSUS-v1.md`). Switching to a minimal 4-file declarative ADK agent scored **`0.17`** (`fleet consensus v1`, vs `0.24` public LB top). Accordingly, all Language-U storage, retrieval, and native kernel results below are evaluated as **standalone offline edge components**.

```text
+===================================================================================+
|                      THE MACRO-GLYPH: ONE SYMBOL, THREE NATURES                   |
+===================================================================================+
|  NATURE 1: STORAGE               NATURE 2: SEMANTICS          NATURE 3: EXECUTION |
|  -----------------               -------------------          ------------------- |
|  256 neural weights packed       Closed ATCC Proposition:     24-bit Glyph Opcode |
|  into 36B/48B (Opt5) or          [Action (+) Target (+)       (rc | rf<<8 |       |
|  132B (Opt4 Dual-Stage RVQ)      Context (+) Constraint]      ra<<16) -> 16-Op    |
|  on S^7 via Hadamard H_8.        + 6-Channel S^7 Compass      Semantic ALU ->     |
|  Gemma 4 E4B (29.37M LM) &       & Two-Tier Trust Law         Constant-Folded C   |
|  31B (114.4M): R = 0 (Exact).    (96.0x RAG compression).     -> C GEMV/KV/LM.    |
+===================================================================================+
|       SHARED GEOMETRY: S^7 Unit Hypersphere + Fail-Closed Byte & Trust Laws       |
+===================================================================================+
```

---

## 1. Introduction: The Fragmentation Problem in Local Coding Agents

Autonomous developer agents operating offline on consumer or edge hardware face three compounding walls:
* **The Storage Wall:** Standard dense models store billions of FP16/BF16 weights that must be paged through memory bandwidth on every autoregressive step, while dense FP32 RAG vector indexes compete for scarce unified RAM.
* **The Context & Attention Wall:** Classical tokenizers split source code and natural language into statistical BPE fragments (`Ġun`, `pre`, `dict`) lacking conceptual closure. Dumping multi-thousand-token raw source files per turn quadratically inflates $\mathcal{O}(N^2)$ attention FLOPs and saturates local KV-cache budgets.
* **The Execution & Trust Wall:** Existing quantization methods (QuIP#, GPTVQ, AWQ, BitNet) treat compressed weights strictly as passive data, while unconstrained top-$k$ cosine retrieval forces nearest-neighbor matches even on out-of-domain queries, triggering multi-turn hallucination cascades.

To address these constraints under one mathematical substrate, we investigate **The Macro-Glyph**: a 256-weight / 8D hyperspherical primitive on $S^7$ connecting storage containers, codebase retrieval, and deterministic C execution.

---

## 2. Mathematical & Systems Architecture

### 2.1 Shared Geometry: Sylvester-Hadamard $H_8$ Rotation on the Octonion 7-Sphere ($S^7$)

All three natures operate on the parallelizable unit 7-sphere $S^7 = \{x \in \mathbb{R}^8 : \|x\|_2 = 1\}$ associated with the octonion normed division algebra ($\mathbb{O}$). Every 8-element sub-block $w \in \mathbb{R}^8$ is rotated via the normalized orthogonal **Sylvester-Hadamard matrix $H_8$**:

$$H_1 = [1], \quad H_{2^k} = \frac{1}{\sqrt{2}} \begin{bmatrix} H_{2^{k-1}} & H_{2^{k-1}} \\ H_{2^{k-1}} & -H_{2^{k-1}} \end{bmatrix}, \quad \|H_8 H_8^T - I_8\|_F < 10^{-12}$$

Rotation by $H_8$ requires only fast additions/subtractions in $\mathcal{O}(d \log d)$ operations, dispersing channel outliers evenly across the 8 axes prior to quantization or geodesic resonance scoring on $S^7$.

### 2.2 Nature 1 (Storage): Container Byte-Laws & Universal $R \equiv 0 \pmod{256}$ Census

A macro-glyph encapsulates **256 neural weights** ($32$ octonions on $S^7$). Before decoding any tensor slice, our layout validator verifies modulo-256 divisibility and exact byte-length framing:

1. **Exact Divisibility Invariant ($R \equiv 0 \pmod{256}$):**
   Because the core dimensions of Gemma 4 and Qwen3.5 share large power-of-two factors ($\gcd(2560, 10240, 256, 262144) = 256$ on Gemma 4 E4B), every 2D projection matrix $M \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ partitions cleanly into 256-weight macro-glyphs with zero remainder ($R = (d_{\text{out}} \cdot d_{\text{in}}) \pmod{256} = 0$):

| Architecture / Checkpoint Family | Verified Matrix Dimensions | Total 2D Weights | Discrete Macro-Glyphs ($\div 256$) | Remainder ($R$) |
| :--- | :--- | ---: | ---: | :---: |
| **`google/gemma-4-E4B-it` (`embed_tokens`)** | $262,144 \times 2,560$ | $671,088,640$ | $2,621,440$ | **$R = 0$** |
| **`google/gemma-4-E4B-it` (`embed_tokens_per_layer`)** | $262,144 \times 10,752$ ($42 \times 256$ PLE) | $2,818,572,288$ | $11,010,048$ | **$R = 0$** |
| **`google/gemma-4-E4B-it` (`MLP` 42L $\times 3$)** | $126 \times 10,240 \times 2,560$ | $3,303,014,400$ | $12,902,400$ | **$R = 0$** |
| **`google/gemma-4-E4B-it` (`Attn Q/O` + `K/V` + `PLE Proj/Gate`)** | 35 sliding (`2048/512`) + 7 full (`4096/1024`) $\times 2560$ + 85 PLE projs | $724,828,160$ | $2,831,360$ | **$R = 0$** |
| **`google/gemma-4-E4B-it` (Total Pure LM 2D Core)** | Exact HF `model.safetensors` LM tensors (excl. 1D norms) | **$7,517,503,488$** | **$29,365,248$** | **$R = 0$** |
| **`google/gemma-4-31b-it-qat-w4a16-ct` (410 2D Proj.)** | $d=5376, d_{\text{ff}}=21504, 60\text{L}$ (50 sliding + 10 full) | $29,286,727,680$ | **$114,401,280$** | **$R = 0$** |
| **`google/embeddinggemma-2` (1,962-Chunk Index)** | $1,962 \times 768\text{-D}$ MRL $\to 1,962 \times 32\text{ B}$ ($S^7$ 8D FP32) | $6,027,264\text{ B}$ ($6.03\text{ MB}$ / $5.75\text{ MiB}$) | **$62,784\text{ B}$ ($61.31\text{ KiB}$, $96.0\times$)** | **$R = 0$** |
| **Reference `Qwen3.5-0.8B` (244 Matrices)** | $d=1024$ | $615,579,648$ | $2,404,608$ | **$R = 0$** |
| **Reference `Qwen3.5-9B` (367 Matrices)** | $d=4096, d_{\text{ff}}=12288$ | $8,630,071,296$ | $33,711,216$ | **$R = 0$** |

*(Including multimodal audio/vision 2D projectors brings `google/gemma-4-E4B-it` to `7,523,991,552` 2D weights / `29,390,592` glyphs, still $R = 0$. On `gemma-4-E4B-it`, `num_kv_shared_layers = 18` shares KV activations across the final 18 layers during forward execution even though tensor shapes are present in `model.safetensors`.)*

2. **Container Byte-Laws & Dual-Road Sandwich Topology:**
   * **Law IIA (Option 5 Ternary Glyph — 36B FULL8 / 48B Outlier-Preserved):** `2B FP16 scale` + `2B FP16 bias` + `32B sign bitmask` (256 bits selecting $\pm\text{scale} + \text{bias}$); 48B appends 4 `uint8` indices + 4 `FP16` heavy-tail residuals (`sum_bytes == header + N * bpg`). During fused C GEMV (`trinity_gemv.c`), dequantization factors out of the inner dot loop: $\sum w_i x_i = \text{scale}\sum(\pm x_i) + \text{bias}\sum x_i + \sum_{k=1}^4 \Delta w_{i_k} x_{i_k}$.
   * **Law IIB (Option 4 Gyroscopic Astrolabe RVQ — 132B / 4.125 bpw):** `2 x 2B FP16 scales` + `2 x 32 uint16 indices` into a 2,048-vertex $S^7$ codebook (`sum_bytes == header + N * 132`).
   * **31B Dual-Road Stream (`7,080,837,136` bytes = `7.08 GB` dec / `6.59 GiB`):** Uses Option 4 on boundary layers `0..4` & `55..59` (`18,923,520` glyphs = `2,497,904,640` B) and Option 5 48B on middle layers `5..54` (`95,477,760` glyphs = `4,582,932,480` B) behind a 16B `LANU` header, validated fail-closed across all 410 projections in `layout_map.py`.

### 2.3 Nature 2 (Semantics & SWE Navigation): ATCC Propositions, $S^7$ RRF, and Two-Tier Trust Law

$$\mathbf{Macro\text{-}Glyph} = [\text{Action } (\mathcal{A})] \oplus [\text{Target } (\mathcal{T})] \oplus [\text{Context } (\mathcal{C})] \oplus [\text{Constraint } (\mathcal{K})]$$

In `trinity_retrieval/` (consolidated into `Language-U-Gemma4`), we project 768-D `embeddinggemma-2` Matryoshka vectors via $H_8$ onto $S^7$, compressing the 1,962-chunk repository index from **`6,027,264` bytes (`6.03 MB` decimal / `5.75 MiB`) down to `62,784` bytes (`61.31 KiB`, `96.0x` reduction)** for L1/L2 CPU cache residency (`0.0119–0.0146 ms` scan time). Multi-channel Hybrid Reciprocal Rank Fusion combines lexical/AST signals with $S^7$ resonance, gated by our **Two-Tier Trust Law**:
* **`STRONG` (Execute):** High $S^7$ resonance + lexical consensus $\to$ injects a surgical **50-token code slice** (>95% reduction vs 1,370-token BM25 dumps).
* **`WEAK` (Warn):** Marginal agreement $\to$ returns slice flagged for manual check.
* **`MISS` (Halt):** Out-of-domain query $\to$ abstains cleanly (**`0/6` OOD false positives** across OOD probes in both our 35-query dev battery `roadmap_hybrid_v5_gemma.py` and our executed companion Kaggle notebook on `devs01`).

### 2.4 Nature 3 (Execution): 24-Bit Cuneiform Opcode & Deterministic C Lowering

Each symbolic action maps 6D Cuneiform 4-bit coordinates `[Domain, Subdomain, Operation, Modality, Depth, Polarity]` into three 8-bit fields (`RC = ((domain & 0xF) << 4) | (subdomain & 0xF)`, `RF = ((op & 0xF) << 4) | (modality & 0xF)`, `RA = ((depth & 0xF) << 4) | (polarity & 0xF)`) forming a **non-overlapping 24-bit instruction opcode**:

```text
code_24bit = rc | (rf << 8) | (ra << 16)   // 24 bits: 0x000000 .. 0xFFFFFF
```

*(Legacy docstrings in prototype scripts referred to this packing as "28-bit"; three non-overlapping 8-bit fields span 24 bits.)* Our **16-Operation Semantic ALU** (`hardware_verification_gate.py`) lowers selected states to constant-folded C checked directly against the Python reference ALU (`< 1e-12` error when compiled with `gcc`/`clang`/SDK C compiler).

---

## 3. Experimental Verification & The Honest Ledger

All artifacts cited below match checked-in reports and executable test scripts.

### 3.1 Physical Hardware & Notebook Evidence Summary

| Component / Gate | Verified Measurement | Artifact / Scope |
| :--- | :--- | :--- |
| **Gemma 4 E4B & 31B $R \equiv 0$ Census** | **$29,365,248$** E4B LM glyphs ($29,390,592$ w/ projectors) & **$114,401,280$** 31B glyphs ($R = 0$) | Verified against official HF `google/gemma-4-E4B-it` header & `model.safetensors` (`23,265,352,448` B) |
| **31B Dual-Road Stream Footprint** | **$7,080,837,136\text{ B}$ ($7.08\text{ GB}$ dec / $6.59\text{ GiB}$)** (`16.54x` vs FP32; `3.29x` vs QAT `safetensors`) | `OFFICIAL_SAFETENSORS_LANGUAGE_U_VERIFIED_REPORT.json` (`8.46 dB` Opt4 canonical / `4.52 dB` Opt5) & `PHYSICAL_PI4_31B_VERIFIED_REPORT.json` |
| **`embeddinggemma-2` $S^7$ Index** | **$6.03\text{ MB}$ (`5.75 MiB`) $\to$ `61.31 KiB` (`96.0x`)**, `0.0119–0.0146 ms` scan | Executed Kaggle Notebook (`COMPLETE`) & `swe_navigation_benchmark_results.json` |
| **Two-Tier Trust Law Abstention** | **`0/6` OOD false positives (`17/23` total)** on Kaggle; **`35/35` passing (`0` OOD FP)** in `v5` dev suite | Companion Kaggle notebook & `trinity_retrieval/roadmap_hybrid_v5_gemma.py` |
| **Pi 4 ARMv8-A 31B Real-Stream GEMV** | **V3 Opt4 (`8192×5376`): `235.51 ms`** (`6.64e-07` parity; V2: `126.93 ms`); **V3 Opt5 (`16384×5376`): `237.82 ms`** (`7.21e-07`); **`62.29 MB` RSS** | Physical Raspberry Pi 4 Cortex-A72 against real `7.08 GB` stream (`PHYSICAL_PI4_31B_VERIFIED_REPORT.json`); **kernel data-path test only (synthetic `x`, not token generation)** |
| **Pi 4 `GlyphAttentionV2` (`v2.1.0`)** | **Sliding (`W=1,024`): `4.36 ms`** (`1.76x` vs vLLM PagedBF16 `7.67 ms`, `6.52x` RAM cut); **Global `128K`: `713.98 ms`** (`1.60x` vs `1,140.71 ms`, `21.2x` RAM cut to `48.27 MB`) | `Language-U-Gemma4/docs/NEON_RUNNER_INTEGRATION_V2.md` |
| **Host Optical Carrier + L0 GEMV (`v2.2.0`)** | **DMA ingest `0.277–0.290 ms`** (100% CRC32); **L0 Q/K/V (`8192/4096 × 5376`) `23.1–26.5 tok/s`** | Host x86-64 silicon via `neon_stream_accelerator.dll` against `7.08 GB` stream (`scripts/benchmark_optical_gemma31b_live.py`) |
| **Python-Free E2E C Runtime (`langu_c`)** | **`5.1–5.5 tok/s` coherent E2E C generation** on ARM64 (`SmolLM2-135M`, `'Paris'` / `'The capital'`, all-FP16 decoded cache, codec 12+13) | Standalone C runtime (`langu_c/`, commit `3167576` merged on `Neon-Runner` `main`) |

### 3.2 Prior Art Comparison (The Unification Matrix)

| System / Prior Art | Weight Container | Semantic Index / Proposition | C / Native Kernel Path | Fail-Closed Hardware Verifier |
| :--- | :---: | :---: | :---: | :---: |
| **QuIP# (ICML'24) / GPTVQ / Leech-VQ ('25)** | Yes ($H_8$ / Lattice VQ) | No | No (Passive data) | Partial (GPU PTQ) |
| **BitNet b1.58 (2024)** | Yes (Ternary) | No | No | No |
| **Gist Tokens (NeurIPS'23) / Sub-Sentence ('23)** | No | Yes (Prompt compression) | No | No |
| **Byte Latent Transformer (Meta, Dec '24)** | No | Partial (Entropy patches) | No | No |
| **TransO (2022) / HoloByte (2026)** | No | Partial ($S^7$ / Hypersphere) | No | No |
| **THE GLYPH STANDARD (This Work)** | **Yes (256w $\to$ 36B/48B/132B, $R=0$)** | **Yes (ATCC + $96.0\times$ $S^7$ RAG)** | **Yes (24-bit ALU + C GEMV/KV/LM)** | **Yes (Fail-Closed Byte & Trust Laws)** |

### 3.3 The Honest Ledger (Competition Scope, Sandbox Boundaries & Scaling Limits)

1. **Agent Competition `0.00` Builds vs. `0.17` Declarative ADK Pivot (Public Record):**
   All three `Consider-Trinity` submissions to the companion *Gemma 4 Developer Agent Competition* (`Consider-Trinity-Gemma4-E4B` ×2, `Consider-Trinity-Gemma4-31B` ×1) scored **`0.00`** (`Language-U-Gemma4/docs/AUTOPSY-0.00-and-FLEET-CONSENSUS-v1.md`) because Kaggle's sealed harness only invokes 9 predefined tools over Kaggle-served QAT weights and ignores custom runtimes/LoRA/RAG indexes. Switching to a minimal 4-file declarative ADK agent + prompt (`agent/build.py`, `fleet consensus v1`) scored **`0.17`** (`0.15` on `v3-converged`, `0.13` on `v5-restored`; public LB top `0.24`). All storage, retrieval, and C kernel components are **standalone offline edge artifacts**.
2. **Real-Stream Kernel Probes vs. Multi-Layer Generative Coherence Bounds:**
   Our Pi 4 31B probes (`235.51 ms` Opt4, `237.82 ms` Opt5, `62.29 MB` RSS) verify the real `7.08 GB` file and bit-exact decode parity using deterministic probe vectors, not full 60-layer generation. Moreover, our C runtime ablations (`langu_c`, commit `3167576`) empirically reveal the exact rate-distortion boundary: while codec 12+13 with an all-FP16 decoded cache produces coherent E2E text at `5.1–5.5 tok/s` on `SmolLM2-135M`, aggressive 2-stage 132B RVQ (`10.8 dB` per matrix on `Qwen3-0.6B`) compounds across 28 layers to low cosine (`0.08`) incoherent text, Q8-int8 decoded caching without dotprod instructions runs slower (`1.7–2.0 tok/s`) and degrades logits (`'7777'`), and single-pass 48B 31B middle layers (`0.80–0.88` cos) lie below the generative coherence floor without `compass`/Opt4 or `Astrolabe` calibration (`reencode_gemma4_31b_compass.py` lifts Opt4 slice fidelity to `12.46 dB` / `cos = 0.9744`).
3. **Evaluation Hygiene & Artifact Corrections:**
   An earlier telemetry log (`gemma4_31b_physical_pi_inference_metrics.json`) came from a 32 MB uniform `0xAA` dummy slab (with `throttled=0x50000` latched since boot), and an earlier template script (`triple_battery`) used hardcoded generation/hashing; both are excluded from this verified release. Note also that `kaggle_submission/benchmarks/run_swe_navigation_eval.py` contains a synthetic noise harness, whereas our real benchmark numbers come directly from `HybridEngineV5` (`test_35_hardware_suite.py`) and the live executed Kaggle notebook.
4. **Compression Baselines (`16.54x` vs. `3.29x`):**
   Compared to FP32 (`117.15 GB` dec / `109.10 GiB`), the `7.08 GB` (`6.59 GiB`) 31B stream achieves `16.54x` compression (`8.27x` vs FP16); compared to the `23.27 GB` W4A16 QAT checkpoint (`model.safetensors`), the physical size ratio is **`3.29x`**.
5. **Analytic Sequence Scaling vs. Realized Agent Gains & Shared KV Invariant:**
   The theoretical $(20.4)^2 = 416.2\times$ attention FLOP scaling describes sequence-length collapse, whereas realized retrieval prompt savings in `trinity_retrieval/` exceed `95%` via 50-token slices (`0/6` OOD FP in our evaluation suite). On `gemma-4-E4B-it`, `num_kv_shared_layers = 18` requires owning vs shared KV awareness during quantization and adapter updates.

---

## 4. Conclusion

The Glyph Standard aligns **256-weight container compression** ($R = 0$ across Gemma 4 E4B `29.37M` LM glyphs and Gemma 4 31B `114.4M` glyphs), **`96.0x` L1/L2-resident `embeddinggemma-2` codebase indexing** with fail-closed `MISS` abstention, and **24-bit deterministic C kernels** (`trinity_gemv.c`, `GlyphAttentionV2`, and `langu_c`). Publishing both verified edge wins and exact rate-distortion/sandbox limits (`0.00` custom vs `0.17` ADK) provides an honest, reproducible blueprint for local Gemma 4 developer agents.

---

## References

1. Tseng, A., et al. (2024). *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks*. ICML 2024 (arXiv:2402.04396).
2. van Baalen, M., et al. (2024). *GPTVQ: The Blessing of Dimensionality for LLM Quantization*. Qualcomm AI Research (arXiv:2402.15319).
3. *Leech Lattice Vector Quantization for Efficient LLM Compression* (2025/2026). arXiv:2603.11021.
4. Ma, S., et al. (2024). *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*. arXiv:2402.17764.
5. Mu, J., Li, X. L., & Goodman, N. (2023). *Learning to Compress Prompts with Gist Tokens*. NeurIPS 2023.
6. Pagnoni, A., et al. (2024). *Byte Latent Transformer: Patches Scale Better Than Tokens*. Meta AI (arXiv:2412.09871).
7. Chen, Z., et al. (2023). *Sub-Sentence Encoder: Contrastive Learning of Propositional Semantic Representations*. arXiv:2311.04335.
8. Cai, L., et al. (2022). *Translation-Based Embeddings with Octonion for Knowledge Graph Completion (TransO)*. Applied Sciences, 12(8), 3935.
9. Khasia, V. (2026). *HoloByte: Continuous Hyperspherical Distillation for Tokenizer-Free Modeling*. arXiv:2603.16917.
10. Kusupati, A., et al. (2022). *Matryoshka Representation Learning*. NeurIPS 2022.
11. Gilbert, H., et al. (2023). *Semantic Compression with Large Language Models*. arXiv:2304.12512.
12. Viazovska, M. S. (2017). *The sphere packing problem in dimension 8*. Annals of Mathematics, 185(3), 991–1015.
