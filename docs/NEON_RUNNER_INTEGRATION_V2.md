# Language-U-Gemma4 <-> Neon-Runner Unified Integration & GlyphAttentionV2 Spec (`v2.1.0`)

**Component Tags:** `Language-U-Gemma4-NeonRunner-Bridge-V2` & `Language-U-Gemma4-GlyphAttention-V2`  
**Release Version:** `v2.1.0`  
**Paired Upstream Target:** [`DannyB-bit/Neon-Runner`](https://github.com/DannyB-bit/Neon-Runner)

---

## 1. Executive Summary

`Language-U-Gemma4` (`v2.1.0`) unifies the Dual-Road macro-glyph weight stream and **Single-Pass Flash-Glyph GQA KV-Cache Attention Engine (`GlyphAttentionV2`)** with the in-register SIMD execution engine of [`DannyB-bit/Neon-Runner`](https://github.com/DannyB-bit/Neon-Runner).

Where vLLM / PagedAttention hits two physical bottlenecks on edge and long-context hardware — **(1) DRAM memory-bandwidth saturation** reading uncompressed 16-bit KV states and **(2) indirect 16-token page-table pointer chasing** — `GlyphAttentionV2` stores KV cache states directly as discrete $S^7$ macro-glyphs (`head_dim = 256` $\to$ 1 glyph, `global_head_dim = 512` $\to$ 2 glyphs, $R = 0$) and computes $Q K^\top$, online softmax, and $P V$ directly in L1 cache without materializing dense FP32 $K$ or $V$ tensors across the sequence.

### Core Execution Subsystems (`v2.1.0`)
1. **Unified Native V2 Stream & Attention Library (`neon_runner/neon_stream_accelerator_v2.c`)**:
   - **Weight Projection GEMV**: Drop-in `libneon_stream_accelerator.so` / `neon_stream_accelerator.dll` (and `libtrinity_gemv.so`) implementing Option 4 (`132 B` RVQ, FWHT-8 & Direct) and Option 5 (`36 B` standard & `48 B` Top-4 outlier) across ARMv8-A NEON (`__aarch64__`), x86-64 AVX2+FMA (`__AVX2__`), and OpenMP.
   - **$O(1)$ Online KV Glyph Encoder (`encode_kv_glyph_fast_c`)**: 19-comparator sorting network + 13 KB ternary-lattice hash table (`compass_s7` / `trit_rvq`) with joint $2 \times 2$ Gram least-squares RVQ scaling and exact $L_2$ sphere norm preservation (`46.3 us/tok` Opt5 48B; `165.2 us/tok` Opt4 132B across 8 global glyphs on Cortex-A72).
   - **Single-Pass Flash-Glyph GQA Kernel (`flash_glyph_attention_gqa_c`)**:
     - *Algebraic 4-Bit Nibble-LUT $Q K^\top$ + Nibble-Histogram $P V$ Engine* (`codec_id=5`, `8:1` & `2:1` GQA): Pre-tabulates 4-bit sign-pattern query dot products (`32 KB`/glyph, L1-resident) and accumulates softmax weights into a 4-bit nibble histogram flushed once per 512 tokens, completely eliminating per-token FP32 $K/V$ materialization.
     - *8-Head Register-Fused RVQ Engine* (`codec_id=4`): Unpacks each 8D $S^7$ octonion once in NEON/AVX2 registers and broadcasts FMA across all 8 GQA query heads simultaneously, backed by coprime-stride (`head_dim + 16 = 33` cache lines) per-job accumulators to prevent 64 KB 2-way L1D set-conflict thrashing.
     - *Hadamard Orthogonality & Associative Dual-Road LSE*: Exploits $\langle H_8 k_t, H_8 q_g \rangle = \langle k_t, q_g \rangle$ and $\sum_t w_t (H_8 \tilde{v}_t) = H_8 (\sum_t w_t \tilde{v}_t)$, applying FWHT-8 once per query head at kernel exit and returning exact `(out_max_score, out_sum_exp)` statistics for lossless log-sum-exp fusion with pinned FP32 attention sinks and rolling hot-tail tokens (`GlyphKVCacheV2`).
2. **Unmodified `Neon-Runner` v1 Binary Interoperability**: When `NeonRunnerBridgeV2` loads an external v1-only `libneon_stream_accelerator.so` from `DannyB-bit/Neon-Runner`, it automatically pre-rotates activations via blockwise orthonormal FWHT-8 ($\tilde{x} = H_8 x$) before `gemv_opt4_mmap`.

---

## 2. V1 $\to$ V2.0 $\to$ V2.1 Forensic Comparison Matrix

| Dimension | V1 (`ad0d7d0` + `Neon-Runner` v1) | V2.0 (`Bridge-V2` `v2.0.0`) | V2.1 (`GlyphAttentionV2` `v2.1.0`) |
| :--- | :--- | :--- | :--- |
| **1. Mathematical Geometry & Architecture** | Weight-only E4B Option 5 / 31B Option 4 split; `gemv_opt4_mmap` omitted `fwht8`. Dense KV cache only. | Unified E4B (`28.7M` glyphs) & 31B (`114.4M` glyphs) weight stream; `gemv_opt4_fwht_mmap` via $H_8^\top = H_8$ (`max_err = 6.10e-05`). | Extends $S^7$ macro-glyph geometry to **KV Cache & GQA Attention**: Hadamard-invariant $Q K^\top$, linear $H_8$ value aggregation, algebraic 4-bit Nibble-LUT/Histogram, and associative Dual-Road LSE (`sink + glyph + hot-tail`). |
| **2. Memory & Latency Footprint (Component Benchmark)** | Linux/ARMv8-only weight GEMV; uncompressed KV (`1,024 MB` naive BF16 / `512 MB` `K==V` per global layer at `128K`). | Cross-platform zero-heap weight GEMV (`3.23M` glyphs/s = `825.9M` params/s on Cortex-A72). | **128K Global Layer KV**: `1,024 MB` $\to$ **`48.27 MB` (`21.2x`, Opt5 Dual-Road)** at **`713.98 ms` (`1.60x` faster than vLLM PagedBF16 `1,140.71 ms`, `1.26x` faster than `K==V` BF16 `899.69 ms`)**, or **`132.24 MB` (`7.7x`, Opt4 Dual-Road)** at `o_proj_cos = 0.97574`. **1K Sliding Layer**: **`4.36 ms` (`1.76x` faster, `o_proj_cos = 0.99597`)** / **`1.95 ms` (`3.92x` faster, Pure Opt5)**. |
| **3. Weights, Codebooks & Convergence** | Incompatible 64 KB codebooks (`compass_s7`, `trit_rvq`, `golden_s7`) collided on row `#1419`. Greedy $O(2048)$ brute-force RVQ search. | `CodebookRegistryV2` fail-closed `#1419` sentinels (`compass_diff = 2.98e-08`, `trit_diff = 0.00e+00`). | Adds **$O(1)$ 19-comparator sorting-network + 13 KB ternary-lattice LUT encoder** (`encode_kv_glyph_fast_c`) with joint $2 \times 2$ Gram solve + $L_2$ sphere norm preservation (`opt4_cos = 0.99731`; C vs FP32 attention parity `6.79e-06` Opt4 / `3.87e-06` Opt5). |
| **4. Wire / Packet Protocols & Arena Layout** | Raw headerless `.glyph.bin`; scattered 16-token PagedAttention `block_table` indirection in vLLM. | Canonical 16-byte `LANU` stream header + Size Law (`16 + sum(tensor_bytes)`). | Contiguous linear glyph arena (`[n_kv_heads, max_seq, glyphs_per_head * bpg]`) for global layers (`k_eq_v=True` single-buffer) + $O(1)$ permutation-invariant circular glyph ring buffer for sliding-window layers (`W=1,024`). |
| **5. Newly Integrated Capabilities** | Separate Option 4 & Option 5 weight libraries. | Unified `libneon_stream_accelerator` + 6-gate `bridge.py --self-test`. | Exports `encode_kv_glyph_fast_c`, `flash_glyph_attention_gqa_c`, `paged_bf16_attention_gqa_c`, `GlyphKVCacheV2` Python runtime (`glyph_attention.py` 6-gate `--self-test`), and Gemma 4 31B silicon benchmark (`scripts/benchmark_glyph_attention_silicon.py`). |

---

## 3. Physical Silicon Benchmark: Agent 01 `AstronautSHE` (Raspberry Pi 4 Cortex-A72)

> **Component Benchmark Disambiguation:** The table below reports the **Attention Kernel & KV Cache Component Benchmark** (`flash_glyph_attention_gqa_c` + `encode_kv_glyph_fast_c` vs. native C/NEON/OpenMP `paged_bf16_attention_gqa_c` with 16-token permuted `block_table` pages and exact FP32 reference) evaluated on **physical Gemma 4 31B Layer 0 & Layer 5 Q/K/V/O activations** streamed from `gemma4_31b_language_u_sandwich_stream.bin` (`15,100,968,976` B) on external SSD (`$LANGUAGE_U_STREAM_DIR`). Full end-to-end token generation additionally couples BPE tokenization (`tokenizer.json`), BF16 embeddings (`embed_tokens`), RMSNorm/QK-Norm + RoPE, 60-layer FFN orchestration, and vocabulary sampling.

* **Hardware:** `AstronautSHE` — Raspberry Pi 4 Model B 8GB (`aarch64`, ARMv8-A Cortex-A72, 4 OpenMP threads, Linux `6.18.50+rpt-rpi-v8`, external SSD)
* **Online Single-Token KV Encode (Global L5, 4 KV heads $\times$ 512d = 8 glyphs):** **`46.3 us/tok`** (Opt5 48B Top-4 Outlier) | **`165.2 us/tok`** (Opt4 132B $O(1)$ Lattice)
* **Raw JSON Telemetry Receipt:** [`docs/glyph_attention_v2_pi4_benchmark_report.json`](glyph_attention_v2_pi4_benchmark_report.json)

### Regime A — Layer 0 Sliding-Window Attention (`W = 1,024`, `32Q / 16KV / 256d`, `V != K`)

| Kernel / Cache Mode | Latency (ms) | Speedup vs. PagedBF16 | Resident KV RAM | Compression | `ctx` Cosine | `o_proj` Cosine (SNR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `vLLM PagedBF16 (16-tok blocks)` | `7.67 ms` | `1.00x` | `16,384.0 KB` | `1.00x` | `0.999999` | `1.00000` |
| `Opt4_Ring_DualRoad (132B + 4 sink / 64 tail)` | `13.28 ms` | `0.58x` | `5,116.0 KB` | `3.20x` | `0.92786` | **`0.99452` (`19.3 dB`)** |
| **`Opt5_Ring_DualRoad (48B + 4 sink / 64 tail)`** | **`4.36 ms`** | **`1.76x`** | **`2,512.0 KB`** | **`6.52x`** | **`0.94210`** | **`0.99597` (`20.8 dB`)** |
| **`Opt5_Ring_Pure (48B Nibble-LUT/Hist)`** | **`1.95 ms`** | **`3.92x`** | **`1,536.0 KB`** | **`10.67x`** | `0.72786` | **`0.97716` (`13.1 dB`)** |

### Regime B — Layer 5 Global Attention (`32Q / 4KV / 512d`, `attention_k_eq_v = True`, `4K -> 128K`)

| Context $S$ | Kernel / Cache Mode | Latency (ms) | Speedup vs. vLLM / `K==V` BF16 | Resident KV RAM | Compression vs. vLLM | `o_proj` Cosine (SNR) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **4,096** | `vLLM PagedBF16 (K+V)` | `35.98 ms` | `1.00x` / `0.81x` | `32.00 MB` | `1.0x` | `1.00000` |
| **4,096** | `PagedBF16 (K==V dedup)` | `29.20 ms` | `1.23x` / `1.00x` | `16.00 MB` | `2.0x` | `1.00000` |
| **4,096** | `Opt4_DualRoad_132B` | `44.74 ms` | `0.80x` / `0.65x` | `4.37 MB` | `7.3x` | **`0.97389` (`12.8 dB`)** |
| **4,096** | **`Opt5_DualRoad_48B`** | **`24.14 ms`** | **`1.49x` / `1.21x`** | **`1.77 MB`** | **`18.1x`** | `0.87724` (`6.1 dB`) |
| **4,096** | **`Opt5_Pure_48B`** | **`20.28 ms`** | **`1.77x` / `1.44x`** | **`1.50 MB`** | **`21.3x`** | `0.85899` (`5.5 dB`) |
| **16,384** | `vLLM PagedBF16 (K+V)` | `138.29 ms` | `1.00x` / `0.81x` | `128.00 MB` | `1.0x` | `1.00000` |
| **16,384** | `PagedBF16 (K==V dedup)` | `111.66 ms` | `1.24x` / `1.00x` | `64.00 MB` | `2.0x` | `1.00000` |
| **16,384** | `Opt4_DualRoad_132B` | `155.26 ms` | `0.89x` / `0.72x` | `16.74 MB` | `7.6x` | **`0.97413` (`12.9 dB`)** |
| **16,384** | **`Opt5_DualRoad_48B`** | **`92.35 ms`** | **`1.50x` / `1.21x`** | **`6.27 MB`** | **`20.4x`** | `0.87620` (`6.0 dB`) |
| **65,536** | `vLLM PagedBF16 (K+V)` | `542.24 ms` | `1.00x` / `0.83x` | `512.00 MB` | `1.0x` | `1.00000` |
| **65,536** | `PagedBF16 (K==V dedup)` | `447.46 ms` | `1.21x` / `1.00x` | `256.00 MB` | `2.0x` | `1.00000` |
| **65,536** | `Opt4_DualRoad_132B` | `594.58 ms` | `0.91x` / `0.75x` | `66.24 MB` | `7.7x` | **`0.97613` (`13.2 dB`)** |
| **65,536** | **`Opt5_DualRoad_48B`** | **`311.57 ms`** | **`1.74x` / `1.44x`** | **`24.27 MB`** | **`21.1x`** | `0.87203` (`5.9 dB`) |
| **131,072** | `vLLM PagedBF16 (K+V)` | `1,140.71 ms` | `1.00x` / `0.79x` | `1,024.00 MB` | `1.0x` | `1.00000` |
| **131,072** | `PagedBF16 (K==V dedup)` | `899.69 ms` | `1.27x` / `1.00x` | `512.00 MB` | `2.0x` | `1.00000` |
| **131,072** | `Opt4_DualRoad_132B` | `1,233.76 ms` | `0.92x` / `0.73x` | `132.24 MB` | `7.7x` | **`0.97574` (`13.2 dB`)** |
| **131,072** | **`Opt5_DualRoad_48B`** | **`713.98 ms`** | **`1.60x` / `1.26x`** | **`48.27 MB`** | **`21.2x`** | `0.87654` (`6.0 dB`) |

---

## 4. Quickstart Usage (`v2.1.0`)

### Build Native Library & Run All 12 Fail-Closed Verification Gates
```bash
# 1. Cross-platform build (ARMv8 NEON .so on Linux/Pi4, AVX2 .so/.dll on x86-64)
python neon_runner/build_native.py

# 2. Execute 6-gate Stream & GEMV physical verification battery
python neon_runner/bridge.py --self-test

# 3. Execute 6-gate GlyphAttentionV2 physical verification battery
python neon_runner/glyph_attention.py --self-test
```

### Run Physical Gemma 4 31B Silicon Benchmark (`4K -> 128K`)
```bash
python scripts/benchmark_glyph_attention_silicon.py --max-seq 131072 --threads 4
```

### Encode Gemma 4 Safetensors into a Neon-Runner `.lanu` Stream
```bash
# Pure Option 4 (132B RVQ + compass_s7) — works directly with unmodified Neon-Runner v1 & V2
python gemma4_e4b/encode_gemma4_e4b_to_glyphs.py \
  --safetensors /path/to/model.safetensors \
  --mode neon-opt4 \
  --codebook compass_s7 \
  --out-stream model/gemma4_e4b_neon_opt4.lanu \
  --out-layout model/gemma4_e4b_neon_opt4.layout.json

# Dual-Road Hybrid (Boundary Opt4 132B + Middle Trunk Opt5 48B)
python gemma4_e4b/encode_gemma4_e4b_to_glyphs.py \
  --safetensors /path/to/model.safetensors \
  --mode hybrid \
  --codebook compass_s7 \
  --out-stream model/gemma4_e4b_hybrid_v2.lanu \
  --out-layout model/gemma4_e4b_hybrid_v2.layout.json
```
