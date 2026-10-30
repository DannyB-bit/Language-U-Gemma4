# Language-U-Gemma4 <-> Neon-Runner Unified Integration Spec (`v2.0.0`)

**Component Tag:** `Language-U-Gemma4-NeonRunner-Bridge-V2`  
**Release Version:** `v2.0.0`  
**Paired Upstream Target:** [`DannyB-bit/Neon-Runner`](https://github.com/DannyB-bit/Neon-Runner)

---

## 1. Executive Summary

`Language-U-Gemma4-NeonRunner-Bridge-V2` unifies the Dual-Road macro-glyph compression and encoding pipeline of `DannyB-bit/Language-U-Gemma4` (supporting both **Gemma 4 E4B** and **Gemma 4 31B**) with the in-register SIMD execution engine of `DannyB-bit/Neon-Runner`.

It supports two execution modes seamlessly:
1. **Unified Native V2 Library (`neon_runner/neon_stream_accelerator_v2.c`)**: Compiles into a drop-in `libneon_stream_accelerator.so` / `neon_stream_accelerator.dll` (and `libtrinity_gemv.so`) implementing both **Option 4 (132B RVQ, FWHT-8 & Direct)** and **Option 5 (36B standard & 48B Top-4 outlier)** with ARMv8-A NEON (`__aarch64__`), x86-64 AVX2+FMA (`__AVX2__`), and OpenMP.
2. **Unmodified `Neon-Runner` v1 Binary Interoperability**: When `NeonRunnerBridgeV2` loads an external v1-only `libneon_stream_accelerator.so` from `DannyB-bit/Neon-Runner`, it automatically pre-rotates activations via blockwise orthonormal FWHT-8 ($\tilde{x} = H_8 x$) before `gemv_opt4_mmap` and executes vectorized algebraic Option 5 factoring for middle trunk layers.

---

## 2. V1 vs. V2 Forensic Comparison Matrix

| Dimension | V1 (`Language-U-Gemma4` `ad0d7d0` + `Neon-Runner` v1) | V2 (`Language-U-Gemma4-NeonRunner-Bridge-V2` `v2.0.0`) |
| :--- | :--- | :--- |
| **1. Mathematical Geometry & Architecture** | `Language-U-Gemma4` encoded E4B (`42L`, `d=2560`) Option 5 tiles; `Neon-Runner` hardcoded 31B (`60L`, `d=5376`) all-Option-4 paths. `gemv_opt4_mmap` omitted `fwht8` while `encode_glyph_132` pre-rotated weights by $H_8$. | Unified parameterized layout & stream engine supporting both **Gemma 4 E4B** (`28,715,008` glyphs, $R=0$) and **Gemma 4 31B** (`114,401,280` glyphs, $R=0$). `gemv_opt4_fwht_mmap` exploits $H_8^T = H_8$ via $O(d_{\text{in}})$ input pre-rotation (`max_err = 6.10e-05` vs full decode). |
| **2. Memory & Latency Footprint (Component Benchmark)** | `Neon-Runner` v1 required Linux POSIX `<sys/mman.h>` and ARMv8 flags (`-mcpu=cortex-a72`); `gemv_opt4_mmap_multi` lacked a non-NEON path. | Zero-heap memory-mapped C11 engine across ARMv8 NEON, x86-64 AVX2+FMA (`_mm256_fmadd_ps`), and Windows/POSIX. Measured **3,213,629 glyphs/s (822.7 M FP32 params/s, 53.53 ms)** on physical 44.04M-param Gemma 4 31B Layer 0 Q-proj slice. |
| **3. Weights, Codebooks & Convergence** | Three incompatible 64 KB codebooks (`compass_s7`, `trit_rvq`, `golden_s7`) collided on row `#1419` sentinel checks. | `CodebookRegistryV2` ships verified `s7_codebook_2048x8.bin` (`#1419` diff `2.98e-08`) and `trit_codebook_2048x8.bin` (`#1419` diff `0.00e+00`) with fail-closed sentinel detection. Real Gemma 4 E4B Layer 0 Q-proj (`[16, 2560]`): **Cosine `0.9743` / SNR `12.88 dB`** (Opt4 132B). |
| **4. Wire / Packet Protocols & Container Law** | Sample `.glyph.bin` was raw headerless 48B Option 5; `Neon-Runner` hardcoded `15,100,968,976` B file-size assertions. | Canonical 16-byte `LANU` header (`magic="LANU"`, `version=2`, `flags=45`, `total_glyphs`, `boundary_glyphs`) + strict Size Law (`16 + sum(tensor_bytes)`) + Lying-Header fail-closed gate. |
| **5. Newly Integrated Capabilities** | Separate Option 4 (`Neon-Runner`) and Option 5 (`trinity_gemv`) C libraries. | Single unified C/SIMD library exporting both `Neon-Runner` (`decode_glyphs_c`, `gemv_opt4_mmap`, `benchmark_projection_neon`) and `trinity_gemv` (`gemv_opt5_mmap`, `trinity_gemv_self_test`) symbols, plus CLI stream encoder (`encode_gemma4_e4b_to_glyphs.py`) and 6-gate `_self_test()`. |

---

## 3. Scope Disambiguation (Component Benchmark vs. Full Inference)

Per engineering verification standards, `neon_runner/bridge.py` and `neon_stream_accelerator_v2.c` provide the **Macro-Glyph Stream Decoder & Fused Projection GEMV Micro-Kernel / Substrate Bridge**. Full end-to-end autoregressive token generation additionally couples:
1. BPE Tokenizer (`tokenizer.json`)
2. BF16 Embedding Table (`embed_tokens`)
3. Lossless FP32/BF16 RMSNorm & QK-Norm Sidecar (`norms.safetensors`) + RoPE
4. Hybrid Sliding/Full GQA Attention + Rolling KV Cache
5. GELU-Tanh / SwiGLU Feed-Forward Orchestration
6. Logit Softcapping & Vocabulary Sampling Loop

---

## 4. Quickstart Usage

### Build Native Library & Run 6-Gate Physical Verification
```bash
# Cross-platform build (ARMv8 NEON .so on Linux/Pi4, AVX2 .so/.dll on x86-64)
python neon_runner/build_native.py

# Execute all 6 fail-closed verification gates
python neon_runner/bridge.py --self-test
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

### Benchmark or Execute GEMV via `NeonRunnerBridgeV2`
```bash
python neon_runner/bridge.py \
  --stream model/gemma4_e4b_sample_layer0_q_proj.neon_opt4.lanu \
  --layout model/gemma4_e4b_sample_layer0_q_proj.neon_opt4.layout.json
```
