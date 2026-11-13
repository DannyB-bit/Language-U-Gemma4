/*
 * neon_stream_accelerator_v2.h
 * ============================================================================
 * Language-U-Gemma4-NeonRunner-Bridge-V2 (v2.1.0 — GlyphAttentionV2)
 * Unified ARMv8 NEON / x86-64 OpenMP Macro-Glyph Stream & Attention Header
 *
 * Bridges DannyB-bit/Language-U-Gemma4 (Gemma 4 E4B & 31B Dual-Road Option 4 +
 * Option 5 streams) with DannyB-bit/Neon-Runner (in-register RVQ + FWHT-8 ABI)
 * and UniversalLayerOrderedCortex (trinity_gemv ABI).
 *
 * Exports:
 *   1. Full Neon-Runner v1 C ABI (100% drop-in compatible):
 *      - decode_glyphs_c
 *      - benchmark_projection_neon
 *      - gemv_opt4_mmap
 *      - gemv_opt4_mmap_multi
 *      - neon_accelerator_self_test
 *   2. Unified V2 Dual-Road & Hadamard-Fused Extensions:
 *      - fwht8_vec_c
 *      - decode_glyphs_opt4_mode_c
 *      - gemv_opt4_fwht_mmap
 *      - decode_glyphs_opt5_c
 *      - gemv_opt5_mmap
 *      - trinity_gemv_self_test
 *   3. GlyphAttentionV2 Single-Pass Flash-Glyph GQA & O(1) KV Cache Engine:
 *      - encode_kv_glyph_fast_c
 *      - flash_glyph_attention_gqa_c
 *      - paged_bf16_attention_gqa_c
 * ============================================================================
 */

#ifndef NEON_STREAM_ACCELERATOR_V2_H
#define NEON_STREAM_ACCELERATOR_V2_H

#include <stdint.h>
#if !defined(NEON_MSVC_FREESTANDING_DLL)
  #include <stddef.h>
#else
  #ifndef _SIZE_T_DEFINED
    #define _SIZE_T_DEFINED
    typedef unsigned __int64 size_t;
  #endif
  #ifndef NULL
    #define NULL ((void*)0)
  #endif
#endif

#if defined(_MSC_VER)
  #define NEON_RESTRICT __restrict
  #ifdef NEON_BUILDING_DLL
    #define NEON_API __declspec(dllexport)
  #else
    #define NEON_API __declspec(dllexport)
  #endif
#else
  #define NEON_RESTRICT restrict
  #define NEON_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 1. Canonical Neon-Runner v1 C ABI (Option 4 132-Byte RVQ + FWHT-8)
 * ============================================================================ */

/**
 * Decodes raw 132-byte Option 4 macro-glyphs into 256 FP32 weights per glyph,
 * applying Sylvester-Hadamard H_8 rotation via in-register FWHT-8.
 */
NEON_API void decode_glyphs_c(
    const uint8_t* NEON_RESTRICT glyphs,
    const float* NEON_RESTRICT cb,
    float* NEON_RESTRICT out_weights,
    int n_glyphs,
    int num_threads
);

/* Packed Opt4: same two fp16 scales, followed by 64 LSB-first 11-bit
 * codebook indices (stage 1 then stage 2). Exactly 92 bytes per glyph.
 * These symbols are additive; existing 132-byte ABIs retain their layout. */
NEON_API void decode_glyphs_opt4_packed_c(
    const uint8_t* NEON_RESTRICT glyphs,
    const float* NEON_RESTRICT cb,
    float* NEON_RESTRICT out_weights,
    int n_glyphs, int apply_fwht8, int num_threads
);

/* x is already pre-rotated if the stream uses fwht8. No tensor expansion. */
NEON_API void gemv_opt4_packed_mmap(
    const uint8_t* NEON_RESTRICT mmap_base, size_t tensor_byte_offset,
    const float* NEON_RESTRICT cb, const float* NEON_RESTRICT x,
    float* NEON_RESTRICT y, int d_out, int d_in
);

/**
 * High-speed in-cache projection benchmark across n_glyphs (132B Option 4).
 * Returns elapsed wall-clock seconds.
 */
NEON_API double benchmark_projection_neon(
    const uint8_t* NEON_RESTRICT mmap_base,
    size_t byte_offset,
    int n_glyphs,
    const float* NEON_RESTRICT cb,
    int num_threads
);

/**
 * Fused zero-heap Option 4 GEMV (direct codebook inner product: y = W_direct * x,
 * or y = W_fwht * x when caller passes FWHT-8 pre-rotated x).
 */
NEON_API void gemv_opt4_mmap(
    const uint8_t* NEON_RESTRICT mmap_base,
    size_t tensor_byte_offset,
    const float* NEON_RESTRICT codebook_s7,
    const float* NEON_RESTRICT x,
    float* NEON_RESTRICT y,
    int d_out,
    int d_in
);

/**
 * Multi-K speculative verify GEMV kernel across K input vectors simultaneously.
 * Portable across ARMv8 NEON and x86-64 / scalar OpenMP targets.
 */
NEON_API void gemv_opt4_mmap_multi(
    const uint8_t* NEON_RESTRICT mmap_base,
    size_t tensor_byte_offset,
    const float* NEON_RESTRICT codebook_s7,
    const float* const* NEON_RESTRICT x_ptrs,
    float* const* NEON_RESTRICT y_ptrs,
    int K,
    int d_out,
    int d_in
);

/**
 * Neon-Runner self-verification battery (FWHT-8 self-inverse + Option 4 + Option 5 parity).
 * Returns 1 if all gates pass, 0 otherwise.
 */
NEON_API int neon_accelerator_self_test(void);

/* ============================================================================
 * 2. Unified V2 Extensions (Dual-Road Option 5 + FWHT-8 Fused GEMV)
 * ============================================================================ */

/**
 * Applies blockwise orthonormal Sylvester-Hadamard FWHT-8 to an array of length n
 * (n must be a multiple of 8). Supports x_in == x_out; partial overlap is not
 * supported. Self-inverse: fwht8_vec_c(fwht8_vec_c(x)) == x.
 */
NEON_API void fwht8_vec_c(
    const float* x_in,
    float* x_out,
    int n
);

/**
 * Decodes 132-byte Option 4 glyphs with selectable FWHT-8 rotation:
 *   apply_fwht8 = 1 -> Hadamard-rotated RVQ (encode_glyph_132 / Neon-Runner law)
 *   apply_fwht8 = 0 -> Real-space RVQ (RVQCompressor / 31B sandwich stream law)
 */
NEON_API void decode_glyphs_opt4_mode_c(
    const uint8_t* NEON_RESTRICT glyphs,
    const float* NEON_RESTRICT cb,
    float* NEON_RESTRICT out_weights,
    int n_glyphs,
    int apply_fwht8,
    int num_threads
);

/**
 * Fused zero-heap Option 4 GEMV for FWHT-8 rotated streams (encode_glyph_132 law).
 * Pre-rotates x once via O(d_in) FWHT-8 (exploiting (H_8 u) . x == u . (H_8 x))
 * and dispatches to gemv_opt4_mmap for bit-exact parity with decode_glyphs_c(W) @ x.
 */
NEON_API void gemv_opt4_fwht_mmap(
    const uint8_t* NEON_RESTRICT mmap_base,
    size_t tensor_byte_offset,
    const float* NEON_RESTRICT codebook_s7,
    const float* NEON_RESTRICT x,
    float* NEON_RESTRICT y,
    int d_out,
    int d_in
);

/**
 * Decodes Option 5 macro-glyphs (bytes_per_glyph = 36 standard or 48 Top-4 outlier)
 * into 256 uncompressed FP32 weights per glyph.
 */
NEON_API void decode_glyphs_opt5_c(
    const uint8_t* NEON_RESTRICT glyphs,
    float* NEON_RESTRICT out_weights,
    int n_glyphs,
    int bytes_per_glyph,
    int num_threads
);

/**
 * Fused zero-heap Option 5 GEMV (36B standard or 48B Top-4 outlier-preserved).
 * Matches trinity_gemv / UniversalLayerOrderedCortex C ABI exactly.
 */
NEON_API void gemv_opt5_mmap(
    const uint8_t* NEON_RESTRICT mmap_base,
    size_t tensor_byte_offset,
    const float* NEON_RESTRICT x,
    float* NEON_RESTRICT y,
    int d_out,
    int d_in,
    int bytes_per_glyph
);

/**
 * Multi-K batched prefill / speculative verify Option 5 GEMV across K input vectors
 * in a single pass over the memory-mapped glyph stream (36B standard or 48B Top-4 outlier).
 */
NEON_API void gemv_opt5_mmap_multi(
    const uint8_t* NEON_RESTRICT mmap_base,
    size_t tensor_byte_offset,
    const float* const* NEON_RESTRICT x_ptrs,
    float* const* NEON_RESTRICT y_ptrs,
    int K,
    int d_out,
    int d_in,
    int bytes_per_glyph
);

/**
 * Compatibility alias for UniversalLayerOrderedCortex (cortex.py).
 * Returns 1 if all self-tests pass, 0 otherwise.
 */
NEON_API int trinity_gemv_self_test(void);

/* ============================================================================
 * 3. GlyphAttentionV2 Extensions (O(1) Online KV Encode + Single-Pass Flash GQA)
 * ============================================================================ */

/**
 * Inspects a 2048x8 S^7 codebook and populates a 6,561-entry (13 KB) base-3
 * ternary lattice lookup table (`lut_6561[state] = centroid_idx` or `0xFFFF`).
 * Returns 1 if `cb` is a pure unit-normalized ternary lattice codebook
 * (`compass_s7` or `trit_rvq`), or 0 otherwise (e.g. `golden_s7`).
 */
NEON_API int build_ternary_lut_c(
    const float* NEON_RESTRICT cb,
    uint16_t* NEON_RESTRICT lut_6561
);

/**
 * Decodes Option 5 macro-glyphs (36B standard or 48B Top-4 outlier) with
 * optional inverse FWHT-8 (`apply_fwht8 = 1` or `0`).
 */
NEON_API void decode_glyphs_opt5_mode_c(
    const uint8_t* NEON_RESTRICT glyphs,
    float* NEON_RESTRICT out_weights,
    int n_glyphs,
    int bytes_per_glyph,
    int apply_fwht8,
    int num_threads
);

/**
 * Online sub-microsecond KV macro-glyph encoder for Option 4 (132B RVQ) and
 * Option 5 (36B standard / 48B Top-4 outlier) with optional in-register FWHT-8
 * outlier smoothing (`apply_fwht8 = 1`).
 *
 * When `codec_id == 4` and `lut_6561 != NULL`, uses O(1) 8-element sorting-network
 * ternary lattice projection (~300x faster than 2,048-centroid brute-force scan);
 * if `lut_6561 == NULL`, uses SIMD 2,048-centroid exact search.
 */
NEON_API void encode_kv_glyph_fast_c(
    const float* NEON_RESTRICT kv_in,
    const float* NEON_RESTRICT cb,
    const uint16_t* NEON_RESTRICT lut_6561,
    uint8_t* NEON_RESTRICT glyphs_out,
    int n_glyphs,
    int codec_id,
    int bytes_per_glyph,
    int apply_fwht8,
    int num_threads
);

/**
 * Single-pass Flash-Glyph Grouped-Query Attention (GQA) kernel:
 *   - Rotates and scales queries `Q` once in O(n_q_heads * head_dim).
 *   - Streams compressed `K` (and `V`, or reuses `K` in registers when `k_eq_v == 1`)
 *     across Split-K OpenMP sequence tiles with zero page-table indirection.
 *   - Evaluates all `G = n_q_heads / n_kv_heads` GQA dot products simultaneously
 *     per unpacked glyph, skipping value unpack/FMA when `s - m < -17.5f` (< 1 ULP FP32).
 *   - Optionally writes per-query log-sum-exp state (`out_max_score`, `out_sum_exp`)
 *     for exact associative merging with pinned sink / rolling hot-tail tokens.
 *   - Returns 0 on success, non-zero on invalid geometry arguments.
 */
NEON_API int flash_glyph_attention_gqa_c(
    const float* NEON_RESTRICT q,
    const uint8_t* NEON_RESTRICT k_glyphs,
    const uint8_t* NEON_RESTRICT v_glyphs,
    const float* NEON_RESTRICT codebook_s7,
    float* NEON_RESTRICT out_context,
    float* NEON_RESTRICT out_max_score,
    float* NEON_RESTRICT out_sum_exp,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int seq_len,
    size_t kv_seq_stride_bytes,
    size_t kv_head_stride_bytes,
    float attn_scale,
    int codec_id,
    int bytes_per_glyph,
    int apply_fwht8,
    int k_eq_v,
    int num_threads
);

/**
 * Native C/SIMD vLLM-style PagedAttention BF16 GQA kernel (for apples-to-apples
 * silicon benchmarking against `flash_glyph_attention_gqa_c`):
 *   - `k_pages`, `v_pages`: physical BF16 page pools `[num_pages, page_size, n_kv_heads, head_dim]`
 *   - `block_table`: logical-to-physical page index array `[num_logical_blocks]`
 */
NEON_API int paged_bf16_attention_gqa_c(
    const float* NEON_RESTRICT q,
    const uint16_t* NEON_RESTRICT k_pages,
    const uint16_t* NEON_RESTRICT v_pages,
    const int32_t* NEON_RESTRICT block_table,
    float* NEON_RESTRICT out_context,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int seq_len,
    int page_size,
    float attn_scale,
    int k_eq_v,
    int num_threads
);

#ifdef __cplusplus
}
#endif

#endif /* NEON_STREAM_ACCELERATOR_V2_H */
