/*
 * neon_stream_accelerator_v2.c
 * ============================================================================
 * Language-U-Gemma4-NeonRunner-Bridge-V2 (v2.0.0)
 * Unified ARMv8 NEON + x86-64 AVX2/FMA Macro-Glyph Stream Accelerator
 *
 * Bridges DannyB-bit/Language-U-Gemma4 (Gemma 4 E4B & 31B Dual-Road Option 4
 * + Option 5 streams) with DannyB-bit/Neon-Runner and UniversalLayerOrderedCortex.
 *
 * Hardware Vector Engines:
 *   - ARMv8-A NEON (`__aarch64__` / `__ARM_NEON`): 2x float32x4_t per S^7 octonion
 *   - x86-64 AVX2+FMA (`__AVX2__`): 1x __m256 (256-bit = 8x float32 = 1 S^7 octonion!)
 *   - Portable Scalar C fallback for any other architecture
 * ============================================================================
 */

#ifndef _WIN32
#define _GNU_SOURCE
#endif

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

#if defined(_WIN32) && defined(NEON_MSVC_FREESTANDING_DLL)
  #include <vcruntime_string.h>
  #include <intrin.h>
  __declspec(dllimport) int __stdcall QueryPerformanceFrequency(int64_t* lpFrequency);
  __declspec(dllimport) int __stdcall QueryPerformanceCounter(int64_t* lpPerformanceCount);
  __declspec(dllimport) void* __stdcall GetProcessHeap(void);
  __declspec(dllimport) void* __stdcall HeapAlloc(void* hHeap, uint32_t dwFlags, size_t dwBytes);
  __declspec(dllimport) int __stdcall HeapFree(void* hHeap, uint32_t dwFlags, void* lpMem);
  static inline void* neon_alloc(size_t n) { return HeapAlloc(GetProcessHeap(), 0, n); }
  static inline void  neon_free(void* p)   { if (p) HeapFree(GetProcessHeap(), 0, p); }
  static inline float neon_fabsf(float v)  { return (v < 0.0f) ? -v : v; }
  int _fltused = 0;
#else
  #include <stdio.h>
  #include <stdlib.h>
  #include <string.h>
  #include <math.h>
  #ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
  #else
    #include <time.h>
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>
  #endif
  static inline void* neon_alloc(size_t n) { return malloc(n); }
  static inline void  neon_free(void* p)   { free(p); }
  static inline float neon_fabsf(float v)  { return fabsf(v); }
#endif

#if defined(_OPENMP) && !defined(NEON_NO_OPENMP)
  #include <omp.h>
  #define HAS_OMP 1
#else
  #define HAS_OMP 0
#endif

#include "neon_stream_accelerator_v2.h"

#if defined(__aarch64__) || defined(__ARM_NEON)
  #include <arm_neon.h>
  #define HAS_NEON 1
#else
  #define HAS_NEON 0
#endif

#if defined(__AVX2__)
  #include <immintrin.h>
  #define HAS_AVX2 1
#else
  #define HAS_AVX2 0
#endif

#if defined(_MSC_VER)
  #define NEON_ALIGN64 __declspec(align(64))
#else
  #define NEON_ALIGN64 __attribute__((aligned(64)))
#endif

/* ----------------------------------------------------------------------------
 * Safe Little-Endian uint16 + IEEE-754 Binary16 -> Binary32 Conversion
 * ---------------------------------------------------------------------------- */
static inline uint16_t read_u16_le(const uint8_t* ptr) {
    return (uint16_t)ptr[0] | ((uint16_t)ptr[1] << 8);
}

static inline float fp16_to_fp32(uint16_t h) {
#if defined(__aarch64__) && (defined(__ARM_FP16_FORMAT_IEEE) || defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC))
    __fp16 f16;
    memcpy(&f16, &h, sizeof(uint16_t));
    return (float)f16;
#else
    uint32_t sign = ((uint32_t)(h & 0x8000u)) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t man  = h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (man == 0) {
            f = sign;
        } else {
            exp = 127u - 15u + 1u;
            while (!(man & 0x400u)) {
                man <<= 1;
                exp--;
            }
            man &= 0x3FFu;
            f = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 0x1Fu) {
        f = sign | 0x7F800000u | (man << 13);
    } else {
        f = sign | ((exp - 15u + 127u) << 23) | (man << 13);
    }
    float out;
    memcpy(&out, &f, sizeof(float));
    return out;
#endif
}

/* ----------------------------------------------------------------------------
 * High-Resolution Monotonic Wall-Clock Timer
 * ---------------------------------------------------------------------------- */
static inline double get_time_sec(void) {
#if defined(_WIN32) && defined(NEON_MSVC_FREESTANDING_DLL)
    int64_t freq = 1, counter = 0;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter / (double)(freq ? freq : 1);
#elif defined(_WIN32)
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

#if HAS_AVX2
static inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(s4);
    __m128 s2 = _mm_add_ps(s4, shuf);
    shuf = _mm_movehl_ps(shuf, s2);
    return _mm_cvtss_f32(_mm_add_ss(s2, shuf));
}
#endif

/* ----------------------------------------------------------------------------
 * Option 5 L1-Resident 8 KB Sign LUT (MSB-first: bit 7..0 -> +1.0f / -1.0f)
 * ---------------------------------------------------------------------------- */
static NEON_ALIGN64 float g_sign_lut[256][8];
static volatile int g_sign_lut_ready = 0;

static void ensure_sign_lut_initialized(void) {
    if (g_sign_lut_ready) return;
    for (int v = 0; v < 256; v++) {
        for (int i = 0; i < 8; i++) {
            g_sign_lut[v][i] = ((v >> (7 - i)) & 1) ? 1.0f : -1.0f;
        }
    }
    g_sign_lut_ready = 1;
}

#if !defined(_MSC_VER)
__attribute__((constructor))
static void neon_v2_auto_init(void) {
    ensure_sign_lut_initialized();
}
#endif

/* ----------------------------------------------------------------------------
 * Fast Walsh-Hadamard Transform Order 8 (Orthonormal H_8 / sqrt(8))
 * ---------------------------------------------------------------------------- */
static inline void fwht8(const float* in, float* out) {
    const float inv_sqrt8 = 0.3535533905932737622f;

    float a0 = in[0] + in[1], a1 = in[0] - in[1];
    float a2 = in[2] + in[3], a3 = in[2] - in[3];
    float a4 = in[4] + in[5], a5 = in[4] - in[5];
    float a6 = in[6] + in[7], a7 = in[6] - in[7];

    float b0 = a0 + a2, b2 = a0 - a2;
    float b1 = a1 + a3, b3 = a1 - a3;
    float b4 = a4 + a6, b6 = a4 - a6;
    float b5 = a5 + a7, b7 = a5 - a7;

    out[0] = (b0 + b4) * inv_sqrt8;
    out[4] = (b0 - b4) * inv_sqrt8;
    out[1] = (b1 + b5) * inv_sqrt8;
    out[5] = (b1 - b5) * inv_sqrt8;
    out[2] = (b2 + b6) * inv_sqrt8;
    out[6] = (b2 - b6) * inv_sqrt8;
    out[3] = (b3 + b7) * inv_sqrt8;
    out[7] = (b3 - b7) * inv_sqrt8;
}

NEON_API void fwht8_vec_c(
    const float* NEON_RESTRICT x_in,
    float* NEON_RESTRICT x_out,
    int n
) {
    int octets = n / 8;
    int o;
    for (o = 0; o < octets; o++) {
        float tmp[8];
        memcpy(tmp, x_in + (size_t)o * 8, 8 * sizeof(float));
        fwht8(tmp, x_out + (size_t)o * 8);
    }
}

/* ----------------------------------------------------------------------------
 * Option 4 (132-Byte Dual-Stage RVQ) Single-Glyph Decode
 * ---------------------------------------------------------------------------- */
static inline void decode_single_glyph_mode(
    const uint8_t* NEON_RESTRICT glyph,
    const float* NEON_RESTRICT cb,
    float* NEON_RESTRICT out_weights,
    int apply_fwht8
) {
    const float s1 = fp16_to_fp32(read_u16_le(glyph + 0));
    const float s2 = fp16_to_fp32(read_u16_le(glyph + 2));
    const uint8_t* p_idx1 = glyph + 4;
    const uint8_t* p_idx2 = glyph + 68;

#if HAS_NEON
    const float32x4_t s1v = vdupq_n_f32(s1);
    const float32x4_t s2v = vdupq_n_f32(s2);

    for (int oct = 0; oct < 32; oct++) {
        uint16_t i1 = read_u16_le(p_idx1 + oct * 2) & 2047u;
        uint16_t i2 = read_u16_le(p_idx2 + oct * 2) & 2047u;
        const float* cb1 = cb + ((size_t)i1 << 3);
        const float* cb2 = cb + ((size_t)i2 << 3);

        float32x4_t v0 = vfmaq_f32(vmulq_f32(vld1q_f32(cb1), s1v), vld1q_f32(cb2), s2v);
        float32x4_t v4 = vfmaq_f32(vmulq_f32(vld1q_f32(cb1 + 4), s1v), vld1q_f32(cb2 + 4), s2v);

        if (apply_fwht8) {
            float temp[8];
            vst1q_f32(temp, v0);
            vst1q_f32(temp + 4, v4);
            fwht8(temp, out_weights + (oct << 3));
        } else {
            vst1q_f32(out_weights + (oct << 3), v0);
            vst1q_f32(out_weights + (oct << 3) + 4, v4);
        }
    }
#elif HAS_AVX2
    const __m256 s1v = _mm256_set1_ps(s1);
    const __m256 s2v = _mm256_set1_ps(s2);

    for (int oct = 0; oct < 32; oct++) {
        uint16_t i1 = read_u16_le(p_idx1 + oct * 2) & 2047u;
        uint16_t i2 = read_u16_le(p_idx2 + oct * 2) & 2047u;
        const float* cb1 = cb + ((size_t)i1 << 3);
        const float* cb2 = cb + ((size_t)i2 << 3);
        __m256 v = _mm256_fmadd_ps(_mm256_loadu_ps(cb2), s2v, _mm256_mul_ps(_mm256_loadu_ps(cb1), s1v));
        if (apply_fwht8) {
            float temp[8];
            _mm256_storeu_ps(temp, v);
            fwht8(temp, out_weights + (oct << 3));
        } else {
            _mm256_storeu_ps(out_weights + (oct << 3), v);
        }
    }
#else
    for (int oct = 0; oct < 32; oct++) {
        uint16_t i1 = read_u16_le(p_idx1 + oct * 2) & 2047u;
        uint16_t i2 = read_u16_le(p_idx2 + oct * 2) & 2047u;
        const float* cb1 = cb + ((size_t)i1 << 3);
        const float* cb2 = cb + ((size_t)i2 << 3);
        if (apply_fwht8) {
            float temp[8];
            for (int t = 0; t < 8; t++) temp[t] = s1 * cb1[t] + s2 * cb2[t];
            fwht8(temp, out_weights + (oct << 3));
        } else {
            float* dst = out_weights + (oct << 3);
            for (int t = 0; t < 8; t++) dst[t] = s1 * cb1[t] + s2 * cb2[t];
        }
    }
#endif
}

NEON_API void decode_glyphs_opt4_mode_c(
    const uint8_t* NEON_RESTRICT glyphs,
    const float* NEON_RESTRICT cb,
    float* NEON_RESTRICT out_weights,
    int n_glyphs,
    int apply_fwht8,
    int num_threads
) {
    int g;
#if HAS_OMP
    if (num_threads > 0) omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(static)
#else
    (void)num_threads;
#endif
    for (g = 0; g < n_glyphs; g++) {
        decode_single_glyph_mode(
            glyphs + (size_t)g * 132,
            cb,
            out_weights + (size_t)g * 256,
            apply_fwht8
        );
    }
}

NEON_API void decode_glyphs_c(
    const uint8_t* NEON_RESTRICT glyphs,
    const float* NEON_RESTRICT cb,
    float* NEON_RESTRICT out_weights,
    int n_glyphs,
    int num_threads
) {
    decode_glyphs_opt4_mode_c(glyphs, cb, out_weights, n_glyphs, 1, num_threads);
}

NEON_API double benchmark_projection_neon(
    const uint8_t* NEON_RESTRICT mmap_base,
    size_t byte_offset,
    int n_glyphs,
    const float* NEON_RESTRICT cb,
    int num_threads
) {
#if HAS_OMP
    if (num_threads > 0) omp_set_num_threads(num_threads);
#else
    (void)num_threads;
#endif
    double t0 = get_time_sec();
#if HAS_OMP
    #pragma omp parallel
#endif
    {
        float* local_ring = (float*)neon_alloc(256 * 256 * sizeof(float));
        if (local_ring) {
            int g;
#if HAS_OMP
            #pragma omp for schedule(static)
#endif
            for (g = 0; g < n_glyphs; g++) {
                const uint8_t* glyph_ptr = mmap_base + byte_offset + (size_t)g * 132;
                float* out_ptr = local_ring + ((size_t)g & 255u) * 256;
                decode_single_glyph_mode(glyph_ptr, cb, out_ptr, 1);
            }
            neon_free(local_ring);
        }
    }
    double t1 = get_time_sec();
    return t1 - t0;
}

/* ----------------------------------------------------------------------------
 * Option 4 Fused Zero-Heap GEMV (ARMv8 NEON + x86-64 AVX2/FMA + Scalar)
 * ---------------------------------------------------------------------------- */
#if HAS_NEON
static inline float hsum4_pair(float32x4_t a, float32x4_t b) {
    return vaddvq_f32(a) + vaddvq_f32(b);
}
#endif

NEON_API void gemv_opt4_mmap(
    const uint8_t* NEON_RESTRICT mmap_base,
    size_t tensor_byte_offset,
    const float* NEON_RESTRICT codebook_s7,
    const float* NEON_RESTRICT x,
    float* NEON_RESTRICT y,
    int d_out,
    int d_in
) {
    const int glyphs_per_row = d_in / 256;
    const uint8_t* tensor_ptr = mmap_base + tensor_byte_offset;
    int row;

#if HAS_OMP
    #pragma omp parallel for schedule(static)
#endif
    for (row = 0; row < d_out; row++) {
        const uint8_t* row_ptr = tensor_ptr + (size_t)row * (size_t)glyphs_per_row * 132;
        float row_acc = 0.0f;
        int g = 0;

#if HAS_NEON
        for (; g + 4 <= glyphs_per_row; g += 4) {
            float32x4_t acc[4][2][2];
            for (int c = 0; c < 4; c++)
                for (int s = 0; s < 2; s++)
                    for (int h = 0; h < 2; h++)
                        acc[c][s][h] = vdupq_n_f32(0.0f);

            for (int oct = 0; oct < 32; oct++) {
                for (int c = 0; c < 4; c++) {
                    const uint8_t* glyph = row_ptr + (size_t)(g + c) * 132;
                    uint16_t i1 = read_u16_le(glyph + 4 + oct * 2) & 2047u;
                    uint16_t i2 = read_u16_le(glyph + 68 + oct * 2) & 2047u;
                    const float* xg = x + (size_t)(g + c) * 256 + (oct << 3);
                    float32x4_t x0 = vld1q_f32(xg);
                    float32x4_t x4 = vld1q_f32(xg + 4);
                    const float* cb1 = codebook_s7 + ((size_t)i1 << 3);
                    const float* cb2 = codebook_s7 + ((size_t)i2 << 3);
                    acc[c][0][0] = vfmaq_f32(acc[c][0][0], vld1q_f32(cb1),     x0);
                    acc[c][0][1] = vfmaq_f32(acc[c][0][1], vld1q_f32(cb1 + 4), x4);
                    acc[c][1][0] = vfmaq_f32(acc[c][1][0], vld1q_f32(cb2),     x0);
                    acc[c][1][1] = vfmaq_f32(acc[c][1][1], vld1q_f32(cb2 + 4), x4);
                }
            }
            for (int c = 0; c < 4; c++) {
                const uint8_t* glyph = row_ptr + (size_t)(g + c) * 132;
                float s1 = fp16_to_fp32(read_u16_le(glyph + 0));
                float s2 = fp16_to_fp32(read_u16_le(glyph + 2));
                row_acc += s1 * hsum4_pair(acc[c][0][0], acc[c][0][1])
                         + s2 * hsum4_pair(acc[c][1][0], acc[c][1][1]);
            }
        }
#endif
        for (; g < glyphs_per_row; g++) {
            const uint8_t* glyph = row_ptr + (size_t)g * 132;
            const float* x_slice = x + (size_t)g * 256;
            const float scale1 = fp16_to_fp32(read_u16_le(glyph + 0));
            const float scale2 = fp16_to_fp32(read_u16_le(glyph + 2));
#if HAS_NEON
            float32x4_t a1 = vdupq_n_f32(0.0f), b1 = vdupq_n_f32(0.0f);
            float32x4_t a2 = vdupq_n_f32(0.0f), b2 = vdupq_n_f32(0.0f);
            for (int oct = 0; oct < 32; oct++) {
                uint16_t i1 = read_u16_le(glyph + 4 + oct * 2) & 2047u;
                uint16_t i2 = read_u16_le(glyph + 68 + oct * 2) & 2047u;
                const float* cb1 = codebook_s7 + ((size_t)i1 << 3);
                const float* cb2 = codebook_s7 + ((size_t)i2 << 3);
                const float* xo  = x_slice + (oct << 3);
                float32x4_t x0 = vld1q_f32(xo);
                float32x4_t x4 = vld1q_f32(xo + 4);
                a1 = vfmaq_f32(a1, vld1q_f32(cb1),     x0);
                b1 = vfmaq_f32(b1, vld1q_f32(cb1 + 4), x4);
                a2 = vfmaq_f32(a2, vld1q_f32(cb2),     x0);
                b2 = vfmaq_f32(b2, vld1q_f32(cb2 + 4), x4);
            }
            row_acc += scale1 * hsum4_pair(a1, b1) + scale2 * hsum4_pair(a2, b2);
#elif HAS_AVX2
            __m256 a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps();
            for (int oct = 0; oct < 32; oct++) {
                uint16_t i1 = read_u16_le(glyph + 4 + oct * 2) & 2047u;
                uint16_t i2 = read_u16_le(glyph + 68 + oct * 2) & 2047u;
                __m256 xv = _mm256_loadu_ps(x_slice + (oct << 3));
                a1 = _mm256_fmadd_ps(_mm256_loadu_ps(codebook_s7 + ((size_t)i1 << 3)), xv, a1);
                a2 = _mm256_fmadd_ps(_mm256_loadu_ps(codebook_s7 + ((size_t)i2 << 3)), xv, a2);
            }
            row_acc += scale1 * hsum256_ps(a1) + scale2 * hsum256_ps(a2);
#else
            float s1d = 0.0f, s2d = 0.0f;
            for (int oct = 0; oct < 32; oct++) {
                uint16_t i1 = read_u16_le(glyph + 4 + oct * 2) & 2047u;
                uint16_t i2 = read_u16_le(glyph + 68 + oct * 2) & 2047u;
                const float* cb1 = codebook_s7 + ((size_t)i1 << 3);
                const float* cb2 = codebook_s7 + ((size_t)i2 << 3);
                const float* xo  = x_slice + (oct << 3);
                for (int t = 0; t < 8; t++) {
                    s1d += cb1[t] * xo[t];
                    s2d += cb2[t] * xo[t];
                }
            }
            row_acc += scale1 * s1d + scale2 * s2d;
#endif
        }
        y[row] = row_acc;
    }
}

NEON_API void gemv_opt4_fwht_mmap(
    const uint8_t* NEON_RESTRICT mmap_base,
    size_t tensor_byte_offset,
    const float* NEON_RESTRICT codebook_s7,
    const float* NEON_RESTRICT x,
    float* NEON_RESTRICT y,
    int d_out,
    int d_in
) {
    float stack_x[512];
    float* x_rot = (d_in <= 512) ? stack_x : (float*)neon_alloc((size_t)d_in * sizeof(float));
    if (!x_rot) return;
    fwht8_vec_c(x, x_rot, d_in);
    gemv_opt4_mmap(mmap_base, tensor_byte_offset, codebook_s7, x_rot, y, d_out, d_in);
    if (x_rot != stack_x) {
        neon_free(x_rot);
    }
}

NEON_API void gemv_opt4_mmap_multi(
    const uint8_t* NEON_RESTRICT mmap_base,
    size_t tensor_byte_offset,
    const float* NEON_RESTRICT codebook_s7,
    const float* const* NEON_RESTRICT x_ptrs,
    float* const* NEON_RESTRICT y_ptrs,
    int K,
    int d_out,
    int d_in
) {
    const int glyphs_per_row = d_in / 256;
    const uint8_t* tensor_ptr = mmap_base + tensor_byte_offset;
    const int k_clamped = (K > 64) ? 64 : K;
    int row;

#if HAS_OMP
    #pragma omp parallel for schedule(static)
#endif
    for (row = 0; row < d_out; row++) {
        const uint8_t* row_ptr = tensor_ptr + (size_t)row * (size_t)glyphs_per_row * 132;
        float acc[64];
        for (int k = 0; k < k_clamped; k++) acc[k] = 0.0f;

        for (int g = 0; g < glyphs_per_row; g++) {
            const uint8_t* glyph = row_ptr + (size_t)g * 132;
            const float scale1 = fp16_to_fp32(read_u16_le(glyph + 0));
            const float scale2 = fp16_to_fp32(read_u16_le(glyph + 2));

#if HAS_NEON
            const float32x4_t s2v = vdupq_n_f32(scale2);
            for (int oct = 0; oct < 32; oct++) {
                uint16_t i1 = read_u16_le(glyph + 4 + oct * 2) & 2047u;
                uint16_t i2 = read_u16_le(glyph + 68 + oct * 2) & 2047u;
                const float* cb1 = codebook_s7 + ((size_t)i1 << 3);
                const float* cb2 = codebook_s7 + ((size_t)i2 << 3);
                const float32x4_t w0 = vfmaq_f32(vmulq_n_f32(vld1q_f32(cb1), scale1),
                                                  vld1q_f32(cb2), s2v);
                const float32x4_t w4 = vfmaq_f32(vmulq_n_f32(vld1q_f32(cb1 + 4), scale1),
                                                  vld1q_f32(cb2 + 4), s2v);
                for (int k = 0; k < k_clamped; k++) {
                    const float* xg = x_ptrs[k] + (size_t)g * 256 + (oct << 3);
                    const float32x4_t p0 = vfmaq_f32(vmulq_f32(w0, vld1q_f32(xg)),
                                                     w4, vld1q_f32(xg + 4));
                    acc[k] += vaddvq_f32(p0);
                }
            }
#elif HAS_AVX2
            const __m256 s1v = _mm256_set1_ps(scale1);
            const __m256 s2v = _mm256_set1_ps(scale2);
            for (int oct = 0; oct < 32; oct++) {
                uint16_t i1 = read_u16_le(glyph + 4 + oct * 2) & 2047u;
                uint16_t i2 = read_u16_le(glyph + 68 + oct * 2) & 2047u;
                __m256 w = _mm256_fmadd_ps(
                    _mm256_loadu_ps(codebook_s7 + ((size_t)i2 << 3)),
                    s2v,
                    _mm256_mul_ps(_mm256_loadu_ps(codebook_s7 + ((size_t)i1 << 3)), s1v)
                );
                for (int k = 0; k < k_clamped; k++) {
                    const float* xg = x_ptrs[k] + (size_t)g * 256 + (oct << 3);
                    acc[k] += hsum256_ps(_mm256_mul_ps(w, _mm256_loadu_ps(xg)));
                }
            }
#else
            for (int oct = 0; oct < 32; oct++) {
                uint16_t i1 = read_u16_le(glyph + 4 + oct * 2) & 2047u;
                uint16_t i2 = read_u16_le(glyph + 68 + oct * 2) & 2047u;
                const float* cb1 = codebook_s7 + ((size_t)i1 << 3);
                const float* cb2 = codebook_s7 + ((size_t)i2 << 3);
                float w[8];
                for (int t = 0; t < 8; t++) w[t] = scale1 * cb1[t] + scale2 * cb2[t];
                for (int k = 0; k < k_clamped; k++) {
                    const float* xg = x_ptrs[k] + (size_t)g * 256 + (oct << 3);
                    float dot = 0.0f;
                    for (int t = 0; t < 8; t++) dot += w[t] * xg[t];
                    acc[k] += dot;
                }
            }
#endif
        }
        for (int k = 0; k < k_clamped; k++) y_ptrs[k][row] = acc[k];
    }
}

/* ----------------------------------------------------------------------------
 * Option 5 (36B Standard / 48B Top-4 Outlier) Decode & Fused Zero-Heap GEMV
 * ---------------------------------------------------------------------------- */
static inline void decode_single_opt5_glyph(
    const uint8_t* NEON_RESTRICT glyph,
    float* NEON_RESTRICT out_weights,
    int bpg
) {
    const float scale = fp16_to_fp32(read_u16_le(glyph + 0));
    const float bias  = fp16_to_fp32(read_u16_le(glyph + 2));
    const uint8_t* bitmask = glyph + 4;

#if HAS_AVX2
    const __m256 sv = _mm256_set1_ps(scale);
    const __m256 bv = _mm256_set1_ps(bias);
    for (int b = 0; b < 32; b++) {
        __m256 signs = _mm256_load_ps(g_sign_lut[bitmask[b]]);
        _mm256_storeu_ps(out_weights + b * 8, _mm256_fmadd_ps(signs, sv, bv));
    }
#else
    for (int b = 0; b < 32; b++) {
        const float* signs = g_sign_lut[bitmask[b]];
        float* dst = out_weights + b * 8;
        for (int i = 0; i < 8; i++) dst[i] = scale * signs[i] + bias;
    }
#endif

    if (bpg == 48) {
        const uint8_t* outlier_idx = glyph + 36;
        const uint8_t* outlier_res = glyph + 40;
        for (int o = 0; o < 4; o++) {
            uint8_t oi = outlier_idx[o];
            float ores = fp16_to_fp32(read_u16_le(outlier_res + o * 2));
            out_weights[oi] += ores;
        }
    }
}

NEON_API void decode_glyphs_opt5_c(
    const uint8_t* NEON_RESTRICT glyphs,
    float* NEON_RESTRICT out_weights,
    int n_glyphs,
    int bytes_per_glyph,
    int num_threads
) {
    ensure_sign_lut_initialized();
    const int bpg = (bytes_per_glyph == 48) ? 48 : 36;
    int g;
#if HAS_OMP
    if (num_threads > 0) omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(static)
#else
    (void)num_threads;
#endif
    for (g = 0; g < n_glyphs; g++) {
        decode_single_opt5_glyph(
            glyphs + (size_t)g * (size_t)bpg,
            out_weights + (size_t)g * 256,
            bpg
        );
    }
}

NEON_API void gemv_opt5_mmap(
    const uint8_t* NEON_RESTRICT mmap_base,
    size_t tensor_byte_offset,
    const float* NEON_RESTRICT x,
    float* NEON_RESTRICT y,
    int d_out,
    int d_in,
    int bytes_per_glyph
) {
    ensure_sign_lut_initialized();
    const int bpg = (bytes_per_glyph == 48) ? 48 : 36;
    const int glyphs_per_row = d_in / 256;
    const uint8_t* tensor_ptr = mmap_base + tensor_byte_offset;

    float stack_sums[128];
    float* xg_sum = (glyphs_per_row <= 128)
        ? stack_sums
        : (float*)neon_alloc((size_t)glyphs_per_row * sizeof(float));
    if (!xg_sum) return;

    for (int g = 0; g < glyphs_per_row; g++) {
        const float* xs = x + (size_t)g * 256;
#if HAS_NEON
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (int i = 0; i < 256; i += 16) {
            acc = vaddq_f32(acc, vld1q_f32(xs + i));
            acc = vaddq_f32(acc, vld1q_f32(xs + i + 4));
            acc = vaddq_f32(acc, vld1q_f32(xs + i + 8));
            acc = vaddq_f32(acc, vld1q_f32(xs + i + 12));
        }
        xg_sum[g] = vaddvq_f32(acc);
#elif HAS_AVX2
        __m256 acc = _mm256_setzero_ps();
        for (int i = 0; i < 256; i += 32) {
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(xs + i));
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(xs + i + 8));
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(xs + i + 16));
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(xs + i + 24));
        }
        xg_sum[g] = hsum256_ps(acc);
#else
        float s = 0.0f;
        for (int i = 0; i < 256; i++) s += xs[i];
        xg_sum[g] = s;
#endif
    }

    int row;
#if HAS_OMP
    #pragma omp parallel for schedule(static)
#endif
    for (row = 0; row < d_out; row++) {
        const uint8_t* row_ptr = tensor_ptr + (size_t)row * (size_t)glyphs_per_row * (size_t)bpg;
        float row_acc = 0.0f;

        for (int g = 0; g < glyphs_per_row; g++) {
            const uint8_t* glyph = row_ptr + (size_t)g * (size_t)bpg;
            const float* x_slice = x + (size_t)g * 256;

            float scale = fp16_to_fp32(read_u16_le(glyph + 0));
            float bias  = fp16_to_fp32(read_u16_le(glyph + 2));
            const uint8_t* bitmask = glyph + 4;

#if HAS_NEON
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (int b = 0; b < 32; b++) {
                const float* lut = g_sign_lut[bitmask[b]];
                acc = vfmaq_f32(acc, vld1q_f32(lut),     vld1q_f32(x_slice + b * 8));
                acc = vfmaq_f32(acc, vld1q_f32(lut + 4), vld1q_f32(x_slice + b * 8 + 4));
            }
            float sign_dot = vaddvq_f32(acc);
#elif HAS_AVX2
            __m256 acc = _mm256_setzero_ps();
            for (int b = 0; b < 32; b++) {
                acc = _mm256_fmadd_ps(
                    _mm256_load_ps(g_sign_lut[bitmask[b]]),
                    _mm256_loadu_ps(x_slice + b * 8),
                    acc
                );
            }
            float sign_dot = hsum256_ps(acc);
#else
            float sign_dot = 0.0f;
            for (int b = 0; b < 32; b++) {
                const float* lut = g_sign_lut[bitmask[b]];
                const float* xb  = x_slice + b * 8;
                for (int i = 0; i < 8; i++) sign_dot += lut[i] * xb[i];
            }
#endif
            float base_acc = (scale * sign_dot) + (bias * xg_sum[g]);

            if (bpg == 48) {
                const uint8_t* outlier_idx = glyph + 36;
                const uint8_t* outlier_res = glyph + 40;
                for (int o = 0; o < 4; o++) {
                    uint8_t oi = outlier_idx[o];
                    float ores = fp16_to_fp32(read_u16_le(outlier_res + o * 2));
                    base_acc += ores * x_slice[oi];
                }
            }
            row_acc += base_acc;
        }
        y[row] = row_acc;
    }

    if (xg_sum != stack_sums) {
        neon_free(xg_sum);
    }
}

/* ----------------------------------------------------------------------------
 * Self-Verification Battery (FWHT-8 + Option 4 + Option 5 Parity)
 * ---------------------------------------------------------------------------- */
NEON_API int neon_accelerator_self_test(void) {
    ensure_sign_lut_initialized();

    /* Gate A: FWHT-8 self-inverse orthogonality H_8 * H_8 = I_8 */
    float in[8] = {1.0f, -2.0f, 3.5f, 4.0f, -5.0f, 6.25f, -7.0f, 8.0f};
    float mid[8], out[8];
    fwht8(in, mid);
    fwht8(mid, out);
    for (int i = 0; i < 8; i++) {
        if (neon_fabsf(out[i] - in[i]) > 1e-5f) return 0;
    }

    /* Gate B: Option 4 decode_glyphs_c vs gemv_opt4_fwht_mmap mathematical parity */
    float* cb = (float*)neon_alloc(2048 * 8 * sizeof(float));
    if (!cb) return 0;
    memset(cb, 0, 2048 * 8 * sizeof(float));
    for (int c = 0; c < 2048; c++) {
        cb[c * 8 + (c & 7)] = (c & 8) ? -1.0f : 1.0f;
    }
    uint8_t glyph4[132];
    memset(glyph4, 0, sizeof(glyph4));
    glyph4[0] = 0x00; glyph4[1] = 0x38; /* FP16 0.5 */
    glyph4[2] = 0x00; glyph4[3] = 0x34; /* FP16 0.25 */
    for (int oct = 0; oct < 32; oct++) {
        uint16_t i1 = (uint16_t)((oct * 7 + 3) & 2047);
        uint16_t i2 = (uint16_t)((oct * 13 + 11) & 2047);
        glyph4[4 + oct * 2]      = (uint8_t)(i1 & 0xFF);
        glyph4[4 + oct * 2 + 1]  = (uint8_t)(i1 >> 8);
        glyph4[68 + oct * 2]     = (uint8_t)(i2 & 0xFF);
        glyph4[68 + oct * 2 + 1] = (uint8_t)(i2 >> 8);
    }
    float w4[256], x[256], y_gemv[1];
    for (int i = 0; i < 256; i++) x[i] = 0.01f * (float)((i % 19) - 9);
    decode_glyphs_c(glyph4, cb, w4, 1, 1);
    gemv_opt4_fwht_mmap(glyph4, 0, cb, x, y_gemv, 1, 256);
    neon_free(cb);

    float y_ref4 = 0.0f;
    for (int i = 0; i < 256; i++) y_ref4 += w4[i] * x[i];
    if (neon_fabsf(y_gemv[0] - y_ref4) > 1e-4f) return 0;

    /* Gate C: Option 5 (48B Outlier) decode_glyphs_opt5_c vs gemv_opt5_mmap parity */
    uint8_t glyph5[48];
    memset(glyph5, 0, sizeof(glyph5));
    glyph5[0] = 0x00; glyph5[1] = 0x38; /* scale = 0.5 */
    glyph5[2] = 0x00; glyph5[3] = 0x34; /* bias = 0.25 */
    for (int b = 0; b < 32; b++) glyph5[4 + b] = (uint8_t)(0xA5u ^ (uint8_t)(b * 3));
    glyph5[36] = 5; glyph5[37] = 77; glyph5[38] = 140; glyph5[39] = 250;
    glyph5[40] = 0x00; glyph5[41] = 0x3C;
    glyph5[42] = 0x00; glyph5[43] = 0xBC;
    glyph5[44] = 0x00; glyph5[45] = 0x38;
    glyph5[46] = 0x00; glyph5[47] = 0xB8;

    float w5[256], y5[1];
    decode_glyphs_opt5_c(glyph5, w5, 1, 48, 1);
    gemv_opt5_mmap(glyph5, 0, x, y5, 1, 256, 48);
    float y_ref5 = 0.0f;
    for (int i = 0; i < 256; i++) y_ref5 += w5[i] * x[i];
    if (neon_fabsf(y5[0] - y_ref5) > 1e-4f) return 0;

    return 1;
}

NEON_API int trinity_gemv_self_test(void) {
    return neon_accelerator_self_test();
}

#ifndef NEON_SHARED_LIB_ONLY
int main(int argc, char** argv) {
    printf("================================================================================\n");
    printf("NEON RUNNER V2 — UNIFIED LANGUAGE-U GEMMA 4 (E4B & 31B) STREAM ACCELERATOR\n");
    printf("================================================================================\n");

    if (!neon_accelerator_self_test()) {
        fprintf(stderr, "FATAL: neon_accelerator_self_test() FAILED!\n");
        return 1;
    }
    printf("[PASS] In-register FWHT-8 + Option 4 RVQ + Option 5 Outlier parity self-test\n");

    if (argc < 3) {
        printf("Usage: %s <stream.bin> <codebook_2048x8.bin> [num_glyphs] [byte_offset]\n", argv[0]);
        return 0;
    }

    const char* stream_path = argv[1];
    const char* cb_path = argv[2];
    int n_glyphs = (argc > 3) ? atoi(argv[3]) : 1024;
    size_t byte_offset = (argc > 4) ? (size_t)atoll(argv[4]) : 16;

    FILE* fcb = fopen(cb_path, "rb");
    if (!fcb) { perror("fopen codebook"); return 1; }
    NEON_ALIGN64 float codebook[2048 * 8];
    size_t cb_elems = fread(codebook, sizeof(float), 2048 * 8, fcb);
    fclose(fcb);
    if (cb_elems != 2048 * 8) return 1;

#ifdef _WIN32
    HANDLE hFile = CreateFileA(stream_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) return 1;
    LARGE_INTEGER liSize;
    GetFileSizeEx(hFile, &liSize);
    size_t stream_size = (size_t)liSize.QuadPart;
    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    const uint8_t* mmap_base = hMap ? (const uint8_t*)MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0) : NULL;
    if (!mmap_base) return 1;
#else
    int fd = open(stream_path, O_RDONLY);
    if (fd < 0) { perror("open stream"); return 1; }
    struct stat st;
    fstat(fd, &st);
    size_t stream_size = (size_t)st.st_size;
    const uint8_t* mmap_base = (const uint8_t*)mmap(NULL, stream_size, PROT_READ, MAP_SHARED, fd, 0);
    if (mmap_base == MAP_FAILED) { perror("mmap stream"); close(fd); return 1; }
#endif

    if (byte_offset + (size_t)n_glyphs * 132 > stream_size) {
        n_glyphs = (int)((stream_size > byte_offset) ? ((stream_size - byte_offset) / 132) : 0);
    }
    double elapsed = benchmark_projection_neon(mmap_base, byte_offset, n_glyphs, codebook, 4);
    double glyphs_per_sec = (double)n_glyphs / (elapsed > 1e-9 ? elapsed : 1e-9);
    printf("[PASS] Stream %s (%zu bytes): decoded %d glyphs in %.3f ms (%.0f glyphs/s, %.1fM params/s)\n",
           stream_path, stream_size, n_glyphs, elapsed * 1000.0, glyphs_per_sec, (glyphs_per_sec * 256.0) / 1e6);

#ifdef _WIN32
    UnmapViewOfFile(mmap_base);
    CloseHandle(hMap);
    CloseHandle(hFile);
#else
    munmap((void*)mmap_base, stream_size);
    close(fd);
#endif
    return 0;
}
#endif /* NEON_SHARED_LIB_ONLY */
