/*
 * neon_stream_accelerator_v2.c
 * ============================================================================
 * Language-U-Gemma4-NeonRunner-Bridge-V2 (v2.1.0 — GlyphAttentionV2)
 * Unified ARMv8 NEON + x86-64 AVX2/FMA Macro-Glyph Stream & Attention Engine
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
  static inline float neon_sqrtf(float v)  { return (v <= 0.0f) ? 0.0f : _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(v))); }
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
  static inline float neon_sqrtf(float v)  { return (v <= 0.0f) ? 0.0f : sqrtf(v); }
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

static inline void write_u16_le(uint8_t* ptr, uint16_t val) {
    ptr[0] = (uint8_t)(val & 0xFFu);
    ptr[1] = (uint8_t)(val >> 8);
}

static inline uint16_t fp32_to_fp16(float val) {
    uint32_t f;
    memcpy(&f, &val, sizeof(float));
    uint32_t sign = (f >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((f >> 23) & 0xFFu) - 127 + 15;
    uint32_t man  = f & 0x007FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        man |= 0x00800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_bit = 1u << (shift - 1);
        uint32_t rem_mask = half_bit - 1u;
        uint32_t h_man = man >> shift;
        if ((man & half_bit) && ((man & rem_mask) || (h_man & 1u))) h_man++;
        return (uint16_t)(sign | h_man);
    } else if (exp >= 31) {
        if (((f >> 23) & 0xFFu) == 0xFFu && man != 0) {
            return (uint16_t)(sign | 0x7E00u);
        }
        return (uint16_t)(sign | 0x7C00u);
    } else {
        if ((man & 0x00001000u) && ((man & 0x00000FFFu) || ((man >> 13) & 1u))) {
            man += 0x00002000u;
            if (man & 0x00800000u) {
                man = 0;
                exp++;
                if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
            }
        }
        return (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
    }
}

static inline float neon_expf(float x) {
#if !defined(NEON_MSVC_FREESTANDING_DLL)
    return expf(x);
#else
    if (x <= -87.33654475f) return 0.0f;
    if (x >=  88.72283905f) x = 88.72283905f;
    const float inv_ln2 = 1.44269504088896341f;
    const float ln2_hi  = 0.693145751953125f;
    const float ln2_lo  = 1.428606765330187045e-6f;
    float fn = (x >= 0.0f) ? (float)((int32_t)(x * inv_ln2 + 0.5f))
                           : (float)((int32_t)(x * inv_ln2 - 0.5f));
    int32_t n = (int32_t)fn;
    float r = (x - fn * ln2_hi) - fn * ln2_lo;
    float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.1666666716f +
              r * (0.0416666679f + r * (0.00833333377f + r * 0.00138888892f)))));
    uint32_t pow2n_bits = (uint32_t)(n + 127) << 23;
    float pow2n;
    memcpy(&pow2n, &pow2n_bits, sizeof(float));
    return p * pow2n;
#endif
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
    if (K <= 0 || d_out <= 0 || d_in <= 0) return;
    if (K == 1) {
        gemv_opt4_mmap(mmap_base, tensor_byte_offset, codebook_s7, x_ptrs[0], y_ptrs[0], d_out, d_in);
        return;
    }

    const int glyphs_per_row = d_in / 256;
    const uint8_t* tensor_ptr = mmap_base + tensor_byte_offset;
    int row;

#if HAS_OMP
    #pragma omp parallel for schedule(static)
#endif
    for (row = 0; row < d_out; row++) {
        const uint8_t* row_ptr = tensor_ptr + (size_t)row * (size_t)glyphs_per_row * 132;

#if HAS_NEON
        int kb = 0;
        /* 8-token SSA register-blocked tile: row_ptr stays hot in 32KB L1D across tiles */
        for (; kb + 8 <= K; kb += 8) {
            const float* x0p = x_ptrs[kb + 0];
            const float* x1p = x_ptrs[kb + 1];
            const float* x2p = x_ptrs[kb + 2];
            const float* x3p = x_ptrs[kb + 3];
            const float* x4p = x_ptrs[kb + 4];
            const float* x5p = x_ptrs[kb + 5];
            const float* x6p = x_ptrs[kb + 6];
            const float* x7p = x_ptrs[kb + 7];
            float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
            float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
            float32x4_t a4 = vdupq_n_f32(0.0f), a5 = vdupq_n_f32(0.0f);
            float32x4_t a6 = vdupq_n_f32(0.0f), a7 = vdupq_n_f32(0.0f);

            for (int g = 0; g < glyphs_per_row; g++) {
                const uint8_t* glyph = row_ptr + (size_t)g * 132;
                const float scale1 = fp16_to_fp32(read_u16_le(glyph + 0));
                const float scale2 = fp16_to_fp32(read_u16_le(glyph + 2));
                const float32x4_t s2v = vdupq_n_f32(scale2);
                const size_t g_off = (size_t)g * 256;

                for (int oct = 0; oct < 32; oct++) {
                    uint16_t i1 = read_u16_le(glyph + 4 + oct * 2) & 2047u;
                    uint16_t i2 = read_u16_le(glyph + 68 + oct * 2) & 2047u;
                    const float* cb1 = codebook_s7 + ((size_t)i1 << 3);
                    const float* cb2 = codebook_s7 + ((size_t)i2 << 3);
                    const float32x4_t w0 = vfmaq_f32(vmulq_n_f32(vld1q_f32(cb1), scale1),
                                                      vld1q_f32(cb2), s2v);
                    const float32x4_t w4 = vfmaq_f32(vmulq_n_f32(vld1q_f32(cb1 + 4), scale1),
                                                      vld1q_f32(cb2 + 4), s2v);
                    const size_t col = g_off + ((size_t)oct << 3);
                    a0 = vfmaq_f32(vfmaq_f32(a0, w0, vld1q_f32(x0p + col)), w4, vld1q_f32(x0p + col + 4));
                    a1 = vfmaq_f32(vfmaq_f32(a1, w0, vld1q_f32(x1p + col)), w4, vld1q_f32(x1p + col + 4));
                    a2 = vfmaq_f32(vfmaq_f32(a2, w0, vld1q_f32(x2p + col)), w4, vld1q_f32(x2p + col + 4));
                    a3 = vfmaq_f32(vfmaq_f32(a3, w0, vld1q_f32(x3p + col)), w4, vld1q_f32(x3p + col + 4));
                    a4 = vfmaq_f32(vfmaq_f32(a4, w0, vld1q_f32(x4p + col)), w4, vld1q_f32(x4p + col + 4));
                    a5 = vfmaq_f32(vfmaq_f32(a5, w0, vld1q_f32(x5p + col)), w4, vld1q_f32(x5p + col + 4));
                    a6 = vfmaq_f32(vfmaq_f32(a6, w0, vld1q_f32(x6p + col)), w4, vld1q_f32(x6p + col + 4));
                    a7 = vfmaq_f32(vfmaq_f32(a7, w0, vld1q_f32(x7p + col)), w4, vld1q_f32(x7p + col + 4));
                }
            }
            y_ptrs[kb + 0][row] = vaddvq_f32(a0);
            y_ptrs[kb + 1][row] = vaddvq_f32(a1);
            y_ptrs[kb + 2][row] = vaddvq_f32(a2);
            y_ptrs[kb + 3][row] = vaddvq_f32(a3);
            y_ptrs[kb + 4][row] = vaddvq_f32(a4);
            y_ptrs[kb + 5][row] = vaddvq_f32(a5);
            y_ptrs[kb + 6][row] = vaddvq_f32(a6);
            y_ptrs[kb + 7][row] = vaddvq_f32(a7);
        }

        /* 4-token SSA register-blocked tile */
        if (kb + 4 <= K) {
            const float* x0p = x_ptrs[kb + 0];
            const float* x1p = x_ptrs[kb + 1];
            const float* x2p = x_ptrs[kb + 2];
            const float* x3p = x_ptrs[kb + 3];
            float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
            float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);

            for (int g = 0; g < glyphs_per_row; g++) {
                const uint8_t* glyph = row_ptr + (size_t)g * 132;
                const float scale1 = fp16_to_fp32(read_u16_le(glyph + 0));
                const float scale2 = fp16_to_fp32(read_u16_le(glyph + 2));
                const float32x4_t s2v = vdupq_n_f32(scale2);
                const size_t g_off = (size_t)g * 256;

                for (int oct = 0; oct < 32; oct++) {
                    uint16_t i1 = read_u16_le(glyph + 4 + oct * 2) & 2047u;
                    uint16_t i2 = read_u16_le(glyph + 68 + oct * 2) & 2047u;
                    const float* cb1 = codebook_s7 + ((size_t)i1 << 3);
                    const float* cb2 = codebook_s7 + ((size_t)i2 << 3);
                    const float32x4_t w0 = vfmaq_f32(vmulq_n_f32(vld1q_f32(cb1), scale1),
                                                      vld1q_f32(cb2), s2v);
                    const float32x4_t w4 = vfmaq_f32(vmulq_n_f32(vld1q_f32(cb1 + 4), scale1),
                                                      vld1q_f32(cb2 + 4), s2v);
                    const size_t col = g_off + ((size_t)oct << 3);
                    a0 = vfmaq_f32(vfmaq_f32(a0, w0, vld1q_f32(x0p + col)), w4, vld1q_f32(x0p + col + 4));
                    a1 = vfmaq_f32(vfmaq_f32(a1, w0, vld1q_f32(x1p + col)), w4, vld1q_f32(x1p + col + 4));
                    a2 = vfmaq_f32(vfmaq_f32(a2, w0, vld1q_f32(x2p + col)), w4, vld1q_f32(x2p + col + 4));
                    a3 = vfmaq_f32(vfmaq_f32(a3, w0, vld1q_f32(x3p + col)), w4, vld1q_f32(x3p + col + 4));
                }
            }
            y_ptrs[kb + 0][row] = vaddvq_f32(a0);
            y_ptrs[kb + 1][row] = vaddvq_f32(a1);
            y_ptrs[kb + 2][row] = vaddvq_f32(a2);
            y_ptrs[kb + 3][row] = vaddvq_f32(a3);
            kb += 4;
        }

        /* 1-token dual-accumulator tail */
        for (; kb < K; kb++) {
            const float* xp = x_ptrs[kb];
            float32x4_t a0 = vdupq_n_f32(0.0f), b0 = vdupq_n_f32(0.0f);
            for (int g = 0; g < glyphs_per_row; g++) {
                const uint8_t* glyph = row_ptr + (size_t)g * 132;
                const float scale1 = fp16_to_fp32(read_u16_le(glyph + 0));
                const float scale2 = fp16_to_fp32(read_u16_le(glyph + 2));
                const float32x4_t s2v = vdupq_n_f32(scale2);
                const size_t g_off = (size_t)g * 256;
                for (int oct = 0; oct < 32; oct++) {
                    uint16_t i1 = read_u16_le(glyph + 4 + oct * 2) & 2047u;
                    uint16_t i2 = read_u16_le(glyph + 68 + oct * 2) & 2047u;
                    const float* cb1 = codebook_s7 + ((size_t)i1 << 3);
                    const float* cb2 = codebook_s7 + ((size_t)i2 << 3);
                    const float32x4_t w0 = vfmaq_f32(vmulq_n_f32(vld1q_f32(cb1), scale1),
                                                      vld1q_f32(cb2), s2v);
                    const float32x4_t w4 = vfmaq_f32(vmulq_n_f32(vld1q_f32(cb1 + 4), scale1),
                                                      vld1q_f32(cb2 + 4), s2v);
                    const size_t col = g_off + ((size_t)oct << 3);
                    a0 = vfmaq_f32(a0, w0, vld1q_f32(xp + col));
                    b0 = vfmaq_f32(b0, w4, vld1q_f32(xp + col + 4));
                }
            }
            y_ptrs[kb][row] = vaddvq_f32(vaddq_f32(a0, b0));
        }
#elif HAS_AVX2
        int kb = 0;
        for (; kb + 8 <= K; kb += 8) {
            const float* x0p = x_ptrs[kb + 0];
            const float* x1p = x_ptrs[kb + 1];
            const float* x2p = x_ptrs[kb + 2];
            const float* x3p = x_ptrs[kb + 3];
            const float* x4p = x_ptrs[kb + 4];
            const float* x5p = x_ptrs[kb + 5];
            const float* x6p = x_ptrs[kb + 6];
            const float* x7p = x_ptrs[kb + 7];
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            __m256 a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps();
            __m256 a6 = _mm256_setzero_ps(), a7 = _mm256_setzero_ps();

            for (int g = 0; g < glyphs_per_row; g++) {
                const uint8_t* glyph = row_ptr + (size_t)g * 132;
                const float scale1 = fp16_to_fp32(read_u16_le(glyph + 0));
                const float scale2 = fp16_to_fp32(read_u16_le(glyph + 2));
                const __m256 s1v = _mm256_set1_ps(scale1);
                const __m256 s2v = _mm256_set1_ps(scale2);
                const size_t g_off = (size_t)g * 256;
                for (int oct = 0; oct < 32; oct++) {
                    uint16_t i1 = read_u16_le(glyph + 4 + oct * 2) & 2047u;
                    uint16_t i2 = read_u16_le(glyph + 68 + oct * 2) & 2047u;
                    __m256 w = _mm256_fmadd_ps(
                        _mm256_loadu_ps(codebook_s7 + ((size_t)i2 << 3)),
                        s2v,
                        _mm256_mul_ps(_mm256_loadu_ps(codebook_s7 + ((size_t)i1 << 3)), s1v)
                    );
                    const size_t col = g_off + ((size_t)oct << 3);
                    a0 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x0p + col), a0);
                    a1 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x1p + col), a1);
                    a2 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x2p + col), a2);
                    a3 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x3p + col), a3);
                    a4 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x4p + col), a4);
                    a5 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x5p + col), a5);
                    a6 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x6p + col), a6);
                    a7 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x7p + col), a7);
                }
            }
            y_ptrs[kb + 0][row] = hsum256_ps(a0);
            y_ptrs[kb + 1][row] = hsum256_ps(a1);
            y_ptrs[kb + 2][row] = hsum256_ps(a2);
            y_ptrs[kb + 3][row] = hsum256_ps(a3);
            y_ptrs[kb + 4][row] = hsum256_ps(a4);
            y_ptrs[kb + 5][row] = hsum256_ps(a5);
            y_ptrs[kb + 6][row] = hsum256_ps(a6);
            y_ptrs[kb + 7][row] = hsum256_ps(a7);
        }
        for (; kb < K; kb++) {
            const float* xp = x_ptrs[kb];
            __m256 a0 = _mm256_setzero_ps();
            for (int g = 0; g < glyphs_per_row; g++) {
                const uint8_t* glyph = row_ptr + (size_t)g * 132;
                const __m256 s1v = _mm256_set1_ps(fp16_to_fp32(read_u16_le(glyph + 0)));
                const __m256 s2v = _mm256_set1_ps(fp16_to_fp32(read_u16_le(glyph + 2)));
                const size_t g_off = (size_t)g * 256;
                for (int oct = 0; oct < 32; oct++) {
                    uint16_t i1 = read_u16_le(glyph + 4 + oct * 2) & 2047u;
                    uint16_t i2 = read_u16_le(glyph + 68 + oct * 2) & 2047u;
                    __m256 w = _mm256_fmadd_ps(
                        _mm256_loadu_ps(codebook_s7 + ((size_t)i2 << 3)),
                        s2v,
                        _mm256_mul_ps(_mm256_loadu_ps(codebook_s7 + ((size_t)i1 << 3)), s1v)
                    );
                    a0 = _mm256_fmadd_ps(w, _mm256_loadu_ps(xp + g_off + ((size_t)oct << 3)), a0);
                }
            }
            y_ptrs[kb][row] = hsum256_ps(a0);
        }
#else
        for (int kb = 0; kb < K; kb += 16) {
            const int kt = (K - kb < 16) ? (K - kb) : 16;
            float acc[16];
            for (int t = 0; t < kt; t++) acc[t] = 0.0f;
            for (int g = 0; g < glyphs_per_row; g++) {
                const uint8_t* glyph = row_ptr + (size_t)g * 132;
                const float scale1 = fp16_to_fp32(read_u16_le(glyph + 0));
                const float scale2 = fp16_to_fp32(read_u16_le(glyph + 2));
                for (int oct = 0; oct < 32; oct++) {
                    uint16_t i1 = read_u16_le(glyph + 4 + oct * 2) & 2047u;
                    uint16_t i2 = read_u16_le(glyph + 68 + oct * 2) & 2047u;
                    const float* cb1 = codebook_s7 + ((size_t)i1 << 3);
                    const float* cb2 = codebook_s7 + ((size_t)i2 << 3);
                    float w[8];
                    for (int i = 0; i < 8; i++) w[i] = scale1 * cb1[i] + scale2 * cb2[i];
                    for (int t = 0; t < kt; t++) {
                        const float* xg = x_ptrs[kb + t] + (size_t)g * 256 + (oct << 3);
                        float dot = 0.0f;
                        for (int i = 0; i < 8; i++) dot += w[i] * xg[i];
                        acc[t] += dot;
                    }
                }
            }
            for (int t = 0; t < kt; t++) y_ptrs[kb + t][row] = acc[t];
        }
#endif
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
#elif HAS_NEON
    const float32x4_t sv = vdupq_n_f32(scale);
    const float32x4_t bv = vdupq_n_f32(bias);
    for (int b = 0; b < 32; b++) {
        const float* signs = g_sign_lut[bitmask[b]];
        float* dst = out_weights + b * 8;
        vst1q_f32(dst,     vfmaq_f32(bv, vld1q_f32(signs),     sv));
        vst1q_f32(dst + 4, vfmaq_f32(bv, vld1q_f32(signs + 4), sv));
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

NEON_API void gemv_opt5_mmap_multi(
    const uint8_t* NEON_RESTRICT mmap_base,
    size_t tensor_byte_offset,
    const float* const* NEON_RESTRICT x_ptrs,
    float* const* NEON_RESTRICT y_ptrs,
    int K,
    int d_out,
    int d_in,
    int bytes_per_glyph
) {
    if (K <= 0 || d_out <= 0 || d_in <= 0) return;
    if (K == 1) {
        gemv_opt5_mmap(mmap_base, tensor_byte_offset, x_ptrs[0], y_ptrs[0], d_out, d_in, bytes_per_glyph);
        return;
    }

    ensure_sign_lut_initialized();
    const int bpg = (bytes_per_glyph == 48) ? 48 : 36;
    const int glyphs_per_row = d_in / 256;
    const uint8_t* tensor_ptr = mmap_base + tensor_byte_offset;
    const size_t total_sums = (size_t)K * (size_t)glyphs_per_row;

    float stack_sums[2048];
    float* xg_sum = (total_sums <= 2048)
        ? stack_sums
        : (float*)neon_alloc(total_sums * sizeof(float));
    if (!xg_sum) return;

    for (int k = 0; k < K; k++) {
        const float* xk = x_ptrs[k];
        float* sk = xg_sum + (size_t)k * (size_t)glyphs_per_row;
        for (int g = 0; g < glyphs_per_row; g++) {
            const float* xs = xk + (size_t)g * 256;
#if HAS_NEON
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (int i = 0; i < 256; i += 16) {
                acc = vaddq_f32(acc, vld1q_f32(xs + i));
                acc = vaddq_f32(acc, vld1q_f32(xs + i + 4));
                acc = vaddq_f32(acc, vld1q_f32(xs + i + 8));
                acc = vaddq_f32(acc, vld1q_f32(xs + i + 12));
            }
            sk[g] = vaddvq_f32(acc);
#elif HAS_AVX2
            __m256 acc = _mm256_setzero_ps();
            for (int i = 0; i < 256; i += 32) {
                acc = _mm256_add_ps(acc, _mm256_loadu_ps(xs + i));
                acc = _mm256_add_ps(acc, _mm256_loadu_ps(xs + i + 8));
                acc = _mm256_add_ps(acc, _mm256_loadu_ps(xs + i + 16));
                acc = _mm256_add_ps(acc, _mm256_loadu_ps(xs + i + 24));
            }
            sk[g] = hsum256_ps(acc);
#else
            float s = 0.0f;
            for (int i = 0; i < 256; i++) s += xs[i];
            sk[g] = s;
#endif
        }
    }

    int row;
#if HAS_OMP
    #pragma omp parallel for schedule(static)
#endif
    for (row = 0; row < d_out; row++) {
        const uint8_t* row_ptr = tensor_ptr + (size_t)row * (size_t)glyphs_per_row * (size_t)bpg;
        int kb = 0;
#if HAS_NEON
        for (; kb + 4 <= K; kb += 4) {
            const float* x0p = x_ptrs[kb + 0];
            const float* x1p = x_ptrs[kb + 1];
            const float* x2p = x_ptrs[kb + 2];
            const float* x3p = x_ptrs[kb + 3];
            const float* s0p = xg_sum + (size_t)(kb + 0) * (size_t)glyphs_per_row;
            const float* s1p = xg_sum + (size_t)(kb + 1) * (size_t)glyphs_per_row;
            const float* s2p = xg_sum + (size_t)(kb + 2) * (size_t)glyphs_per_row;
            const float* s3p = xg_sum + (size_t)(kb + 3) * (size_t)glyphs_per_row;
            float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;

            for (int g = 0; g < glyphs_per_row; g++) {
                const uint8_t* glyph = row_ptr + (size_t)g * (size_t)bpg;
                const float scale = fp16_to_fp32(read_u16_le(glyph + 0));
                const float bias  = fp16_to_fp32(read_u16_le(glyph + 2));
                const uint8_t* bitmask = glyph + 4;
                const size_t g_off = (size_t)g * 256;

                float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
                float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
                for (int b = 0; b < 32; b++) {
                    const float* lut = g_sign_lut[bitmask[b]];
                    const float32x4_t l0 = vld1q_f32(lut);
                    const float32x4_t l4 = vld1q_f32(lut + 4);
                    const size_t col = g_off + (size_t)b * 8;
                    a0 = vfmaq_f32(vfmaq_f32(a0, l0, vld1q_f32(x0p + col)), l4, vld1q_f32(x0p + col + 4));
                    a1 = vfmaq_f32(vfmaq_f32(a1, l0, vld1q_f32(x1p + col)), l4, vld1q_f32(x1p + col + 4));
                    a2 = vfmaq_f32(vfmaq_f32(a2, l0, vld1q_f32(x2p + col)), l4, vld1q_f32(x2p + col + 4));
                    a3 = vfmaq_f32(vfmaq_f32(a3, l0, vld1q_f32(x3p + col)), l4, vld1q_f32(x3p + col + 4));
                }
                float b0 = scale * vaddvq_f32(a0) + bias * s0p[g];
                float b1 = scale * vaddvq_f32(a1) + bias * s1p[g];
                float b2 = scale * vaddvq_f32(a2) + bias * s2p[g];
                float b3 = scale * vaddvq_f32(a3) + bias * s3p[g];
                if (bpg == 48) {
                    const uint8_t* outlier_idx = glyph + 36;
                    const uint8_t* outlier_res = glyph + 40;
                    for (int o = 0; o < 4; o++) {
                        size_t oi = g_off + (size_t)outlier_idx[o];
                        float ores = fp16_to_fp32(read_u16_le(outlier_res + o * 2));
                        b0 += ores * x0p[oi];
                        b1 += ores * x1p[oi];
                        b2 += ores * x2p[oi];
                        b3 += ores * x3p[oi];
                    }
                }
                r0 += b0; r1 += b1; r2 += b2; r3 += b3;
            }
            y_ptrs[kb + 0][row] = r0;
            y_ptrs[kb + 1][row] = r1;
            y_ptrs[kb + 2][row] = r2;
            y_ptrs[kb + 3][row] = r3;
        }
#endif
        for (; kb < K; kb++) {
            const float* xp = x_ptrs[kb];
            const float* sp = xg_sum + (size_t)kb * (size_t)glyphs_per_row;
            float row_acc = 0.0f;
            for (int g = 0; g < glyphs_per_row; g++) {
                const uint8_t* glyph = row_ptr + (size_t)g * (size_t)bpg;
                const float* x_slice = xp + (size_t)g * 256;
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
                    acc = _mm256_fmadd_ps(_mm256_load_ps(g_sign_lut[bitmask[b]]),
                                          _mm256_loadu_ps(x_slice + b * 8), acc);
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
                float base_acc = (scale * sign_dot) + (bias * sp[g]);
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
            y_ptrs[kb][row] = row_acc;
        }
    }

    if (xg_sum != stack_sums) {
        neon_free(xg_sum);
    }
}

/* ============================================================================
 * 3. GlyphAttentionV2 Native Kernels (O(1) KV Encode + Single-Pass Flash GQA)
 * ============================================================================ */

NEON_API void decode_glyphs_opt5_mode_c(
    const uint8_t* NEON_RESTRICT glyphs,
    float* NEON_RESTRICT out_weights,
    int n_glyphs,
    int bytes_per_glyph,
    int apply_fwht8,
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
        float* dst = out_weights + (size_t)g * 256;
        decode_single_opt5_glyph(glyphs + (size_t)g * (size_t)bpg, dst, bpg);
        if (apply_fwht8) {
            for (int oct = 0; oct < 32; oct++) {
                float tmp[8];
                memcpy(tmp, dst + (oct << 3), 8 * sizeof(float));
                fwht8(tmp, dst + (oct << 3));
            }
        }
    }
}

NEON_API int build_ternary_lut_c(
    const float* NEON_RESTRICT cb,
    uint16_t* NEON_RESTRICT lut_6561
) {
    static const int pow3[8] = {1, 3, 9, 27, 81, 243, 729, 2187};
    if (!cb || !lut_6561) return 0;
    for (int s = 0; s < 6561; s++) lut_6561[s] = 0xFFFFu;

    for (int c = 0; c < 2048; c++) {
        const float* row = cb + ((size_t)c << 3);
        float max_abs = 0.0f;
        float norm2 = 0.0f;
        for (int d = 0; d < 8; d++) {
            float v = row[d];
            float av = neon_fabsf(v);
            if (av > max_abs) max_abs = av;
            norm2 += v * v;
        }
        if (max_abs < 0.1f || neon_fabsf(norm2 - 1.0f) > 2e-3f) {
            return 0;
        }
        float inv_max = 1.0f / max_abs;
        int state = 0;
        for (int d = 0; d < 8; d++) {
            float sc = row[d] * inv_max;
            int digit;
            if (neon_fabsf(sc) < 0.03f) {
                digit = 1;
            } else if (neon_fabsf(sc - 1.0f) < 0.03f) {
                digit = 2;
            } else if (neon_fabsf(sc + 1.0f) < 0.03f) {
                digit = 0;
            } else {
                return 0;
            }
            state += digit * pow3[d];
        }
        if (state < 0 || state >= 6561) return 0;
        lut_6561[state] = (uint16_t)c;
    }
    return 1;
}

static inline uint16_t quantize_oct8_fast(
    const float* NEON_RESTRICT x8,
    const float* NEON_RESTRICT cb,
    const uint16_t* NEON_RESTRICT lut_6561,
    float* NEON_RESTRICT out_dot
) {
    if (lut_6561 != NULL) {
        static const int pow3[8] = {1, 3, 9, 27, 81, 243, 729, 2187};
        static const float inv_sqrt_k[9] = {
            0.0f, 1.0f, 0.7071067812f, 0.5773502692f, 0.5f,
            0.4472135955f, 0.4082482905f, 0.3779644730f, 0.3535533906f
        };
        float ax[8];
        int ord[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        for (int d = 0; d < 8; d++) ax[d] = neon_fabsf(x8[d]);

        #define NEON_SWAP8(a, b) do { \
            if (ax[ord[a]] < ax[ord[b]]) { int _t = ord[a]; ord[a] = ord[b]; ord[b] = _t; } \
        } while (0)
        NEON_SWAP8(0, 1); NEON_SWAP8(2, 3); NEON_SWAP8(4, 5); NEON_SWAP8(6, 7);
        NEON_SWAP8(0, 2); NEON_SWAP8(1, 3); NEON_SWAP8(4, 6); NEON_SWAP8(5, 7);
        NEON_SWAP8(1, 2); NEON_SWAP8(5, 6);
        NEON_SWAP8(0, 4); NEON_SWAP8(1, 5); NEON_SWAP8(2, 6); NEON_SWAP8(3, 7);
        NEON_SWAP8(2, 4); NEON_SWAP8(3, 5);
        NEON_SWAP8(1, 2); NEON_SWAP8(3, 4); NEON_SWAP8(5, 6);
        #undef NEON_SWAP8

        int state = 3280; /* (0,0,0,0,0,0,0,0) in base-3 digit=trit+1 */
        float abs_sum = 0.0f;
        float best_score = -1e30f;
        uint16_t best_idx = 0;

        for (int k = 1; k <= 8; k++) {
            int prev_state = state;
            float prev_abs = abs_sum;
            int d = ord[k - 1];
            int delta = (x8[d] >= 0.0f) ? pow3[d] : -pow3[d];
            state = prev_state + delta;
            abs_sum = prev_abs + ax[d];

            uint16_t idx = lut_6561[state];
            if (idx != 0xFFFFu) {
                float sc = abs_sum * inv_sqrt_k[k];
                if (sc > best_score) {
                    best_score = sc;
                    best_idx = idx;
                }
            } else {
                if (k < 8) {
                    int d2 = ord[k];
                    int alt_st = prev_state + ((x8[d2] >= 0.0f) ? pow3[d2] : -pow3[d2]);
                    uint16_t idx_alt = lut_6561[alt_st];
                    if (idx_alt != 0xFFFFu) {
                        float sc = (prev_abs + ax[d2]) * inv_sqrt_k[k];
                        if (sc > best_score) {
                            best_score = sc;
                            best_idx = idx_alt;
                        }
                    }
                }
                int flip_st = prev_state - delta;
                uint16_t idx_fl = lut_6561[flip_st];
                if (idx_fl != 0xFFFFu) {
                    float sc = (prev_abs - ax[d]) * inv_sqrt_k[k];
                    if (sc > best_score) {
                        best_score = sc;
                        best_idx = idx_fl;
                    }
                }
            }
        }
        *out_dot = best_score;
        return best_idx;
    }

    /* Exact SIMD 2,048-centroid search fallback (for golden_s7 or arbitrary S^7 codebooks) */
    float best_score = -1e30f;
    uint16_t best_idx = 0;
#if HAS_AVX2
    __m256 xv = _mm256_loadu_ps(x8);
    for (int c = 0; c < 2048; c++) {
        float sc = hsum256_ps(_mm256_mul_ps(xv, _mm256_loadu_ps(cb + ((size_t)c << 3))));
        if (sc > best_score) {
            best_score = sc;
            best_idx = (uint16_t)c;
        }
    }
#elif HAS_NEON
    float32x4_t x0 = vld1q_f32(x8);
    float32x4_t x4 = vld1q_f32(x8 + 4);
    for (int c = 0; c < 2048; c++) {
        const float* row = cb + ((size_t)c << 3);
        float sc = hsum4_pair(vmulq_f32(x0, vld1q_f32(row)), vmulq_f32(x4, vld1q_f32(row + 4)));
        if (sc > best_score) {
            best_score = sc;
            best_idx = (uint16_t)c;
        }
    }
#else
    for (int c = 0; c < 2048; c++) {
        const float* row = cb + ((size_t)c << 3);
        float sc = 0.0f;
        for (int d = 0; d < 8; d++) sc += x8[d] * row[d];
        if (sc > best_score) {
            best_score = sc;
            best_idx = (uint16_t)c;
        }
    }
#endif
    *out_dot = best_score;
    return best_idx;
}

static inline void encode_single_opt4_glyph_fast(
    const float* NEON_RESTRICT in_256,
    const float* NEON_RESTRICT cb,
    const uint16_t* NEON_RESTRICT lut_6561,
    uint8_t* NEON_RESTRICT glyph_out,
    int apply_fwht8
) {
    float target[256];
    if (apply_fwht8) {
        for (int oct = 0; oct < 32; oct++) {
            fwht8(in_256 + (oct << 3), target + (oct << 3));
        }
    } else {
        memcpy(target, in_256, 256 * sizeof(float));
    }

    float norm2_target = 0.0f;
    for (int i = 0; i < 256; i++) norm2_target += target[i] * target[i];

    uint16_t idx1[32];
    uint16_t idx2[32];
    float b1 = 0.0f;
    for (int oct = 0; oct < 32; oct++) {
        float dot1 = 0.0f;
        idx1[oct] = quantize_oct8_fast(target + (oct << 3), cb, lut_6561, &dot1);
        b1 += dot1;
    }

    float s1_init = b1 * (1.0f / 32.0f);
    float sum_dot2 = 0.0f;
    float g12 = 0.0f;
    for (int oct = 0; oct < 32; oct++) {
        const float* t8 = target + (oct << 3);
        const float* c1 = cb + ((size_t)idx1[oct] << 3);
        float res[8];
        for (int d = 0; d < 8; d++) res[d] = t8[d] - s1_init * c1[d];
        float dot2 = 0.0f;
        idx2[oct] = quantize_oct8_fast(res, cb, lut_6561, &dot2);
        sum_dot2 += dot2;
        const float* c2 = cb + ((size_t)idx2[oct] << 3);
        float c12 = 0.0f;
        for (int d = 0; d < 8; d++) c12 += c1[d] * c2[d];
        g12 += c12;
    }

    /* Joint 2x2 Gram least-squares projection + L2 sphere norm preservation */
    float b2 = sum_dot2 + s1_init * g12;
    float det = 1024.0f - g12 * g12;
    float s1_opt, s2_opt;
    if (det > 1e-2f) {
        float inv_det = 1.0f / det;
        s1_opt = (32.0f * b1 - g12 * b2) * inv_det;
        s2_opt = (32.0f * b2 - g12 * b1) * inv_det;
    } else {
        s1_opt = s1_init;
        s2_opt = 0.0f;
    }
    float norm2_rec = 32.0f * (s1_opt * s1_opt + s2_opt * s2_opt) + 2.0f * g12 * s1_opt * s2_opt;
    if (norm2_rec > 1e-20f && norm2_target > 1e-20f) {
        float gamma = neon_sqrtf(norm2_target / norm2_rec);
        s1_opt *= gamma;
        s2_opt *= gamma;
    }

    write_u16_le(glyph_out + 0, fp32_to_fp16(s1_opt));
    write_u16_le(glyph_out + 2, fp32_to_fp16(s2_opt));
    for (int oct = 0; oct < 32; oct++) {
        write_u16_le(glyph_out + 4 + oct * 2, idx1[oct]);
        write_u16_le(glyph_out + 68 + oct * 2, idx2[oct]);
    }
}

static inline void encode_single_opt5_glyph_fast(
    const float* NEON_RESTRICT in_256,
    uint8_t* NEON_RESTRICT glyph_out,
    int bpg,
    int apply_fwht8
) {
    float target[256];
    if (apply_fwht8) {
        for (int oct = 0; oct < 32; oct++) {
            fwht8(in_256 + (oct << 3), target + (oct << 3));
        }
    } else {
        memcpy(target, in_256, 256 * sizeof(float));
    }

    float sum = 0.0f;
    for (int i = 0; i < 256; i++) sum += target[i];
    uint16_t bias_h = fp32_to_fp16(sum * (1.0f / 256.0f));
    float bias = fp16_to_fp32(bias_h);

    float abs_sum = 0.0f;
    float var_sum = 0.0f;
    float top_abs[4] = {-1.0f, -1.0f, -1.0f, -1.0f};
    float top_diff[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint8_t top_idx[4] = {0, 0, 0, 0};

    for (int b = 0; b < 32; b++) {
        uint8_t mask = 0;
        for (int bit = 0; bit < 8; bit++) {
            int idx = (b << 3) + bit;
            float v = target[idx];
            int pos = (v >= bias) ? 1 : 0;
            mask = (uint8_t)((mask << 1) | pos);
            float diff = v - bias;
            float adiff = neon_fabsf(diff);
            abs_sum += adiff;
            var_sum += diff * diff;
            if (bpg == 48 && adiff > top_abs[3]) {
                int ins = 3;
                while (ins > 0 && adiff > top_abs[ins - 1]) {
                    top_abs[ins] = top_abs[ins - 1];
                    top_diff[ins] = top_diff[ins - 1];
                    top_idx[ins] = top_idx[ins - 1];
                    ins--;
                }
                top_abs[ins] = adiff;
                top_diff[ins] = diff;
                top_idx[ins] = (uint8_t)idx;
            }
        }
        glyph_out[4 + b] = mask;
    }

    write_u16_le(glyph_out + 2, bias_h);

    if (bpg == 48) {
        float top_abs_sum = top_abs[0] + top_abs[1] + top_abs[2] + top_abs[3];
        float top_var_sum = top_diff[0] * top_diff[0] + top_diff[1] * top_diff[1]
                          + top_diff[2] * top_diff[2] + top_diff[3] * top_diff[3];
        float abs_rem = (abs_sum > top_abs_sum) ? (abs_sum - top_abs_sum) : 0.0f;
        float s_mse = abs_rem * (1.0f / 252.0f);
        float var_rec = 252.0f * s_mse * s_mse + top_var_sum;
        float gamma = (var_rec > 1e-20f && var_sum > 1e-20f) ? neon_sqrtf(var_sum / var_rec) : 1.0f;

        uint16_t scale_h = fp32_to_fp16(s_mse * gamma);
        float scale_actual = fp16_to_fp32(scale_h);
        write_u16_le(glyph_out + 0, scale_h);

        for (int o = 0; o < 4; o++) {
            float sgn = (target[top_idx[o]] >= bias) ? 1.0f : -1.0f;
            float res_o = gamma * top_diff[o] - sgn * scale_actual;
            glyph_out[36 + o] = top_idx[o];
            write_u16_le(glyph_out + 40 + o * 2, fp32_to_fp16(res_o));
        }
    } else {
        uint16_t scale_h = fp32_to_fp16(neon_sqrtf(var_sum * (1.0f / 256.0f)));
        write_u16_le(glyph_out + 0, scale_h);
    }
}

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
) {
    if (n_glyphs <= 0) return;
    const int bpg = (codec_id == 4) ? 132 : ((bytes_per_glyph == 48) ? 48 : 36);
    int g;
#if HAS_OMP
    if (n_glyphs >= 16 && num_threads != 1) {
        if (num_threads > 0) omp_set_num_threads(num_threads);
        #pragma omp parallel for schedule(static)
        for (g = 0; g < n_glyphs; g++) {
            if (codec_id == 4) {
                encode_single_opt4_glyph_fast(
                    kv_in + (size_t)g * 256, cb, lut_6561, glyphs_out + (size_t)g * 132, apply_fwht8
                );
            } else {
                encode_single_opt5_glyph_fast(
                    kv_in + (size_t)g * 256, glyphs_out + (size_t)g * (size_t)bpg, bpg, apply_fwht8
                );
            }
        }
        return;
    }
#else
    (void)num_threads;
#endif
    for (g = 0; g < n_glyphs; g++) {
        if (codec_id == 4) {
            encode_single_opt4_glyph_fast(
                kv_in + (size_t)g * 256, cb, lut_6561, glyphs_out + (size_t)g * 132, apply_fwht8
            );
        } else {
            encode_single_opt5_glyph_fast(
                kv_in + (size_t)g * 256, glyphs_out + (size_t)g * (size_t)bpg, bpg, apply_fwht8
            );
        }
    }
}

static inline void simd_rescale_and_fma(
    float* NEON_RESTRICT acc,
    const float* NEON_RESTRICT v_vec,
    float alpha,
    float beta,
    int dim
) {
    int d = 0;
#if HAS_AVX2
    __m256 va = _mm256_set1_ps(alpha);
    __m256 vb = _mm256_set1_ps(beta);
    for (; d + 8 <= dim; d += 8) {
        __m256 a = _mm256_mul_ps(_mm256_loadu_ps(acc + d), va);
        _mm256_storeu_ps(acc + d, _mm256_fmadd_ps(_mm256_loadu_ps(v_vec + d), vb, a));
    }
#elif HAS_NEON
    float32x4_t va = vdupq_n_f32(alpha);
    float32x4_t vb = vdupq_n_f32(beta);
    for (; d + 8 <= dim; d += 8) {
        vst1q_f32(acc + d,     vfmaq_f32(vmulq_f32(vld1q_f32(acc + d),     va), vld1q_f32(v_vec + d),     vb));
        vst1q_f32(acc + d + 4, vfmaq_f32(vmulq_f32(vld1q_f32(acc + d + 4), va), vld1q_f32(v_vec + d + 4), vb));
    }
#endif
    for (; d < dim; d++) {
        acc[d] = acc[d] * alpha + beta * v_vec[d];
    }
}

static inline void simd_fma_only(
    float* NEON_RESTRICT acc,
    const float* NEON_RESTRICT v_vec,
    float beta,
    int dim
) {
    int d = 0;
#if HAS_AVX2
    __m256 vb = _mm256_set1_ps(beta);
    for (; d + 8 <= dim; d += 8) {
        _mm256_storeu_ps(acc + d, _mm256_fmadd_ps(_mm256_loadu_ps(v_vec + d), vb, _mm256_loadu_ps(acc + d)));
    }
#elif HAS_NEON
    float32x4_t vb = vdupq_n_f32(beta);
    for (; d + 8 <= dim; d += 8) {
        vst1q_f32(acc + d,     vfmaq_f32(vld1q_f32(acc + d),     vld1q_f32(v_vec + d),     vb));
        vst1q_f32(acc + d + 4, vfmaq_f32(vld1q_f32(acc + d + 4), vld1q_f32(v_vec + d + 4), vb));
    }
#endif
    for (; d < dim; d++) {
        acc[d] += beta * v_vec[d];
    }
}

static inline void simd_scale_inplace(float* NEON_RESTRICT acc, float alpha, int dim) {
    int d = 0;
#if HAS_AVX2
    __m256 va = _mm256_set1_ps(alpha);
    for (; d + 8 <= dim; d += 8) {
        _mm256_storeu_ps(acc + d, _mm256_mul_ps(_mm256_loadu_ps(acc + d), va));
    }
#elif HAS_NEON
    float32x4_t va = vdupq_n_f32(alpha);
    for (; d + 8 <= dim; d += 8) {
        vst1q_f32(acc + d,     vmulq_f32(vld1q_f32(acc + d),     va));
        vst1q_f32(acc + d + 4, vmulq_f32(vld1q_f32(acc + d + 4), va));
    }
#endif
    for (; d < dim; d++) acc[d] *= alpha;
}

static inline void simd_fma_8heads(
    float* const acc_ptrs[8],
    const float* NEON_RESTRICT v_vec,
    const float w[8],
    int dim
) {
    float* NEON_RESTRICT a0 = acc_ptrs[0];
    float* NEON_RESTRICT a1 = acc_ptrs[1];
    float* NEON_RESTRICT a2 = acc_ptrs[2];
    float* NEON_RESTRICT a3 = acc_ptrs[3];
    float* NEON_RESTRICT a4 = acc_ptrs[4];
    float* NEON_RESTRICT a5 = acc_ptrs[5];
    float* NEON_RESTRICT a6 = acc_ptrs[6];
    float* NEON_RESTRICT a7 = acc_ptrs[7];
    int d = 0;
#if HAS_AVX2
    const __m256 w0 = _mm256_set1_ps(w[0]), w1 = _mm256_set1_ps(w[1]);
    const __m256 w2 = _mm256_set1_ps(w[2]), w3 = _mm256_set1_ps(w[3]);
    const __m256 w4 = _mm256_set1_ps(w[4]), w5 = _mm256_set1_ps(w[5]);
    const __m256 w6 = _mm256_set1_ps(w[6]), w7 = _mm256_set1_ps(w[7]);
    for (; d + 8 <= dim; d += 8) {
        __m256 vv = _mm256_loadu_ps(v_vec + d);
        _mm256_storeu_ps(a0 + d, _mm256_fmadd_ps(vv, w0, _mm256_loadu_ps(a0 + d)));
        _mm256_storeu_ps(a1 + d, _mm256_fmadd_ps(vv, w1, _mm256_loadu_ps(a1 + d)));
        _mm256_storeu_ps(a2 + d, _mm256_fmadd_ps(vv, w2, _mm256_loadu_ps(a2 + d)));
        _mm256_storeu_ps(a3 + d, _mm256_fmadd_ps(vv, w3, _mm256_loadu_ps(a3 + d)));
        _mm256_storeu_ps(a4 + d, _mm256_fmadd_ps(vv, w4, _mm256_loadu_ps(a4 + d)));
        _mm256_storeu_ps(a5 + d, _mm256_fmadd_ps(vv, w5, _mm256_loadu_ps(a5 + d)));
        _mm256_storeu_ps(a6 + d, _mm256_fmadd_ps(vv, w6, _mm256_loadu_ps(a6 + d)));
        _mm256_storeu_ps(a7 + d, _mm256_fmadd_ps(vv, w7, _mm256_loadu_ps(a7 + d)));
    }
#elif HAS_NEON
    const float32x4_t w0 = vdupq_n_f32(w[0]), w1 = vdupq_n_f32(w[1]);
    const float32x4_t w2 = vdupq_n_f32(w[2]), w3 = vdupq_n_f32(w[3]);
    const float32x4_t w4 = vdupq_n_f32(w[4]), w5 = vdupq_n_f32(w[5]);
    const float32x4_t w6 = vdupq_n_f32(w[6]), w7 = vdupq_n_f32(w[7]);
    for (; d + 8 <= dim; d += 8) {
        float32x4_t v0 = vld1q_f32(v_vec + d);
        float32x4_t v4 = vld1q_f32(v_vec + d + 4);
        vst1q_f32(a0 + d,     vfmaq_f32(vld1q_f32(a0 + d),     v0, w0));
        vst1q_f32(a1 + d,     vfmaq_f32(vld1q_f32(a1 + d),     v0, w1));
        vst1q_f32(a2 + d,     vfmaq_f32(vld1q_f32(a2 + d),     v0, w2));
        vst1q_f32(a3 + d,     vfmaq_f32(vld1q_f32(a3 + d),     v0, w3));
        vst1q_f32(a4 + d,     vfmaq_f32(vld1q_f32(a4 + d),     v0, w4));
        vst1q_f32(a5 + d,     vfmaq_f32(vld1q_f32(a5 + d),     v0, w5));
        vst1q_f32(a6 + d,     vfmaq_f32(vld1q_f32(a6 + d),     v0, w6));
        vst1q_f32(a7 + d,     vfmaq_f32(vld1q_f32(a7 + d),     v0, w7));
        vst1q_f32(a0 + d + 4, vfmaq_f32(vld1q_f32(a0 + d + 4), v4, w0));
        vst1q_f32(a1 + d + 4, vfmaq_f32(vld1q_f32(a1 + d + 4), v4, w1));
        vst1q_f32(a2 + d + 4, vfmaq_f32(vld1q_f32(a2 + d + 4), v4, w2));
        vst1q_f32(a3 + d + 4, vfmaq_f32(vld1q_f32(a3 + d + 4), v4, w3));
        vst1q_f32(a4 + d + 4, vfmaq_f32(vld1q_f32(a4 + d + 4), v4, w4));
        vst1q_f32(a5 + d + 4, vfmaq_f32(vld1q_f32(a5 + d + 4), v4, w5));
        vst1q_f32(a6 + d + 4, vfmaq_f32(vld1q_f32(a6 + d + 4), v4, w6));
        vst1q_f32(a7 + d + 4, vfmaq_f32(vld1q_f32(a7 + d + 4), v4, w7));
    }
#endif
    for (; d < dim; d++) {
        float vv = v_vec[d];
        a0[d] += w[0] * vv; a1[d] += w[1] * vv; a2[d] += w[2] * vv; a3[d] += w[3] * vv;
        a4[d] += w[4] * vv; a5[d] += w[5] * vv; a6[d] += w[6] * vv; a7[d] += w[7] * vv;
    }
}

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
) {
    if (n_q_heads <= 0 || n_kv_heads <= 0 || (n_q_heads % n_kv_heads) != 0) return -1;
    if (head_dim <= 0 || (head_dim % 256) != 0 || seq_len < 0) return -2;
    if (codec_id == 4 && !codebook_s7) return -3;
    if (!k_eq_v && seq_len > 0 && !v_glyphs) return -4;

    if (codec_id != 4) {
        ensure_sign_lut_initialized();
    }

    if (seq_len == 0) {
        memset(out_context, 0, (size_t)n_q_heads * (size_t)head_dim * sizeof(float));
        if (out_max_score) {
            for (int qh = 0; qh < n_q_heads; qh++) out_max_score[qh] = -1e30f;
        }
        if (out_sum_exp) {
            for (int qh = 0; qh < n_q_heads; qh++) out_sum_exp[qh] = 0.0f;
        }
        return 0;
    }

    const int gqa_group = n_q_heads / n_kv_heads;
    const int glyphs_per_head = head_dim / 256;
    const int bpg = (codec_id == 4) ? 132 : ((bytes_per_glyph == 48) ? 48 : 36);
    const int use_opt5_nibble_engine = (codec_id != 4 && (gqa_group == 8 || gqa_group == 2)) ? 1 : 0;

    /* Cap Split-K tiles per KV head at <= 32 so tile scratch stays <= 2 MB in L3 cache */
    int tile_size = 512;
    if (seq_len < 512) {
        tile_size = seq_len;
    } else if ((seq_len + 511) / 512 > 32) {
        tile_size = (seq_len + 31) / 32;
    }
    const int n_tiles = (seq_len + tile_size - 1) / tile_size;
    const int total_jobs = n_kv_heads * n_tiles;

    const size_t gl_lut_stride = 64u * 16u * (size_t)gqa_group + 16u; /* +64B skew prevents L1 set aliasing */
    const size_t q_elems    = (size_t)n_q_heads * (size_t)head_dim;
    const size_t qsum_elems = (size_t)n_q_heads * (size_t)glyphs_per_head;
    const size_t meta_elems = (size_t)n_q_heads * (size_t)n_tiles;
    const size_t acc_elems  = meta_elems * (size_t)head_dim;
    const size_t qlut_elems = use_opt5_nibble_engine
        ? ((size_t)n_kv_heads * (size_t)glyphs_per_head * gl_lut_stride)
        : 0u;
    const size_t total_floats = q_elems + qsum_elems + 2 * meta_elems + acc_elems + qlut_elems;

    float* slab = (float*)neon_alloc(total_floats * sizeof(float));
    if (!slab) return -5;

    float* q_scaled = slab;
    float* q_gsum   = q_scaled + q_elems;
    float* tile_m   = q_gsum + qsum_elems;
    float* tile_l   = tile_m + meta_elems;
    float* tile_acc = tile_l + meta_elems;
    float* qlut_all = tile_acc + acc_elems;

    /* 1. Pre-rotate (optional FWHT-8) and pre-scale queries once in O(n_q_heads * head_dim) */
    for (int qh = 0; qh < n_q_heads; qh++) {
        const float* q_src = q + (size_t)qh * (size_t)head_dim;
        float* q_dst = q_scaled + (size_t)qh * (size_t)head_dim;
        if (apply_fwht8) {
            fwht8_vec_c(q_src, q_dst, head_dim);
            for (int d = 0; d < head_dim; d++) q_dst[d] *= attn_scale;
        } else {
            for (int d = 0; d < head_dim; d++) q_dst[d] = q_src[d] * attn_scale;
        }
        if (codec_id != 4) {
            for (int g = 0; g < glyphs_per_head; g++) {
                float s = 0.0f;
                const float* sl = q_dst + (size_t)g * 256;
                for (int i = 0; i < 256; i++) s += sl[i];
                q_gsum[(size_t)qh * (size_t)glyphs_per_head + (size_t)g] = s;
            }
        }
    }

    /* 1b. Precompute L1-resident 4-Bit Nibble GQA-Transposed Query LUT (32 KB/glyph for 8:1, 8 KB/glyph for 2:1) */
    if (use_opt5_nibble_engine) {
        for (int kvh = 0; kvh < n_kv_heads; kvh++) {
            for (int gl = 0; gl < glyphs_per_head; gl++) {
                float* lut_gl = qlut_all + ((size_t)kvh * (size_t)glyphs_per_head + (size_t)gl) * gl_lut_stride;
                for (int nib = 0; nib < 64; nib++) {
                    int d_off = gl * 256 + nib * 4;
                    for (int pat = 0; pat < 16; pat++) {
                        float s0 = ((pat >> 3) & 1) ? 1.0f : -1.0f;
                        float s1 = ((pat >> 2) & 1) ? 1.0f : -1.0f;
                        float s2 = ((pat >> 1) & 1) ? 1.0f : -1.0f;
                        float s3 = (pat & 1)        ? 1.0f : -1.0f;
                        float* dst = lut_gl + ((size_t)nib * 16u + (size_t)pat) * (size_t)gqa_group;
                        for (int g = 0; g < gqa_group; g++) {
                            const float* qg = q_scaled + (size_t)(kvh * gqa_group + g) * (size_t)head_dim + d_off;
                            dst[g] = s0 * qg[0] + s1 * qg[1] + s2 * qg[2] + s3 * qg[3];
                        }
                    }
                }
            }
        }
    }

    int job;
#if HAS_OMP
    if (total_jobs > 1 && num_threads != 1) {
        if (num_threads > 0) omp_set_num_threads(num_threads);
    }
    #pragma omp parallel for schedule(dynamic, 1) if(total_jobs > 1 && num_threads != 1)
#else
    (void)num_threads;
#endif
    for (job = 0; job < total_jobs; job++) {
        const int kvh = job / n_tiles;
        const int tidx = job % n_tiles;
        const int t_start = tidx * tile_size;
        const int t_end = (t_start + tile_size < seq_len) ? (t_start + tile_size) : seq_len;
        const int qh_base = kvh * gqa_group;

        float stack_k[512];
        float stack_v[512];
        float stack_scores[16];
        float stack_m[16];
        float stack_l[16];
        float stack_local_acc[8 * (512 + 16)]; /* L1-private coprime-stride (33 cache lines) accumulator */
        float* stack_acc_ptrs[16];

        const size_t acc_stride = (size_t)head_dim + 16u;
        float* k_rot = (head_dim <= 512) ? stack_k : (float*)neon_alloc((size_t)head_dim * sizeof(float));
        float* v_rot = (head_dim <= 512) ? stack_v : (float*)neon_alloc((size_t)head_dim * sizeof(float));
        float* scores = (gqa_group <= 16) ? stack_scores : (float*)neon_alloc((size_t)gqa_group * sizeof(float));
        float* loc_m  = (gqa_group <= 16) ? stack_m : (float*)neon_alloc((size_t)gqa_group * sizeof(float));
        float* loc_l  = (gqa_group <= 16) ? stack_l : (float*)neon_alloc((size_t)gqa_group * sizeof(float));
        float* local_acc_buf = (gqa_group <= 8 && head_dim <= 512)
            ? stack_local_acc
            : (float*)neon_alloc((size_t)gqa_group * acc_stride * sizeof(float));
        float** acc_ptrs = (gqa_group <= 16)
            ? stack_acc_ptrs
            : (float**)neon_alloc((size_t)gqa_group * sizeof(float*));

        for (int g = 0; g < gqa_group; g++) {
            loc_m[g] = -1e30f;
            loc_l[g] = 0.0f;
            float* acc_g = local_acc_buf + (size_t)g * acc_stride;
            memset(acc_g, 0, (size_t)head_dim * sizeof(float));
            acc_ptrs[g] = acc_g;
        }

        const uint8_t* k_head_base = k_glyphs + (size_t)kvh * kv_head_stride_bytes;
        const uint8_t* v_head_base = (!k_eq_v && v_glyphs) ? (v_glyphs + (size_t)kvh * kv_head_stride_bytes) : NULL;
        int handled_by_nibble = 0;

        /* ====================================================================
         * FAST PATH: Algebraic Nibble-LUT QK^T + Nibble-Histogram PV Engine
         * Never materializes FP32 K or V vectors in memory across the sequence!
         * ==================================================================== */
        if (use_opt5_nibble_engine) {
            const size_t hist_floats = (size_t)glyphs_per_head * gl_lut_stride;
            const size_t bias_floats = (size_t)glyphs_per_head * (size_t)gqa_group;
            float* nib_hist = (float*)neon_alloc((hist_floats + 512u * (size_t)gqa_group + bias_floats) * sizeof(float));
            if (nib_hist) {
                float* scores_blk = nib_hist + hist_floats;
                float* bias_acc   = scores_blk + 512u * (size_t)gqa_group;

                for (int blk_start = t_start; blk_start < t_end; blk_start += 512) {
                    const int blk_end = (blk_start + 512 < t_end) ? (blk_start + 512) : t_end;
                    const int blk_len = blk_end - blk_start;
                    float blk_max[8] = {-1e30f, -1e30f, -1e30f, -1e30f, -1e30f, -1e30f, -1e30f, -1e30f};

                    /* Step A: Score 512 tokens directly via L1 Nibble LUT */
                    for (int u = 0; u < blk_len; u++) {
                        const uint8_t* k_tok = k_head_base + (size_t)(blk_start + u) * kv_seq_stride_bytes;
                        float* sc_u = scores_blk + (size_t)u * (size_t)gqa_group;
                        for (int g = 0; g < gqa_group; g++) sc_u[g] = 0.0f;

                        for (int gl = 0; gl < glyphs_per_head; gl++) {
                            const uint8_t* glyph = k_tok + (size_t)gl * (size_t)bpg;
                            const float scale = fp16_to_fp32(read_u16_le(glyph + 0));
                            const float bias  = fp16_to_fp32(read_u16_le(glyph + 2));
                            const uint8_t* bitmask = glyph + 4;
                            const float* lut_gl = qlut_all + ((size_t)kvh * (size_t)glyphs_per_head + (size_t)gl) * gl_lut_stride;

                            if (gqa_group == 8) {
                                float sdot[8];
#if HAS_AVX2
                                __m256 a07 = _mm256_setzero_ps();
                                for (int b = 0; b < 32; b++) {
                                    uint8_t m = bitmask[b];
                                    const float* p_hi = lut_gl + ((size_t)(b * 2) * 16u + (size_t)(m >> 4)) * 8u;
                                    const float* p_lo = lut_gl + ((size_t)(b * 2 + 1) * 16u + (size_t)(m & 15u)) * 8u;
                                    a07 = _mm256_add_ps(a07, _mm256_add_ps(_mm256_loadu_ps(p_hi), _mm256_loadu_ps(p_lo)));
                                }
                                _mm256_storeu_ps(sdot, a07);
#elif HAS_NEON
                                float32x4_t a03 = vdupq_n_f32(0.0f), a47 = vdupq_n_f32(0.0f);
                                for (int b = 0; b < 32; b++) {
                                    uint8_t m = bitmask[b];
                                    const float* p_hi = lut_gl + ((size_t)(b * 2) * 16u + (size_t)(m >> 4)) * 8u;
                                    const float* p_lo = lut_gl + ((size_t)(b * 2 + 1) * 16u + (size_t)(m & 15u)) * 8u;
                                    a03 = vaddq_f32(a03, vaddq_f32(vld1q_f32(p_hi),     vld1q_f32(p_lo)));
                                    a47 = vaddq_f32(a47, vaddq_f32(vld1q_f32(p_hi + 4), vld1q_f32(p_lo + 4)));
                                }
                                vst1q_f32(sdot,     a03);
                                vst1q_f32(sdot + 4, a47);
#else
                                for (int g = 0; g < 8; g++) sdot[g] = 0.0f;
                                for (int b = 0; b < 32; b++) {
                                    uint8_t m = bitmask[b];
                                    const float* p_hi = lut_gl + ((size_t)(b * 2) * 16u + (size_t)(m >> 4)) * 8u;
                                    const float* p_lo = lut_gl + ((size_t)(b * 2 + 1) * 16u + (size_t)(m & 15u)) * 8u;
                                    for (int g = 0; g < 8; g++) sdot[g] += p_hi[g] + p_lo[g];
                                }
#endif
                                for (int g = 0; g < 8; g++) {
                                    sc_u[g] += scale * sdot[g] + bias * q_gsum[(size_t)(qh_base + g) * glyphs_per_head + gl];
                                }
                                if (bpg == 48) {
                                    for (int o = 0; o < 4; o++) {
                                        uint8_t oi = glyph[36 + o];
                                        float ores = fp16_to_fp32(read_u16_le(glyph + 40 + o * 2));
                                        int q_off = gl * 256 + (int)oi;
                                        for (int g = 0; g < 8; g++) {
                                            sc_u[g] += ores * q_scaled[(size_t)(qh_base + g) * head_dim + q_off];
                                        }
                                    }
                                }
                            } else { /* gqa_group == 2 */
                                float s0 = 0.0f, s1 = 0.0f;
#if HAS_NEON
                                float32x2_t a01 = vdup_n_f32(0.0f);
                                for (int b = 0; b < 32; b++) {
                                    uint8_t m = bitmask[b];
                                    const float* p_hi = lut_gl + ((size_t)(b * 2) * 16u + (size_t)(m >> 4)) * 2u;
                                    const float* p_lo = lut_gl + ((size_t)(b * 2 + 1) * 16u + (size_t)(m & 15u)) * 2u;
                                    a01 = vadd_f32(a01, vadd_f32(vld1_f32(p_hi), vld1_f32(p_lo)));
                                }
                                s0 = vget_lane_f32(a01, 0);
                                s1 = vget_lane_f32(a01, 1);
#else
                                for (int b = 0; b < 32; b++) {
                                    uint8_t m = bitmask[b];
                                    const float* p_hi = lut_gl + ((size_t)(b * 2) * 16u + (size_t)(m >> 4)) * 2u;
                                    const float* p_lo = lut_gl + ((size_t)(b * 2 + 1) * 16u + (size_t)(m & 15u)) * 2u;
                                    s0 += p_hi[0] + p_lo[0];
                                    s1 += p_hi[1] + p_lo[1];
                                }
#endif
                                sc_u[0] += scale * s0 + bias * q_gsum[(size_t)(qh_base + 0) * glyphs_per_head + gl];
                                sc_u[1] += scale * s1 + bias * q_gsum[(size_t)(qh_base + 1) * glyphs_per_head + gl];
                                if (bpg == 48) {
                                    const float* q0 = q_scaled + (size_t)(qh_base + 0) * head_dim + (size_t)gl * 256;
                                    const float* q1 = q_scaled + (size_t)(qh_base + 1) * head_dim + (size_t)gl * 256;
                                    for (int o = 0; o < 4; o++) {
                                        uint8_t oi = glyph[36 + o];
                                        float ores = fp16_to_fp32(read_u16_le(glyph + 40 + o * 2));
                                        sc_u[0] += ores * q0[oi];
                                        sc_u[1] += ores * q1[oi];
                                    }
                                }
                            }
                        }
                        for (int g = 0; g < gqa_group; g++) {
                            if (sc_u[g] > blk_max[g]) blk_max[g] = sc_u[g];
                        }
                    }

                    /* Step B: Block-level early-out & single accumulator rescale per 512 tokens */
                    int blk_active = 0;
                    for (int g = 0; g < gqa_group; g++) {
                        if (blk_max[g] - loc_m[g] > -17.5f) blk_active = 1;
                        if (blk_max[g] > loc_m[g]) {
                            float alpha = (loc_m[g] <= -1e29f) ? 0.0f : neon_expf(loc_m[g] - blk_max[g]);
                            loc_m[g] = blk_max[g];
                            loc_l[g] *= alpha;
                            if (alpha == 0.0f) {
                                memset(acc_ptrs[g], 0, (size_t)head_dim * sizeof(float));
                            } else {
                                simd_scale_inplace(acc_ptrs[g], alpha, head_dim);
                            }
                        }
                    }
                    if (!blk_active) continue;

                    /* Step C: Accumulate V weights directly into 4-bit Nibble Histogram */
                    memset(nib_hist, 0, hist_floats * sizeof(float));
                    memset(bias_acc, 0, (size_t)glyphs_per_head * (size_t)gqa_group * sizeof(float));

                    for (int u = 0; u < blk_len; u++) {
                        const float* sc_u = scores_blk + (size_t)u * (size_t)gqa_group;
                        float w[8];
                        int tok_active = 0;
                        for (int g = 0; g < gqa_group; g++) {
                            float diff = sc_u[g] - loc_m[g];
                            if (diff > -17.5f) {
                                float wg = neon_expf(diff);
                                w[g] = wg;
                                loc_l[g] += wg;
                                tok_active = 1;
                            } else {
                                w[g] = 0.0f;
                            }
                        }
                        if (!tok_active) continue;

                        const size_t t_idx = (size_t)(blk_start + u);
                        const uint8_t* v_tok = k_eq_v
                            ? (k_head_base + t_idx * kv_seq_stride_bytes)
                            : (v_head_base + t_idx * kv_seq_stride_bytes);

                        for (int gl = 0; gl < glyphs_per_head; gl++) {
                            const uint8_t* v_gl = v_tok + (size_t)gl * (size_t)bpg;
                            const float v_scale = fp16_to_fp32(read_u16_le(v_gl + 0));
                            const float v_bias  = fp16_to_fp32(read_u16_le(v_gl + 2));
                            const uint8_t* v_mask = v_gl + 4;
                            float* hist_gl = nib_hist + (size_t)gl * gl_lut_stride;

                            if (gqa_group == 8) {
                                for (int g = 0; g < 8; g++) bias_acc[gl * 8 + g] += w[g] * v_bias;
#if HAS_AVX2
                                __m256 ws = _mm256_mul_ps(_mm256_loadu_ps(w), _mm256_set1_ps(v_scale));
                                for (int b = 0; b < 32; b++) {
                                    uint8_t m = v_mask[b];
                                    float* h_hi = hist_gl + ((size_t)(b * 2) * 16u + (size_t)(m >> 4)) * 8u;
                                    float* h_lo = hist_gl + ((size_t)(b * 2 + 1) * 16u + (size_t)(m & 15u)) * 8u;
                                    _mm256_storeu_ps(h_hi, _mm256_add_ps(_mm256_loadu_ps(h_hi), ws));
                                    _mm256_storeu_ps(h_lo, _mm256_add_ps(_mm256_loadu_ps(h_lo), ws));
                                }
#elif HAS_NEON
                                float32x4_t vs = vdupq_n_f32(v_scale);
                                float32x4_t ws03 = vmulq_f32(vld1q_f32(w),     vs);
                                float32x4_t ws47 = vmulq_f32(vld1q_f32(w + 4), vs);
                                for (int b = 0; b < 32; b++) {
                                    uint8_t m = v_mask[b];
                                    float* h_hi = hist_gl + ((size_t)(b * 2) * 16u + (size_t)(m >> 4)) * 8u;
                                    float* h_lo = hist_gl + ((size_t)(b * 2 + 1) * 16u + (size_t)(m & 15u)) * 8u;
                                    vst1q_f32(h_hi,     vaddq_f32(vld1q_f32(h_hi),     ws03));
                                    vst1q_f32(h_hi + 4, vaddq_f32(vld1q_f32(h_hi + 4), ws47));
                                    vst1q_f32(h_lo,     vaddq_f32(vld1q_f32(h_lo),     ws03));
                                    vst1q_f32(h_lo + 4, vaddq_f32(vld1q_f32(h_lo + 4), ws47));
                                }
#else
                                for (int b = 0; b < 32; b++) {
                                    uint8_t m = v_mask[b];
                                    float* h_hi = hist_gl + ((size_t)(b * 2) * 16u + (size_t)(m >> 4)) * 8u;
                                    float* h_lo = hist_gl + ((size_t)(b * 2 + 1) * 16u + (size_t)(m & 15u)) * 8u;
                                    for (int g = 0; g < 8; g++) {
                                        float ws = w[g] * v_scale;
                                        h_hi[g] += ws;
                                        h_lo[g] += ws;
                                    }
                                }
#endif
                                if (bpg == 48) {
                                    for (int o = 0; o < 4; o++) {
                                        int d_idx = gl * 256 + (int)v_gl[36 + o];
                                        float ores = fp16_to_fp32(read_u16_le(v_gl + 40 + o * 2));
                                        for (int g = 0; g < 8; g++) acc_ptrs[g][d_idx] += w[g] * ores;
                                    }
                                }
                            } else { /* gqa_group == 2 */
                                bias_acc[gl * 2 + 0] += w[0] * v_bias;
                                bias_acc[gl * 2 + 1] += w[1] * v_bias;
#if HAS_NEON
                                float32x2_t ws01 = vmul_n_f32(vld1_f32(w), v_scale);
                                for (int b = 0; b < 32; b++) {
                                    uint8_t m = v_mask[b];
                                    float* h_hi = hist_gl + ((size_t)(b * 2) * 16u + (size_t)(m >> 4)) * 2u;
                                    float* h_lo = hist_gl + ((size_t)(b * 2 + 1) * 16u + (size_t)(m & 15u)) * 2u;
                                    vst1_f32(h_hi, vadd_f32(vld1_f32(h_hi), ws01));
                                    vst1_f32(h_lo, vadd_f32(vld1_f32(h_lo), ws01));
                                }
#else
                                float ws0 = w[0] * v_scale, ws1 = w[1] * v_scale;
                                for (int b = 0; b < 32; b++) {
                                    uint8_t m = v_mask[b];
                                    float* h_hi = hist_gl + ((size_t)(b * 2) * 16u + (size_t)(m >> 4)) * 2u;
                                    float* h_lo = hist_gl + ((size_t)(b * 2 + 1) * 16u + (size_t)(m & 15u)) * 2u;
                                    h_hi[0] += ws0; h_hi[1] += ws1;
                                    h_lo[0] += ws0; h_lo[1] += ws1;
                                }
#endif
                                if (bpg == 48) {
                                    for (int o = 0; o < 4; o++) {
                                        int d_idx = gl * 256 + (int)v_gl[36 + o];
                                        float ores = fp16_to_fp32(read_u16_le(v_gl + 40 + o * 2));
                                        acc_ptrs[0][d_idx] += w[0] * ores;
                                        acc_ptrs[1][d_idx] += w[1] * ores;
                                    }
                                }
                            }
                        }
                    }

                    /* Step D: Flush Nibble Histogram into head accumulators once per 512 tokens */
                    for (int gl = 0; gl < glyphs_per_head; gl++) {
                        const float* hist_gl = nib_hist + (size_t)gl * gl_lut_stride;
                        for (int nib = 0; nib < 64; nib++) {
                            float out4[4][8];
                            for (int g = 0; g < gqa_group; g++) {
                                float bg = bias_acc[gl * gqa_group + g];
                                out4[0][g] = bg; out4[1][g] = bg; out4[2][g] = bg; out4[3][g] = bg;
                            }
                            for (int pat = 0; pat < 16; pat++) {
                                const float* hw = hist_gl + ((size_t)nib * 16u + (size_t)pat) * (size_t)gqa_group;
                                float s0 = ((pat >> 3) & 1) ? 1.0f : -1.0f;
                                float s1 = ((pat >> 2) & 1) ? 1.0f : -1.0f;
                                float s2 = ((pat >> 1) & 1) ? 1.0f : -1.0f;
                                float s3 = (pat & 1)        ? 1.0f : -1.0f;
                                for (int g = 0; g < gqa_group; g++) {
                                    float hwg = hw[g];
                                    out4[0][g] += s0 * hwg;
                                    out4[1][g] += s1 * hwg;
                                    out4[2][g] += s2 * hwg;
                                    out4[3][g] += s3 * hwg;
                                }
                            }
                            int d0 = gl * 256 + nib * 4;
                            for (int g = 0; g < gqa_group; g++) {
                                acc_ptrs[g][d0 + 0] += out4[0][g];
                                acc_ptrs[g][d0 + 1] += out4[1][g];
                                acc_ptrs[g][d0 + 2] += out4[2][g];
                                acc_ptrs[g][d0 + 3] += out4[3][g];
                            }
                        }
                    }
                }

                neon_free(nib_hist);
                handled_by_nibble = 1;
            }
        }

        if (!handled_by_nibble) {
        for (int t = t_start; t < t_end; t++) {
            const uint8_t* k_tok = k_head_base + (size_t)t * kv_seq_stride_bytes;
            for (int g = 0; g < gqa_group; g++) scores[g] = 0.0f;

            if (codec_id == 4) {
                /* Option 4: Unpack each 8D RVQ octet once in registers & dot against all G queries */
                for (int gl = 0; gl < glyphs_per_head; gl++) {
                    const uint8_t* glyph = k_tok + (size_t)gl * 132;
                    const float s1 = fp16_to_fp32(read_u16_le(glyph + 0));
                    const float s2 = fp16_to_fp32(read_u16_le(glyph + 2));
                    const uint8_t* p_idx1 = glyph + 4;
                    const uint8_t* p_idx2 = glyph + 68;
                    float* k_dst_gl = k_rot + (size_t)gl * 256;

#if HAS_AVX2
                    const __m256 s1v = _mm256_set1_ps(s1);
                    const __m256 s2v = _mm256_set1_ps(s2);
                    if (gqa_group == 8) {
                        __m256 d0 = _mm256_setzero_ps(), d1 = _mm256_setzero_ps();
                        __m256 d2 = _mm256_setzero_ps(), d3 = _mm256_setzero_ps();
                        __m256 d4 = _mm256_setzero_ps(), d5 = _mm256_setzero_ps();
                        __m256 d6 = _mm256_setzero_ps(), d7 = _mm256_setzero_ps();
                        const float* q0 = q_scaled + (size_t)(qh_base + 0) * head_dim + (size_t)gl * 256;
                        const float* q1 = q_scaled + (size_t)(qh_base + 1) * head_dim + (size_t)gl * 256;
                        const float* q2 = q_scaled + (size_t)(qh_base + 2) * head_dim + (size_t)gl * 256;
                        const float* q3 = q_scaled + (size_t)(qh_base + 3) * head_dim + (size_t)gl * 256;
                        const float* q4 = q_scaled + (size_t)(qh_base + 4) * head_dim + (size_t)gl * 256;
                        const float* q5 = q_scaled + (size_t)(qh_base + 5) * head_dim + (size_t)gl * 256;
                        const float* q6 = q_scaled + (size_t)(qh_base + 6) * head_dim + (size_t)gl * 256;
                        const float* q7 = q_scaled + (size_t)(qh_base + 7) * head_dim + (size_t)gl * 256;
                        for (int oct = 0; oct < 32; oct++) {
                            uint16_t i1 = read_u16_le(p_idx1 + oct * 2) & 2047u;
                            uint16_t i2 = read_u16_le(p_idx2 + oct * 2) & 2047u;
                            __m256 kv = _mm256_fmadd_ps(
                                _mm256_loadu_ps(codebook_s7 + ((size_t)i2 << 3)), s2v,
                                _mm256_mul_ps(_mm256_loadu_ps(codebook_s7 + ((size_t)i1 << 3)), s1v)
                            );
                            if (k_eq_v) _mm256_storeu_ps(k_dst_gl + (oct << 3), kv);
                            int off = oct << 3;
                            d0 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(q0 + off), d0);
                            d1 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(q1 + off), d1);
                            d2 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(q2 + off), d2);
                            d3 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(q3 + off), d3);
                            d4 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(q4 + off), d4);
                            d5 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(q5 + off), d5);
                            d6 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(q6 + off), d6);
                            d7 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(q7 + off), d7);
                        }
                        scores[0] += hsum256_ps(d0); scores[1] += hsum256_ps(d1);
                        scores[2] += hsum256_ps(d2); scores[3] += hsum256_ps(d3);
                        scores[4] += hsum256_ps(d4); scores[5] += hsum256_ps(d5);
                        scores[6] += hsum256_ps(d6); scores[7] += hsum256_ps(d7);
                    } else {
                        for (int oct = 0; oct < 32; oct++) {
                            uint16_t i1 = read_u16_le(p_idx1 + oct * 2) & 2047u;
                            uint16_t i2 = read_u16_le(p_idx2 + oct * 2) & 2047u;
                            __m256 kv = _mm256_fmadd_ps(
                                _mm256_loadu_ps(codebook_s7 + ((size_t)i2 << 3)), s2v,
                                _mm256_mul_ps(_mm256_loadu_ps(codebook_s7 + ((size_t)i1 << 3)), s1v)
                            );
                            _mm256_storeu_ps(k_dst_gl + (oct << 3), kv);
                        }
                        for (int g = 0; g < gqa_group; g++) {
                            const float* qg = q_scaled + (size_t)(qh_base + g) * head_dim + (size_t)gl * 256;
                            __m256 dg = _mm256_setzero_ps();
                            for (int d = 0; d < 256; d += 8) {
                                dg = _mm256_fmadd_ps(_mm256_loadu_ps(k_dst_gl + d), _mm256_loadu_ps(qg + d), dg);
                            }
                            scores[g] += hsum256_ps(dg);
                        }
                    }
#elif HAS_NEON
                    if (gqa_group == 8) {
                        const float32x4_t s1v = vdupq_n_f32(s1);
                        const float32x4_t s2v = vdupq_n_f32(s2);
                        float32x4_t d0 = vdupq_n_f32(0.0f), d1 = vdupq_n_f32(0.0f);
                        float32x4_t d2 = vdupq_n_f32(0.0f), d3 = vdupq_n_f32(0.0f);
                        float32x4_t d4 = vdupq_n_f32(0.0f), d5 = vdupq_n_f32(0.0f);
                        float32x4_t d6 = vdupq_n_f32(0.0f), d7 = vdupq_n_f32(0.0f);
                        const float* q0 = q_scaled + (size_t)(qh_base + 0) * head_dim + (size_t)gl * 256;
                        const float* q1 = q_scaled + (size_t)(qh_base + 1) * head_dim + (size_t)gl * 256;
                        const float* q2 = q_scaled + (size_t)(qh_base + 2) * head_dim + (size_t)gl * 256;
                        const float* q3 = q_scaled + (size_t)(qh_base + 3) * head_dim + (size_t)gl * 256;
                        const float* q4 = q_scaled + (size_t)(qh_base + 4) * head_dim + (size_t)gl * 256;
                        const float* q5 = q_scaled + (size_t)(qh_base + 5) * head_dim + (size_t)gl * 256;
                        const float* q6 = q_scaled + (size_t)(qh_base + 6) * head_dim + (size_t)gl * 256;
                        const float* q7 = q_scaled + (size_t)(qh_base + 7) * head_dim + (size_t)gl * 256;
                        for (int oct = 0; oct < 32; oct++) {
                            uint16_t i1 = read_u16_le(p_idx1 + oct * 2) & 2047u;
                            uint16_t i2 = read_u16_le(p_idx2 + oct * 2) & 2047u;
                            const float* cb1 = codebook_s7 + ((size_t)i1 << 3);
                            const float* cb2 = codebook_s7 + ((size_t)i2 << 3);
                            float32x4_t kv0 = vfmaq_f32(vmulq_f32(vld1q_f32(cb1), s1v), vld1q_f32(cb2), s2v);
                            float32x4_t kv4 = vfmaq_f32(vmulq_f32(vld1q_f32(cb1 + 4), s1v), vld1q_f32(cb2 + 4), s2v);
                            int off = oct << 3;
                            if (k_eq_v) {
                                vst1q_f32(k_dst_gl + off, kv0);
                                vst1q_f32(k_dst_gl + off + 4, kv4);
                            }
                            /* Interleave all 8 lo-half FMAs then all 8 hi-half FMAs to hide 4-cycle Cortex-A72 FMLA latency */
                            d0 = vfmaq_f32(d0, kv0, vld1q_f32(q0 + off));
                            d1 = vfmaq_f32(d1, kv0, vld1q_f32(q1 + off));
                            d2 = vfmaq_f32(d2, kv0, vld1q_f32(q2 + off));
                            d3 = vfmaq_f32(d3, kv0, vld1q_f32(q3 + off));
                            d4 = vfmaq_f32(d4, kv0, vld1q_f32(q4 + off));
                            d5 = vfmaq_f32(d5, kv0, vld1q_f32(q5 + off));
                            d6 = vfmaq_f32(d6, kv0, vld1q_f32(q6 + off));
                            d7 = vfmaq_f32(d7, kv0, vld1q_f32(q7 + off));
                            d0 = vfmaq_f32(d0, kv4, vld1q_f32(q0 + off + 4));
                            d1 = vfmaq_f32(d1, kv4, vld1q_f32(q1 + off + 4));
                            d2 = vfmaq_f32(d2, kv4, vld1q_f32(q2 + off + 4));
                            d3 = vfmaq_f32(d3, kv4, vld1q_f32(q3 + off + 4));
                            d4 = vfmaq_f32(d4, kv4, vld1q_f32(q4 + off + 4));
                            d5 = vfmaq_f32(d5, kv4, vld1q_f32(q5 + off + 4));
                            d6 = vfmaq_f32(d6, kv4, vld1q_f32(q6 + off + 4));
                            d7 = vfmaq_f32(d7, kv4, vld1q_f32(q7 + off + 4));
                        }
                        scores[0] += vaddvq_f32(d0); scores[1] += vaddvq_f32(d1);
                        scores[2] += vaddvq_f32(d2); scores[3] += vaddvq_f32(d3);
                        scores[4] += vaddvq_f32(d4); scores[5] += vaddvq_f32(d5);
                        scores[6] += vaddvq_f32(d6); scores[7] += vaddvq_f32(d7);
                    } else {
                        decode_single_glyph_mode(glyph, codebook_s7, k_dst_gl, 0);
                        for (int g = 0; g < gqa_group; g++) {
                            const float* qg = q_scaled + (size_t)(qh_base + g) * head_dim + (size_t)gl * 256;
                            float32x4_t dg0 = vdupq_n_f32(0.0f), dg1 = vdupq_n_f32(0.0f);
                            for (int d = 0; d < 256; d += 8) {
                                dg0 = vfmaq_f32(dg0, vld1q_f32(k_dst_gl + d),     vld1q_f32(qg + d));
                                dg1 = vfmaq_f32(dg1, vld1q_f32(k_dst_gl + d + 4), vld1q_f32(qg + d + 4));
                            }
                            scores[g] += hsum4_pair(dg0, dg1);
                        }
                    }
#else
                    decode_single_glyph_mode(glyph, codebook_s7, k_dst_gl, 0);
                    for (int g = 0; g < gqa_group; g++) {
                        const float* qg = q_scaled + (size_t)(qh_base + g) * head_dim + (size_t)gl * 256;
                        float s = 0.0f;
                        for (int d = 0; d < 256; d++) s += k_dst_gl[d] * qg[d];
                        scores[g] += s;
                    }
#endif
                }
            } else {
                /* Generic Option 5 fallback for arbitrary GQA group sizes */
                for (int gl = 0; gl < glyphs_per_head; gl++) {
                    const uint8_t* glyph = k_tok + (size_t)gl * (size_t)bpg;
                    const float scale = fp16_to_fp32(read_u16_le(glyph + 0));
                    const float bias  = fp16_to_fp32(read_u16_le(glyph + 2));
                    const uint8_t* bitmask = glyph + 4;
                    for (int g = 0; g < gqa_group; g++) {
                        const float* qg = q_scaled + (size_t)(qh_base + g) * head_dim + (size_t)gl * 256;
#if HAS_AVX2
                        __m256 dg = _mm256_setzero_ps();
                        for (int b = 0; b < 32; b++) {
                            dg = _mm256_fmadd_ps(_mm256_load_ps(g_sign_lut[bitmask[b]]), _mm256_loadu_ps(qg + (b << 3)), dg);
                        }
                        float sdot = hsum256_ps(dg);
#elif HAS_NEON
                        float32x4_t dg = vdupq_n_f32(0.0f);
                        for (int b = 0; b < 32; b++) {
                            const float* lut = g_sign_lut[bitmask[b]];
                            dg = vfmaq_f32(dg, vld1q_f32(lut),     vld1q_f32(qg + (b << 3)));
                            dg = vfmaq_f32(dg, vld1q_f32(lut + 4), vld1q_f32(qg + (b << 3) + 4));
                        }
                        float sdot = vaddvq_f32(dg);
#else
                        float sdot = 0.0f;
                        for (int b = 0; b < 32; b++) {
                            const float* lut = g_sign_lut[bitmask[b]];
                            for (int i = 0; i < 8; i++) sdot += lut[i] * qg[(b << 3) + i];
                        }
#endif
                        float base = scale * sdot + bias * q_gsum[(size_t)(qh_base + g) * glyphs_per_head + gl];
                        if (bpg == 48) {
                            for (int o = 0; o < 4; o++) {
                                uint8_t oi = glyph[36 + o];
                                float ores = fp16_to_fp32(read_u16_le(glyph + 40 + o * 2));
                                base += ores * qg[oi];
                            }
                        }
                        scores[g] += base;
                    }
                }
            }

            /* Online Softmax FP32-Lossless Early-Out Gate (exp(-17.5) = 2.5e-8 < 0.5 ULP FP32) */
            int any_active = 0;
            int any_new_max = 0;
            for (int g = 0; g < gqa_group; g++) {
                float diff = scores[g] - loc_m[g];
                if (diff > -17.5f) any_active = 1;
                if (diff > 0.0f)   any_new_max = 1;
            }
            if (!any_active) continue;

            const float* v_vec;
            if (k_eq_v) {
                if (codec_id != 4) {
                    for (int gl = 0; gl < glyphs_per_head; gl++) {
                        decode_single_opt5_glyph(k_tok + (size_t)gl * (size_t)bpg, k_rot + (size_t)gl * 256, bpg);
                    }
                }
                v_vec = k_rot;
            } else {
                const uint8_t* v_tok = v_head_base + (size_t)t * kv_seq_stride_bytes;
                for (int gl = 0; gl < glyphs_per_head; gl++) {
                    if (codec_id == 4) {
                        decode_single_glyph_mode(v_tok + (size_t)gl * 132, codebook_s7, v_rot + (size_t)gl * 256, 0);
                    } else {
                        decode_single_opt5_glyph(v_tok + (size_t)gl * (size_t)bpg, v_rot + (size_t)gl * 256, bpg);
                    }
                }
                v_vec = v_rot;
            }

            if (gqa_group == 8 && !any_new_max) {
                /* 99.95% fast path: single-load 8-head register-fused V accumulation */
                float w8[8];
                for (int g = 0; g < 8; g++) {
                    float diff = scores[g] - loc_m[g];
                    if (diff > -17.5f) {
                        float wg = neon_expf(diff);
                        w8[g] = wg;
                        loc_l[g] += wg;
                    } else {
                        w8[g] = 0.0f;
                    }
                }
                simd_fma_8heads(acc_ptrs, v_vec, w8, head_dim);
            } else {
                for (int g = 0; g < gqa_group; g++) {
                    float s = scores[g];
                    float m_old = loc_m[g];
                    if (s - m_old <= -17.5f) continue;

                    float* acc_g = acc_ptrs[g];
                    if (s > m_old) {
                        float alpha = (m_old <= -1e29f) ? 0.0f : neon_expf(m_old - s);
                        loc_m[g] = s;
                        loc_l[g] = loc_l[g] * alpha + 1.0f;
                        simd_rescale_and_fma(acc_g, v_vec, alpha, 1.0f, head_dim);
                    } else {
                        float w = neon_expf(s - m_old);
                        loc_l[g] += w;
                        simd_fma_only(acc_g, v_vec, w, head_dim);
                    }
                }
            }
        }
        }

        for (int g = 0; g < gqa_group; g++) {
            size_t slot = (size_t)(qh_base + g) * (size_t)n_tiles + (size_t)tidx;
            tile_m[slot] = loc_m[g];
            tile_l[slot] = loc_l[g];
            memcpy(tile_acc + slot * (size_t)head_dim, acc_ptrs[g], (size_t)head_dim * sizeof(float));
        }

        if (k_rot != stack_k) neon_free(k_rot);
        if (v_rot != stack_v) neon_free(v_rot);
        if (scores != stack_scores) neon_free(scores);
        if (loc_m != stack_m) neon_free(loc_m);
        if (loc_l != stack_l) neon_free(loc_l);
        if (local_acc_buf != stack_local_acc) neon_free(local_acc_buf);
        if (acc_ptrs != stack_acc_ptrs) neon_free(acc_ptrs);
    }

    /* 3. Associative Log-Sum-Exp Reduction across Split-K Tiles + Single Output FWHT-8 */
    for (int qh = 0; qh < n_q_heads; qh++) {
        float m_max = -1e30f;
        for (int tidx = 0; tidx < n_tiles; tidx++) {
            float mt = tile_m[(size_t)qh * (size_t)n_tiles + (size_t)tidx];
            if (mt > m_max) m_max = mt;
        }

        float* out_q = out_context + (size_t)qh * (size_t)head_dim;
        memset(out_q, 0, (size_t)head_dim * sizeof(float));
        float l_sum = 0.0f;

        for (int tidx = 0; tidx < n_tiles; tidx++) {
            size_t slot = (size_t)qh * (size_t)n_tiles + (size_t)tidx;
            float lt = tile_l[slot];
            if (lt <= 0.0f) continue;
            float scale_t = neon_expf(tile_m[slot] - m_max);
            l_sum += lt * scale_t;
            const float* acc_t = tile_acc + slot * (size_t)head_dim;
            simd_fma_only(out_q, acc_t, scale_t, head_dim);
        }

        float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
        for (int d = 0; d < head_dim; d++) out_q[d] *= inv_l;

        if (apply_fwht8) {
            fwht8_vec_c(out_q, out_q, head_dim);
        }
        if (out_max_score) out_max_score[qh] = m_max;
        if (out_sum_exp)   out_sum_exp[qh]   = l_sum;
    }

    neon_free(slab);
    return 0;
}

static inline void bf16_to_fp32_vec(const uint16_t* NEON_RESTRICT src, float* NEON_RESTRICT dst, int n) {
    int i = 0;
#if HAS_AVX2
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm_loadu_si128((const __m128i*)(src + i));
        __m256i w = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16);
        _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(w));
    }
#elif HAS_NEON
    for (; i + 8 <= n; i += 8) {
        uint16x8_t h = vld1q_u16(src + i);
        uint32x4_t w0 = vshll_n_u16(vget_low_u16(h), 16);
        uint32x4_t w1 = vshll_n_u16(vget_high_u16(h), 16);
        vst1q_f32(dst + i,     vreinterpretq_f32_u32(w0));
        vst1q_f32(dst + i + 4, vreinterpretq_f32_u32(w1));
    }
#endif
    for (; i < n; i++) {
        uint32_t bits = ((uint32_t)src[i]) << 16;
        memcpy(dst + i, &bits, sizeof(float));
    }
}

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
) {
    if (n_q_heads <= 0 || n_kv_heads <= 0 || (n_q_heads % n_kv_heads) != 0) return -1;
    if (head_dim <= 0 || seq_len <= 0 || page_size <= 0) return -2;

    const int gqa_group = n_q_heads / n_kv_heads;
    int tile_size = 512;
    if (seq_len < 512) tile_size = seq_len;
    else if ((seq_len + 511) / 512 > 32) tile_size = (seq_len + 31) / 32;
    const int n_tiles = (seq_len + tile_size - 1) / tile_size;
    const int total_jobs = n_kv_heads * n_tiles;

    const size_t q_elems = (size_t)n_q_heads * (size_t)head_dim;
    const size_t meta_elems = (size_t)n_q_heads * (size_t)n_tiles;
    const size_t acc_elems = meta_elems * (size_t)head_dim;
    float* slab = (float*)neon_alloc((q_elems + 2 * meta_elems + acc_elems) * sizeof(float));
    if (!slab) return -3;

    float* q_scaled = slab;
    float* tile_m   = q_scaled + q_elems;
    float* tile_l   = tile_m + meta_elems;
    float* tile_acc = tile_l + meta_elems;

    for (size_t i = 0; i < q_elems; i++) q_scaled[i] = q[i] * attn_scale;

    const size_t page_stride = (size_t)page_size * (size_t)n_kv_heads * (size_t)head_dim;
    const size_t slot_stride = (size_t)n_kv_heads * (size_t)head_dim;

    int job;
#if HAS_OMP
    if (total_jobs > 1 && num_threads != 1) {
        if (num_threads > 0) omp_set_num_threads(num_threads);
    }
    #pragma omp parallel for schedule(dynamic, 1) if(total_jobs > 1 && num_threads != 1)
#else
    (void)num_threads;
#endif
    for (job = 0; job < total_jobs; job++) {
        const int kvh = job / n_tiles;
        const int tidx = job % n_tiles;
        const int t_start = tidx * tile_size;
        const int t_end = (t_start + tile_size < seq_len) ? (t_start + tile_size) : seq_len;
        const int qh_base = kvh * gqa_group;

        float stack_k[512], stack_v[512], loc_m[16], loc_l[16], scores[16];
        for (int g = 0; g < gqa_group; g++) {
            loc_m[g] = -1e30f;
            loc_l[g] = 0.0f;
            memset(tile_acc + ((size_t)(qh_base + g) * n_tiles + tidx) * head_dim, 0, (size_t)head_dim * sizeof(float));
        }

        for (int t = t_start; t < t_end; t++) {
            int phys_page = block_table[t / page_size];
            int page_off  = t % page_size;
            size_t elem_off = (size_t)phys_page * page_stride + (size_t)page_off * slot_stride + (size_t)kvh * head_dim;
            bf16_to_fp32_vec(k_pages + elem_off, stack_k, head_dim);

            for (int g = 0; g < gqa_group; g++) {
                const float* qg = q_scaled + (size_t)(qh_base + g) * head_dim;
#if HAS_AVX2
                __m256 dg = _mm256_setzero_ps();
                for (int d = 0; d < head_dim; d += 8) {
                    dg = _mm256_fmadd_ps(_mm256_loadu_ps(stack_k + d), _mm256_loadu_ps(qg + d), dg);
                }
                scores[g] = hsum256_ps(dg);
#elif HAS_NEON
                float32x4_t dg0 = vdupq_n_f32(0.0f), dg1 = vdupq_n_f32(0.0f);
                for (int d = 0; d < head_dim; d += 8) {
                    dg0 = vfmaq_f32(dg0, vld1q_f32(stack_k + d),     vld1q_f32(qg + d));
                    dg1 = vfmaq_f32(dg1, vld1q_f32(stack_k + d + 4), vld1q_f32(qg + d + 4));
                }
                scores[g] = hsum4_pair(dg0, dg1);
#else
                float s = 0.0f;
                for (int d = 0; d < head_dim; d++) s += stack_k[d] * qg[d];
                scores[g] = s;
#endif
            }

            const float* v_vec = stack_k;
            if (!k_eq_v) {
                bf16_to_fp32_vec(v_pages + elem_off, stack_v, head_dim);
                v_vec = stack_v;
            }

            for (int g = 0; g < gqa_group; g++) {
                float s = scores[g];
                float m_old = loc_m[g];
                float* acc_g = tile_acc + ((size_t)(qh_base + g) * n_tiles + tidx) * head_dim;
                if (s > m_old) {
                    float alpha = (m_old <= -1e29f) ? 0.0f : neon_expf(m_old - s);
                    loc_m[g] = s;
                    loc_l[g] = loc_l[g] * alpha + 1.0f;
                    simd_rescale_and_fma(acc_g, v_vec, alpha, 1.0f, head_dim);
                } else {
                    float w = neon_expf(s - m_old);
                    loc_l[g] += w;
                    simd_fma_only(acc_g, v_vec, w, head_dim);
                }
            }
        }

        for (int g = 0; g < gqa_group; g++) {
            size_t slot = (size_t)(qh_base + g) * n_tiles + tidx;
            tile_m[slot] = loc_m[g];
            tile_l[slot] = loc_l[g];
        }
    }

    for (int qh = 0; qh < n_q_heads; qh++) {
        float m_max = -1e30f;
        for (int tidx = 0; tidx < n_tiles; tidx++) {
            float mt = tile_m[(size_t)qh * n_tiles + tidx];
            if (mt > m_max) m_max = mt;
        }
        float* out_q = out_context + (size_t)qh * head_dim;
        memset(out_q, 0, (size_t)head_dim * sizeof(float));
        float l_sum = 0.0f;
        for (int tidx = 0; tidx < n_tiles; tidx++) {
            size_t slot = (size_t)qh * n_tiles + tidx;
            float lt = tile_l[slot];
            if (lt <= 0.0f) continue;
            float scale_t = neon_expf(tile_m[slot] - m_max);
            l_sum += lt * scale_t;
            simd_fma_only(out_q, tile_acc + slot * head_dim, scale_t, head_dim);
        }
        float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
        for (int d = 0; d < head_dim; d++) out_q[d] *= inv_l;
    }

    neon_free(slab);
    return 0;
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

    /* Gate D: GlyphAttentionV2 encode_kv_glyph_fast_c + flash_glyph_attention_gqa_c parity */
    {
        float kv_src[2 * 256];
        uint8_t kv_g5[2 * 48];
        float kv_dec[2 * 256];
        float out_flash[256];
        for (int i = 0; i < 512; i++) {
            kv_src[i] = 0.03f * (float)(((i * 17 + 5) % 29) - 14);
        }
        encode_kv_glyph_fast_c(kv_src, NULL, NULL, kv_g5, 2, 5, 48, 1, 1);
        decode_glyphs_opt5_mode_c(kv_g5, kv_dec, 2, 48, 1, 1);
        int rc = flash_glyph_attention_gqa_c(
            x, kv_g5, NULL, NULL, out_flash, NULL, NULL,
            1, 1, 256, 2, 48, 2 * 48, 0.0625f, 5, 48, 1, 1, 1
        );
        if (rc != 0) return 0;
        float s0 = 0.0f, s1 = 0.0f;
        for (int i = 0; i < 256; i++) {
            s0 += kv_dec[i] * x[i] * 0.0625f;
            s1 += kv_dec[256 + i] * x[i] * 0.0625f;
        }
        float smax = (s0 > s1) ? s0 : s1;
        float e0 = neon_expf(s0 - smax), e1 = neon_expf(s1 - smax);
        float w0 = e0 / (e0 + e1), w1 = e1 / (e0 + e1);
        for (int i = 0; i < 256; i++) {
            float ref_i = w0 * kv_dec[i] + w1 * kv_dec[256 + i];
            if (neon_fabsf(out_flash[i] - ref_i) > 1e-4f) return 0;
        }
    }

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
