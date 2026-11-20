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

#if defined(_MSC_VER)
  #if HAS_AVX2 || defined(NEON_MSVC_FREESTANDING_DLL)
    #define neon_prefetch(p) _mm_prefetch((const char*)(p), _MM_HINT_NTA)
  #else
    #define neon_prefetch(p) ((void)(p))
  #endif
#elif defined(__GNUC__) || defined(__clang__)
  #define neon_prefetch(p) __builtin_prefetch((const void*)(p), 0, 0)
#else
  #define neon_prefetch(p) ((void)(p))
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
    const float* x_in,
    float* x_out,
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

static inline uint16_t read_packed11(const uint8_t* glyph, unsigned index) {
    const unsigned bit = index * 11u;
    const unsigned shift = bit & 7u;
    const uint8_t* p = glyph + 4 + (bit >> 3);
    uint32_t v = (uint32_t)p[0] | ((uint32_t)p[1] << 8);
    /* The last field starts at shift=5, so this never reads past byte 91. */
    if (shift > 5u) v |= (uint32_t)p[2] << 16;
    return (uint16_t)((v >> shift) & 2047u);
}

NEON_API void decode_glyphs_opt4_packed_c(
    const uint8_t* NEON_RESTRICT glyphs,
    const float* NEON_RESTRICT cb,
    float* NEON_RESTRICT out_weights,
    int n_glyphs, int apply_fwht8, int num_threads
) {
    int g;
#if HAS_OMP
    if (num_threads > 0) omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(static)
#else
    (void)num_threads;
#endif
    for (g = 0; g < n_glyphs; ++g) {
        const uint8_t* src = glyphs + (size_t)g * 92;
        uint8_t expanded[132]; /* One glyph on the worker stack, never a matrix. */
        memcpy(expanded, src, 4);
        for (unsigned j = 0; j < 64; ++j)
            write_u16_le(expanded + 4 + j * 2, read_packed11(src, j));
        decode_single_glyph_mode(expanded, cb, out_weights + (size_t)g * 256, apply_fwht8);
    }
}

NEON_API void gemv_opt4_packed_mmap(
    const uint8_t* NEON_RESTRICT mmap_base, size_t tensor_byte_offset,
    const float* NEON_RESTRICT cb, const float* NEON_RESTRICT x,
    float* NEON_RESTRICT y, int d_out, int d_in
) {
    const int glyphs_per_row = d_in / 256;
    const uint8_t* tensor = mmap_base + tensor_byte_offset;
    int row;
#if HAS_OMP
    #pragma omp parallel for schedule(static)
#endif
    for (row = 0; row < d_out; ++row) {
        float total = 0.0f;
        for (int g = 0; g < glyphs_per_row; ++g) {
            const uint8_t* p = tensor + ((size_t)row * glyphs_per_row + g) * 92;
            const float* xg = x + (size_t)g * 256;
            const float s1 = fp16_to_fp32(read_u16_le(p));
            const float s2 = fp16_to_fp32(read_u16_le(p + 2));
            float a = 0.0f, b = 0.0f;
            for (unsigned oct = 0; oct < 32; ++oct) {
                const float* c1 = cb + ((size_t)read_packed11(p, oct) << 3);
                const float* c2 = cb + ((size_t)read_packed11(p, oct + 32) << 3);
                for (unsigned j = 0; j < 8; ++j) {
                    a += c1[j] * xg[oct * 8 + j];
                    b += c2[j] * xg[oct * 8 + j];
                }
            }
            total += s1 * a + s2 * b;
        }
        y[row] = total;
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
    int alloc_failed = 0;
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
        float* stack_acc_ptrs[16];

        const size_t acc_stride = (size_t)head_dim + 16u;
        float* k_rot = (head_dim <= 512) ? stack_k : (float*)neon_alloc((size_t)head_dim * sizeof(float));
        float* v_rot = (head_dim <= 512) ? stack_v : (float*)neon_alloc((size_t)head_dim * sizeof(float));
        float* scores = (gqa_group <= 16) ? stack_scores : (float*)neon_alloc((size_t)gqa_group * sizeof(float));
        float* loc_m  = (gqa_group <= 16) ? stack_m : (float*)neon_alloc((size_t)gqa_group * sizeof(float));
        float* loc_l  = (gqa_group <= 16) ? stack_l : (float*)neon_alloc((size_t)gqa_group * sizeof(float));
        /* Heap-allocate L1-private accumulator (~17 KB+) rather than risking OpenMP worker stack guard overflow */
        float* local_acc_buf = (float*)neon_alloc((size_t)gqa_group * acc_stride * sizeof(float));
        float** acc_ptrs = (gqa_group <= 16)
            ? stack_acc_ptrs
            : (float**)neon_alloc((size_t)gqa_group * sizeof(float*));

        if (!k_rot || !v_rot || !scores || !loc_m || !loc_l || !local_acc_buf || !acc_ptrs) {
#if HAS_OMP
            #pragma omp critical
#endif
            { alloc_failed = 1; }
        } else {
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
#if HAS_AVX2 || HAS_NEON
                    const float s1 = fp16_to_fp32(read_u16_le(glyph + 0));
                    const float s2 = fp16_to_fp32(read_u16_le(glyph + 2));
                    const uint8_t* p_idx1 = glyph + 4;
                    const uint8_t* p_idx2 = glyph + 68;
#endif
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
        }

        if (k_rot && k_rot != stack_k) neon_free(k_rot);
        if (v_rot && v_rot != stack_v) neon_free(v_rot);
        if (scores && scores != stack_scores) neon_free(scores);
        if (loc_m && loc_m != stack_m) neon_free(loc_m);
        if (loc_l && loc_l != stack_l) neon_free(loc_l);
        if (local_acc_buf) neon_free(local_acc_buf);
        if (acc_ptrs && acc_ptrs != stack_acc_ptrs) neon_free(acc_ptrs);
    }

    if (alloc_failed) {
        neon_free(slab);
        return -5;
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
 * Bit-Serial Tile-16 Sub64 L1-Budget LUT Kernel (1..4-bit, Sym/Asym, CS Skip)
 * ---------------------------------------------------------------------------- */

static inline int neon_round_i32(float v) {
    return (int)((v >= 0.0f) ? (v + 0.5f) : (v - 0.5f));
}

static inline void fill_l1_quad_lut_s8(
    const float* NEON_RESTRICT qx4,
    float inv,
    int8_t* NEON_RESTRICT tab,
    int* NEON_RESTRICT out_q4
) {
    int q0 = neon_round_i32(qx4[0] * inv);
    int q1 = neon_round_i32(qx4[1] * inv);
    int q2 = neon_round_i32(qx4[2] * inv);
    int q3 = neon_round_i32(qx4[3] * inv);
    if (q0 > 125) q0 = 125; else if (q0 < -125) q0 = -125;
    if (q1 > 125) q1 = 125; else if (q1 < -125) q1 = -125;
    if (q2 > 125) q2 = 125; else if (q2 < -125) q2 = -125;
    if (q3 > 125) q3 = 125; else if (q3 < -125) q3 = -125;
    if (out_q4) {
        out_q4[0] = q0;
        out_q4[1] = q1;
        out_q4[2] = q2;
        out_q4[3] = q3;
    }
    for (int m = 0; m < 16; m++) {
        int tot = ((m & 8) ? q0 : -q0) + ((m & 4) ? q1 : -q1) +
                  ((m & 2) ? q2 : -q2) + ((m & 1) ? q3 : -q3);
        if (tot > 127) tot = 127;
        if (tot < -127) tot = -127;
        tab[m] = (int8_t)tot;
    }
}

static void quantize_x_sub64_l1_tile16(
    const float* NEON_RESTRICT x,
    int d_in,
    int8_t* NEON_RESTRICT luts,
    float* NEON_RESTRICT sub_scales
) {
    const int n_subs = d_in / 64;
    for (int s = 0; s < n_subs; s++) {
        const float* xs = x + (size_t)s * 64;
        float smax = 0.0f;
        for (int n = 0; n < 16; n++) {
            float sa = neon_fabsf(xs[n * 4 + 0]) + neon_fabsf(xs[n * 4 + 1]) +
                       neon_fabsf(xs[n * 4 + 2]) + neon_fabsf(xs[n * 4 + 3]);
            if (sa > smax) smax = sa;
        }
        float sc = (smax > 1e-12f) ? (smax / 125.0f) : 0.0f;
        float inv = (sc > 0.0f) ? (1.0f / sc) : 0.0f;
        double dot_xq = 0.0;
        double dot_qq = 0.0;
        for (int n = 0; n < 16; n++) {
            int q4[4];
            fill_l1_quad_lut_s8(xs + n * 4, inv, luts + ((size_t)s * 16 + (size_t)n) * 16, q4);
            for (int k = 0; k < 4; k++) {
                double xf = (double)xs[n * 4 + k];
                double qf = (double)q4[k];
                dot_xq += xf * qf;
                dot_qq += qf * qf;
            }
        }
        sub_scales[s] = (dot_qq > 0.0) ? (float)(dot_xq / dot_qq) : sc;
    }
}

static void quantize_x_sub64_l1_dual_tile16(
    const float* NEON_RESTRICT x,
    int d_in,
    int8_t* NEON_RESTRICT luts_hi,
    float* NEON_RESTRICT sub_scales_hi,
    int8_t* NEON_RESTRICT luts_lo,
    float* NEON_RESTRICT sub_scales_lo
) {
    const int n_subs = d_in / 64;
    float res[64];
    for (int s = 0; s < n_subs; s++) {
        const float* xs = x + (size_t)s * 64;
        float smax_hi = 0.0f;
        for (int n = 0; n < 16; n++) {
            float sa = neon_fabsf(xs[n * 4 + 0]) + neon_fabsf(xs[n * 4 + 1]) +
                       neon_fabsf(xs[n * 4 + 2]) + neon_fabsf(xs[n * 4 + 3]);
            if (sa > smax_hi) smax_hi = sa;
        }
        float sc_hi = (smax_hi > 1e-12f) ? (smax_hi / 125.0f) : 0.0f;
        sub_scales_hi[s] = sc_hi;
        float inv_hi = (sc_hi > 0.0f) ? (1.0f / sc_hi) : 0.0f;
        float smax_lo = 0.0f;
        for (int n = 0; n < 16; n++) {
            int q4[4];
            fill_l1_quad_lut_s8(xs + n * 4, inv_hi, luts_hi + ((size_t)s * 16 + (size_t)n) * 16, q4);
            float e0 = xs[n * 4 + 0] - sc_hi * (float)q4[0];
            float e1 = xs[n * 4 + 1] - sc_hi * (float)q4[1];
            float e2 = xs[n * 4 + 2] - sc_hi * (float)q4[2];
            float e3 = xs[n * 4 + 3] - sc_hi * (float)q4[3];
            res[n * 4 + 0] = e0;
            res[n * 4 + 1] = e1;
            res[n * 4 + 2] = e2;
            res[n * 4 + 3] = e3;
            float sa_lo = neon_fabsf(e0) + neon_fabsf(e1) + neon_fabsf(e2) + neon_fabsf(e3);
            if (sa_lo > smax_lo) smax_lo = sa_lo;
        }
        float sc_lo = (smax_lo > 1e-30f) ? (smax_lo / 125.0f) : 0.0f;
        sub_scales_lo[s] = sc_lo;
        float inv_lo = (sc_lo > 0.0f) ? (1.0f / sc_lo) : 0.0f;
        for (int n = 0; n < 16; n++) {
            fill_l1_quad_lut_s8(res + n * 4, inv_lo, luts_lo + ((size_t)s * 16 + (size_t)n) * 16, NULL);
        }
    }
}

/* Build 256-entry FP32 quintet LUT (0..242 valid base-3, 243..255 zeroed). */
static inline void build_tern5_lut_f32(
    float x0, float x1, float x2, float x3, float x4, float* NEON_RESTRICT dst_lut
) {
    const float a0[3] = { -x0, 0.0f, x0 };
    const float a1[3] = { -x1, 0.0f, x1 };
    const float a2[3] = { -x2, 0.0f, x2 };
    const float a3[3] = { -x3, 0.0f, x3 };
    const float a4[3] = { -x4, 0.0f, x4 };
    float b[9];
    for (int d1 = 0, i = 0; d1 < 3; d1++) {
        const float v1 = a1[d1];
        for (int d0 = 0; d0 < 3; d0++, i++) b[i] = a0[d0] + v1;
    }
    float c[27];
    for (int d2 = 0, i = 0; d2 < 3; d2++) {
        const float v2 = a2[d2];
        for (int j = 0; j < 9; j++, i++) c[i] = b[j] + v2;
    }
    for (int d4 = 0, out = 0; d4 < 3; d4++) {
        const float v4 = a4[d4];
        for (int d3 = 0; d3 < 3; d3++) {
            const float v34 = v4 + a3[d3];
            for (int j = 0; j < 27; j++, out++) {
                dst_lut[out] = c[j] + v34;
            }
        }
    }
    for (int out = 243; out < 256; out++) {
        dst_lut[out] = 0.0f;
    }
}

#if HAS_NEON && (defined(__aarch64__) || defined(_M_ARM64))
static const int8_t NEON_T5_LUT27_D0[32] = {
    -1, 0, 1, -1, 0, 1, -1, 0, 1,
    -1, 0, 1, -1, 0, 1, -1, 0, 1,
    -1, 0, 1, -1, 0, 1, -1, 0, 1,
     0, 0, 0,  0, 0
};
static const int8_t NEON_T5_LUT27_D1[32] = {
    -1, -1, -1,  0,  0,  0,  1,  1,  1,
    -1, -1, -1,  0,  0,  0,  1,  1,  1,
    -1, -1, -1,  0,  0,  0,  1,  1,  1,
     0,  0,  0,  0,  0
};
static const int8_t NEON_T5_LUT27_D2[32] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1,
     0,  0,  0,  0,  0,  0,  0,  0,  0,
     1,  1,  1,  1,  1,  1,  1,  1,  1,
     0,  0,  0,  0,  0
};
static const int8_t NEON_T5_LUT9_D3[16] = {
    -1, 0, 1, -1, 0, 1, -1, 0, 1,  0, 0, 0, 0, 0, 0, 0
};
static const int8_t NEON_T5_LUT9_D4[16] = {
    -1, -1, -1,  0,  0,  0,  1,  1,  1,  0, 0, 0, 0, 0, 0, 0
};

/* Exact single-step DIV27 + parallel AArch64 TBL unpack of 16 packed 5-trit uint8 bytes
 * -> five {-1, 0, +1} vectors with no 4-stage serial dependency chain.
 * Proof: for all q in [0, 242], write q = 27*a + r with 0 <= a <= 8, 0 <= r <= 26.
 * Since 27 * 19 = 513 = 512 + 1 and a + 19*r <= 8 + 19*26 = 502 < 512,
 * (q * 19) >> 9 == a and q - 27*a == r identically. */
static inline void unpack5_neon_tile16(
    uint8x16_t q0,
    int8x16x2_t t27_0, int8x16x2_t t27_1, int8x16x2_t t27_2,
    int8x16_t t9_3, int8x16_t t9_4,
    int8x16_t* w0, int8x16_t* w1, int8x16_t* w2, int8x16_t* w3, int8x16_t* w4
) {
    const uint8x16_t c242 = vdupq_n_u8(242);
    const uint8x16_t c19 = vdupq_n_u8(19);
    const uint8x8_t c19_lo = vget_low_u8(c19);
    const uint8x16_t c27 = vdupq_n_u8(27);

    uint8x16_t q = vminq_u8(q0, c242);
    uint16x8_t p_lo = vmull_u8(vget_low_u8(q), c19_lo);
    uint16x8_t p_hi = vmull_high_u8(q, c19);
    uint8x16_t a = vcombine_u8(vmovn_u16(vshrq_n_u16(p_lo, 9)), vmovn_u16(vshrq_n_u16(p_hi, 9)));
    uint8x16_t r = vmlsq_u8(q, a, c27);

    *w3 = vqtbl1q_s8(t9_3, a);
    *w4 = vqtbl1q_s8(t9_4, a);
    *w0 = vqtbl2q_s8(t27_0, r);
    *w1 = vqtbl2q_s8(t27_1, r);
    *w2 = vqtbl2q_s8(t27_2, r);
}

static inline float quantize_t5_subblk_i8(
    const float* NEON_RESTRICT src,
    int len,
    int pad_to,
    int8_t* NEON_RESTRICT dst
) {
    float mx = 0.0f;
    for (int i = 0; i < len; i++) {
        float a = neon_fabsf(src[i]);
        if (a > mx) mx = a;
    }
    const float inv = (mx > 1e-30f) ? (127.0f / mx) : 0.0f;
    double dot_xq = 0.0, dot_qq = 0.0;
    for (int i = 0; i < len; i++) {
        float sv = src[i] * inv;
        int v = (int)(sv >= 0.0f ? (sv + 0.5f) : (sv - 0.5f));
        if (v > 127) v = 127;
        else if (v < -127) v = -127;
        dst[i] = (int8_t)v;
        dot_xq += (double)src[i] * (double)v;
        dot_qq += (double)v * (double)v;
    }
    for (int i = len; i < pad_to; i++) dst[i] = 0;
    return (dot_qq > 0.0) ? (float)(dot_xq / dot_qq) : ((mx > 1e-30f) ? (mx / 127.0f) : 0.0f);
}
#endif

#define T16C_GAUSS_BIT    0x80
#define T16C_ROWMAJOR_BIT 0x40

static volatile int g_andvcnt_abits = 5;

NEON_API int neon_set_andvcnt_abits(int a_bits) {
    if (a_bits >= 4 && a_bits <= 6) {
        g_andvcnt_abits = a_bits;
    }
    return g_andvcnt_abits;
}

NEON_API int t16c_transpose_col_to_rowmajor_c(
    uint8_t* packed,
    int d_out,
    int d_in,
    int block_size,
    int n_bits,
    int is_asym
) {
    if (!packed || d_out <= 0 || d_in <= 0) return -1;
    if (block_size != 64 && block_size != 128 && block_size != 256) return -1;
    if ((d_in % block_size) != 0) return -1;
    const int eff_bits = n_bits & 0x3F;
    if (eff_bits < 1 || eff_bits > 4) return -1;
    if (is_asym != 0 && is_asym != 1) return -1;

    const int n_tiles = (d_out + 15) / 16;
    const int n_blks = d_in / block_size;
    const int hdr_bytes = is_asym ? 64 : 32;
    const int bytes_per_row = block_size / 8;
    const size_t plane_bytes = (size_t)bytes_per_row * 16u;
    const size_t tblk_bytes = (size_t)hdr_bytes + (size_t)eff_bits * plane_bytes;
    uint8_t tmp[512]; /* max plane_bytes = 32 * 16 = 512 */

    for (int t = 0; t < n_tiles; t++) {
        for (int blk = 0; blk < n_blks; blk++) {
            uint8_t* bptr = packed + ((size_t)t * (size_t)n_blks + (size_t)blk) * tblk_bytes;
            for (int p = 0; p < eff_bits; p++) {
                uint8_t* pptr = bptr + (size_t)hdr_bytes + (size_t)p * plane_bytes;
                for (int r = 0; r < 16; r++) {
                    for (int bo = 0; bo < bytes_per_row; bo++) {
                        tmp[r * bytes_per_row + bo] = pptr[bo * 16 + r];
                    }
                }
                memcpy(pptr, tmp, plane_bytes);
            }
        }
    }
    return 0;
}

static const uint8_t POPCNT_BYTE_LUT[256] = {
    0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
    3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8
};

#if HAS_NEON && (defined(__aarch64__) || defined(_M_ARM64))
static inline float32x4_t sum4_u8x16_exact_neon(uint8x16_t a0, uint8x16_t a1, uint8x16_t a2, uint8x16_t a3) {
    uint16x8_t s01 = vpaddq_u16(vpaddlq_u8(a0), vpaddlq_u8(a1));
    uint16x8_t s23 = vpaddq_u16(vpaddlq_u8(a2), vpaddlq_u8(a3));
    uint16x8_t p   = vpaddq_u16(s01, s23);
    return vcvtq_f32_u32(vpaddlq_u16(p));
}
#endif

/* Quantize one activation block xb[0..block_size-1] into a_bits MSB-first bitplanes
 * for exact AND+VCNT bipolar inner product:
 *   sum_j (2*wb_j - 1) * x_j ~= asc * (2 * sum_bp 2^bp popcnt(wb & ab_bp) - asum_u - half * (2*popcnt(wb) - B)) */
static inline void quantize_x_blk_andvcnt(
    const float* NEON_RESTRICT xb,
    int block_size,
    int a_bits,
    uint8_t* NEON_RESTRICT ab_out,
    float* NEON_RESTRICT out_asc,
    float* NEON_RESTRICT out_asum_u,
    float* NEON_RESTRICT out_xnorm,
    float* NEON_RESTRICT out_xsum
) {
    const int bytes_per_row = block_size / 8;
    const int half = 1 << (a_bits - 1);
    const int maxq = half - 1;

    float mx = 0.0f;
    float n2 = 0.0f;
    float xs = 0.0f;
    for (int c = 0; c < block_size; c++) {
        float v = xb[c];
        float av = neon_fabsf(v);
        if (av > mx) mx = av;
        n2 += v * v;
        xs += v;
    }
    if (out_xnorm) *out_xnorm = neon_sqrtf(n2);
    if (out_xsum)  *out_xsum  = xs;

    memset(ab_out, 0, (size_t)a_bits * (size_t)bytes_per_row);
    if (mx <= 1e-12f) {
        *out_asc = 0.0f;
        *out_asum_u = 0.0f;
        return;
    }

    const float sc = mx / (float)maxq;
    const float inv = 1.0f / sc;
    double dot_xq = 0.0;
    double dot_qq = 0.0;
    int sum_u = 0;
    for (int c = 0; c < block_size; c++) {
        float sv = xb[c] * inv;
        int q = (int)(sv >= 0.0f ? (sv + 0.5f) : (sv - 0.5f));
        if (q > maxq) q = maxq;
        else if (q < -maxq) q = -maxq;
        dot_xq += (double)xb[c] * (double)q;
        dot_qq += (double)q * (double)q;
        int u = q + half;
        sum_u += u;
        const int byte_idx = c >> 3;
        const uint8_t mask = (uint8_t)(1u << (7 - (c & 7)));
        for (int bp = 0; bp < a_bits; bp++) {
            if ((u >> bp) & 1) {
                ab_out[(size_t)bp * (size_t)bytes_per_row + (size_t)byte_idx] |= mask;
            }
        }
    }
    *out_asc = (dot_qq > 0.0) ? (float)(dot_xq / dot_qq) : sc;
    *out_asum_u = (float)sum_u;
}

/* Detect sparse / spiky outlier sub-blocks where top-2 elements carry >90% of L2 energy
 * (e.g. Gate H 1000.0f / +-800.0f with ~0.03f payload, or sparse [1,2,4,8,16,0..] = 93.84%).
 * FWHT-rotated Gemma 4 activations distribute energy across all channels (measured max ~44%),
 * so both k=1 and k>1 deterministically execute identical single-plane Q8 (dual=0, kc_step=4). */
static inline int x_needs_dual_plane_masked(
    const float* x,
    int d_in,
    int block_size,
    int k_tokens,
    const uint8_t* col_blk_mask
) {
    for (int tok = 0; tok < k_tokens; tok++) {
        const float* xk = x + (size_t)tok * (size_t)d_in;
        for (int off = 0; off < d_in; off += 64) {
            if (col_blk_mask && block_size > 0 && col_blk_mask[off / block_size] == 0) {
                continue;
            }
            int len = (d_in - off < 64) ? (d_in - off) : 64;
            float s1 = 0.0f, s2 = 0.0f, sum_sq = 0.0f;
            for (int j = 0; j < len; j++) {
                float a = neon_fabsf(xk[off + j]);
                float q = a * a;
                sum_sq += q;
                if (q >= s1) { s2 = s1; s1 = q; }
                else if (q > s2) { s2 = q; }
            }
            if (sum_sq > 0.0f && (s1 + s2) > 0.90f * sum_sq) {
                return 1;
            }
        }
    }
    return 0;
}

static inline int x_needs_dual_plane(const float* x, int d_in, int k_tokens) {
    return x_needs_dual_plane_masked(x, d_in, 0, k_tokens, NULL);
}

static inline float t16c_plane_weight(int gauss_lut, int eff_bits, int p) {
    static const float GAUSS_W2[2] = { 1.087f, 2.087f };
    static const float GAUSS_W3[3] = { 1.000f, 2.080f, 4.350f };
    if (gauss_lut && eff_bits == 2 && p >= 0 && p < 2) return GAUSS_W2[p];
    if (gauss_lut && eff_bits == 3 && p >= 0 && p < 3) return GAUSS_W3[p];
    return (float)(1u << p);
}

static int gemm_ternary5_tile16_batch(
    const uint8_t* NEON_RESTRICT packed,
    const uint16_t* NEON_RESTRICT tile_norm_fp16,
    const float* NEON_RESTRICT x,
    float* NEON_RESTRICT y,
    int d_out,
    int d_in,
    int block_size,
    int k_tokens,
    float cs_skip_thresh,
    int num_threads,
    int64_t* NEON_RESTRICT out_skipped
) {
    const int n_tiles = (d_out + 15) / 16;
    const int n_blks = d_in / block_size;
    const int quintets = (block_size + 4) / 5;
    const size_t tblk_bytes = (size_t)32 + (size_t)quintets * (size_t)16;
    const int use_cs = (cs_skip_thresh > 0.0f && tile_norm_fp16 != NULL);

#if HAS_NEON && (defined(__aarch64__) || defined(_M_ARM64))
    if (k_tokens <= 0 || d_out <= 0 || n_blks <= 0 || quintets <= 0) {
        if (out_skipped) *out_skipped = 0;
        return 0;
    }

    /* 13 quintets = 65 ternary weights per sub-block -> |sum| <= 65*127 = 8255 (fits int16x8_t cleanly). */
    const int sub_q = 13;
    const int n_subs_blk = (quintets + sub_q - 1) / sub_q;
    const int q_pad = n_subs_blk * sub_q * 5;

    const size_t xq_bytes = (size_t)4 * (size_t)n_blks * (size_t)q_pad;
    const size_t xsc_elems = (size_t)4 * (size_t)n_blks * (size_t)n_subs_blk;
    int8_t* xq = (int8_t*)neon_alloc(xq_bytes);
    float* xsc = (float*)neon_alloc(xsc_elems * sizeof(float));
    float* blk_xnorm = use_cs ? (float*)neon_alloc((size_t)4 * (size_t)n_blks * sizeof(float)) : NULL;
    int32_t* tile_skips = (out_skipped != NULL) ? (int32_t*)neon_alloc((size_t)n_tiles * sizeof(int32_t)) : NULL;

    if (!xq || !xsc || (use_cs && !blk_xnorm) || (out_skipped != NULL && !tile_skips)) {
        neon_free(xq);
        neon_free(xsc);
        neon_free(blk_xnorm);
        neon_free(tile_skips);
        return -1;
    }

    memset(xq, 0, xq_bytes);
    memset(y, 0, (size_t)k_tokens * (size_t)d_out * sizeof(float));

#if HAS_OMP
    if (num_threads > 0) omp_set_num_threads(num_threads);
#else
    (void)num_threads;
#endif

    int64_t total_skipped = 0;
    const int dual = x_needs_dual_plane(x, d_in, k_tokens);
    const int kc_step = dual ? 2 : 4;

    for (int k0 = 0; k0 < k_tokens; k0 += kc_step) {
        const int kc = (k_tokens - k0 >= kc_step) ? kc_step : (k_tokens - k0);
        const int slots = dual ? (kc * 2) : kc;

        for (int kk = 0; kk < kc; kk++) {
            const float* xk = x + (size_t)(k0 + kk) * (size_t)d_in;
            for (int blk = 0; blk < n_blks; blk++) {
                const float* xb = xk + (size_t)blk * (size_t)block_size;
                if (use_cs) {
                    float n2 = 0.0f;
                    for (int i = 0; i < block_size; i++) {
                        float v = xb[i];
                        n2 += v * v;
                    }
                    blk_xnorm[(size_t)kk * (size_t)n_blks + (size_t)blk] = neon_sqrtf(n2);
                }
                for (int sb = 0; sb < n_subs_blk; sb++) {
                    const int s0 = sb * sub_q * 5;
                    int s1 = s0 + sub_q * 5;
                    if (s1 > block_size) s1 = block_size;
                    const int len = s1 - s0;
                    const int pad_len = sub_q * 5;
                    const int slot_hi = dual ? (kk * 2) : kk;
                    int8_t* dst_hi = xq + ((size_t)slot_hi * (size_t)n_blks + (size_t)blk) * (size_t)q_pad + (size_t)s0;
                    float sc_hi = quantize_t5_subblk_i8(xb + s0, len, pad_len, dst_hi);
                    xsc[((size_t)slot_hi * (size_t)n_blks + (size_t)blk) * (size_t)n_subs_blk + (size_t)sb] = sc_hi;
                    if (dual) {
                        float res[65];
                        for (int i = 0; i < len; i++) {
                            res[i] = xb[s0 + i] - sc_hi * (float)dst_hi[i];
                        }
                        const int slot_lo = kk * 2 + 1;
                        int8_t* dst_lo = xq + ((size_t)slot_lo * (size_t)n_blks + (size_t)blk) * (size_t)q_pad + (size_t)s0;
                        float sc_lo = quantize_t5_subblk_i8(res, len, pad_len, dst_lo);
                        xsc[((size_t)slot_lo * (size_t)n_blks + (size_t)blk) * (size_t)n_subs_blk + (size_t)sb] = sc_lo;
                    }
                }
            }
        }

        if (tile_skips) {
            memset(tile_skips, 0, (size_t)n_tiles * sizeof(int32_t));
        }

        int t;
#if HAS_OMP
        #pragma omp parallel for schedule(static)
#endif
        for (t = 0; t < n_tiles; t++) {
            const uint8_t* tptr = packed + (size_t)t * (size_t)n_blks * tblk_bytes;
            const int r_lim = ((d_out - t * 16) < 16) ? (d_out - t * 16) : 16;
            int32_t t_skip = 0;
            int8x16x2_t t27_0, t27_1, t27_2;
            t27_0.val[0] = vld1q_s8(NEON_T5_LUT27_D0); t27_0.val[1] = vld1q_s8(NEON_T5_LUT27_D0 + 16);
            t27_1.val[0] = vld1q_s8(NEON_T5_LUT27_D1); t27_1.val[1] = vld1q_s8(NEON_T5_LUT27_D1 + 16);
            t27_2.val[0] = vld1q_s8(NEON_T5_LUT27_D2); t27_2.val[1] = vld1q_s8(NEON_T5_LUT27_D2 + 16);
            const int8x16_t t9_3 = vld1q_s8(NEON_T5_LUT9_D3);
            const int8x16_t t9_4 = vld1q_s8(NEON_T5_LUT9_D4);

            if (slots == 1) {
                /* Zero-spill Cortex-A72 k=1 decode fast path (<= 25 live Q-registers, register-resident b0..b3). */
                float32x4_t acc00 = vdupq_n_f32(0), acc01 = vdupq_n_f32(0), acc02 = vdupq_n_f32(0), acc03 = vdupq_n_f32(0);
                for (int blk = 0; blk < n_blks; blk++) {
                    if (use_cs) {
                        float xn = blk_xnorm[(size_t)blk];
                        float wnorm = fp16_to_fp32(tile_norm_fp16[(size_t)t * (size_t)n_blks + (size_t)blk]);
                        if (wnorm * xn <= cs_skip_thresh) {
                            t_skip++;
                            continue;
                        }
                    }

                    const uint8_t* bptr = tptr + (size_t)blk * tblk_bytes;
                    if (blk + 1 < n_blks) {
                        const uint8_t* nptr = bptr + tblk_bytes;
                        neon_prefetch(nptr);
                        neon_prefetch(nptr + 256);
                        neon_prefetch(nptr + 512);
                    }
                    float32x4_t vs0, vs1, vs2, vs3;
                    if (r_lim == 16) {
                        uint16_t raw_sc[8];
                        memcpy(raw_sc, bptr + 0, 16);
                        float16x8_t hsc0 = vreinterpretq_f16_u16(vld1q_u16(raw_sc));
                        memcpy(raw_sc, bptr + 16, 16);
                        float16x8_t hsc1 = vreinterpretq_f16_u16(vld1q_u16(raw_sc));
                        vs0 = vcvt_f32_f16(vget_low_f16(hsc0));
                        vs1 = vcvt_f32_f16(vget_high_f16(hsc0));
                        vs2 = vcvt_f32_f16(vget_low_f16(hsc1));
                        vs3 = vcvt_f32_f16(vget_high_f16(hsc1));
                    } else {
                        float sc[16];
                        for (int r = 0; r < r_lim; r++) sc[r] = fp16_to_fp32(read_u16_le(bptr + (size_t)r * 2u));
                        for (int r = r_lim; r < 16; r++) sc[r] = 0.0f;
                        vs0 = vld1q_f32(sc); vs1 = vld1q_f32(sc + 4);
                        vs2 = vld1q_f32(sc + 8); vs3 = vld1q_f32(sc + 12);
                    }

                    const uint8_t* qp = bptr + 32;
                    float32x4_t b0 = vdupq_n_f32(0), b1 = vdupq_n_f32(0), b2 = vdupq_n_f32(0), b3 = vdupq_n_f32(0);

                    for (int sb = 0; sb < n_subs_blk; sb++) {
                        int g_end = (sb + 1) * sub_q;
                        if (g_end > quintets) g_end = quintets;
                        int16x8_t i0L = vdupq_n_s16(0), i0H = vdupq_n_s16(0);
                        const int8_t* x0p = xq + (size_t)blk * (size_t)q_pad + (size_t)sb * (size_t)sub_q * 5u;

                        for (int g = sb * sub_q; g < g_end; g++, qp += 16, x0p += 5) {
                            uint8x16_t bv = vld1q_u8(qp);
                            int8x16_t w0, w1, w2, w3, w4;
                            unpack5_neon_tile16(bv, t27_0, t27_1, t27_2, t9_3, t9_4, &w0, &w1, &w2, &w3, &w4);
                            int8x16_t vx;
                            vx = vdupq_n_s8(x0p[0]); i0L = vmlal_s8(i0L, vget_low_s8(w0), vget_low_s8(vx)); i0H = vmlal_high_s8(i0H, w0, vx);
                            vx = vdupq_n_s8(x0p[1]); i0L = vmlal_s8(i0L, vget_low_s8(w1), vget_low_s8(vx)); i0H = vmlal_high_s8(i0H, w1, vx);
                            vx = vdupq_n_s8(x0p[2]); i0L = vmlal_s8(i0L, vget_low_s8(w2), vget_low_s8(vx)); i0H = vmlal_high_s8(i0H, w2, vx);
                            vx = vdupq_n_s8(x0p[3]); i0L = vmlal_s8(i0L, vget_low_s8(w3), vget_low_s8(vx)); i0H = vmlal_high_s8(i0H, w3, vx);
                            vx = vdupq_n_s8(x0p[4]); i0L = vmlal_s8(i0L, vget_low_s8(w4), vget_low_s8(vx)); i0H = vmlal_high_s8(i0H, w4, vx);
                        }
                        float ss = xsc[(size_t)blk * (size_t)n_subs_blk + (size_t)sb];
                        float32x4_t vss = vdupq_n_f32(ss);
                        b0 = vaddq_f32(b0, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(i0L))), vss));
                        b1 = vaddq_f32(b1, vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(i0L)), vss));
                        b2 = vaddq_f32(b2, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(i0H))), vss));
                        b3 = vaddq_f32(b3, vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(i0H)), vss));
                    }
                    acc00 = vfmaq_f32(acc00, vs0, b0);
                    acc01 = vfmaq_f32(acc01, vs1, b1);
                    acc02 = vfmaq_f32(acc02, vs2, b2);
                    acc03 = vfmaq_f32(acc03, vs3, b3);
                }

                float* y0 = y + (size_t)k0 * (size_t)d_out + (size_t)t * 16u;
                if (r_lim == 16) {
                    vst1q_f32(y0 + 0,  acc00);
                    vst1q_f32(y0 + 4,  acc01);
                    vst1q_f32(y0 + 8,  acc02);
                    vst1q_f32(y0 + 12, acc03);
                } else {
                    float tmp[16];
                    vst1q_f32(tmp, acc00); vst1q_f32(tmp + 4, acc01); vst1q_f32(tmp + 8, acc02); vst1q_f32(tmp + 12, acc03);
                    for (int r = 0; r < r_lim; r++) y0[r] = tmp[r];
                }
                if (tile_skips) tile_skips[t] = t_skip;
                continue;
            }

            if (dual && kc == 1) {
                /* Bit-exact zero-spill Cortex-A72 k=1 dual-plane decode fast path (matches k=4 IEEE-754 rounding). */
                float32x4_t acc00 = vdupq_n_f32(0), acc01 = vdupq_n_f32(0), acc02 = vdupq_n_f32(0), acc03 = vdupq_n_f32(0);
                float32x4_t acc10 = vdupq_n_f32(0), acc11 = vdupq_n_f32(0), acc12 = vdupq_n_f32(0), acc13 = vdupq_n_f32(0);
                for (int blk = 0; blk < n_blks; blk++) {
                    if (use_cs) {
                        float xn = blk_xnorm[(size_t)blk];
                        float wnorm = fp16_to_fp32(tile_norm_fp16[(size_t)t * (size_t)n_blks + (size_t)blk]);
                        if (wnorm * xn <= cs_skip_thresh) {
                            t_skip++;
                            continue;
                        }
                    }

                    const uint8_t* bptr = tptr + (size_t)blk * tblk_bytes;
                    if (blk + 1 < n_blks) {
                        const uint8_t* nptr = bptr + tblk_bytes;
                        neon_prefetch(nptr);
                        neon_prefetch(nptr + 256);
                        neon_prefetch(nptr + 512);
                    }
                    float32x4_t vs0, vs1, vs2, vs3;
                    if (r_lim == 16) {
                        uint16_t raw_sc[8];
                        memcpy(raw_sc, bptr + 0, 16);
                        float16x8_t hsc0 = vreinterpretq_f16_u16(vld1q_u16(raw_sc));
                        memcpy(raw_sc, bptr + 16, 16);
                        float16x8_t hsc1 = vreinterpretq_f16_u16(vld1q_u16(raw_sc));
                        vs0 = vcvt_f32_f16(vget_low_f16(hsc0));
                        vs1 = vcvt_f32_f16(vget_high_f16(hsc0));
                        vs2 = vcvt_f32_f16(vget_low_f16(hsc1));
                        vs3 = vcvt_f32_f16(vget_high_f16(hsc1));
                    } else {
                        float sc[16];
                        for (int r = 0; r < r_lim; r++) sc[r] = fp16_to_fp32(read_u16_le(bptr + (size_t)r * 2u));
                        for (int r = r_lim; r < 16; r++) sc[r] = 0.0f;
                        vs0 = vld1q_f32(sc); vs1 = vld1q_f32(sc + 4);
                        vs2 = vld1q_f32(sc + 8); vs3 = vld1q_f32(sc + 12);
                    }

                    const uint8_t* qp = bptr + 32;
                    float32x4_t b0 = vdupq_n_f32(0), b1 = vdupq_n_f32(0), b2 = vdupq_n_f32(0), b3 = vdupq_n_f32(0);
                    float32x4_t c0 = vdupq_n_f32(0), c1 = vdupq_n_f32(0), c2 = vdupq_n_f32(0), c3 = vdupq_n_f32(0);

                    for (int sb = 0; sb < n_subs_blk; sb++) {
                        int g_end = (sb + 1) * sub_q;
                        if (g_end > quintets) g_end = quintets;
                        int16x8_t i0L = vdupq_n_s16(0), i0H = vdupq_n_s16(0);
                        int16x8_t i1L = vdupq_n_s16(0), i1H = vdupq_n_s16(0);
                        const size_t x_off = (size_t)blk * (size_t)q_pad + (size_t)sb * (size_t)sub_q * 5u;
                        const int8_t* x0p = xq + x_off;
                        const int8_t* x1p = xq + (size_t)n_blks * (size_t)q_pad + x_off;

                        for (int g = sb * sub_q; g < g_end; g++, qp += 16, x0p += 5, x1p += 5) {
                            uint8x16_t bv = vld1q_u8(qp);
                            int8x16_t w[5];
                            unpack5_neon_tile16(bv, t27_0, t27_1, t27_2, t9_3, t9_4, &w[0], &w[1], &w[2], &w[3], &w[4]);
                            for (int p = 0; p < 5; p++) {
                                int8x16_t wp = w[p];
                                int8x8_t wl = vget_low_s8(wp);
                                int8x16_t vx0 = vdupq_n_s8(x0p[p]);
                                int8x16_t vx1 = vdupq_n_s8(x1p[p]);
                                i0L = vmlal_s8(i0L, wl, vget_low_s8(vx0)); i0H = vmlal_high_s8(i0H, wp, vx0);
                                i1L = vmlal_s8(i1L, wl, vget_low_s8(vx1)); i1H = vmlal_high_s8(i1H, wp, vx1);
                            }
                        }
                        float32x4_t vss0 = vdupq_n_f32(xsc[(size_t)blk * (size_t)n_subs_blk + (size_t)sb]);
                        float32x4_t vss1 = vdupq_n_f32(xsc[((size_t)n_blks + (size_t)blk) * (size_t)n_subs_blk + (size_t)sb]);
                        b0 = vaddq_f32(b0, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(i0L))), vss0));
                        b1 = vaddq_f32(b1, vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(i0L)), vss0));
                        b2 = vaddq_f32(b2, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(i0H))), vss0));
                        b3 = vaddq_f32(b3, vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(i0H)), vss0));
                        c0 = vaddq_f32(c0, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(i1L))), vss1));
                        c1 = vaddq_f32(c1, vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(i1L)), vss1));
                        c2 = vaddq_f32(c2, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(i1H))), vss1));
                        c3 = vaddq_f32(c3, vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(i1H)), vss1));
                    }
                    acc00 = vfmaq_f32(acc00, vs0, b0);
                    acc01 = vfmaq_f32(acc01, vs1, b1);
                    acc02 = vfmaq_f32(acc02, vs2, b2);
                    acc03 = vfmaq_f32(acc03, vs3, b3);
                    acc10 = vfmaq_f32(acc10, vs0, c0);
                    acc11 = vfmaq_f32(acc11, vs1, c1);
                    acc12 = vfmaq_f32(acc12, vs2, c2);
                    acc13 = vfmaq_f32(acc13, vs3, c3);
                }

                acc00 = vaddq_f32(acc00, acc10);
                acc01 = vaddq_f32(acc01, acc11);
                acc02 = vaddq_f32(acc02, acc12);
                acc03 = vaddq_f32(acc03, acc13);
                float* y0 = y + (size_t)k0 * (size_t)d_out + (size_t)t * 16u;
                if (r_lim == 16) {
                    vst1q_f32(y0 + 0,  acc00);
                    vst1q_f32(y0 + 4,  acc01);
                    vst1q_f32(y0 + 8,  acc02);
                    vst1q_f32(y0 + 12, acc03);
                } else {
                    float tmp[16];
                    vst1q_f32(tmp, acc00); vst1q_f32(tmp + 4, acc01); vst1q_f32(tmp + 8, acc02); vst1q_f32(tmp + 12, acc03);
                    for (int r = 0; r < r_lim; r++) y0[r] = tmp[r];
                }
                if (tile_skips) tile_skips[t] = t_skip;
                continue;
            }

            float32x4_t acc00 = vdupq_n_f32(0), acc01 = vdupq_n_f32(0), acc02 = vdupq_n_f32(0), acc03 = vdupq_n_f32(0);
            float32x4_t acc10 = vdupq_n_f32(0), acc11 = vdupq_n_f32(0), acc12 = vdupq_n_f32(0), acc13 = vdupq_n_f32(0);
            float32x4_t acc20 = vdupq_n_f32(0), acc21 = vdupq_n_f32(0), acc22 = vdupq_n_f32(0), acc23 = vdupq_n_f32(0);
            float32x4_t acc30 = vdupq_n_f32(0), acc31 = vdupq_n_f32(0), acc32 = vdupq_n_f32(0), acc33 = vdupq_n_f32(0);

            for (int blk = 0; blk < n_blks; blk++) {
                if (use_cs) {
                    float max_xnorm = 0.0f;
                    for (int kk = 0; kk < kc; kk++) {
                        float xn = blk_xnorm[(size_t)kk * (size_t)n_blks + (size_t)blk];
                        if (xn > max_xnorm) max_xnorm = xn;
                    }
                    float wnorm = fp16_to_fp32(tile_norm_fp16[(size_t)t * (size_t)n_blks + (size_t)blk]);
                    if (wnorm * max_xnorm <= cs_skip_thresh) {
                        t_skip++;
                        continue;
                    }
                }

                const uint8_t* bptr = tptr + (size_t)blk * tblk_bytes;
                if (blk + 1 < n_blks) {
                    const uint8_t* nptr = bptr + tblk_bytes;
                    neon_prefetch(nptr);
                    neon_prefetch(nptr + 256);
                    neon_prefetch(nptr + 512);
                }
                float32x4_t vs0, vs1, vs2, vs3;
                if (r_lim == 16) {
                    uint16_t raw_sc[8];
                    memcpy(raw_sc, bptr + 0, 16);
                    float16x8_t hsc0 = vreinterpretq_f16_u16(vld1q_u16(raw_sc));
                    memcpy(raw_sc, bptr + 16, 16);
                    float16x8_t hsc1 = vreinterpretq_f16_u16(vld1q_u16(raw_sc));
                    vs0 = vcvt_f32_f16(vget_low_f16(hsc0));
                    vs1 = vcvt_f32_f16(vget_high_f16(hsc0));
                    vs2 = vcvt_f32_f16(vget_low_f16(hsc1));
                    vs3 = vcvt_f32_f16(vget_high_f16(hsc1));
                } else {
                    float sc[16];
                    for (int r = 0; r < r_lim; r++) {
                        sc[r] = fp16_to_fp32(read_u16_le(bptr + (size_t)r * 2u));
                    }
                    for (int r = r_lim; r < 16; r++) sc[r] = 0.0f;
                    vs0 = vld1q_f32(sc);
                    vs1 = vld1q_f32(sc + 4);
                    vs2 = vld1q_f32(sc + 8);
                    vs3 = vld1q_f32(sc + 12);
                }

                const uint8_t* qp = bptr + 32;
                float bsum[4][16] = {{0}};

                for (int sb = 0; sb < n_subs_blk; sb++) {
                    int g_end = (sb + 1) * sub_q;
                    if (g_end > quintets) g_end = quintets;
                    int16x8_t i0L = vdupq_n_s16(0), i0H = vdupq_n_s16(0);
                    int16x8_t i1L = vdupq_n_s16(0), i1H = vdupq_n_s16(0);
                    int16x8_t i2L = vdupq_n_s16(0), i2H = vdupq_n_s16(0);
                    int16x8_t i3L = vdupq_n_s16(0), i3H = vdupq_n_s16(0);

                    const size_t x_off = (size_t)blk * (size_t)q_pad + (size_t)sb * (size_t)sub_q * 5u;
                    const int8_t* x0p = xq + ((size_t)0 * (size_t)n_blks) * (size_t)q_pad + x_off;
                    const int8_t* x1p = xq + ((size_t)1 * (size_t)n_blks) * (size_t)q_pad + x_off;
                    const int8_t* x2p = xq + ((size_t)2 * (size_t)n_blks) * (size_t)q_pad + x_off;
                    const int8_t* x3p = xq + ((size_t)3 * (size_t)n_blks) * (size_t)q_pad + x_off;

                    for (int g = sb * sub_q; g < g_end; g++, qp += 16, x0p += 5, x1p += 5, x2p += 5, x3p += 5) {
                        uint8x16_t bv = vld1q_u8(qp);
                        int8x16_t w[5];
                        unpack5_neon_tile16(bv, t27_0, t27_1, t27_2, t9_3, t9_4, &w[0], &w[1], &w[2], &w[3], &w[4]);
                        if (slots == 1) {
                            for (int p = 0; p < 5; p++) {
                                int8x16_t wp = w[p];
                                int8x16_t vx0 = vdupq_n_s8(x0p[p]);
                                i0L = vmlal_s8(i0L, vget_low_s8(wp), vget_low_s8(vx0));
                                i0H = vmlal_high_s8(i0H, wp, vx0);
                            }
                        } else {
                            for (int p = 0; p < 5; p++) {
                                int8x16_t wp = w[p];
                                int8x8_t wl = vget_low_s8(wp);
                                int8x16_t vx0 = vdupq_n_s8(x0p[p]);
                                int8x16_t vx1 = vdupq_n_s8(x1p[p]);
                                i0L = vmlal_s8(i0L, wl, vget_low_s8(vx0)); i0H = vmlal_high_s8(i0H, wp, vx0);
                                i1L = vmlal_s8(i1L, wl, vget_low_s8(vx1)); i1H = vmlal_high_s8(i1H, wp, vx1);
                                if (slots > 2) {
                                    int8x16_t vx2 = vdupq_n_s8(x2p[p]);
                                    i2L = vmlal_s8(i2L, wl, vget_low_s8(vx2)); i2H = vmlal_high_s8(i2H, wp, vx2);
                                }
                                if (slots > 3) {
                                    int8x16_t vx3 = vdupq_n_s8(x3p[p]);
                                    i3L = vmlal_s8(i3L, wl, vget_low_s8(vx3)); i3H = vmlal_high_s8(i3H, wp, vx3);
                                }
                            }
                        }
                    }

                    #define T5_FLUSH_SB(sl, iL, iH) do { \
                        float ss = xsc[((size_t)(sl) * (size_t)n_blks + (size_t)blk) * (size_t)n_subs_blk + (size_t)sb]; \
                        float32x4_t vss = vdupq_n_f32(ss); \
                        float32x4_t f0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(iL))), vss); \
                        float32x4_t f1 = vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(iL)), vss); \
                        float32x4_t f2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(iH))), vss); \
                        float32x4_t f3 = vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(iH)), vss); \
                        vst1q_f32(bsum[sl],      vaddq_f32(vld1q_f32(bsum[sl]),      f0)); \
                        vst1q_f32(bsum[sl] + 4,  vaddq_f32(vld1q_f32(bsum[sl] + 4),  f1)); \
                        vst1q_f32(bsum[sl] + 8,  vaddq_f32(vld1q_f32(bsum[sl] + 8),  f2)); \
                        vst1q_f32(bsum[sl] + 12, vaddq_f32(vld1q_f32(bsum[sl] + 12), f3)); \
                    } while (0)
                    T5_FLUSH_SB(0, i0L, i0H);
                    if (slots > 1) T5_FLUSH_SB(1, i1L, i1H);
                    if (slots > 2) T5_FLUSH_SB(2, i2L, i2H);
                    if (slots > 3) T5_FLUSH_SB(3, i3L, i3H);
                    #undef T5_FLUSH_SB
                }

                acc00 = vfmaq_f32(acc00, vs0, vld1q_f32(bsum[0]));
                acc01 = vfmaq_f32(acc01, vs1, vld1q_f32(bsum[0] + 4));
                acc02 = vfmaq_f32(acc02, vs2, vld1q_f32(bsum[0] + 8));
                acc03 = vfmaq_f32(acc03, vs3, vld1q_f32(bsum[0] + 12));
                if (slots > 1) {
                    acc10 = vfmaq_f32(acc10, vs0, vld1q_f32(bsum[1]));
                    acc11 = vfmaq_f32(acc11, vs1, vld1q_f32(bsum[1] + 4));
                    acc12 = vfmaq_f32(acc12, vs2, vld1q_f32(bsum[1] + 8));
                    acc13 = vfmaq_f32(acc13, vs3, vld1q_f32(bsum[1] + 12));
                }
                if (slots > 2) {
                    acc20 = vfmaq_f32(acc20, vs0, vld1q_f32(bsum[2]));
                    acc21 = vfmaq_f32(acc21, vs1, vld1q_f32(bsum[2] + 4));
                    acc22 = vfmaq_f32(acc22, vs2, vld1q_f32(bsum[2] + 8));
                    acc23 = vfmaq_f32(acc23, vs3, vld1q_f32(bsum[2] + 12));
                }
                if (slots > 3) {
                    acc30 = vfmaq_f32(acc30, vs0, vld1q_f32(bsum[3]));
                    acc31 = vfmaq_f32(acc31, vs1, vld1q_f32(bsum[3] + 4));
                    acc32 = vfmaq_f32(acc32, vs2, vld1q_f32(bsum[3] + 8));
                    acc33 = vfmaq_f32(acc33, vs3, vld1q_f32(bsum[3] + 12));
                }
            }

            float tmp[4][16];
            vst1q_f32(tmp[0], acc00); vst1q_f32(tmp[0] + 4, acc01); vst1q_f32(tmp[0] + 8, acc02); vst1q_f32(tmp[0] + 12, acc03);
            if (slots > 1) {
                vst1q_f32(tmp[1], acc10); vst1q_f32(tmp[1] + 4, acc11); vst1q_f32(tmp[1] + 8, acc12); vst1q_f32(tmp[1] + 12, acc13);
            }
            if (slots > 2) {
                vst1q_f32(tmp[2], acc20); vst1q_f32(tmp[2] + 4, acc21); vst1q_f32(tmp[2] + 8, acc22); vst1q_f32(tmp[2] + 12, acc23);
            }
            if (slots > 3) {
                vst1q_f32(tmp[3], acc30); vst1q_f32(tmp[3] + 4, acc31); vst1q_f32(tmp[3] + 8, acc32); vst1q_f32(tmp[3] + 12, acc33);
            }
            if (dual) {
                for (int kk = 0; kk < kc; kk++) {
                    float* yk = y + (size_t)(k0 + kk) * (size_t)d_out + (size_t)t * 16u;
                    const float* t_hi = tmp[kk * 2];
                    const float* t_lo = tmp[kk * 2 + 1];
                    for (int r = 0; r < r_lim; r++) {
                        yk[r] += t_hi[r] + t_lo[r];
                    }
                }
            } else {
                for (int kk = 0; kk < kc; kk++) {
                    float* yk = y + (size_t)(k0 + kk) * (size_t)d_out + (size_t)t * 16u;
                    for (int r = 0; r < r_lim; r++) {
                        yk[r] += tmp[kk][r];
                    }
                }
            }
            if (tile_skips) tile_skips[t] = t_skip;
        }

        if (tile_skips) {
            for (int ti = 0; ti < n_tiles; ti++) total_skipped += tile_skips[ti];
        }
    }

    if (out_skipped) *out_skipped = total_skipped;
    neon_free(xq);
    neon_free(xsc);
    neon_free(blk_xnorm);
    neon_free(tile_skips);
    return 0;
#else
    /* Keep L2 LUT budget <= 8 * 52 * 256 * 4 B = 426 KB across kc <= 4 tokens. */
    const size_t max_lut_elems = (size_t)8 * (size_t)quintets * (size_t)256;
    float* luts = (float*)neon_alloc(max_lut_elems * sizeof(float));
    float* blk_xnorm = use_cs ? (float*)neon_alloc((size_t)4 * (size_t)n_blks * sizeof(float)) : NULL;
    int32_t* tile_skips = (out_skipped != NULL) ? (int32_t*)neon_alloc((size_t)n_tiles * sizeof(int32_t)) : NULL;

    if (!luts || (use_cs && !blk_xnorm) || (out_skipped != NULL && !tile_skips)) {
        neon_free(luts);
        neon_free(blk_xnorm);
        neon_free(tile_skips);
        return -1;
    }

    memset(y, 0, (size_t)k_tokens * (size_t)d_out * sizeof(float));

#if HAS_OMP
    if (num_threads > 0) omp_set_num_threads(num_threads);
#else
    (void)num_threads;
#endif

    int64_t total_skipped = 0;

    for (int k0 = 0; k0 < k_tokens; k0 += 4) {
        const int kc = (k_tokens - k0 >= 4) ? 4 : (k_tokens - k0);
        const int b_chunk = (kc <= 2) ? 4 : 2;

        if (use_cs) {
            for (int kk = 0; kk < kc; kk++) {
                const float* xk = x + (size_t)(k0 + kk) * (size_t)d_in;
                for (int blk = 0; blk < n_blks; blk++) {
                    const float* xb = xk + (size_t)blk * (size_t)block_size;
                    float n2 = 0.0f;
                    for (int i = 0; i < block_size; i++) {
                        float v = xb[i];
                        n2 += v * v;
                    }
                    blk_xnorm[(size_t)kk * (size_t)n_blks + (size_t)blk] = neon_sqrtf(n2);
                }
            }
        }

        for (int blk0 = 0; blk0 < n_blks; blk0 += b_chunk) {
            const int b_cnt = (n_blks - blk0 < b_chunk) ? (n_blks - blk0) : b_chunk;
            for (int kk = 0; kk < kc; kk++) {
                const float* xk = x + (size_t)(k0 + kk) * (size_t)d_in;
                for (int bi = 0; bi < b_cnt; bi++) {
                    const float* xblk = xk + (size_t)(blk0 + bi) * (size_t)block_size;
                    float* dst_b = luts + ((size_t)kk * (size_t)b_cnt + (size_t)bi) * (size_t)quintets * (size_t)256;
                    for (int g = 0; g < quintets; g++) {
                        const int base = g * 5;
                        const float x0 = (base + 0 < block_size) ? xblk[base + 0] : 0.0f;
                        const float x1 = (base + 1 < block_size) ? xblk[base + 1] : 0.0f;
                        const float x2 = (base + 2 < block_size) ? xblk[base + 2] : 0.0f;
                        const float x3 = (base + 3 < block_size) ? xblk[base + 3] : 0.0f;
                        const float x4 = (base + 4 < block_size) ? xblk[base + 4] : 0.0f;
                        build_tern5_lut_f32(x0, x1, x2, x3, x4, dst_b + (size_t)g * 256u);
                    }
                }
            }

            if (tile_skips) {
                memset(tile_skips, 0, (size_t)n_tiles * sizeof(int32_t));
            }

            int t;
#if HAS_OMP
            #pragma omp parallel for schedule(static)
#endif
            for (t = 0; t < n_tiles; t++) {
                const uint8_t* tptr = packed + (size_t)t * (size_t)n_blks * tblk_bytes;
                const int r_lim = ((d_out - t * 16) < 16) ? (d_out - t * 16) : 16;
                int32_t t_skip = 0;
                float acc[4][16];
                for (int kk = 0; kk < kc; kk++) {
                    for (int r = 0; r < 16; r++) acc[kk][r] = 0.0f;
                }

                for (int bi = 0; bi < b_cnt; bi++) {
                    const int blk = blk0 + bi;
                    if (use_cs) {
                        float max_xnorm = 0.0f;
                        for (int kk = 0; kk < kc; kk++) {
                            float xn = blk_xnorm[(size_t)kk * (size_t)n_blks + (size_t)blk];
                            if (xn > max_xnorm) max_xnorm = xn;
                        }
                        float wnorm = fp16_to_fp32(tile_norm_fp16[(size_t)t * (size_t)n_blks + (size_t)blk]);
                        if (wnorm * max_xnorm <= cs_skip_thresh) {
                            t_skip++;
                            continue;
                        }
                    }

                    const uint8_t* bptr = tptr + (size_t)blk * tblk_bytes;
                    float sc[16];
                    for (int r = 0; r < r_lim; r++) {
                        sc[r] = fp16_to_fp32(read_u16_le(bptr + (size_t)r * 2u));
                    }
                    for (int r = r_lim; r < 16; r++) {
                        sc[r] = 0.0f;
                    }

                    if (kc == 1) {
                        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                        float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;
                        float s8 = 0.0f, s9 = 0.0f, s10 = 0.0f, s11 = 0.0f;
                        float s12 = 0.0f, s13 = 0.0f, s14 = 0.0f, s15 = 0.0f;
                        const uint8_t* qp = bptr + 32;
                        const float* lg = luts + (size_t)bi * (size_t)quintets * (size_t)256;
                        for (int g = 0; g < quintets; g++, qp += 16, lg += 256) {
                            s0  += lg[qp[0]];  s1  += lg[qp[1]];  s2  += lg[qp[2]];  s3  += lg[qp[3]];
                            s4  += lg[qp[4]];  s5  += lg[qp[5]];  s6  += lg[qp[6]];  s7  += lg[qp[7]];
                            s8  += lg[qp[8]];  s9  += lg[qp[9]];  s10 += lg[qp[10]]; s11 += lg[qp[11]];
                            s12 += lg[qp[12]]; s13 += lg[qp[13]]; s14 += lg[qp[14]]; s15 += lg[qp[15]];
                        }
                        acc[0][0]  += sc[0]  * s0;  acc[0][1]  += sc[1]  * s1;
                        acc[0][2]  += sc[2]  * s2;  acc[0][3]  += sc[3]  * s3;
                        acc[0][4]  += sc[4]  * s4;  acc[0][5]  += sc[5]  * s5;
                        acc[0][6]  += sc[6]  * s6;  acc[0][7]  += sc[7]  * s7;
                        acc[0][8]  += sc[8]  * s8;  acc[0][9]  += sc[9]  * s9;
                        acc[0][10] += sc[10] * s10; acc[0][11] += sc[11] * s11;
                        acc[0][12] += sc[12] * s12; acc[0][13] += sc[13] * s13;
                        acc[0][14] += sc[14] * s14; acc[0][15] += sc[15] * s15;
                    } else {
                        float bs[4][16];
                        for (int kk = 0; kk < kc; kk++) {
                            for (int r = 0; r < 16; r++) bs[kk][r] = 0.0f;
                        }
                        const uint8_t* qp = bptr + 32;
                        for (int g = 0; g < quintets; g++, qp += 16) {
                            const uint8_t c0 = qp[0],  c1 = qp[1],  c2 = qp[2],  c3 = qp[3];
                            const uint8_t c4 = qp[4],  c5 = qp[5],  c6 = qp[6],  c7 = qp[7];
                            const uint8_t c8 = qp[8],  c9 = qp[9],  c10 = qp[10], c11 = qp[11];
                            const uint8_t c12 = qp[12], c13 = qp[13], c14 = qp[14], c15 = qp[15];
                            for (int kk = 0; kk < kc; kk++) {
                                const float* lg = luts + (((size_t)kk * (size_t)b_cnt + (size_t)bi) * (size_t)quintets + (size_t)g) * (size_t)256;
                                bs[kk][0]  += lg[c0];  bs[kk][1]  += lg[c1];  bs[kk][2]  += lg[c2];  bs[kk][3]  += lg[c3];
                                bs[kk][4]  += lg[c4];  bs[kk][5]  += lg[c5];  bs[kk][6]  += lg[c6];  bs[kk][7]  += lg[c7];
                                bs[kk][8]  += lg[c8];  bs[kk][9]  += lg[c9];  bs[kk][10] += lg[c10]; bs[kk][11] += lg[c11];
                                bs[kk][12] += lg[c12]; bs[kk][13] += lg[c13]; bs[kk][14] += lg[c14]; bs[kk][15] += lg[c15];
                            }
                        }
                        for (int kk = 0; kk < kc; kk++) {
                            for (int r = 0; r < r_lim; r++) {
                                acc[kk][r] += sc[r] * bs[kk][r];
                            }
                        }
                    }
                }

                for (int kk = 0; kk < kc; kk++) {
                    float* yk = y + (size_t)(k0 + kk) * (size_t)d_out + (size_t)t * 16;
                    for (int r = 0; r < r_lim; r++) {
                        yk[r] += acc[kk][r];
                    }
                }
                if (tile_skips) tile_skips[t] = t_skip;
            }

            if (tile_skips) {
                for (int ti = 0; ti < n_tiles; ti++) total_skipped += tile_skips[ti];
            }
        }
    }

    if (out_skipped) *out_skipped = total_skipped;
    neon_free(luts);
    neon_free(blk_xnorm);
    neon_free(tile_skips);
    return 0;
#endif
}

/* ============================================================================
 * Packed Tile-16 Symmetric Signed Int8 (n_bits=8) GEMV/GEMM Kernel
 * ============================================================================
 * Tile block layout: 32 B (16 x FP16 per-row scales) + block_size * 16 B int8
 * weights stored in [col][row=0..15] order. On ARMv8 NEON without SDOT,
 * accumulating two columns at a time into int16x8 guarantees |sum| <= 2*127*127
 * = 32,258 < 32,767 (zero int16 overflow) before widening to int32x4.
 */
static int gemm_int8_tile16_batch(
    const uint8_t* NEON_RESTRICT packed,
    const uint16_t* NEON_RESTRICT tile_norm_fp16,
    const float* NEON_RESTRICT x,
    float* NEON_RESTRICT y,
    int d_out,
    int d_in,
    int block_size,
    int k_tokens,
    float cs_skip_thresh,
    int num_threads,
    int64_t* NEON_RESTRICT out_skipped
) {
    const int n_tiles = (d_out + 15) / 16;
    const int n_blks = d_in / block_size;
    const size_t tblk_bytes = (size_t)32 + (size_t)block_size * 16u;
    const int use_cs = (cs_skip_thresh > 0.0f && tile_norm_fp16 != NULL) ? 1 : 0;

    if (k_tokens <= 0 || d_out <= 0 || n_blks <= 0) {
        if (out_skipped) *out_skipped = 0;
        return 0;
    }

    memset(y, 0, (size_t)k_tokens * (size_t)d_out * sizeof(float));

#if HAS_OMP
    if (num_threads > 0) omp_set_num_threads(num_threads);
#else
    (void)num_threads;
#endif

    int64_t total_skipped = 0;
    int32_t* tile_skips = (out_skipped != NULL) ? (int32_t*)neon_alloc((size_t)n_tiles * sizeof(int32_t)) : NULL;
    if (out_skipped != NULL && !tile_skips) {
        return -1;
    }

#if HAS_NEON && (defined(__aarch64__) || defined(_M_ARM64))
    int8_t* xq = (int8_t*)neon_alloc((size_t)4 * (size_t)d_in);
    float* xsc = (float*)neon_alloc((size_t)4 * (size_t)n_blks * sizeof(float));
    float* blk_xnorm = use_cs ? (float*)neon_alloc((size_t)4 * (size_t)n_blks * sizeof(float)) : NULL;

    if (!xq || !xsc || (use_cs && !blk_xnorm)) {
        neon_free(xq);
        neon_free(xsc);
        neon_free(blk_xnorm);
        neon_free(tile_skips);
        return -1;
    }

    const int dual = x_needs_dual_plane(x, d_in, k_tokens);
    const int kc_step = dual ? 2 : 4;

    for (int k0 = 0; k0 < k_tokens; k0 += kc_step) {
        const int kc = (k_tokens - k0 >= kc_step) ? kc_step : (k_tokens - k0);
        const int slots = dual ? (kc * 2) : kc;

        for (int kk = 0; kk < kc; kk++) {
            const float* xk = x + (size_t)(k0 + kk) * (size_t)d_in;
            for (int blk = 0; blk < n_blks; blk++) {
                const float* xb = xk + (size_t)blk * (size_t)block_size;
                if (use_cs) {
                    float n2 = 0.0f;
                    for (int i = 0; i < block_size; i++) {
                        float v = xb[i];
                        n2 += v * v;
                    }
                    blk_xnorm[(size_t)kk * (size_t)n_blks + (size_t)blk] = neon_sqrtf(n2);
                }
                const int slot_hi = dual ? (kk * 2) : kk;
                int8_t* dst_hi = xq + (size_t)slot_hi * (size_t)d_in + (size_t)blk * (size_t)block_size;
                float sc_hi = quantize_t5_subblk_i8(xb, block_size, block_size, dst_hi);
                xsc[(size_t)slot_hi * (size_t)n_blks + (size_t)blk] = sc_hi;
                if (dual) {
                    float res[256];
                    for (int i = 0; i < block_size; i++) {
                        res[i] = xb[i] - sc_hi * (float)dst_hi[i];
                    }
                    const int slot_lo = kk * 2 + 1;
                    int8_t* dst_lo = xq + (size_t)slot_lo * (size_t)d_in + (size_t)blk * (size_t)block_size;
                    float sc_lo = quantize_t5_subblk_i8(res, block_size, block_size, dst_lo);
                    xsc[(size_t)slot_lo * (size_t)n_blks + (size_t)blk] = sc_lo;
                }
            }
        }

        if (tile_skips) {
            memset(tile_skips, 0, (size_t)n_tiles * sizeof(int32_t));
        }

        int t;
#if HAS_OMP
        #pragma omp parallel for schedule(static)
#endif
        for (t = 0; t < n_tiles; t++) {
            const uint8_t* tptr = packed + (size_t)t * (size_t)n_blks * tblk_bytes;
            const int r_lim = ((d_out - t * 16) < 16) ? (d_out - t * 16) : 16;
            int32_t t_skip = 0;
            float acc[4][16];
            for (int sl = 0; sl < slots; sl++) {
                for (int r = 0; r < 16; r++) acc[sl][r] = 0.0f;
            }

            for (int blk = 0; blk < n_blks; blk++) {
                if (use_cs) {
                    float max_xnorm = 0.0f;
                    for (int kk = 0; kk < kc; kk++) {
                        float xn = blk_xnorm[(size_t)kk * (size_t)n_blks + (size_t)blk];
                        if (xn > max_xnorm) max_xnorm = xn;
                    }
                    float wnorm = fp16_to_fp32(tile_norm_fp16[(size_t)t * (size_t)n_blks + (size_t)blk]);
                    if (wnorm * max_xnorm <= cs_skip_thresh) {
                        t_skip++;
                        continue;
                    }
                }

                const uint8_t* bptr = tptr + (size_t)blk * tblk_bytes;
                if (blk + 2 < n_blks) {
                    neon_prefetch(tptr + ((size_t)blk + 2) * tblk_bytes);
                }
                float sc[16];
                for (int r = 0; r < r_lim; r++) {
                    sc[r] = fp16_to_fp32(read_u16_le(bptr + (size_t)r * 2u));
                }
                for (int r = r_lim; r < 16; r++) sc[r] = 0.0f;
                const float32x4_t vs0 = vld1q_f32(sc);
                const float32x4_t vs1 = vld1q_f32(sc + 4);
                const float32x4_t vs2 = vld1q_f32(sc + 8);
                const float32x4_t vs3 = vld1q_f32(sc + 12);
                const int8_t* wp = (const int8_t*)(bptr + 32);

                if (slots == 2) {
                    const int8_t* x0p = xq + (size_t)0 * (size_t)d_in + (size_t)blk * (size_t)block_size;
                    const int8_t* x1p = xq + (size_t)1 * (size_t)d_in + (size_t)blk * (size_t)block_size;
                    int32x4_t d00 = vdupq_n_s32(0), d01 = d00, d02 = d00, d03 = d00;
                    int32x4_t d10 = vdupq_n_s32(0), d11 = d10, d12 = d10, d13 = d10;
                    for (int c = 0; c < block_size; c += 2, wp += 32) {
                        int8x16_t w0 = vld1q_s8(wp);
                        int8x16_t w1 = vld1q_s8(wp + 16);
                        int8x8_t w0l = vget_low_s8(w0), w0h = vget_high_s8(w0);
                        int8x8_t w1l = vget_low_s8(w1), w1h = vget_high_s8(w1);

                        int8x8_t x0a = vdup_n_s8(x0p[c]), x0b = vdup_n_s8(x0p[c + 1]);
                        int16x8_t l0 = vmlal_s8(vmull_s8(w0l, x0a), w1l, x0b);
                        int16x8_t h0 = vmlal_s8(vmull_s8(w0h, x0a), w1h, x0b);
                        d00 = vaddw_s16(d00, vget_low_s16(l0));
                        d01 = vaddw_s16(d01, vget_high_s16(l0));
                        d02 = vaddw_s16(d02, vget_low_s16(h0));
                        d03 = vaddw_s16(d03, vget_high_s16(h0));

                        int8x8_t x1a = vdup_n_s8(x1p[c]), x1b = vdup_n_s8(x1p[c + 1]);
                        int16x8_t l1 = vmlal_s8(vmull_s8(w0l, x1a), w1l, x1b);
                        int16x8_t h1 = vmlal_s8(vmull_s8(w0h, x1a), w1h, x1b);
                        d10 = vaddw_s16(d10, vget_low_s16(l1));
                        d11 = vaddw_s16(d11, vget_high_s16(l1));
                        d12 = vaddw_s16(d12, vget_low_s16(h1));
                        d13 = vaddw_s16(d13, vget_high_s16(h1));
                    }
                    const float32x4_t m0 = vdupq_n_f32(xsc[(size_t)0 * (size_t)n_blks + (size_t)blk]);
                    vst1q_f32(acc[0],      vfmaq_f32(vld1q_f32(acc[0]),      vmulq_f32(vs0, m0), vcvtq_f32_s32(d00)));
                    vst1q_f32(acc[0] + 4,  vfmaq_f32(vld1q_f32(acc[0] + 4),  vmulq_f32(vs1, m0), vcvtq_f32_s32(d01)));
                    vst1q_f32(acc[0] + 8,  vfmaq_f32(vld1q_f32(acc[0] + 8),  vmulq_f32(vs2, m0), vcvtq_f32_s32(d02)));
                    vst1q_f32(acc[0] + 12, vfmaq_f32(vld1q_f32(acc[0] + 12), vmulq_f32(vs3, m0), vcvtq_f32_s32(d03)));
                    const float32x4_t m1 = vdupq_n_f32(xsc[(size_t)1 * (size_t)n_blks + (size_t)blk]);
                    vst1q_f32(acc[1],      vfmaq_f32(vld1q_f32(acc[1]),      vmulq_f32(vs0, m1), vcvtq_f32_s32(d10)));
                    vst1q_f32(acc[1] + 4,  vfmaq_f32(vld1q_f32(acc[1] + 4),  vmulq_f32(vs1, m1), vcvtq_f32_s32(d11)));
                    vst1q_f32(acc[1] + 8,  vfmaq_f32(vld1q_f32(acc[1] + 8),  vmulq_f32(vs2, m1), vcvtq_f32_s32(d12)));
                    vst1q_f32(acc[1] + 12, vfmaq_f32(vld1q_f32(acc[1] + 12), vmulq_f32(vs3, m1), vcvtq_f32_s32(d13)));
                } else {
                    int32x4_t d[4][4];
                    for (int sl = 0; sl < slots; sl++) {
                        d[sl][0] = vdupq_n_s32(0);
                        d[sl][1] = vdupq_n_s32(0);
                        d[sl][2] = vdupq_n_s32(0);
                        d[sl][3] = vdupq_n_s32(0);
                    }
                    const int8_t* xp[4];
                    for (int sl = 0; sl < slots; sl++) {
                        xp[sl] = xq + (size_t)sl * (size_t)d_in + (size_t)blk * (size_t)block_size;
                    }
                    for (int c = 0; c < block_size; c += 2, wp += 32) {
                        int8x16_t w0 = vld1q_s8(wp);
                        int8x16_t w1 = vld1q_s8(wp + 16);
                        int8x8_t w0l = vget_low_s8(w0), w0h = vget_high_s8(w0);
                        int8x8_t w1l = vget_low_s8(w1), w1h = vget_high_s8(w1);
                        for (int sl = 0; sl < slots; sl++) {
                            int8x8_t xa = vdup_n_s8(xp[sl][c]), xb_v = vdup_n_s8(xp[sl][c + 1]);
                            int16x8_t l = vmlal_s8(vmull_s8(w0l, xa), w1l, xb_v);
                            int16x8_t h = vmlal_s8(vmull_s8(w0h, xa), w1h, xb_v);
                            d[sl][0] = vaddw_s16(d[sl][0], vget_low_s16(l));
                            d[sl][1] = vaddw_s16(d[sl][1], vget_high_s16(l));
                            d[sl][2] = vaddw_s16(d[sl][2], vget_low_s16(h));
                            d[sl][3] = vaddw_s16(d[sl][3], vget_high_s16(h));
                        }
                    }
                    for (int sl = 0; sl < slots; sl++) {
                        const float32x4_t m = vdupq_n_f32(xsc[(size_t)sl * (size_t)n_blks + (size_t)blk]);
                        vst1q_f32(acc[sl],      vfmaq_f32(vld1q_f32(acc[sl]),      vmulq_f32(vs0, m), vcvtq_f32_s32(d[sl][0])));
                        vst1q_f32(acc[sl] + 4,  vfmaq_f32(vld1q_f32(acc[sl] + 4),  vmulq_f32(vs1, m), vcvtq_f32_s32(d[sl][1])));
                        vst1q_f32(acc[sl] + 8,  vfmaq_f32(vld1q_f32(acc[sl] + 8),  vmulq_f32(vs2, m), vcvtq_f32_s32(d[sl][2])));
                        vst1q_f32(acc[sl] + 12, vfmaq_f32(vld1q_f32(acc[sl] + 12), vmulq_f32(vs3, m), vcvtq_f32_s32(d[sl][3])));
                    }
                }
            }

            if (dual) {
                for (int kk = 0; kk < kc; kk++) {
                    float* yk = y + (size_t)(k0 + kk) * (size_t)d_out + (size_t)t * 16u;
                    const float* t_hi = acc[kk * 2];
                    const float* t_lo = acc[kk * 2 + 1];
                    for (int r = 0; r < r_lim; r++) {
                        yk[r] = t_hi[r] + t_lo[r];
                    }
                }
            } else {
                for (int kk = 0; kk < kc; kk++) {
                    float* yk = y + (size_t)(k0 + kk) * (size_t)d_out + (size_t)t * 16u;
                    for (int r = 0; r < r_lim; r++) {
                        yk[r] = acc[kk][r];
                    }
                }
            }
            if (tile_skips) tile_skips[t] = t_skip;
        }

        if (tile_skips) {
            for (int ti = 0; ti < n_tiles; ti++) total_skipped += tile_skips[ti];
        }
    }

    neon_free(xq);
    neon_free(xsc);
    neon_free(blk_xnorm);
    neon_free(tile_skips);
    if (out_skipped) *out_skipped = total_skipped;
    return 0;
#else
    int t;
#if HAS_OMP
    #pragma omp parallel for schedule(static)
#endif
    for (t = 0; t < n_tiles; t++) {
        const uint8_t* tptr = packed + (size_t)t * (size_t)n_blks * tblk_bytes;
        const int r_lim = ((d_out - t * 16) < 16) ? (d_out - t * 16) : 16;
        int32_t t_skip = 0;
        for (int k0 = 0; k0 < k_tokens; k0 += 4) {
            const int kc = (k_tokens - k0 >= 4) ? 4 : (k_tokens - k0);
            double acc[4][16];
            for (int kk = 0; kk < kc; kk++) {
                for (int r = 0; r < 16; r++) acc[kk][r] = 0.0;
            }
            for (int blk = 0; blk < n_blks; blk++) {
                if (use_cs) {
                    float max_xnorm = 0.0f;
                    for (int kk = 0; kk < kc; kk++) {
                        const float* xb = x + (size_t)(k0 + kk) * (size_t)d_in + (size_t)blk * (size_t)block_size;
                        float n2 = 0.0f;
                        for (int i = 0; i < block_size; i++) {
                            float v = xb[i];
                            n2 += v * v;
                        }
                        float xn = neon_sqrtf(n2);
                        if (xn > max_xnorm) max_xnorm = xn;
                    }
                    float wnorm = fp16_to_fp32(tile_norm_fp16[(size_t)t * (size_t)n_blks + (size_t)blk]);
                    if (wnorm * max_xnorm <= cs_skip_thresh) {
                        t_skip++;
                        continue;
                    }
                }
                const uint8_t* bptr = tptr + (size_t)blk * tblk_bytes;
                double sc[16];
                for (int r = 0; r < r_lim; r++) {
                    sc[r] = (double)fp16_to_fp32(read_u16_le(bptr + (size_t)r * 2u));
                }
                const int8_t* wp = (const int8_t*)(bptr + 32);
                for (int kk = 0; kk < kc; kk++) {
                    const float* xb = x + (size_t)(k0 + kk) * (size_t)d_in + (size_t)blk * (size_t)block_size;
                    double bsum[16] = {0};
                    for (int c = 0; c < block_size; c++) {
                        const int8_t* wr = wp + (size_t)c * 16u;
                        double xv = (double)xb[c];
                        for (int r = 0; r < r_lim; r++) bsum[r] += (double)wr[r] * xv;
                    }
                    for (int r = 0; r < r_lim; r++) acc[kk][r] += sc[r] * bsum[r];
                }
            }
            for (int kk = 0; kk < kc; kk++) {
                float* yk = y + (size_t)(k0 + kk) * (size_t)d_out + (size_t)t * 16u;
                for (int r = 0; r < r_lim; r++) {
                    yk[r] = (float)acc[kk][r];
                }
            }
        }
        if (tile_skips) tile_skips[t] = t_skip;
    }

    if (tile_skips) {
        for (int ti = 0; ti < n_tiles; ti++) total_skipped += tile_skips[ti];
        neon_free(tile_skips);
    }
    if (out_skipped) *out_skipped = total_skipped;
    return 0;
#endif
}

#if !defined(NEON_MSVC_FREESTANDING_DLL)
  #if defined(_MSC_VER)
    #define NEON_TLS __declspec(thread)
  #else
    #define NEON_TLS __thread
  #endif
typedef struct {
    uint8_t* ptr;
    size_t cap;
} neon_tile16_tls_buf_t;
static NEON_TLS neon_tile16_tls_buf_t g_tile16_tls_scratch = { NULL, 0 };
#endif

static inline size_t neon_align64_sz(size_t n) {
    return (n + 63u) & ~(size_t)63u;
}

static int gemm_bitserial_rowmajor_andvcnt_sparse_batch(
    const uint8_t* NEON_RESTRICT packed,
    const uint16_t* NEON_RESTRICT tile_norm_fp16,
    const float* NEON_RESTRICT x,
    float* NEON_RESTRICT y,
    int d_out,
    int d_in,
    int block_size,
    int n_bits,
    int is_asym,
    int k_tokens,
    float cs_skip_thresh,
    const uint8_t* NEON_RESTRICT row_tile_mask,
    const uint8_t* NEON_RESTRICT col_blk_mask,
    int num_threads,
    int64_t* NEON_RESTRICT out_skipped
) {
    const int gauss_lut = (n_bits & T16C_GAUSS_BIT) != 0;
    const int eff_bits = n_bits & 0x3F;
    if (eff_bits < 1 || eff_bits > 4) return -1;
    if (gauss_lut && eff_bits != 2 && eff_bits != 3) return -1;

    int a_bits = g_andvcnt_abits;
    if (a_bits < 4) a_bits = 4;
    if (a_bits > 6) a_bits = 6;
    const float half_f = (float)(1 << (a_bits - 1));
    const float bsize_f = (float)block_size;

    const int n_tiles = (d_out + 15) / 16;
    const int n_blks = d_in / block_size;
    const int hdr_bytes = is_asym ? 64 : 32;
    const int bytes_per_row = block_size / 8;
    const int bytes_per_plane_blk = bytes_per_row * 16;
    const size_t tile_blk_bytes = (size_t)hdr_bytes + (size_t)eff_bits * (size_t)bytes_per_plane_blk;
    const int kc_step = 8;
    const int kc_max = (k_tokens < kc_step) ? k_tokens : kc_step;
    const int need_norm = (cs_skip_thresh > 0.0f && tile_norm_fp16 != NULL) ? 1 : 0;
    const int need_sum = (is_asym != 0) ? 1 : 0;
    const size_t a_blk_bytes = (size_t)a_bits * (size_t)bytes_per_row;

    const size_t sz_ab    = neon_align64_sz((size_t)kc_max * (size_t)n_blks * a_blk_bytes);
    const size_t sz_asc   = neon_align64_sz((size_t)kc_max * (size_t)n_blks * sizeof(float));
    const size_t sz_asum  = neon_align64_sz((size_t)kc_max * (size_t)n_blks * sizeof(float));
    const size_t sz_xnorm = need_norm ? neon_align64_sz((size_t)kc_max * (size_t)n_blks * sizeof(float)) : 0u;
    const size_t sz_xsum  = need_sum ? neon_align64_sz((size_t)kc_max * (size_t)n_blks * sizeof(float)) : 0u;
    const size_t sz_tsk   = (out_skipped != NULL) ? neon_align64_sz((size_t)n_tiles * sizeof(int32_t)) : 0u;
    const size_t sz_ablks = neon_align64_sz((size_t)n_blks * sizeof(int32_t));
    const size_t sz_atile = (row_tile_mask != NULL) ? neon_align64_sz((size_t)n_tiles * sizeof(int32_t)) : 0u;
    const size_t total_scratch = sz_ab + sz_asc + sz_asum + sz_xnorm + sz_xsum + sz_tsk + sz_ablks + sz_atile + 64u;

    uint8_t* scratch_raw = NULL;
    int scratch_heap_owned = 0;
#if !defined(NEON_MSVC_FREESTANDING_DLL)
    if (total_scratch <= (size_t)(4u * 1024u * 1024u)) {
        if (g_tile16_tls_scratch.cap < total_scratch) {
            uint8_t* nb = (uint8_t*)neon_alloc(total_scratch);
            if (!nb) return -1;
            neon_free(g_tile16_tls_scratch.ptr);
            g_tile16_tls_scratch.ptr = nb;
            g_tile16_tls_scratch.cap = total_scratch;
        }
        scratch_raw = g_tile16_tls_scratch.ptr;
    } else
#endif
    {
        scratch_raw = (uint8_t*)neon_alloc(total_scratch);
        if (!scratch_raw) return -1;
        scratch_heap_owned = 1;
    }

    uintptr_t p_cur = ((uintptr_t)scratch_raw + 63u) & ~(uintptr_t)63u;
    uint8_t* ab_buf      = (uint8_t*)p_cur; p_cur += sz_ab;
    float* asc_buf       = (float*)p_cur;   p_cur += sz_asc;
    float* asum_buf      = (float*)p_cur;   p_cur += sz_asum;
    float* blk_xnorm     = need_norm ? (float*)p_cur : NULL; p_cur += sz_xnorm;
    float* blk_xsum      = need_sum  ? (float*)p_cur : NULL; p_cur += sz_xsum;
    int32_t* tile_skips  = (out_skipped != NULL) ? (int32_t*)p_cur : NULL; p_cur += sz_tsk;
    int32_t* active_blks = (int32_t*)p_cur; p_cur += sz_ablks;
    int32_t* active_tiles = (row_tile_mask != NULL) ? (int32_t*)p_cur : NULL;

    int n_active_blks = 0;
    if (col_blk_mask != NULL) {
        for (int b = 0; b < n_blks; b++) {
            if (col_blk_mask[b] != 0) active_blks[n_active_blks++] = b;
        }
    } else {
        for (int b = 0; b < n_blks; b++) active_blks[b] = b;
        n_active_blks = n_blks;
    }
    int n_active_tiles = n_tiles;
    if (row_tile_mask != NULL) {
        n_active_tiles = 0;
        for (int t = 0; t < n_tiles; t++) {
            if (row_tile_mask[t] != 0) active_tiles[n_active_tiles++] = t;
        }
    }
    const int32_t col_skipped_per_tile = (int32_t)(n_blks - n_active_blks);
    const int64_t row_skipped_blks_per_pass = (int64_t)(n_tiles - n_active_tiles) * (int64_t)n_blks;

#if HAS_OMP
    if (num_threads > 0) omp_set_num_threads(num_threads);
#else
    (void)num_threads;
#endif

    int64_t total_skipped = 0;

    for (int k0 = 0; k0 < k_tokens; k0 += kc_step) {
        const int kc = (k_tokens - k0 >= kc_step) ? kc_step : (k_tokens - k0);
        for (int kk = 0; kk < kc; kk++) {
            const float* xk = x + (size_t)(k0 + kk) * (size_t)d_in;
            for (int ai = 0; ai < n_active_blks; ai++) {
                const int blk = active_blks[ai];
                const size_t idx = (size_t)kk * (size_t)n_blks + (size_t)blk;
                quantize_x_blk_andvcnt(
                    xk + (size_t)blk * (size_t)block_size,
                    block_size,
                    a_bits,
                    ab_buf + idx * a_blk_bytes,
                    &asc_buf[idx],
                    &asum_buf[idx],
                    need_norm ? &blk_xnorm[idx] : NULL,
                    need_sum ? &blk_xsum[idx] : NULL
                );
            }
        }
        if (tile_skips) {
            memset(tile_skips, 0, (size_t)n_tiles * sizeof(int32_t));
        }
        if (row_tile_mask != NULL && n_active_tiles < n_tiles) {
            memset(y + (size_t)k0 * (size_t)d_out, 0, (size_t)kc * (size_t)d_out * sizeof(float));
        }

        int ati;
#if HAS_OMP
        #pragma omp parallel for schedule(static)
#endif
        for (ati = 0; ati < n_active_tiles; ati++) {
            const int t = active_tiles ? active_tiles[ati] : ati;
            const uint8_t* tptr = packed + (size_t)t * (size_t)n_blks * tile_blk_bytes;
            int32_t t_skip = col_skipped_per_tile;

#if HAS_NEON && (defined(__aarch64__) || defined(_M_ARM64))
            float32x4_t acc[8][4];
            for (int kk = 0; kk < kc; kk++) {
                acc[kk][0] = vdupq_n_f32(0.0f);
                acc[kk][1] = vdupq_n_f32(0.0f);
                acc[kk][2] = vdupq_n_f32(0.0f);
                acc[kk][3] = vdupq_n_f32(0.0f);
            }
            const float32x4_t v_bsize = vdupq_n_f32(bsize_f);

            for (int ai = 0; ai < n_active_blks; ai++) {
                const int blk = active_blks[ai];
                if (need_norm) {
                    float max_xn = 0.0f;
                    for (int kk = 0; kk < kc; kk++) {
                        float xn = blk_xnorm[(size_t)kk * (size_t)n_blks + (size_t)blk];
                        if (xn > max_xn) max_xn = xn;
                    }
                    float wn = fp16_to_fp32(tile_norm_fp16[(size_t)t * (size_t)n_blks + (size_t)blk]);
                    if (wn * max_xn <= cs_skip_thresh) {
                        t_skip++;
                        continue;
                    }
                }
                const uint8_t* bptr = tptr + (size_t)blk * tile_blk_bytes;
                if (ai + 1 < n_active_blks) {
                    const uint8_t* nptr = tptr + (size_t)active_blks[ai + 1] * tile_blk_bytes;
                    neon_prefetch(nptr);
                    neon_prefetch(nptr + 256);
                }

                uint16_t raw_sc[8];
                memcpy(raw_sc, bptr + 0, 16);
                float16x8_t hsc0 = vreinterpretq_f16_u16(vld1q_u16(raw_sc));
                memcpy(raw_sc, bptr + 16, 16);
                float16x8_t hsc1 = vreinterpretq_f16_u16(vld1q_u16(raw_sc));
                float32x4_t sc[4] = {
                    vcvt_f32_f16(vget_low_f16(hsc0)),  vcvt_f32_f16(vget_high_f16(hsc0)),
                    vcvt_f32_f16(vget_low_f16(hsc1)),  vcvt_f32_f16(vget_high_f16(hsc1))
                };

                if (need_sum) {
                    uint16_t raw_bi[8];
                    memcpy(raw_bi, bptr + 32, 16);
                    float16x8_t hbi0 = vreinterpretq_f16_u16(vld1q_u16(raw_bi));
                    memcpy(raw_bi, bptr + 48, 16);
                    float16x8_t hbi1 = vreinterpretq_f16_u16(vld1q_u16(raw_bi));
                    float32x4_t bi0 = vcvt_f32_f16(vget_low_f16(hbi0));
                    float32x4_t bi1 = vcvt_f32_f16(vget_high_f16(hbi0));
                    float32x4_t bi2 = vcvt_f32_f16(vget_low_f16(hbi1));
                    float32x4_t bi3 = vcvt_f32_f16(vget_high_f16(hbi1));
                    for (int kk = 0; kk < kc; kk++) {
                        float sx = blk_xsum[(size_t)kk * (size_t)n_blks + (size_t)blk];
                        acc[kk][0] = vfmaq_n_f32(acc[kk][0], bi0, sx);
                        acc[kk][1] = vfmaq_n_f32(acc[kk][1], bi1, sx);
                        acc[kk][2] = vfmaq_n_f32(acc[kk][2], bi2, sx);
                        acc[kk][3] = vfmaq_n_f32(acc[kk][3], bi3, sx);
                    }
                }

                const uint8_t* planes = bptr + hdr_bytes;
                for (int p = 0; p < eff_bits; p++) {
                    const uint8_t* pptr = planes + (size_t)p * (size_t)bytes_per_plane_blk;
                    const float weight_p = t16c_plane_weight(gauss_lut, eff_bits, p);

                    if (bytes_per_row == 32) {
                        for (int q4 = 0; q4 < 4; q4++) {
                            const uint8_t* wp = pptr + (size_t)q4 * 128u;
                            uint8x16_t w0l = vld1q_u8(wp + 0),   w0h = vld1q_u8(wp + 16);
                            uint8x16_t w1l = vld1q_u8(wp + 32),  w1h = vld1q_u8(wp + 48);
                            uint8x16_t w2l = vld1q_u8(wp + 64),  w2h = vld1q_u8(wp + 80);
                            uint8x16_t w3l = vld1q_u8(wp + 96),  w3h = vld1q_u8(wp + 112);

                            uint8x16_t pc0 = vaddq_u8(vcntq_u8(w0l), vcntq_u8(w0h));
                            uint8x16_t pc1 = vaddq_u8(vcntq_u8(w1l), vcntq_u8(w1h));
                            uint8x16_t pc2 = vaddq_u8(vcntq_u8(w2l), vcntq_u8(w2h));
                            uint8x16_t pc3 = vaddq_u8(vcntq_u8(w3l), vcntq_u8(w3h));
                            float32x4_t wpop = sum4_u8x16_exact_neon(pc0, pc1, pc2, pc3);
                            float32x4_t h_wsum = vmulq_n_f32(vsubq_f32(vaddq_f32(wpop, wpop), v_bsize), half_f);
                            float32x4_t sc_p = vmulq_n_f32(sc[q4], weight_p);

                            for (int kk = 0; kk < kc; kk++) {
                                const size_t kidx = (size_t)kk * (size_t)n_blks + (size_t)blk;
                                const uint8_t* ab = ab_buf + kidx * a_blk_bytes;
                                float32x4_t dwu;
                                if (a_bits == 4) {
                                    uint8x16_t al = vld1q_u8(ab + 96), ah = vld1q_u8(ab + 112);
                                    uint8x16_t c0 = vaddq_u8(vcntq_u8(vandq_u8(w0l, al)), vcntq_u8(vandq_u8(w0h, ah)));
                                    uint8x16_t c1 = vaddq_u8(vcntq_u8(vandq_u8(w1l, al)), vcntq_u8(vandq_u8(w1h, ah)));
                                    uint8x16_t c2 = vaddq_u8(vcntq_u8(vandq_u8(w2l, al)), vcntq_u8(vandq_u8(w2h, ah)));
                                    uint8x16_t c3 = vaddq_u8(vcntq_u8(vandq_u8(w3l, al)), vcntq_u8(vandq_u8(w3h, ah)));
                                    for (int bp = 2; bp >= 0; bp--) {
                                        al = vld1q_u8(ab + (size_t)bp * 32u);
                                        ah = vld1q_u8(ab + (size_t)bp * 32u + 16u);
                                        c0 = vaddq_u8(vaddq_u8(c0, c0), vaddq_u8(vcntq_u8(vandq_u8(w0l, al)), vcntq_u8(vandq_u8(w0h, ah))));
                                        c1 = vaddq_u8(vaddq_u8(c1, c1), vaddq_u8(vcntq_u8(vandq_u8(w1l, al)), vcntq_u8(vandq_u8(w1h, ah))));
                                        c2 = vaddq_u8(vaddq_u8(c2, c2), vaddq_u8(vcntq_u8(vandq_u8(w2l, al)), vcntq_u8(vandq_u8(w2h, ah))));
                                        c3 = vaddq_u8(vaddq_u8(c3, c3), vaddq_u8(vcntq_u8(vandq_u8(w3l, al)), vcntq_u8(vandq_u8(w3h, ah))));
                                    }
                                    dwu = sum4_u8x16_exact_neon(c0, c1, c2, c3);
                                } else if (a_bits == 5) {
                                    uint8x16_t al = vld1q_u8(ab + 128), ah = vld1q_u8(ab + 144);
                                    uint8x16_t c0 = vaddq_u8(vcntq_u8(vandq_u8(w0l, al)), vcntq_u8(vandq_u8(w0h, ah)));
                                    uint8x16_t c1 = vaddq_u8(vcntq_u8(vandq_u8(w1l, al)), vcntq_u8(vandq_u8(w1h, ah)));
                                    uint8x16_t c2 = vaddq_u8(vcntq_u8(vandq_u8(w2l, al)), vcntq_u8(vandq_u8(w2h, ah)));
                                    uint8x16_t c3 = vaddq_u8(vcntq_u8(vandq_u8(w3l, al)), vcntq_u8(vandq_u8(w3h, ah)));
                                    for (int bp = 3; bp >= 1; bp--) {
                                        al = vld1q_u8(ab + (size_t)bp * 32u);
                                        ah = vld1q_u8(ab + (size_t)bp * 32u + 16u);
                                        c0 = vaddq_u8(vaddq_u8(c0, c0), vaddq_u8(vcntq_u8(vandq_u8(w0l, al)), vcntq_u8(vandq_u8(w0h, ah))));
                                        c1 = vaddq_u8(vaddq_u8(c1, c1), vaddq_u8(vcntq_u8(vandq_u8(w1l, al)), vcntq_u8(vandq_u8(w1h, ah))));
                                        c2 = vaddq_u8(vaddq_u8(c2, c2), vaddq_u8(vcntq_u8(vandq_u8(w2l, al)), vcntq_u8(vandq_u8(w2h, ah))));
                                        c3 = vaddq_u8(vaddq_u8(c3, c3), vaddq_u8(vcntq_u8(vandq_u8(w3l, al)), vcntq_u8(vandq_u8(w3h, ah))));
                                    }
                                    float32x4_t hi4 = sum4_u8x16_exact_neon(c0, c1, c2, c3);
                                    al = vld1q_u8(ab + 0); ah = vld1q_u8(ab + 16);
                                    c0 = vaddq_u8(vcntq_u8(vandq_u8(w0l, al)), vcntq_u8(vandq_u8(w0h, ah)));
                                    c1 = vaddq_u8(vcntq_u8(vandq_u8(w1l, al)), vcntq_u8(vandq_u8(w1h, ah)));
                                    c2 = vaddq_u8(vcntq_u8(vandq_u8(w2l, al)), vcntq_u8(vandq_u8(w2h, ah)));
                                    c3 = vaddq_u8(vcntq_u8(vandq_u8(w3l, al)), vcntq_u8(vandq_u8(w3h, ah)));
                                    dwu = vaddq_f32(vaddq_f32(hi4, hi4), sum4_u8x16_exact_neon(c0, c1, c2, c3));
                                } else {
                                    /* a_bits == 6: bp=5..2 (max 240) + bp=1..0 (max 48) */
                                    uint8x16_t al = vld1q_u8(ab + 160), ah = vld1q_u8(ab + 176);
                                    uint8x16_t c0 = vaddq_u8(vcntq_u8(vandq_u8(w0l, al)), vcntq_u8(vandq_u8(w0h, ah)));
                                    uint8x16_t c1 = vaddq_u8(vcntq_u8(vandq_u8(w1l, al)), vcntq_u8(vandq_u8(w1h, ah)));
                                    uint8x16_t c2 = vaddq_u8(vcntq_u8(vandq_u8(w2l, al)), vcntq_u8(vandq_u8(w2h, ah)));
                                    uint8x16_t c3 = vaddq_u8(vcntq_u8(vandq_u8(w3l, al)), vcntq_u8(vandq_u8(w3h, ah)));
                                    for (int bp = 4; bp >= 2; bp--) {
                                        al = vld1q_u8(ab + (size_t)bp * 32u);
                                        ah = vld1q_u8(ab + (size_t)bp * 32u + 16u);
                                        c0 = vaddq_u8(vaddq_u8(c0, c0), vaddq_u8(vcntq_u8(vandq_u8(w0l, al)), vcntq_u8(vandq_u8(w0h, ah))));
                                        c1 = vaddq_u8(vaddq_u8(c1, c1), vaddq_u8(vcntq_u8(vandq_u8(w1l, al)), vcntq_u8(vandq_u8(w1h, ah))));
                                        c2 = vaddq_u8(vaddq_u8(c2, c2), vaddq_u8(vcntq_u8(vandq_u8(w2l, al)), vcntq_u8(vandq_u8(w2h, ah))));
                                        c3 = vaddq_u8(vaddq_u8(c3, c3), vaddq_u8(vcntq_u8(vandq_u8(w3l, al)), vcntq_u8(vandq_u8(w3h, ah))));
                                    }
                                    float32x4_t hi4 = sum4_u8x16_exact_neon(c0, c1, c2, c3);
                                    al = vld1q_u8(ab + 32); ah = vld1q_u8(ab + 48);
                                    c0 = vaddq_u8(vcntq_u8(vandq_u8(w0l, al)), vcntq_u8(vandq_u8(w0h, ah)));
                                    c1 = vaddq_u8(vcntq_u8(vandq_u8(w1l, al)), vcntq_u8(vandq_u8(w1h, ah)));
                                    c2 = vaddq_u8(vcntq_u8(vandq_u8(w2l, al)), vcntq_u8(vandq_u8(w2h, ah)));
                                    c3 = vaddq_u8(vcntq_u8(vandq_u8(w3l, al)), vcntq_u8(vandq_u8(w3h, ah)));
                                    al = vld1q_u8(ab + 0); ah = vld1q_u8(ab + 16);
                                    c0 = vaddq_u8(vaddq_u8(c0, c0), vaddq_u8(vcntq_u8(vandq_u8(w0l, al)), vcntq_u8(vandq_u8(w0h, ah))));
                                    c1 = vaddq_u8(vaddq_u8(c1, c1), vaddq_u8(vcntq_u8(vandq_u8(w1l, al)), vcntq_u8(vandq_u8(w1h, ah))));
                                    c2 = vaddq_u8(vaddq_u8(c2, c2), vaddq_u8(vcntq_u8(vandq_u8(w2l, al)), vcntq_u8(vandq_u8(w2h, ah))));
                                    c3 = vaddq_u8(vaddq_u8(c3, c3), vaddq_u8(vcntq_u8(vandq_u8(w3l, al)), vcntq_u8(vandq_u8(w3h, ah))));
                                    dwu = vfmaq_n_f32(sum4_u8x16_exact_neon(c0, c1, c2, c3), hi4, 4.0f);
                                }
                                float32x4_t bdot = vsubq_f32(vsubq_f32(vaddq_f32(dwu, dwu), vdupq_n_f32(asum_buf[kidx])), h_wsum);
                                acc[kk][q4] = vfmaq_f32(acc[kk][q4], vmulq_n_f32(sc_p, asc_buf[kidx]), bdot);
                            }
                        }
                    } else if (bytes_per_row == 16) {
                        for (int q4 = 0; q4 < 4; q4++) {
                            const uint8_t* wp = pptr + (size_t)q4 * 64u;
                            uint8x16_t w0 = vld1q_u8(wp + 0),  w1 = vld1q_u8(wp + 16);
                            uint8x16_t w2 = vld1q_u8(wp + 32), w3 = vld1q_u8(wp + 48);
                            float32x4_t wpop = sum4_u8x16_exact_neon(vcntq_u8(w0), vcntq_u8(w1), vcntq_u8(w2), vcntq_u8(w3));
                            float32x4_t h_wsum = vmulq_n_f32(vsubq_f32(vaddq_f32(wpop, wpop), v_bsize), half_f);
                            float32x4_t sc_p = vmulq_n_f32(sc[q4], weight_p);
                            for (int kk = 0; kk < kc; kk++) {
                                const size_t kidx = (size_t)kk * (size_t)n_blks + (size_t)blk;
                                const uint8_t* ab = ab_buf + kidx * a_blk_bytes;
                                float32x4_t dwu;
                                if (a_bits <= 5) {
                                    uint8x16_t av = vld1q_u8(ab + (size_t)(a_bits - 1) * 16u);
                                    uint8x16_t c0 = vcntq_u8(vandq_u8(w0, av));
                                    uint8x16_t c1 = vcntq_u8(vandq_u8(w1, av));
                                    uint8x16_t c2 = vcntq_u8(vandq_u8(w2, av));
                                    uint8x16_t c3 = vcntq_u8(vandq_u8(w3, av));
                                    for (int bp = a_bits - 2; bp >= 0; bp--) {
                                        av = vld1q_u8(ab + (size_t)bp * 16u);
                                        c0 = vaddq_u8(vaddq_u8(c0, c0), vcntq_u8(vandq_u8(w0, av)));
                                        c1 = vaddq_u8(vaddq_u8(c1, c1), vcntq_u8(vandq_u8(w1, av)));
                                        c2 = vaddq_u8(vaddq_u8(c2, c2), vcntq_u8(vandq_u8(w2, av)));
                                        c3 = vaddq_u8(vaddq_u8(c3, c3), vcntq_u8(vandq_u8(w3, av)));
                                    }
                                    dwu = sum4_u8x16_exact_neon(c0, c1, c2, c3);
                                } else {
                                    uint8x16_t av = vld1q_u8(ab + 5u * 16u);
                                    uint8x16_t c0 = vcntq_u8(vandq_u8(w0, av));
                                    uint8x16_t c1 = vcntq_u8(vandq_u8(w1, av));
                                    uint8x16_t c2 = vcntq_u8(vandq_u8(w2, av));
                                    uint8x16_t c3 = vcntq_u8(vandq_u8(w3, av));
                                    for (int bp = 4; bp >= 1; bp--) {
                                        av = vld1q_u8(ab + (size_t)bp * 16u);
                                        c0 = vaddq_u8(vaddq_u8(c0, c0), vcntq_u8(vandq_u8(w0, av)));
                                        c1 = vaddq_u8(vaddq_u8(c1, c1), vcntq_u8(vandq_u8(w1, av)));
                                        c2 = vaddq_u8(vaddq_u8(c2, c2), vcntq_u8(vandq_u8(w2, av)));
                                        c3 = vaddq_u8(vaddq_u8(c3, c3), vcntq_u8(vandq_u8(w3, av)));
                                    }
                                    float32x4_t hi5 = sum4_u8x16_exact_neon(c0, c1, c2, c3);
                                    av = vld1q_u8(ab);
                                    dwu = vaddq_f32(vaddq_f32(hi5, hi5), sum4_u8x16_exact_neon(
                                        vcntq_u8(vandq_u8(w0, av)), vcntq_u8(vandq_u8(w1, av)),
                                        vcntq_u8(vandq_u8(w2, av)), vcntq_u8(vandq_u8(w3, av))
                                    ));
                                }
                                float32x4_t bdot = vsubq_f32(vsubq_f32(vaddq_f32(dwu, dwu), vdupq_n_f32(asum_buf[kidx])), h_wsum);
                                acc[kk][q4] = vfmaq_f32(acc[kk][q4], vmulq_n_f32(sc_p, asc_buf[kidx]), bdot);
                            }
                        }
                    } else {
                        /* bytes_per_row == 8 (block_size == 64): zero-extend 8B rows into 16B lanes */
                        const uint8x8_t z8 = vdup_n_u8(0);
                        for (int q4 = 0; q4 < 4; q4++) {
                            const uint8_t* wp = pptr + (size_t)q4 * 32u;
                            uint8x8_t w0 = vld1_u8(wp + 0),  w1 = vld1_u8(wp + 8);
                            uint8x8_t w2 = vld1_u8(wp + 16), w3 = vld1_u8(wp + 24);
                            float32x4_t wpop = sum4_u8x16_exact_neon(
                                vcombine_u8(vcnt_u8(w0), z8), vcombine_u8(vcnt_u8(w1), z8),
                                vcombine_u8(vcnt_u8(w2), z8), vcombine_u8(vcnt_u8(w3), z8)
                            );
                            float32x4_t h_wsum = vmulq_n_f32(vsubq_f32(vaddq_f32(wpop, wpop), v_bsize), half_f);
                            float32x4_t sc_p = vmulq_n_f32(sc[q4], weight_p);
                            for (int kk = 0; kk < kc; kk++) {
                                const size_t kidx = (size_t)kk * (size_t)n_blks + (size_t)blk;
                                const uint8_t* ab = ab_buf + kidx * a_blk_bytes;
                                float32x4_t dwu;
                                if (a_bits <= 5) {
                                    uint8x8_t av = vld1_u8(ab + (size_t)(a_bits - 1) * 8u);
                                    uint8x8_t c0 = vcnt_u8(vand_u8(w0, av));
                                    uint8x8_t c1 = vcnt_u8(vand_u8(w1, av));
                                    uint8x8_t c2 = vcnt_u8(vand_u8(w2, av));
                                    uint8x8_t c3 = vcnt_u8(vand_u8(w3, av));
                                    for (int bp = a_bits - 2; bp >= 0; bp--) {
                                        av = vld1_u8(ab + (size_t)bp * 8u);
                                        c0 = vadd_u8(vadd_u8(c0, c0), vcnt_u8(vand_u8(w0, av)));
                                        c1 = vadd_u8(vadd_u8(c1, c1), vcnt_u8(vand_u8(w1, av)));
                                        c2 = vadd_u8(vadd_u8(c2, c2), vcnt_u8(vand_u8(w2, av)));
                                        c3 = vadd_u8(vadd_u8(c3, c3), vcnt_u8(vand_u8(w3, av)));
                                    }
                                    dwu = sum4_u8x16_exact_neon(
                                        vcombine_u8(c0, z8), vcombine_u8(c1, z8),
                                        vcombine_u8(c2, z8), vcombine_u8(c3, z8)
                                    );
                                } else {
                                    uint8x8_t av = vld1_u8(ab + 5u * 8u);
                                    uint8x8_t c0 = vcnt_u8(vand_u8(w0, av));
                                    uint8x8_t c1 = vcnt_u8(vand_u8(w1, av));
                                    uint8x8_t c2 = vcnt_u8(vand_u8(w2, av));
                                    uint8x8_t c3 = vcnt_u8(vand_u8(w3, av));
                                    for (int bp = 4; bp >= 1; bp--) {
                                        av = vld1_u8(ab + (size_t)bp * 8u);
                                        c0 = vadd_u8(vadd_u8(c0, c0), vcnt_u8(vand_u8(w0, av)));
                                        c1 = vadd_u8(vadd_u8(c1, c1), vcnt_u8(vand_u8(w1, av)));
                                        c2 = vadd_u8(vadd_u8(c2, c2), vcnt_u8(vand_u8(w2, av)));
                                        c3 = vadd_u8(vadd_u8(c3, c3), vcnt_u8(vand_u8(w3, av)));
                                    }
                                    float32x4_t hi5 = sum4_u8x16_exact_neon(
                                        vcombine_u8(c0, z8), vcombine_u8(c1, z8),
                                        vcombine_u8(c2, z8), vcombine_u8(c3, z8)
                                    );
                                    av = vld1_u8(ab);
                                    dwu = vaddq_f32(vaddq_f32(hi5, hi5), sum4_u8x16_exact_neon(
                                        vcombine_u8(vcnt_u8(vand_u8(w0, av)), z8), vcombine_u8(vcnt_u8(vand_u8(w1, av)), z8),
                                        vcombine_u8(vcnt_u8(vand_u8(w2, av)), z8), vcombine_u8(vcnt_u8(vand_u8(w3, av)), z8)
                                    ));
                                }
                                float32x4_t bdot = vsubq_f32(vsubq_f32(vaddq_f32(dwu, dwu), vdupq_n_f32(asum_buf[kidx])), h_wsum);
                                acc[kk][q4] = vfmaq_f32(acc[kk][q4], vmulq_n_f32(sc_p, asc_buf[kidx]), bdot);
                            }
                        }
                    }
                }
            }

            const int base_r = t * 16;
            if (base_r + 16 <= d_out) {
                for (int kk = 0; kk < kc; kk++) {
                    float* yk = y + (size_t)(k0 + kk) * (size_t)d_out + (size_t)base_r;
                    vst1q_f32(yk + 0,  acc[kk][0]);
                    vst1q_f32(yk + 4,  acc[kk][1]);
                    vst1q_f32(yk + 8,  acc[kk][2]);
                    vst1q_f32(yk + 12, acc[kk][3]);
                }
            } else {
                const int valid = d_out - base_r;
                for (int kk = 0; kk < kc; kk++) {
                    float tmp[16];
                    vst1q_f32(tmp + 0,  acc[kk][0]);
                    vst1q_f32(tmp + 4,  acc[kk][1]);
                    vst1q_f32(tmp + 8,  acc[kk][2]);
                    vst1q_f32(tmp + 12, acc[kk][3]);
                    float* yk = y + (size_t)(k0 + kk) * (size_t)d_out + (size_t)base_r;
                    for (int r = 0; r < valid; r++) yk[r] = tmp[r];
                }
            }
#else
            float acc[8][16];
            for (int kk = 0; kk < kc; kk++) {
                for (int r = 0; r < 16; r++) acc[kk][r] = 0.0f;
            }
            for (int ai = 0; ai < n_active_blks; ai++) {
                const int blk = active_blks[ai];
                if (need_norm) {
                    float max_xn = 0.0f;
                    for (int kk = 0; kk < kc; kk++) {
                        float xn = blk_xnorm[(size_t)kk * (size_t)n_blks + (size_t)blk];
                        if (xn > max_xn) max_xn = xn;
                    }
                    float wn = fp16_to_fp32(tile_norm_fp16[(size_t)t * (size_t)n_blks + (size_t)blk]);
                    if (wn * max_xn <= cs_skip_thresh) {
                        t_skip++;
                        continue;
                    }
                }
                const uint8_t* bptr = tptr + (size_t)blk * tile_blk_bytes;
                float sc_arr[16], bi_arr[16];
                for (int r = 0; r < 16; r++) {
                    sc_arr[r] = fp16_to_fp32(read_u16_le(bptr + (size_t)r * 2u));
                    bi_arr[r] = is_asym ? fp16_to_fp32(read_u16_le(bptr + 32u + (size_t)r * 2u)) : 0.0f;
                }
                if (need_sum) {
                    for (int kk = 0; kk < kc; kk++) {
                        float sx = blk_xsum[(size_t)kk * (size_t)n_blks + (size_t)blk];
                        for (int r = 0; r < 16; r++) acc[kk][r] += bi_arr[r] * sx;
                    }
                }
                const uint8_t* planes = bptr + hdr_bytes;
                for (int p = 0; p < eff_bits; p++) {
                    const uint8_t* pptr = planes + (size_t)p * (size_t)bytes_per_plane_blk;
                    const float weight_p = t16c_plane_weight(gauss_lut, eff_bits, p);
                    for (int r = 0; r < 16; r++) {
                        const uint8_t* wr = pptr + (size_t)r * (size_t)bytes_per_row;
                        int wpop = 0;
                        for (int bo = 0; bo < bytes_per_row; bo++) wpop += (int)POPCNT_BYTE_LUT[wr[bo]];
                        float h_wsum = half_f * (float)(2 * wpop - block_size);
                        float sc_p = sc_arr[r] * weight_p;
                        for (int kk = 0; kk < kc; kk++) {
                            const size_t kidx = (size_t)kk * (size_t)n_blks + (size_t)blk;
                            const uint8_t* ab = ab_buf + kidx * a_blk_bytes;
                            int32_t dwu = 0;
                            for (int bp = 0; bp < a_bits; bp++) {
                                const uint8_t* abp = ab + (size_t)bp * (size_t)bytes_per_row;
                                int32_t pc = 0;
                                for (int bo = 0; bo < bytes_per_row; bo++) {
                                    pc += (int32_t)POPCNT_BYTE_LUT[wr[bo] & abp[bo]];
                                }
                                dwu += pc << bp;
                            }
                            float bdot = (2.0f * (float)dwu - asum_buf[kidx]) - h_wsum;
                            acc[kk][r] += (sc_p * asc_buf[kidx]) * bdot;
                        }
                    }
                }
            }
            const int base_r = t * 16;
            const int valid = (base_r + 16 <= d_out) ? 16 : (d_out - base_r);
            for (int kk = 0; kk < kc; kk++) {
                float* yk = y + (size_t)(k0 + kk) * (size_t)d_out + (size_t)base_r;
                for (int r = 0; r < valid; r++) yk[r] = acc[kk][r];
            }
#endif
            if (tile_skips) tile_skips[t] = t_skip;
        }
        if (tile_skips) {
            total_skipped += row_skipped_blks_per_pass;
            for (int ati2 = 0; ati2 < n_active_tiles; ati2++) {
                total_skipped += tile_skips[active_tiles ? active_tiles[ati2] : ati2];
            }
        }
    }

    if (out_skipped) *out_skipped = total_skipped;
    if (scratch_heap_owned) {
        neon_free(scratch_raw);
    }
    return 0;
}

NEON_API int gemm_bitserial_tile16_sparse_batch(
    const uint8_t* NEON_RESTRICT packed,
    const uint16_t* NEON_RESTRICT tile_norm_fp16,
    const float* NEON_RESTRICT x,
    float* NEON_RESTRICT y,
    int d_out,
    int d_in,
    int block_size,
    int n_bits,
    int is_asym,
    int k_tokens,
    float cs_skip_thresh,
    const uint8_t* NEON_RESTRICT row_tile_mask,
    const uint8_t* NEON_RESTRICT col_blk_mask,
    int num_threads,
    int64_t* NEON_RESTRICT out_skipped
) {
    if (!packed || !x || !y) return -1;
    if (d_out <= 0 || d_in <= 0) return -1;
    if (block_size != 64 && block_size != 128 && block_size != 256) return -1;
    if ((d_in % block_size) != 0) return -1;
    const int gauss_lut = (n_bits & T16C_GAUSS_BIT) != 0;
    const int rowmajor = (n_bits & T16C_ROWMAJOR_BIT) != 0;
    const int eff_bits = n_bits & 0x3F;
    if (eff_bits < 1 || (eff_bits > 5 && eff_bits != 8)) return -1;
    if ((is_asym != 0 && is_asym != 1) || k_tokens <= 0) return -1;
    if (gauss_lut && eff_bits != 2 && eff_bits != 3) return -1;
    if (rowmajor) {
        return gemm_bitserial_rowmajor_andvcnt_sparse_batch(
            packed, tile_norm_fp16, x, y,
            d_out, d_in, block_size, n_bits, is_asym, k_tokens,
            cs_skip_thresh, row_tile_mask, col_blk_mask, num_threads, out_skipped
        );
    }
    if (eff_bits == 5) {
        if (is_asym != 0 || row_tile_mask != NULL || col_blk_mask != NULL) return -1;
        return gemm_ternary5_tile16_batch(
            packed, tile_norm_fp16, x, y,
            d_out, d_in, block_size, k_tokens,
            cs_skip_thresh, num_threads, out_skipped
        );
    }
    if (eff_bits == 8) {
        if (is_asym != 0 || row_tile_mask != NULL || col_blk_mask != NULL) return -1;
        return gemm_int8_tile16_batch(
            packed, tile_norm_fp16, x, y,
            d_out, d_in, block_size, k_tokens,
            cs_skip_thresh, num_threads, out_skipped
        );
    }

    const int n_tiles = (d_out + 15) / 16;
    const int n_blks = d_in / block_size;
    const int n_subs = d_in / 64;
    const int subs_per_blk = block_size / 64;
    const int hdr_bytes = is_asym ? 64 : 32;
    const int bytes_per_plane_blk = (block_size / 8) * 16;
    const size_t tile_blk_bytes = (size_t)hdr_bytes + (size_t)eff_bits * (size_t)bytes_per_plane_blk;
    const size_t luts_stride = (size_t)(d_in / 4) * 16;
    const int dual = x_needs_dual_plane_masked(x, d_in, block_size, k_tokens, col_blk_mask);
    const int kc_inner = dual ? 2 : 4;
#if HAS_NEON && (defined(__aarch64__) || defined(_M_ARM64))
    const int kc_step = dual ? 8 : 16; /* L1-cached tile-outer batch: 1 DRAM stream serves up to 8 (dual) or 16 (single) tokens */
#else
    const int kc_step = kc_inner;
#endif
    const int kc_max = (k_tokens < kc_step) ? k_tokens : kc_step;
    const int slot_cap = dual ? (kc_max * 2) : kc_max;
    const int need_norm = (cs_skip_thresh > 0.0f && tile_norm_fp16 != NULL) ? 1 : 0;
    const int need_sum = (is_asym != 0) ? 1 : 0;

    const size_t sz_luts = neon_align64_sz((size_t)slot_cap * luts_stride);
#if HAS_NEON && (defined(__aarch64__) || defined(_M_ARM64))
    const size_t sz_luts_i4 = (slot_cap >= 4) ? sz_luts : 0u;
#else
    const size_t sz_luts_i4 = 0u;
#endif
    const size_t sz_subsc = neon_align64_sz((size_t)slot_cap * (size_t)n_subs * sizeof(float));
    const size_t sz_xnorm = need_norm ? neon_align64_sz((size_t)kc_max * (size_t)n_blks * sizeof(float)) : 0u;
    const size_t sz_xsum = need_sum ? neon_align64_sz((size_t)kc_max * (size_t)n_blks * sizeof(float)) : 0u;
    const size_t sz_tskips = (out_skipped != NULL) ? neon_align64_sz((size_t)n_tiles * sizeof(int32_t)) : 0u;
    const size_t sz_ablks = neon_align64_sz((size_t)n_blks * sizeof(int32_t));
    const size_t sz_atiles = (row_tile_mask != NULL) ? neon_align64_sz((size_t)n_tiles * sizeof(int32_t)) : 0u;
    const size_t total_scratch = sz_luts + sz_luts_i4 + sz_subsc + sz_xnorm + sz_xsum + sz_tskips + sz_ablks + sz_atiles + 64u;

    uint8_t* scratch_raw = NULL;
    int scratch_heap_owned = 0;
#if !defined(NEON_MSVC_FREESTANDING_DLL)
    if (total_scratch <= (size_t)(4u * 1024u * 1024u)) {
        if (g_tile16_tls_scratch.cap < total_scratch) {
            uint8_t* nb = (uint8_t*)neon_alloc(total_scratch);
            if (!nb) return -1;
            neon_free(g_tile16_tls_scratch.ptr);
            g_tile16_tls_scratch.ptr = nb;
            g_tile16_tls_scratch.cap = total_scratch;
        }
        scratch_raw = g_tile16_tls_scratch.ptr;
    } else
#endif
    {
        scratch_raw = (uint8_t*)neon_alloc(total_scratch);
        if (!scratch_raw) return -1;
        scratch_heap_owned = 1;
    }

    uintptr_t p_cur = ((uintptr_t)scratch_raw + 63u) & ~(uintptr_t)63u;
    int8_t* luts = (int8_t*)p_cur; p_cur += sz_luts;
#if HAS_NEON && (defined(__aarch64__) || defined(_M_ARM64))
    int8_t* luts_i4 = (sz_luts_i4 > 0u) ? (int8_t*)p_cur : NULL;
    p_cur += sz_luts_i4;
#endif
    float* sub_scales = (float*)p_cur; p_cur += sz_subsc;
    float* blk_xnorm = need_norm ? (float*)p_cur : NULL; p_cur += sz_xnorm;
    float* blk_xsum = need_sum ? (float*)p_cur : NULL; p_cur += sz_xsum;
    int32_t* tile_skips = (out_skipped != NULL) ? (int32_t*)p_cur : NULL; p_cur += sz_tskips;
    int32_t* active_blks = (int32_t*)p_cur; p_cur += sz_ablks;
    int32_t* active_tiles = (row_tile_mask != NULL) ? (int32_t*)p_cur : NULL;

    int n_active_blks = 0;
    if (col_blk_mask != NULL) {
        for (int b = 0; b < n_blks; b++) {
            if (col_blk_mask[b] != 0) active_blks[n_active_blks++] = b;
        }
    } else {
        for (int b = 0; b < n_blks; b++) active_blks[b] = b;
        n_active_blks = n_blks;
    }
    int n_active_tiles = n_tiles;
    if (row_tile_mask != NULL) {
        n_active_tiles = 0;
        for (int t = 0; t < n_tiles; t++) {
            if (row_tile_mask[t] != 0) active_tiles[n_active_tiles++] = t;
        }
    }
    const int32_t col_skipped_per_tile = (int32_t)(n_blks - n_active_blks);
    const int64_t row_skipped_blks_per_pass = (int64_t)(n_tiles - n_active_tiles) * (int64_t)n_blks;

#if HAS_OMP
    if (num_threads > 0) omp_set_num_threads(num_threads);
#else
    (void)num_threads;
#endif

    int64_t total_skipped = 0;

    for (int k0 = 0; k0 < k_tokens; k0 += kc_step) {
        const int kc = (k_tokens - k0 >= kc_step) ? kc_step : (k_tokens - k0);
        const int slots = dual ? (kc * 2) : kc;
        for (int kk = 0; kk < kc; kk++) {
            const float* xk = x + (size_t)(k0 + kk) * (size_t)d_in;
            if (dual) {
                const size_t shi = (size_t)(kk * 2);
                const size_t slo = shi + 1;
                quantize_x_sub64_l1_dual_tile16(
                    xk, d_in,
                    luts + shi * luts_stride, sub_scales + shi * (size_t)n_subs,
                    luts + slo * luts_stride, sub_scales + slo * (size_t)n_subs
                );
            } else {
                quantize_x_sub64_l1_tile16(
                    xk, d_in,
                    luts + (size_t)kk * luts_stride,
                    sub_scales + (size_t)kk * (size_t)n_subs
                );
            }
            if (need_norm || need_sum) {
                for (int ai = 0; ai < n_active_blks; ai++) {
                    int blk = active_blks[ai];
                    const float* xb = xk + (size_t)blk * (size_t)block_size;
                    float n2 = 0.0f;
                    float bs = 0.0f;
                    for (int i = 0; i < block_size; i++) {
                        float v = xb[i];
                        n2 += v * v;
                        bs += v;
                    }
                    if (need_norm) blk_xnorm[(size_t)kk * (size_t)n_blks + (size_t)blk] = neon_sqrtf(n2);
                    if (need_sum)  blk_xsum[(size_t)kk * (size_t)n_blks + (size_t)blk] = bs;
                }
            }
        }
        if (tile_skips) {
            memset(tile_skips, 0, (size_t)n_tiles * sizeof(int32_t));
        }
        if (row_tile_mask != NULL && n_active_tiles < n_tiles) {
            memset(y + (size_t)k0 * (size_t)d_out, 0, (size_t)kc * (size_t)d_out * sizeof(float));
        }

#if HAS_NEON && (defined(__aarch64__) || defined(_M_ARM64))
        /* Interleave each full 4-slot group as [g_sub*8+bo][4][32] (128B = 2 contiguous L1 cache lines)
           to eliminate 2-way Cortex-A72 L1D conflict misses when luts_stride >= 16 KiB! */
        if (luts_i4 != NULL && slots >= 4) {
            const int n_bo = n_subs * 8;
            for (int s0 = 0; s0 + 4 <= slots; s0 += 4) {
                const int8_t* s_0 = luts + (size_t)(s0 + 0) * luts_stride;
                const int8_t* s_1 = luts + (size_t)(s0 + 1) * luts_stride;
                const int8_t* s_2 = luts + (size_t)(s0 + 2) * luts_stride;
                const int8_t* s_3 = luts + (size_t)(s0 + 3) * luts_stride;
                int8_t* dst_g = luts_i4 + (size_t)s0 * luts_stride;
                for (int bi = 0; bi < n_bo; bi++) {
                    const size_t src_off = (size_t)bi * 32u;
                    int8_t* d = dst_g + (size_t)bi * 128u;
                    memcpy(d + 0,  s_0 + src_off, 32);
                    memcpy(d + 32, s_1 + src_off, 32);
                    memcpy(d + 64, s_2 + src_off, 32);
                    memcpy(d + 96, s_3 + src_off, 32);
                }
            }
        }
        if (kc == 1) {
            /* Zero-branch bit-exact Cortex-A72 k=1 decode fast path (matches k=4 IEEE-754 rounding bit-for-bit). */
#if HAS_OMP
            #pragma omp parallel for schedule(static)
#endif
            for (int ati = 0; ati < n_active_tiles; ati++) {
                const int t = active_tiles ? active_tiles[ati] : ati;
                const uint8x16_t mask0f = vdupq_n_u8(0x0F);
                const int8_t* const l0 = luts;
                const int8_t* const l1 = (slots == 2) ? (luts + luts_stride) : luts;
                const uint8_t* tptr = packed + (size_t)t * (size_t)n_blks * tile_blk_bytes;
                int32_t t_skip = col_skipped_per_tile;
                float32x4_t acc00 = vdupq_n_f32(0.0f), acc01 = vdupq_n_f32(0.0f);
                float32x4_t acc02 = vdupq_n_f32(0.0f), acc03 = vdupq_n_f32(0.0f);
                float32x4_t acc10 = vdupq_n_f32(0.0f), acc11 = vdupq_n_f32(0.0f);
                float32x4_t acc12 = vdupq_n_f32(0.0f), acc13 = vdupq_n_f32(0.0f);

                for (int ai = 0; ai < n_active_blks; ai++) {
                    const int blk = active_blks[ai];
                    if (need_norm) {
                        float xn = blk_xnorm[(size_t)blk];
                        float wnorm = fp16_to_fp32(tile_norm_fp16[(size_t)t * (size_t)n_blks + (size_t)blk]);
                        if (wnorm * xn <= cs_skip_thresh) {
                            t_skip++;
                            continue;
                        }
                    }
                    const uint8_t* bptr = tptr + (size_t)blk * tile_blk_bytes;
                    if (ai + 1 < n_active_blks) {
                        const uint8_t* nptr = tptr + (size_t)active_blks[ai + 1] * tile_blk_bytes;
                        neon_prefetch(nptr);
                        neon_prefetch(nptr + 256);
                        neon_prefetch(nptr + 512);
                    }
                    uint16_t raw_sc[8];
                    memcpy(raw_sc, bptr + 0, 16);
                    float16x8_t hsc0 = vreinterpretq_f16_u16(vld1q_u16(raw_sc));
                    memcpy(raw_sc, bptr + 16, 16);
                    float16x8_t hsc1 = vreinterpretq_f16_u16(vld1q_u16(raw_sc));
                    float32x4_t sc0 = vcvt_f32_f16(vget_low_f16(hsc0));
                    float32x4_t sc1 = vcvt_f32_f16(vget_high_f16(hsc0));
                    float32x4_t sc2 = vcvt_f32_f16(vget_low_f16(hsc1));
                    float32x4_t sc3 = vcvt_f32_f16(vget_high_f16(hsc1));

                    if (need_sum) {
                        uint16_t raw_bi[8];
                        memcpy(raw_bi, bptr + 32, 16);
                        float16x8_t hbi0 = vreinterpretq_f16_u16(vld1q_u16(raw_bi));
                        memcpy(raw_bi, bptr + 48, 16);
                        float16x8_t hbi1 = vreinterpretq_f16_u16(vld1q_u16(raw_bi));
                        float32x4_t bi0 = vcvt_f32_f16(vget_low_f16(hbi0));
                        float32x4_t bi1 = vcvt_f32_f16(vget_high_f16(hbi0));
                        float32x4_t bi2 = vcvt_f32_f16(vget_low_f16(hbi1));
                        float32x4_t bi3 = vcvt_f32_f16(vget_high_f16(hbi1));
                        float sx0 = blk_xsum[blk];
                        acc00 = vfmaq_n_f32(acc00, bi0, sx0); acc01 = vfmaq_n_f32(acc01, bi1, sx0);
                        acc02 = vfmaq_n_f32(acc02, bi2, sx0); acc03 = vfmaq_n_f32(acc03, bi3, sx0);
                    }

                    const uint8_t* planes = bptr + hdr_bytes;
                    if (slots == 1) {
                        for (int p = 0; p < eff_bits; p++) {
                            const uint8_t* pptr = planes + (size_t)p * (size_t)bytes_per_plane_blk;
                            float weight_p = t16c_plane_weight(gauss_lut, eff_bits, p);
                            for (int sb = 0; sb < subs_per_blk; sb++) {
                                int g_sub = blk * subs_per_blk + sb;
                                int16x8_t a0l = vdupq_n_s16(0), a0h = vdupq_n_s16(0);
                                int16x8_t b0l = vdupq_n_s16(0), b0h = vdupq_n_s16(0);
                                const uint8_t* pp_sb = pptr + (size_t)sb * 128u;
                                const int8_t* l0_sb = l0 + (size_t)g_sub * 256u;
                                for (int bo = 0; bo < 8; bo += 2) {
                                    uint8x16_t bv0 = vld1q_u8(pp_sb + (size_t)bo * 16u);
                                    uint8x16_t bv1 = vld1q_u8(pp_sb + ((size_t)bo + 1u) * 16u);
                                    const int8_t* lp0 = l0_sb + (size_t)bo * 32u;
                                    const int8_t* lp1 = lp0 + 32u;
                                    int8x16_t rh0 = vqtbl1q_s8(vld1q_s8(lp0),      vshrq_n_u8(bv0, 4));
                                    int8x16_t rl0 = vqtbl1q_s8(vld1q_s8(lp0 + 16), vandq_u8(bv0, mask0f));
                                    int8x16_t rh1 = vqtbl1q_s8(vld1q_s8(lp1),      vshrq_n_u8(bv1, 4));
                                    int8x16_t rl1 = vqtbl1q_s8(vld1q_s8(lp1 + 16), vandq_u8(bv1, mask0f));
                                    a0l = vaddq_s16(a0l, vaddl_s8(vget_low_s8(rh0), vget_low_s8(rl0)));
                                    a0h = vaddq_s16(a0h, vaddl_high_s8(rh0, rl0));
                                    b0l = vaddq_s16(b0l, vaddl_s8(vget_low_s8(rh1), vget_low_s8(rl1)));
                                    b0h = vaddq_s16(b0h, vaddl_high_s8(rh1, rl1));
                                }
                                a0l = vaddq_s16(a0l, b0l);
                                a0h = vaddq_s16(a0h, b0h);
                                float m0 = sub_scales[g_sub] * weight_p;
                                acc00 = vfmaq_f32(acc00, sc0, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a0l))), m0));
                                acc01 = vfmaq_f32(acc01, sc1, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a0l)), m0));
                                acc02 = vfmaq_f32(acc02, sc2, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a0h))), m0));
                                acc03 = vfmaq_f32(acc03, sc3, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a0h)), m0));
                            }
                        }
                    } else {
                        /* dual == 1, kc == 1 (slots == 2): separate plane accumulators match k=4 bit-for-bit. */
                        for (int p = 0; p < eff_bits; p++) {
                            const uint8_t* pptr = planes + (size_t)p * (size_t)bytes_per_plane_blk;
                            float weight_p = t16c_plane_weight(gauss_lut, eff_bits, p);
                            for (int sb = 0; sb < subs_per_blk; sb++) {
                                int g_sub = blk * subs_per_blk + sb;
                                int16x8_t a0l = vdupq_n_s16(0), a0h = vdupq_n_s16(0);
                                int16x8_t a1l = vdupq_n_s16(0), a1h = vdupq_n_s16(0);
                                const uint8_t* pp_sb = pptr + (size_t)sb * 128u;
                                const int8_t* l0_sb = l0 + (size_t)g_sub * 256u;
                                const int8_t* l1_sb = l1 + (size_t)g_sub * 256u;
                                for (int bo = 0; bo < 8; bo++) {
                                    size_t boff = (size_t)bo * 32u;
                                    uint8x16_t bv = vld1q_u8(pp_sb + (size_t)bo * 16u);
                                    uint8x16_t nhi = vshrq_n_u8(bv, 4);
                                    uint8x16_t nlo = vandq_u8(bv, mask0f);
                                    int8x16_t rh = vqtbl1q_s8(vld1q_s8(l0_sb + boff), nhi);
                                    int8x16_t rl = vqtbl1q_s8(vld1q_s8(l0_sb + boff + 16), nlo);
                                    a0l = vaddw_s8(a0l, vget_low_s8(rh)); a0h = vaddw_high_s8(a0h, rh);
                                    a0l = vaddw_s8(a0l, vget_low_s8(rl)); a0h = vaddw_high_s8(a0h, rl);
                                    rh = vqtbl1q_s8(vld1q_s8(l1_sb + boff), nhi);
                                    rl = vqtbl1q_s8(vld1q_s8(l1_sb + boff + 16), nlo);
                                    a1l = vaddw_s8(a1l, vget_low_s8(rh)); a1h = vaddw_high_s8(a1h, rh);
                                    a1l = vaddw_s8(a1l, vget_low_s8(rl)); a1h = vaddw_high_s8(a1h, rl);
                                }
                                float m0 = sub_scales[g_sub] * weight_p;
                                float m1 = sub_scales[n_subs + g_sub] * weight_p;
                                acc00 = vfmaq_f32(acc00, sc0, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a0l))), m0));
                                acc01 = vfmaq_f32(acc01, sc1, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a0l)), m0));
                                acc02 = vfmaq_f32(acc02, sc2, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a0h))), m0));
                                acc03 = vfmaq_f32(acc03, sc3, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a0h)), m0));
                                acc10 = vfmaq_f32(acc10, sc0, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a1l))), m1));
                                acc11 = vfmaq_f32(acc11, sc1, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a1l)), m1));
                                acc12 = vfmaq_f32(acc12, sc2, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a1h))), m1));
                                acc13 = vfmaq_f32(acc13, sc3, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a1h)), m1));
                            }
                        }
                    }
                }

                if (dual) {
                    acc00 = vaddq_f32(acc00, acc10); acc01 = vaddq_f32(acc01, acc11);
                    acc02 = vaddq_f32(acc02, acc12); acc03 = vaddq_f32(acc03, acc13);
                }
                int base_r = t * 16;
                float* y0 = y + (size_t)k0 * (size_t)d_out + (size_t)base_r;
                if (base_r + 16 <= d_out) {
                    vst1q_f32(y0 + 0, acc00); vst1q_f32(y0 + 4, acc01);
                    vst1q_f32(y0 + 8, acc02); vst1q_f32(y0 + 12, acc03);
                } else {
                    float tmp[16];
                    vst1q_f32(tmp + 0, acc00); vst1q_f32(tmp + 4, acc01);
                    vst1q_f32(tmp + 8, acc02); vst1q_f32(tmp + 12, acc03);
                    int valid = d_out - base_r;
                    for (int r = 0; r < valid; r++) y0[r] = tmp[r];
                }
                if (tile_skips) tile_skips[t] = t_skip;
            }
        } else if (kc > kc_inner) {
            /* L1-cached tile-outer multi-group NEON path (slots in 5..16): streams each 16-row weight tile
               from LPDDR4 ONCE while keeping the 1 KiB tile_acc[16][16] and active block hot in 32 KiB L1D.
               Bit-exact IEEE-754 parity with kc==1 and kc<=kc_inner paths. */
#if HAS_OMP
            #pragma omp parallel for schedule(static)
#endif
            for (int ati = 0; ati < n_active_tiles; ati++) {
                const int t = active_tiles ? active_tiles[ati] : ati;
                const uint8_t* tptr = packed + (size_t)t * (size_t)n_blks * tile_blk_bytes;
                const uint8x16_t mask0f = vdupq_n_u8(0x0F);
                int32_t t_skip = col_skipped_per_tile;
                NEON_ALIGN64 float tile_acc[16][16];
                memset(tile_acc, 0, (size_t)slots * 16u * sizeof(float));

                for (int ai = 0; ai < n_active_blks; ai++) {
                    const int blk = active_blks[ai];
                    if (need_norm) {
                        float max_xnorm = 0.0f;
                        for (int kk = 0; kk < kc; kk++) {
                            float xn = blk_xnorm[(size_t)kk * (size_t)n_blks + (size_t)blk];
                            if (xn > max_xnorm) max_xnorm = xn;
                        }
                        float wnorm = fp16_to_fp32(tile_norm_fp16[(size_t)t * (size_t)n_blks + (size_t)blk]);
                        if (wnorm * max_xnorm <= cs_skip_thresh) {
                            t_skip++;
                            continue;
                        }
                    }
                    const uint8_t* bptr = tptr + (size_t)blk * tile_blk_bytes;
                    if (ai + 1 < n_active_blks) {
                        const uint8_t* nptr = tptr + (size_t)active_blks[ai + 1] * tile_blk_bytes;
                        neon_prefetch(nptr);
                        neon_prefetch(nptr + 256);
                        neon_prefetch(nptr + 512);
                    }

                    if (need_sum) {
                        uint16_t raw_bi[8];
                        memcpy(raw_bi, bptr + 32, 16);
                        float16x8_t hbi0 = vreinterpretq_f16_u16(vld1q_u16(raw_bi));
                        memcpy(raw_bi, bptr + 48, 16);
                        float16x8_t hbi1 = vreinterpretq_f16_u16(vld1q_u16(raw_bi));
                        float32x4_t bi0 = vcvt_f32_f16(vget_low_f16(hbi0));
                        float32x4_t bi1 = vcvt_f32_f16(vget_high_f16(hbi0));
                        float32x4_t bi2 = vcvt_f32_f16(vget_low_f16(hbi1));
                        float32x4_t bi3 = vcvt_f32_f16(vget_high_f16(hbi1));
                        for (int kk = 0; kk < kc; kk++) {
                            const int dst_s = dual ? (kk * 2) : kk;
                            float sx = blk_xsum[(size_t)kk * (size_t)n_blks + (size_t)blk];
                            float* ta = tile_acc[dst_s];
                            vst1q_f32(ta + 0,  vfmaq_n_f32(vld1q_f32(ta + 0),  bi0, sx));
                            vst1q_f32(ta + 4,  vfmaq_n_f32(vld1q_f32(ta + 4),  bi1, sx));
                            vst1q_f32(ta + 8,  vfmaq_n_f32(vld1q_f32(ta + 8),  bi2, sx));
                            vst1q_f32(ta + 12, vfmaq_n_f32(vld1q_f32(ta + 12), bi3, sx));
                        }
                    }

                    const uint8_t* planes = bptr + hdr_bytes;
                    /* Hoist FP16->FP32 tile scale decode ONCE per block into an L1 stack array
                       instead of repeating 4-8x inside every (s0, p, sb) iteration! */
                    NEON_ALIGN64 float sc_blk[16];
                    {
                        uint16_t raw_sc[8];
                        memcpy(raw_sc, bptr + 0, 16);
                        float16x8_t hsc0 = vreinterpretq_f16_u16(vld1q_u16(raw_sc));
                        memcpy(raw_sc, bptr + 16, 16);
                        float16x8_t hsc1 = vreinterpretq_f16_u16(vld1q_u16(raw_sc));
                        vst1q_f32(sc_blk + 0,  vcvt_f32_f16(vget_low_f16(hsc0)));
                        vst1q_f32(sc_blk + 4,  vcvt_f32_f16(vget_high_f16(hsc0)));
                        vst1q_f32(sc_blk + 8,  vcvt_f32_f16(vget_low_f16(hsc1)));
                        vst1q_f32(sc_blk + 12, vcvt_f32_f16(vget_high_f16(hsc1)));
                    }
                    for (int s0 = 0; s0 < slots; s0 += 4) {
                        const int sc_g = (slots - s0 >= 4) ? 4 : (slots - s0);
                        float32x4_t acc00 = vld1q_f32(tile_acc[s0 + 0] + 0), acc01 = vld1q_f32(tile_acc[s0 + 0] + 4);
                        float32x4_t acc02 = vld1q_f32(tile_acc[s0 + 0] + 8), acc03 = vld1q_f32(tile_acc[s0 + 0] + 12);
                        float32x4_t acc10 = vdupq_n_f32(0.0f), acc11 = vdupq_n_f32(0.0f), acc12 = vdupq_n_f32(0.0f), acc13 = vdupq_n_f32(0.0f);
                        float32x4_t acc20 = vdupq_n_f32(0.0f), acc21 = vdupq_n_f32(0.0f), acc22 = vdupq_n_f32(0.0f), acc23 = vdupq_n_f32(0.0f);
                        float32x4_t acc30 = vdupq_n_f32(0.0f), acc31 = vdupq_n_f32(0.0f), acc32 = vdupq_n_f32(0.0f), acc33 = vdupq_n_f32(0.0f);
                        if (sc_g >= 2) {
                            acc10 = vld1q_f32(tile_acc[s0 + 1] + 0); acc11 = vld1q_f32(tile_acc[s0 + 1] + 4);
                            acc12 = vld1q_f32(tile_acc[s0 + 1] + 8); acc13 = vld1q_f32(tile_acc[s0 + 1] + 12);
                        }
                        if (sc_g >= 3) {
                            acc20 = vld1q_f32(tile_acc[s0 + 2] + 0); acc21 = vld1q_f32(tile_acc[s0 + 2] + 4);
                            acc22 = vld1q_f32(tile_acc[s0 + 2] + 8); acc23 = vld1q_f32(tile_acc[s0 + 2] + 12);
                        }
                        if (sc_g == 4) {
                            acc30 = vld1q_f32(tile_acc[s0 + 3] + 0); acc31 = vld1q_f32(tile_acc[s0 + 3] + 4);
                            acc32 = vld1q_f32(tile_acc[s0 + 3] + 8); acc33 = vld1q_f32(tile_acc[s0 + 3] + 12);
                        }

                        for (int p = 0; p < eff_bits; p++) {
                            const uint8_t* pptr = planes + (size_t)p * (size_t)bytes_per_plane_blk;
                            float weight_p = t16c_plane_weight(gauss_lut, eff_bits, p);
                            if (sc_g == 4) {
                                const int8_t* lg = luts_i4 + (size_t)s0 * luts_stride;
                                for (int sb = 0; sb < subs_per_blk; sb++) {
                                    int g_sub = blk * subs_per_blk + sb;
                                    int16x8_t a0l = vdupq_n_s16(0), a0h = vdupq_n_s16(0);
                                    int16x8_t a1l = vdupq_n_s16(0), a1h = vdupq_n_s16(0);
                                    int16x8_t a2l = vdupq_n_s16(0), a2h = vdupq_n_s16(0);
                                    int16x8_t a3l = vdupq_n_s16(0), a3h = vdupq_n_s16(0);
                                    const int8_t* lp = lg + (size_t)g_sub * 8u * 128u;
                                    for (int bo = 0; bo < 8; bo++, lp += 128) {
                                        uint8x16_t bv = vld1q_u8(pptr + ((size_t)sb * 8 + (size_t)bo) * 16);
                                        uint8x16_t nhi = vshrq_n_u8(bv, 4);
                                        uint8x16_t nlo = vandq_u8(bv, mask0f);
                                        int8x16_t rh, rl;
                                        rh = vqtbl1q_s8(vld1q_s8(lp + 0),  nhi); rl = vqtbl1q_s8(vld1q_s8(lp + 16),  nlo);
                                        a0l = vaddq_s16(a0l, vaddl_s8(vget_low_s8(rh), vget_low_s8(rl)));
                                        a0h = vaddq_s16(a0h, vaddl_high_s8(rh, rl));
                                        rh = vqtbl1q_s8(vld1q_s8(lp + 32), nhi); rl = vqtbl1q_s8(vld1q_s8(lp + 48),  nlo);
                                        a1l = vaddq_s16(a1l, vaddl_s8(vget_low_s8(rh), vget_low_s8(rl)));
                                        a1h = vaddq_s16(a1h, vaddl_high_s8(rh, rl));
                                        rh = vqtbl1q_s8(vld1q_s8(lp + 64), nhi); rl = vqtbl1q_s8(vld1q_s8(lp + 80),  nlo);
                                        a2l = vaddq_s16(a2l, vaddl_s8(vget_low_s8(rh), vget_low_s8(rl)));
                                        a2h = vaddq_s16(a2h, vaddl_high_s8(rh, rl));
                                        rh = vqtbl1q_s8(vld1q_s8(lp + 96), nhi); rl = vqtbl1q_s8(vld1q_s8(lp + 112), nlo);
                                        a3l = vaddq_s16(a3l, vaddl_s8(vget_low_s8(rh), vget_low_s8(rl)));
                                        a3h = vaddq_s16(a3h, vaddl_high_s8(rh, rl));
                                    }
                                    float32x4_t sc0 = vld1q_f32(sc_blk + 0);
                                    float32x4_t sc1 = vld1q_f32(sc_blk + 4);
                                    float32x4_t sc2 = vld1q_f32(sc_blk + 8);
                                    float32x4_t sc3 = vld1q_f32(sc_blk + 12);
                                    float m0 = sub_scales[(size_t)(s0 + 0) * (size_t)n_subs + (size_t)g_sub] * weight_p;
                                    float m1 = sub_scales[(size_t)(s0 + 1) * (size_t)n_subs + (size_t)g_sub] * weight_p;
                                    float m2 = sub_scales[(size_t)(s0 + 2) * (size_t)n_subs + (size_t)g_sub] * weight_p;
                                    float m3 = sub_scales[(size_t)(s0 + 3) * (size_t)n_subs + (size_t)g_sub] * weight_p;
                                    acc00 = vfmaq_f32(acc00, sc0, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a0l))), m0));
                                    acc01 = vfmaq_f32(acc01, sc1, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a0l)), m0));
                                    acc02 = vfmaq_f32(acc02, sc2, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a0h))), m0));
                                    acc03 = vfmaq_f32(acc03, sc3, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a0h)), m0));
                                    acc10 = vfmaq_f32(acc10, sc0, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a1l))), m1));
                                    acc11 = vfmaq_f32(acc11, sc1, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a1l)), m1));
                                    acc12 = vfmaq_f32(acc12, sc2, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a1h))), m1));
                                    acc13 = vfmaq_f32(acc13, sc3, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a1h)), m1));
                                    acc20 = vfmaq_f32(acc20, sc0, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a2l))), m2));
                                    acc21 = vfmaq_f32(acc21, sc1, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a2l)), m2));
                                    acc22 = vfmaq_f32(acc22, sc2, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a2h))), m2));
                                    acc23 = vfmaq_f32(acc23, sc3, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a2h)), m2));
                                    acc30 = vfmaq_f32(acc30, sc0, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a3l))), m3));
                                    acc31 = vfmaq_f32(acc31, sc1, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a3l)), m3));
                                    acc32 = vfmaq_f32(acc32, sc2, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a3h))), m3));
                                    acc33 = vfmaq_f32(acc33, sc3, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a3h)), m3));
                                }
                            } else {
                                const int8_t* l0 = luts + (size_t)(s0 + 0) * luts_stride;
                                const int8_t* l1 = (sc_g >= 2) ? (luts + (size_t)(s0 + 1) * luts_stride) : l0;
                                const int8_t* l2 = (sc_g >= 3) ? (luts + (size_t)(s0 + 2) * luts_stride) : l0;
                                for (int sb = 0; sb < subs_per_blk; sb++) {
                                    int g_sub = blk * subs_per_blk + sb;
                                    int16x8_t a0l = vdupq_n_s16(0), a0h = vdupq_n_s16(0);
                                    int16x8_t a1l = vdupq_n_s16(0), a1h = vdupq_n_s16(0);
                                    int16x8_t a2l = vdupq_n_s16(0), a2h = vdupq_n_s16(0);
                                    for (int bo = 0; bo < 8; bo++) {
                                        size_t loff = ((size_t)g_sub * 8 + (size_t)bo) * 32;
                                        uint8x16_t bv = vld1q_u8(pptr + ((size_t)sb * 8 + (size_t)bo) * 16);
                                        uint8x16_t nhi = vshrq_n_u8(bv, 4);
                                        uint8x16_t nlo = vandq_u8(bv, mask0f);
                                        int8x16_t rh = vqtbl1q_s8(vld1q_s8(l0 + loff), nhi);
                                        int8x16_t rl = vqtbl1q_s8(vld1q_s8(l0 + loff + 16), nlo);
                                        a0l = vaddq_s16(a0l, vaddl_s8(vget_low_s8(rh), vget_low_s8(rl)));
                                        a0h = vaddq_s16(a0h, vaddl_high_s8(rh, rl));
                                        if (sc_g >= 2) {
                                            rh = vqtbl1q_s8(vld1q_s8(l1 + loff), nhi); rl = vqtbl1q_s8(vld1q_s8(l1 + loff + 16), nlo);
                                            a1l = vaddq_s16(a1l, vaddl_s8(vget_low_s8(rh), vget_low_s8(rl)));
                                            a1h = vaddq_s16(a1h, vaddl_high_s8(rh, rl));
                                        }
                                        if (sc_g >= 3) {
                                            rh = vqtbl1q_s8(vld1q_s8(l2 + loff), nhi); rl = vqtbl1q_s8(vld1q_s8(l2 + loff + 16), nlo);
                                            a2l = vaddq_s16(a2l, vaddl_s8(vget_low_s8(rh), vget_low_s8(rl)));
                                            a2h = vaddq_s16(a2h, vaddl_high_s8(rh, rl));
                                        }
                                    }
                                    float32x4_t sc0 = vld1q_f32(sc_blk + 0);
                                    float32x4_t sc1 = vld1q_f32(sc_blk + 4);
                                    float32x4_t sc2 = vld1q_f32(sc_blk + 8);
                                    float32x4_t sc3 = vld1q_f32(sc_blk + 12);
                                    float m0 = sub_scales[(size_t)(s0 + 0) * (size_t)n_subs + (size_t)g_sub] * weight_p;
                                    acc00 = vfmaq_f32(acc00, sc0, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a0l))), m0));
                                    acc01 = vfmaq_f32(acc01, sc1, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a0l)), m0));
                                    acc02 = vfmaq_f32(acc02, sc2, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a0h))), m0));
                                    acc03 = vfmaq_f32(acc03, sc3, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a0h)), m0));
                                    if (sc_g >= 2) {
                                        float m1 = sub_scales[(size_t)(s0 + 1) * (size_t)n_subs + (size_t)g_sub] * weight_p;
                                        acc10 = vfmaq_f32(acc10, sc0, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a1l))), m1));
                                        acc11 = vfmaq_f32(acc11, sc1, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a1l)), m1));
                                        acc12 = vfmaq_f32(acc12, sc2, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a1h))), m1));
                                        acc13 = vfmaq_f32(acc13, sc3, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a1h)), m1));
                                    }
                                    if (sc_g >= 3) {
                                        float m2 = sub_scales[(size_t)(s0 + 2) * (size_t)n_subs + (size_t)g_sub] * weight_p;
                                        acc20 = vfmaq_f32(acc20, sc0, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a2l))), m2));
                                        acc21 = vfmaq_f32(acc21, sc1, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a2l)), m2));
                                        acc22 = vfmaq_f32(acc22, sc2, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a2h))), m2));
                                        acc23 = vfmaq_f32(acc23, sc3, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a2h)), m2));
                                    }
                                }
                            }
                        }

                        vst1q_f32(tile_acc[s0 + 0] + 0, acc00); vst1q_f32(tile_acc[s0 + 0] + 4, acc01);
                        vst1q_f32(tile_acc[s0 + 0] + 8, acc02); vst1q_f32(tile_acc[s0 + 0] + 12, acc03);
                        if (sc_g >= 2) {
                            vst1q_f32(tile_acc[s0 + 1] + 0, acc10); vst1q_f32(tile_acc[s0 + 1] + 4, acc11);
                            vst1q_f32(tile_acc[s0 + 1] + 8, acc12); vst1q_f32(tile_acc[s0 + 1] + 12, acc13);
                        }
                        if (sc_g >= 3) {
                            vst1q_f32(tile_acc[s0 + 2] + 0, acc20); vst1q_f32(tile_acc[s0 + 2] + 4, acc21);
                            vst1q_f32(tile_acc[s0 + 2] + 8, acc22); vst1q_f32(tile_acc[s0 + 2] + 12, acc23);
                        }
                        if (sc_g == 4) {
                            vst1q_f32(tile_acc[s0 + 3] + 0, acc30); vst1q_f32(tile_acc[s0 + 3] + 4, acc31);
                            vst1q_f32(tile_acc[s0 + 3] + 8, acc32); vst1q_f32(tile_acc[s0 + 3] + 12, acc33);
                        }
                    }
                }

                const int base_r = t * 16;
                if (base_r + 16 <= d_out) {
                    for (int kk = 0; kk < kc; kk++) {
                        float* yk = y + (size_t)(k0 + kk) * (size_t)d_out + (size_t)base_r;
                        if (dual) {
                            const float* hi = tile_acc[kk * 2];
                            const float* lo = tile_acc[kk * 2 + 1];
                            vst1q_f32(yk + 0,  vaddq_f32(vld1q_f32(hi + 0),  vld1q_f32(lo + 0)));
                            vst1q_f32(yk + 4,  vaddq_f32(vld1q_f32(hi + 4),  vld1q_f32(lo + 4)));
                            vst1q_f32(yk + 8,  vaddq_f32(vld1q_f32(hi + 8),  vld1q_f32(lo + 8)));
                            vst1q_f32(yk + 12, vaddq_f32(vld1q_f32(hi + 12), vld1q_f32(lo + 12)));
                        } else {
                            const float* src = tile_acc[kk];
                            vst1q_f32(yk + 0,  vld1q_f32(src + 0));
                            vst1q_f32(yk + 4,  vld1q_f32(src + 4));
                            vst1q_f32(yk + 8,  vld1q_f32(src + 8));
                            vst1q_f32(yk + 12, vld1q_f32(src + 12));
                        }
                    }
                } else {
                    const int valid = d_out - base_r;
                    for (int kk = 0; kk < kc; kk++) {
                        float* yk = y + (size_t)(k0 + kk) * (size_t)d_out + (size_t)base_r;
                        if (dual) {
                            for (int r = 0; r < valid; r++) yk[r] = tile_acc[kk * 2][r] + tile_acc[kk * 2 + 1][r];
                        } else {
                            for (int r = 0; r < valid; r++) yk[r] = tile_acc[kk][r];
                        }
                    }
                }
                if (tile_skips) tile_skips[t] = t_skip;
            }
        } else
#endif
        {
        int ati;
#if HAS_OMP
        #pragma omp parallel for schedule(static)
#endif
        for (ati = 0; ati < n_active_tiles; ati++) {
            const int t = active_tiles ? active_tiles[ati] : ati;
            const uint8_t* tptr = packed + (size_t)t * (size_t)n_blks * tile_blk_bytes;
            int32_t t_skip = col_skipped_per_tile;

#if HAS_NEON && (defined(__aarch64__) || defined(_M_ARM64))
            const uint8x16_t mask0f = vdupq_n_u8(0x0F);
            float32x4_t acc00 = vdupq_n_f32(0.0f), acc01 = vdupq_n_f32(0.0f), acc02 = vdupq_n_f32(0.0f), acc03 = vdupq_n_f32(0.0f);
            float32x4_t acc10 = vdupq_n_f32(0.0f), acc11 = vdupq_n_f32(0.0f), acc12 = vdupq_n_f32(0.0f), acc13 = vdupq_n_f32(0.0f);
            float32x4_t acc20 = vdupq_n_f32(0.0f), acc21 = vdupq_n_f32(0.0f), acc22 = vdupq_n_f32(0.0f), acc23 = vdupq_n_f32(0.0f);
            float32x4_t acc30 = vdupq_n_f32(0.0f), acc31 = vdupq_n_f32(0.0f), acc32 = vdupq_n_f32(0.0f), acc33 = vdupq_n_f32(0.0f);

            for (int ai = 0; ai < n_active_blks; ai++) {
                const int blk = active_blks[ai];
                if (need_norm) {
                    float max_xnorm = 0.0f;
                    for (int kk = 0; kk < kc; kk++) {
                        float xn = blk_xnorm[(size_t)kk * (size_t)n_blks + (size_t)blk];
                        if (xn > max_xnorm) max_xnorm = xn;
                    }
                    float wnorm = fp16_to_fp32(tile_norm_fp16[(size_t)t * (size_t)n_blks + (size_t)blk]);
                    if (wnorm * max_xnorm <= cs_skip_thresh) {
                        t_skip++;
                        continue;
                    }
                }
                const uint8_t* bptr = tptr + (size_t)blk * tile_blk_bytes;
                if (ai + 1 < n_active_blks) {
                    const uint8_t* nptr = tptr + (size_t)active_blks[ai + 1] * tile_blk_bytes;
                    neon_prefetch(nptr);
                    neon_prefetch(nptr + 256);
                    neon_prefetch(nptr + 512);
                }

                if (need_sum) {
                    uint16_t raw_bi[8];
                    memcpy(raw_bi, bptr + 32, 16);
                    float16x8_t hbi0 = vreinterpretq_f16_u16(vld1q_u16(raw_bi));
                    memcpy(raw_bi, bptr + 48, 16);
                    float16x8_t hbi1 = vreinterpretq_f16_u16(vld1q_u16(raw_bi));
                    float32x4_t bi0 = vcvt_f32_f16(vget_low_f16(hbi0));
                    float32x4_t bi1 = vcvt_f32_f16(vget_high_f16(hbi0));
                    float32x4_t bi2 = vcvt_f32_f16(vget_low_f16(hbi1));
                    float32x4_t bi3 = vcvt_f32_f16(vget_high_f16(hbi1));
                    float sx0 = blk_xsum[0 * n_blks + blk];
                    acc00 = vfmaq_n_f32(acc00, bi0, sx0); acc01 = vfmaq_n_f32(acc01, bi1, sx0);
                    acc02 = vfmaq_n_f32(acc02, bi2, sx0); acc03 = vfmaq_n_f32(acc03, bi3, sx0);
                    if (dual) {
                        if (kc == 2) {
                            float sx1 = blk_xsum[1 * n_blks + blk];
                            acc20 = vfmaq_n_f32(acc20, bi0, sx1); acc21 = vfmaq_n_f32(acc21, bi1, sx1);
                            acc22 = vfmaq_n_f32(acc22, bi2, sx1); acc23 = vfmaq_n_f32(acc23, bi3, sx1);
                        }
                    } else {
                        if (kc >= 2) {
                            float sx1 = blk_xsum[1 * n_blks + blk];
                            acc10 = vfmaq_n_f32(acc10, bi0, sx1); acc11 = vfmaq_n_f32(acc11, bi1, sx1);
                            acc12 = vfmaq_n_f32(acc12, bi2, sx1); acc13 = vfmaq_n_f32(acc13, bi3, sx1);
                        }
                        if (kc >= 3) {
                            float sx2 = blk_xsum[2 * n_blks + blk];
                            acc20 = vfmaq_n_f32(acc20, bi0, sx2); acc21 = vfmaq_n_f32(acc21, bi1, sx2);
                            acc22 = vfmaq_n_f32(acc22, bi2, sx2); acc23 = vfmaq_n_f32(acc23, bi3, sx2);
                        }
                        if (kc == 4) {
                            float sx3 = blk_xsum[3 * n_blks + blk];
                            acc30 = vfmaq_n_f32(acc30, bi0, sx3); acc31 = vfmaq_n_f32(acc31, bi1, sx3);
                            acc32 = vfmaq_n_f32(acc32, bi2, sx3); acc33 = vfmaq_n_f32(acc33, bi3, sx3);
                        }
                    }
                }

                const uint8_t* planes = bptr + hdr_bytes;
                NEON_ALIGN64 float sc_blk[16];
                {
                    uint16_t raw_sc[8];
                    memcpy(raw_sc, bptr + 0, 16);
                    float16x8_t hsc0 = vreinterpretq_f16_u16(vld1q_u16(raw_sc));
                    memcpy(raw_sc, bptr + 16, 16);
                    float16x8_t hsc1 = vreinterpretq_f16_u16(vld1q_u16(raw_sc));
                    vst1q_f32(sc_blk + 0,  vcvt_f32_f16(vget_low_f16(hsc0)));
                    vst1q_f32(sc_blk + 4,  vcvt_f32_f16(vget_high_f16(hsc0)));
                    vst1q_f32(sc_blk + 8,  vcvt_f32_f16(vget_low_f16(hsc1)));
                    vst1q_f32(sc_blk + 12, vcvt_f32_f16(vget_high_f16(hsc1)));
                }
                for (int p = 0; p < eff_bits; p++) {
                    const uint8_t* pptr = planes + (size_t)p * (size_t)bytes_per_plane_blk;
                    float weight_p = t16c_plane_weight(gauss_lut, eff_bits, p);
                    if (slots == 4) {
                        for (int sb = 0; sb < subs_per_blk; sb++) {
                            int g_sub = blk * subs_per_blk + sb;
                            int16x8_t a0l = vdupq_n_s16(0), a0h = vdupq_n_s16(0);
                            int16x8_t a1l = vdupq_n_s16(0), a1h = vdupq_n_s16(0);
                            int16x8_t a2l = vdupq_n_s16(0), a2h = vdupq_n_s16(0);
                            int16x8_t a3l = vdupq_n_s16(0), a3h = vdupq_n_s16(0);
                            const int8_t* lp = luts_i4 + (size_t)g_sub * 8u * 128u;
                            for (int bo = 0; bo < 8; bo++, lp += 128) {
                                uint8x16_t bv = vld1q_u8(pptr + ((size_t)sb * 8 + (size_t)bo) * 16);
                                uint8x16_t nhi = vshrq_n_u8(bv, 4);
                                uint8x16_t nlo = vandq_u8(bv, mask0f);
                                int8x16_t rh, rl;
                                rh = vqtbl1q_s8(vld1q_s8(lp + 0),  nhi); rl = vqtbl1q_s8(vld1q_s8(lp + 16),  nlo);
                                a0l = vaddq_s16(a0l, vaddl_s8(vget_low_s8(rh), vget_low_s8(rl)));
                                a0h = vaddq_s16(a0h, vaddl_high_s8(rh, rl));
                                rh = vqtbl1q_s8(vld1q_s8(lp + 32), nhi); rl = vqtbl1q_s8(vld1q_s8(lp + 48),  nlo);
                                a1l = vaddq_s16(a1l, vaddl_s8(vget_low_s8(rh), vget_low_s8(rl)));
                                a1h = vaddq_s16(a1h, vaddl_high_s8(rh, rl));
                                rh = vqtbl1q_s8(vld1q_s8(lp + 64), nhi); rl = vqtbl1q_s8(vld1q_s8(lp + 80),  nlo);
                                a2l = vaddq_s16(a2l, vaddl_s8(vget_low_s8(rh), vget_low_s8(rl)));
                                a2h = vaddq_s16(a2h, vaddl_high_s8(rh, rl));
                                rh = vqtbl1q_s8(vld1q_s8(lp + 96), nhi); rl = vqtbl1q_s8(vld1q_s8(lp + 112), nlo);
                                a3l = vaddq_s16(a3l, vaddl_s8(vget_low_s8(rh), vget_low_s8(rl)));
                                a3h = vaddq_s16(a3h, vaddl_high_s8(rh, rl));
                            }
                            float32x4_t sc0 = vld1q_f32(sc_blk + 0);
                            float32x4_t sc1 = vld1q_f32(sc_blk + 4);
                            float32x4_t sc2 = vld1q_f32(sc_blk + 8);
                            float32x4_t sc3 = vld1q_f32(sc_blk + 12);
                            float m0 = sub_scales[0 * n_subs + g_sub] * weight_p;
                            float m1 = sub_scales[1 * n_subs + g_sub] * weight_p;
                            float m2 = sub_scales[2 * n_subs + g_sub] * weight_p;
                            float m3 = sub_scales[3 * n_subs + g_sub] * weight_p;
                            acc00 = vfmaq_f32(acc00, sc0, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a0l))), m0));
                            acc01 = vfmaq_f32(acc01, sc1, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a0l)), m0));
                            acc02 = vfmaq_f32(acc02, sc2, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a0h))), m0));
                            acc03 = vfmaq_f32(acc03, sc3, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a0h)), m0));
                            acc10 = vfmaq_f32(acc10, sc0, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a1l))), m1));
                            acc11 = vfmaq_f32(acc11, sc1, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a1l)), m1));
                            acc12 = vfmaq_f32(acc12, sc2, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a1h))), m1));
                            acc13 = vfmaq_f32(acc13, sc3, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a1h)), m1));
                            acc20 = vfmaq_f32(acc20, sc0, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a2l))), m2));
                            acc21 = vfmaq_f32(acc21, sc1, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a2l)), m2));
                            acc22 = vfmaq_f32(acc22, sc2, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a2h))), m2));
                            acc23 = vfmaq_f32(acc23, sc3, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a2h)), m2));
                            acc30 = vfmaq_f32(acc30, sc0, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a3l))), m3));
                            acc31 = vfmaq_f32(acc31, sc1, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a3l)), m3));
                            acc32 = vfmaq_f32(acc32, sc2, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a3h))), m3));
                            acc33 = vfmaq_f32(acc33, sc3, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a3h)), m3));
                        }
                    } else {
                        for (int sb = 0; sb < subs_per_blk; sb++) {
                            int g_sub = blk * subs_per_blk + sb;
                            int16x8_t a0l = vdupq_n_s16(0), a0h = vdupq_n_s16(0);
                            int16x8_t a1l = vdupq_n_s16(0), a1h = vdupq_n_s16(0);
                            int16x8_t a2l = vdupq_n_s16(0), a2h = vdupq_n_s16(0);
                            const int8_t* l0 = luts + 0 * luts_stride;
                            const int8_t* l1 = (slots >= 2) ? (luts + 1 * luts_stride) : l0;
                            const int8_t* l2 = (slots >= 3) ? (luts + 2 * luts_stride) : l0;
                            for (int bo = 0; bo < 8; bo++) {
                                size_t loff = ((size_t)g_sub * 8 + (size_t)bo) * 32;
                                uint8x16_t bv = vld1q_u8(pptr + ((size_t)sb * 8 + (size_t)bo) * 16);
                                uint8x16_t nhi = vshrq_n_u8(bv, 4);
                                uint8x16_t nlo = vandq_u8(bv, mask0f);
                                int8x16_t rh = vqtbl1q_s8(vld1q_s8(l0 + loff), nhi);
                                int8x16_t rl = vqtbl1q_s8(vld1q_s8(l0 + loff + 16), nlo);
                                a0l = vaddq_s16(a0l, vaddl_s8(vget_low_s8(rh), vget_low_s8(rl)));
                                a0h = vaddq_s16(a0h, vaddl_high_s8(rh, rl));
                                if (slots >= 2) {
                                    rh = vqtbl1q_s8(vld1q_s8(l1 + loff), nhi); rl = vqtbl1q_s8(vld1q_s8(l1 + loff + 16), nlo);
                                    a1l = vaddq_s16(a1l, vaddl_s8(vget_low_s8(rh), vget_low_s8(rl)));
                                    a1h = vaddq_s16(a1h, vaddl_high_s8(rh, rl));
                                }
                                if (slots >= 3) {
                                    rh = vqtbl1q_s8(vld1q_s8(l2 + loff), nhi); rl = vqtbl1q_s8(vld1q_s8(l2 + loff + 16), nlo);
                                    a2l = vaddq_s16(a2l, vaddl_s8(vget_low_s8(rh), vget_low_s8(rl)));
                                    a2h = vaddq_s16(a2h, vaddl_high_s8(rh, rl));
                                }
                            }
                            float32x4_t sc0 = vld1q_f32(sc_blk + 0);
                            float32x4_t sc1 = vld1q_f32(sc_blk + 4);
                            float32x4_t sc2 = vld1q_f32(sc_blk + 8);
                            float32x4_t sc3 = vld1q_f32(sc_blk + 12);
                            float m0 = sub_scales[0 * n_subs + g_sub] * weight_p;
                            acc00 = vfmaq_f32(acc00, sc0, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a0l))), m0));
                            acc01 = vfmaq_f32(acc01, sc1, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a0l)), m0));
                            acc02 = vfmaq_f32(acc02, sc2, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a0h))), m0));
                            acc03 = vfmaq_f32(acc03, sc3, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a0h)), m0));
                            if (slots >= 2) {
                                float m1 = sub_scales[1 * n_subs + g_sub] * weight_p;
                                acc10 = vfmaq_f32(acc10, sc0, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a1l))), m1));
                                acc11 = vfmaq_f32(acc11, sc1, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a1l)), m1));
                                acc12 = vfmaq_f32(acc12, sc2, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a1h))), m1));
                                acc13 = vfmaq_f32(acc13, sc3, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a1h)), m1));
                            }
                            if (slots >= 3) {
                                float m2 = sub_scales[2 * n_subs + g_sub] * weight_p;
                                acc20 = vfmaq_f32(acc20, sc0, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a2l))), m2));
                                acc21 = vfmaq_f32(acc21, sc1, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a2l)), m2));
                                acc22 = vfmaq_f32(acc22, sc2, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a2h))), m2));
                                acc23 = vfmaq_f32(acc23, sc3, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a2h)), m2));
                            }
                        }
                    }
                }
            }

            if (dual) {
                acc00 = vaddq_f32(acc00, acc10); acc01 = vaddq_f32(acc01, acc11);
                acc02 = vaddq_f32(acc02, acc12); acc03 = vaddq_f32(acc03, acc13);
                if (kc == 2) {
                    acc10 = vaddq_f32(acc20, acc30); acc11 = vaddq_f32(acc21, acc31);
                    acc12 = vaddq_f32(acc22, acc32); acc13 = vaddq_f32(acc23, acc33);
                }
            }

            int base_r = t * 16;
            if (base_r + 16 <= d_out) {
                float* y0 = y + (size_t)k0 * (size_t)d_out + (size_t)base_r;
                vst1q_f32(y0 + 0, acc00); vst1q_f32(y0 + 4, acc01); vst1q_f32(y0 + 8, acc02); vst1q_f32(y0 + 12, acc03);
                if (kc >= 2) {
                    float* y1 = y + (size_t)(k0 + 1) * (size_t)d_out + (size_t)base_r;
                    vst1q_f32(y1 + 0, acc10); vst1q_f32(y1 + 4, acc11); vst1q_f32(y1 + 8, acc12); vst1q_f32(y1 + 12, acc13);
                }
                if (kc >= 3) {
                    float* y2 = y + (size_t)(k0 + 2) * (size_t)d_out + (size_t)base_r;
                    vst1q_f32(y2 + 0, acc20); vst1q_f32(y2 + 4, acc21); vst1q_f32(y2 + 8, acc22); vst1q_f32(y2 + 12, acc23);
                }
                if (kc == 4) {
                    float* y3 = y + (size_t)(k0 + 3) * (size_t)d_out + (size_t)base_r;
                    vst1q_f32(y3 + 0, acc30); vst1q_f32(y3 + 4, acc31); vst1q_f32(y3 + 8, acc32); vst1q_f32(y3 + 12, acc33);
                }
            } else {
                float tmp[4][16];
                vst1q_f32(tmp[0] + 0, acc00); vst1q_f32(tmp[0] + 4, acc01); vst1q_f32(tmp[0] + 8, acc02); vst1q_f32(tmp[0] + 12, acc03);
                if (kc >= 2) { vst1q_f32(tmp[1] + 0, acc10); vst1q_f32(tmp[1] + 4, acc11); vst1q_f32(tmp[1] + 8, acc12); vst1q_f32(tmp[1] + 12, acc13); }
                if (kc >= 3) { vst1q_f32(tmp[2] + 0, acc20); vst1q_f32(tmp[2] + 4, acc21); vst1q_f32(tmp[2] + 8, acc22); vst1q_f32(tmp[2] + 12, acc23); }
                if (kc == 4) { vst1q_f32(tmp[3] + 0, acc30); vst1q_f32(tmp[3] + 4, acc31); vst1q_f32(tmp[3] + 8, acc32); vst1q_f32(tmp[3] + 12, acc33); }
                int valid = d_out - base_r;
                for (int kk = 0; kk < kc; kk++) {
                    float* yk = y + (size_t)(k0 + kk) * (size_t)d_out + (size_t)base_r;
                    for (int r = 0; r < valid; r++) yk[r] = tmp[kk][r];
                }
            }
#elif HAS_AVX2
            const __m128i mask0f = _mm_set1_epi8(0x0F);
            __m256 acc0_lo = _mm256_setzero_ps(), acc0_hi = _mm256_setzero_ps();
            __m256 acc1_lo = _mm256_setzero_ps(), acc1_hi = _mm256_setzero_ps();
            __m256 acc2_lo = _mm256_setzero_ps(), acc2_hi = _mm256_setzero_ps();
            __m256 acc3_lo = _mm256_setzero_ps(), acc3_hi = _mm256_setzero_ps();

            for (int ai = 0; ai < n_active_blks; ai++) {
                const int blk = active_blks[ai];
                if (need_norm) {
                    float max_xnorm = 0.0f;
                    for (int kk = 0; kk < kc; kk++) {
                        float xn = blk_xnorm[(size_t)kk * (size_t)n_blks + (size_t)blk];
                        if (xn > max_xnorm) max_xnorm = xn;
                    }
                    float wnorm = fp16_to_fp32(tile_norm_fp16[(size_t)t * (size_t)n_blks + (size_t)blk]);
                    if (wnorm * max_xnorm <= cs_skip_thresh) {
                        t_skip++;
                        continue;
                    }
                }
                const uint8_t* bptr = tptr + (size_t)blk * tile_blk_bytes;
                if (ai + 2 < n_active_blks) {
                    neon_prefetch(tptr + (size_t)active_blks[ai + 2] * tile_blk_bytes);
                }
                float sc_arr[16];
                for (int r = 0; r < 16; r++) sc_arr[r] = fp16_to_fp32(read_u16_le(bptr + (size_t)r * 2));
                __m256 sc_lo = _mm256_loadu_ps(sc_arr + 0);
                __m256 sc_hi = _mm256_loadu_ps(sc_arr + 8);

                if (need_sum) {
                    float bi_arr[16];
                    for (int r = 0; r < 16; r++) bi_arr[r] = fp16_to_fp32(read_u16_le(bptr + 32 + (size_t)r * 2));
                    __m256 bi_lo = _mm256_loadu_ps(bi_arr + 0);
                    __m256 bi_hi = _mm256_loadu_ps(bi_arr + 8);
                    __m256 sx0 = _mm256_set1_ps(blk_xsum[0 * n_blks + blk]);
                    acc0_lo = _mm256_fmadd_ps(bi_lo, sx0, acc0_lo);
                    acc0_hi = _mm256_fmadd_ps(bi_hi, sx0, acc0_hi);
                    if (dual) {
                        if (kc == 2) {
                            __m256 sx1 = _mm256_set1_ps(blk_xsum[1 * n_blks + blk]);
                            acc2_lo = _mm256_fmadd_ps(bi_lo, sx1, acc2_lo);
                            acc2_hi = _mm256_fmadd_ps(bi_hi, sx1, acc2_hi);
                        }
                    } else {
                        if (kc >= 2) {
                            __m256 sx1 = _mm256_set1_ps(blk_xsum[1 * n_blks + blk]);
                            acc1_lo = _mm256_fmadd_ps(bi_lo, sx1, acc1_lo);
                            acc1_hi = _mm256_fmadd_ps(bi_hi, sx1, acc1_hi);
                        }
                        if (kc >= 3) {
                            __m256 sx2 = _mm256_set1_ps(blk_xsum[2 * n_blks + blk]);
                            acc2_lo = _mm256_fmadd_ps(bi_lo, sx2, acc2_lo);
                            acc2_hi = _mm256_fmadd_ps(bi_hi, sx2, acc2_hi);
                        }
                        if (kc == 4) {
                            __m256 sx3 = _mm256_set1_ps(blk_xsum[3 * n_blks + blk]);
                            acc3_lo = _mm256_fmadd_ps(bi_lo, sx3, acc3_lo);
                            acc3_hi = _mm256_fmadd_ps(bi_hi, sx3, acc3_hi);
                        }
                    }
                }

                const uint8_t* planes = bptr + hdr_bytes;
                for (int p = 0; p < eff_bits; p++) {
                    const uint8_t* pptr = planes + (size_t)p * (size_t)bytes_per_plane_blk;
                    float weight_p = t16c_plane_weight(gauss_lut, eff_bits, p);
                    for (int sb = 0; sb < subs_per_blk; sb++) {
                        int g_sub = blk * subs_per_blk + sb;
                        __m256i a0 = _mm256_setzero_si256();
                        __m256i a1 = _mm256_setzero_si256();
                        __m256i a2 = _mm256_setzero_si256();
                        __m256i a3 = _mm256_setzero_si256();
                        const int8_t* l0 = luts + 0 * luts_stride;
                        const int8_t* l1 = (slots >= 2) ? (luts + 1 * luts_stride) : l0;
                        const int8_t* l2 = (slots >= 3) ? (luts + 2 * luts_stride) : l0;
                        const int8_t* l3 = (slots == 4) ? (luts + 3 * luts_stride) : l0;
                        for (int bo = 0; bo < 8; bo++) {
                            size_t loff = ((size_t)g_sub * 8 + (size_t)bo) * 32;
                            __m128i bv = _mm_loadu_si128((const __m128i*)(pptr + ((size_t)sb * 8 + (size_t)bo) * 16));
                            __m128i nhi = _mm_and_si128(_mm_srli_epi16(bv, 4), mask0f);
                            __m128i nlo = _mm_and_si128(bv, mask0f);
                            {
                                __m128i rh = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(l0 + loff)), nhi);
                                __m128i rl = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(l0 + loff + 16)), nlo);
                                a0 = _mm256_add_epi16(a0, _mm256_add_epi16(_mm256_cvtepi8_epi16(rh), _mm256_cvtepi8_epi16(rl)));
                            }
                            if (slots >= 2) {
                                __m128i rh = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(l1 + loff)), nhi);
                                __m128i rl = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(l1 + loff + 16)), nlo);
                                a1 = _mm256_add_epi16(a1, _mm256_add_epi16(_mm256_cvtepi8_epi16(rh), _mm256_cvtepi8_epi16(rl)));
                            }
                            if (slots >= 3) {
                                __m128i rh = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(l2 + loff)), nhi);
                                __m128i rl = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(l2 + loff + 16)), nlo);
                                a2 = _mm256_add_epi16(a2, _mm256_add_epi16(_mm256_cvtepi8_epi16(rh), _mm256_cvtepi8_epi16(rl)));
                            }
                            if (slots == 4) {
                                __m128i rh = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(l3 + loff)), nhi);
                                __m128i rl = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(l3 + loff + 16)), nlo);
                                a3 = _mm256_add_epi16(a3, _mm256_add_epi16(_mm256_cvtepi8_epi16(rh), _mm256_cvtepi8_epi16(rl)));
                            }
                        }
                        {
                            __m256 m0 = _mm256_set1_ps(sub_scales[0 * n_subs + g_sub] * weight_p);
                            __m256 f_lo = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(a0))), m0);
                            __m256 f_hi = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(a0, 1))), m0);
                            acc0_lo = _mm256_fmadd_ps(sc_lo, f_lo, acc0_lo);
                            acc0_hi = _mm256_fmadd_ps(sc_hi, f_hi, acc0_hi);
                        }
                        if (slots >= 2) {
                            __m256 m1 = _mm256_set1_ps(sub_scales[1 * n_subs + g_sub] * weight_p);
                            __m256 f_lo = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(a1))), m1);
                            __m256 f_hi = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(a1, 1))), m1);
                            acc1_lo = _mm256_fmadd_ps(sc_lo, f_lo, acc1_lo);
                            acc1_hi = _mm256_fmadd_ps(sc_hi, f_hi, acc1_hi);
                        }
                        if (slots >= 3) {
                            __m256 m2 = _mm256_set1_ps(sub_scales[2 * n_subs + g_sub] * weight_p);
                            __m256 f_lo = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(a2))), m2);
                            __m256 f_hi = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(a2, 1))), m2);
                            acc2_lo = _mm256_fmadd_ps(sc_lo, f_lo, acc2_lo);
                            acc2_hi = _mm256_fmadd_ps(sc_hi, f_hi, acc2_hi);
                        }
                        if (slots == 4) {
                            __m256 m3 = _mm256_set1_ps(sub_scales[3 * n_subs + g_sub] * weight_p);
                            __m256 f_lo = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(a3))), m3);
                            __m256 f_hi = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(a3, 1))), m3);
                            acc3_lo = _mm256_fmadd_ps(sc_lo, f_lo, acc3_lo);
                            acc3_hi = _mm256_fmadd_ps(sc_hi, f_hi, acc3_hi);
                        }
                    }
                }
            }

            if (dual) {
                acc0_lo = _mm256_add_ps(acc0_lo, acc1_lo);
                acc0_hi = _mm256_add_ps(acc0_hi, acc1_hi);
                if (kc == 2) {
                    acc1_lo = _mm256_add_ps(acc2_lo, acc3_lo);
                    acc1_hi = _mm256_add_ps(acc2_hi, acc3_hi);
                }
            }

            int base_r = t * 16;
            if (base_r + 16 <= d_out) {
                float* y0 = y + (size_t)k0 * (size_t)d_out + (size_t)base_r;
                _mm256_storeu_ps(y0 + 0, acc0_lo); _mm256_storeu_ps(y0 + 8, acc0_hi);
                if (kc >= 2) { float* y1 = y + (size_t)(k0 + 1) * (size_t)d_out + (size_t)base_r; _mm256_storeu_ps(y1 + 0, acc1_lo); _mm256_storeu_ps(y1 + 8, acc1_hi); }
                if (kc >= 3) { float* y2 = y + (size_t)(k0 + 2) * (size_t)d_out + (size_t)base_r; _mm256_storeu_ps(y2 + 0, acc2_lo); _mm256_storeu_ps(y2 + 8, acc2_hi); }
                if (kc == 4) { float* y3 = y + (size_t)(k0 + 3) * (size_t)d_out + (size_t)base_r; _mm256_storeu_ps(y3 + 0, acc3_lo); _mm256_storeu_ps(y3 + 8, acc3_hi); }
            } else {
                float tmp[4][16];
                _mm256_storeu_ps(tmp[0] + 0, acc0_lo); _mm256_storeu_ps(tmp[0] + 8, acc0_hi);
                if (kc >= 2) { _mm256_storeu_ps(tmp[1] + 0, acc1_lo); _mm256_storeu_ps(tmp[1] + 8, acc1_hi); }
                if (kc >= 3) { _mm256_storeu_ps(tmp[2] + 0, acc2_lo); _mm256_storeu_ps(tmp[2] + 8, acc2_hi); }
                if (kc == 4) { _mm256_storeu_ps(tmp[3] + 0, acc3_lo); _mm256_storeu_ps(tmp[3] + 8, acc3_hi); }
                int valid = d_out - base_r;
                for (int kk = 0; kk < kc; kk++) {
                    float* yk = y + (size_t)(k0 + kk) * (size_t)d_out + (size_t)base_r;
                    for (int r = 0; r < valid; r++) yk[r] = tmp[kk][r];
                }
            }
#else
            float acc[4][16];
            for (int kk = 0; kk < 4; kk++) for (int r = 0; r < 16; r++) acc[kk][r] = 0.0f;

            for (int ai = 0; ai < n_active_blks; ai++) {
                const int blk = active_blks[ai];
                if (need_norm) {
                    float max_xnorm = 0.0f;
                    for (int kk = 0; kk < kc; kk++) {
                        float xn = blk_xnorm[(size_t)kk * (size_t)n_blks + (size_t)blk];
                        if (xn > max_xnorm) max_xnorm = xn;
                    }
                    float wnorm = fp16_to_fp32(tile_norm_fp16[(size_t)t * (size_t)n_blks + (size_t)blk]);
                    if (wnorm * max_xnorm <= cs_skip_thresh) {
                        t_skip++;
                        continue;
                    }
                }
                const uint8_t* bptr = tptr + (size_t)blk * tile_blk_bytes;
                float sc_arr[16], bi_arr[16];
                for (int r = 0; r < 16; r++) {
                    sc_arr[r] = fp16_to_fp32(read_u16_le(bptr + (size_t)r * 2));
                    bi_arr[r] = is_asym ? fp16_to_fp32(read_u16_le(bptr + 32 + (size_t)r * 2)) : 0.0f;
                }
                if (need_sum) {
                    for (int kk = 0; kk < kc; kk++) {
                        float sx = blk_xsum[(size_t)kk * (size_t)n_blks + (size_t)blk];
                        int dst_sl = dual ? (kk * 2) : kk;
                        for (int r = 0; r < 16; r++) acc[dst_sl][r] += bi_arr[r] * sx;
                    }
                }
                const uint8_t* planes = bptr + hdr_bytes;
                for (int p = 0; p < eff_bits; p++) {
                    const uint8_t* pptr = planes + (size_t)p * (size_t)bytes_per_plane_blk;
                    float weight_p = t16c_plane_weight(gauss_lut, eff_bits, p);
                    for (int sb = 0; sb < subs_per_blk; sb++) {
                        int g_sub = blk * subs_per_blk + sb;
                        for (int sl = 0; sl < slots; sl++) {
                            const int8_t* lk = luts + (size_t)sl * luts_stride;
                            float mk = sub_scales[(size_t)sl * (size_t)n_subs + (size_t)g_sub] * weight_p;
                            int32_t sums[16] = {0};
                            for (int bo = 0; bo < 8; bo++) {
                                size_t loff = ((size_t)g_sub * 8 + (size_t)bo) * 32;
                                const uint8_t* bp = pptr + ((size_t)sb * 8 + (size_t)bo) * 16;
                                const int8_t* lut_hi = lk + loff;
                                const int8_t* lut_lo = lk + loff + 16;
                                for (int r = 0; r < 16; r++) {
                                    uint8_t bv = bp[r];
                                    sums[r] += (int32_t)lut_hi[bv >> 4] + (int32_t)lut_lo[bv & 0x0Fu];
                                }
                            }
                            for (int r = 0; r < 16; r++) {
                                acc[sl][r] += sc_arr[r] * (float)sums[r] * mk;
                            }
                        }
                    }
                }
            }
            int base_r = t * 16;
            int valid = (base_r + 16 <= d_out) ? 16 : (d_out - base_r);
            for (int kk = 0; kk < kc; kk++) {
                float* yk = y + (size_t)(k0 + kk) * (size_t)d_out + (size_t)base_r;
                if (dual) {
                    for (int r = 0; r < valid; r++) yk[r] = acc[kk * 2][r] + acc[kk * 2 + 1][r];
                } else {
                    for (int r = 0; r < valid; r++) yk[r] = acc[kk][r];
                }
            }
#endif
            if (tile_skips) tile_skips[t] = t_skip;
        }
        }
        if (tile_skips) {
            total_skipped += row_skipped_blks_per_pass;
            for (int ati2 = 0; ati2 < n_active_tiles; ati2++) {
                total_skipped += tile_skips[active_tiles ? active_tiles[ati2] : ati2];
            }
        }
    }

    if (out_skipped) *out_skipped = total_skipped;
    if (scratch_heap_owned) {
        neon_free(scratch_raw);
    }
    return 0;
}

NEON_API int gemm_bitserial_tile16_batch(
    const uint8_t* NEON_RESTRICT packed,
    const uint16_t* NEON_RESTRICT tile_norm_fp16,
    const float* NEON_RESTRICT x,
    float* NEON_RESTRICT y,
    int d_out,
    int d_in,
    int block_size,
    int n_bits,
    int is_asym,
    int k_tokens,
    float cs_skip_thresh,
    int num_threads,
    int64_t* NEON_RESTRICT out_skipped
) {
    return gemm_bitserial_tile16_sparse_batch(
        packed, tile_norm_fp16, x, y,
        d_out, d_in, block_size, n_bits, is_asym,
        k_tokens, cs_skip_thresh, NULL, NULL,
        num_threads, out_skipped
    );
}

NEON_API int gemv_bitserial_tile16_mmap(
    const uint8_t* NEON_RESTRICT packed,
    const uint16_t* NEON_RESTRICT tile_norm_fp16,
    const float* NEON_RESTRICT x,
    float* NEON_RESTRICT y,
    int d_out,
    int d_in,
    int block_size,
    int n_bits,
    int is_asym,
    float cs_skip_thresh,
    int num_threads,
    int64_t* NEON_RESTRICT out_skipped
) {
    return gemm_bitserial_tile16_batch(
        packed, tile_norm_fp16, x, y,
        d_out, d_in, block_size, n_bits, is_asym,
        1, cs_skip_thresh, num_threads, out_skipped
    );
}

static inline void t16c_accum_blk_slots(
    const uint8_t* NEON_RESTRICT bptr,
    int blk,
    int subs_per_blk,
    int n_subs,
    int n_blks,
    int eff_bits,
    int gauss_lut,
    int is_asym,
    int dual,
    int kc,
    int slots,
    size_t tile_blk_bytes,
    int bytes_per_plane_blk,
    size_t luts_stride,
    const int8_t* NEON_RESTRICT luts,
    const float* NEON_RESTRICT sub_scales,
    const float* NEON_RESTRICT blk_xsum,
    float acc[4][16]
) {
    (void)tile_blk_bytes;
#if HAS_NEON && (defined(__aarch64__) || defined(_M_ARM64))
    const uint8x16_t mask0f = vdupq_n_u8(0x0F);
    uint16_t raw_sc[8];
    memcpy(raw_sc, bptr + 0, 16);
    float16x8_t hsc0 = vreinterpretq_f16_u16(vld1q_u16(raw_sc));
    memcpy(raw_sc, bptr + 16, 16);
    float16x8_t hsc1 = vreinterpretq_f16_u16(vld1q_u16(raw_sc));
    float32x4_t sc0 = vcvt_f32_f16(vget_low_f16(hsc0));
    float32x4_t sc1 = vcvt_f32_f16(vget_high_f16(hsc0));
    float32x4_t sc2 = vcvt_f32_f16(vget_low_f16(hsc1));
    float32x4_t sc3 = vcvt_f32_f16(vget_high_f16(hsc1));

    float32x4_t a_v[4][4];
    for (int sl = 0; sl < slots; sl++) {
        a_v[sl][0] = vld1q_f32(&acc[sl][0]);
        a_v[sl][1] = vld1q_f32(&acc[sl][4]);
        a_v[sl][2] = vld1q_f32(&acc[sl][8]);
        a_v[sl][3] = vld1q_f32(&acc[sl][12]);
    }
    if (is_asym && blk_xsum) {
        uint16_t raw_bi[8];
        memcpy(raw_bi, bptr + 32, 16);
        float16x8_t hbi0 = vreinterpretq_f16_u16(vld1q_u16(raw_bi));
        memcpy(raw_bi, bptr + 48, 16);
        float16x8_t hbi1 = vreinterpretq_f16_u16(vld1q_u16(raw_bi));
        float32x4_t bi0 = vcvt_f32_f16(vget_low_f16(hbi0));
        float32x4_t bi1 = vcvt_f32_f16(vget_high_f16(hbi0));
        float32x4_t bi2 = vcvt_f32_f16(vget_low_f16(hbi1));
        float32x4_t bi3 = vcvt_f32_f16(vget_high_f16(hbi1));
        for (int kk = 0; kk < kc; kk++) {
            float sx = blk_xsum[(size_t)kk * (size_t)n_blks + (size_t)blk];
            int dst_sl = dual ? (kk * 2) : kk;
            a_v[dst_sl][0] = vfmaq_n_f32(a_v[dst_sl][0], bi0, sx);
            a_v[dst_sl][1] = vfmaq_n_f32(a_v[dst_sl][1], bi1, sx);
            a_v[dst_sl][2] = vfmaq_n_f32(a_v[dst_sl][2], bi2, sx);
            a_v[dst_sl][3] = vfmaq_n_f32(a_v[dst_sl][3], bi3, sx);
        }
    }
    const uint8_t* planes = bptr + (is_asym ? 64 : 32);
    for (int p = 0; p < eff_bits; p++) {
        const uint8_t* pptr = planes + (size_t)p * (size_t)bytes_per_plane_blk;
        float weight_p = t16c_plane_weight(gauss_lut, eff_bits, p);
        for (int sb = 0; sb < subs_per_blk; sb++) {
            int g_sub = blk * subs_per_blk + sb;
            for (int sl = 0; sl < slots; sl++) {
                const int8_t* lk = luts + (size_t)sl * luts_stride;
                int16x8_t al = vdupq_n_s16(0), ah = vdupq_n_s16(0);
                for (int bo = 0; bo < 8; bo++) {
                    size_t loff = ((size_t)g_sub * 8 + (size_t)bo) * 32;
                    uint8x16_t bv = vld1q_u8(pptr + ((size_t)sb * 8 + (size_t)bo) * 16);
                    int8x16_t rh = vqtbl1q_s8(vld1q_s8(lk + loff), vshrq_n_u8(bv, 4));
                    int8x16_t rl = vqtbl1q_s8(vld1q_s8(lk + loff + 16), vandq_u8(bv, mask0f));
                    al = vaddw_s8(al, vget_low_s8(rh)); ah = vaddw_high_s8(ah, rh);
                    al = vaddw_s8(al, vget_low_s8(rl)); ah = vaddw_high_s8(ah, rl);
                }
                float mk = sub_scales[(size_t)sl * (size_t)n_subs + (size_t)g_sub] * weight_p;
                a_v[sl][0] = vfmaq_f32(a_v[sl][0], sc0, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(al))), mk));
                a_v[sl][1] = vfmaq_f32(a_v[sl][1], sc1, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(al)), mk));
                a_v[sl][2] = vfmaq_f32(a_v[sl][2], sc2, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(ah))), mk));
                a_v[sl][3] = vfmaq_f32(a_v[sl][3], sc3, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(ah)), mk));
            }
        }
    }
    for (int sl = 0; sl < slots; sl++) {
        vst1q_f32(&acc[sl][0], a_v[sl][0]);
        vst1q_f32(&acc[sl][4], a_v[sl][1]);
        vst1q_f32(&acc[sl][8], a_v[sl][2]);
        vst1q_f32(&acc[sl][12], a_v[sl][3]);
    }
#else
    float sc_arr[16], bi_arr[16];
    for (int r = 0; r < 16; r++) {
        sc_arr[r] = fp16_to_fp32(read_u16_le(bptr + (size_t)r * 2));
        bi_arr[r] = is_asym ? fp16_to_fp32(read_u16_le(bptr + 32 + (size_t)r * 2)) : 0.0f;
    }
    if (is_asym && blk_xsum) {
        for (int kk = 0; kk < kc; kk++) {
            float sx = blk_xsum[(size_t)kk * (size_t)n_blks + (size_t)blk];
            int dst_sl = dual ? (kk * 2) : kk;
            for (int r = 0; r < 16; r++) acc[dst_sl][r] += bi_arr[r] * sx;
        }
    }
    const uint8_t* planes = bptr + (is_asym ? 64 : 32);
    for (int p = 0; p < eff_bits; p++) {
        const uint8_t* pptr = planes + (size_t)p * (size_t)bytes_per_plane_blk;
        float weight_p = t16c_plane_weight(gauss_lut, eff_bits, p);
        for (int sb = 0; sb < subs_per_blk; sb++) {
            int g_sub = blk * subs_per_blk + sb;
            for (int sl = 0; sl < slots; sl++) {
                const int8_t* lk = luts + (size_t)sl * luts_stride;
                float mk = sub_scales[(size_t)sl * (size_t)n_subs + (size_t)g_sub] * weight_p;
                int32_t sums[16] = {0};
                for (int bo = 0; bo < 8; bo++) {
                    size_t loff = ((size_t)g_sub * 8 + (size_t)bo) * 32;
                    const uint8_t* bp = pptr + ((size_t)sb * 8 + (size_t)bo) * 16;
                    const int8_t* lut_hi = lk + loff;
                    const int8_t* lut_lo = lk + loff + 16;
                    for (int r = 0; r < 16; r++) {
                        uint8_t bv = bp[r];
                        sums[r] += (int32_t)lut_hi[bv >> 4] + (int32_t)lut_lo[bv & 0x0Fu];
                    }
                }
                for (int r = 0; r < 16; r++) {
                    acc[sl][r] += sc_arr[r] * (float)sums[r] * mk;
                }
            }
        }
    }
#endif
}

NEON_API int lmhead_cs_top1_tile16_batch(
    const uint8_t* NEON_RESTRICT packed,
    const uint16_t* NEON_RESTRICT tile_norm_fp16,
    const float* NEON_RESTRICT x,
    float* NEON_RESTRICT y,
    int d_out,
    int d_in,
    int block_size,
    int n_bits,
    int is_asym,
    int k_tokens,
    int top_k_keep,
    float sieve_frac,
    int num_threads,
    int64_t* NEON_RESTRICT out_skipped_blks
) {
    if (!packed || !x || !y || d_out <= 0 || d_in <= 0 || k_tokens < 0) return -1;
    if (k_tokens == 0) {
        if (out_skipped_blks) *out_skipped_blks = 0;
        return 0;
    }
    if ((block_size != 64 && block_size != 128 && block_size != 256) || (d_in % block_size) != 0) return -1;
    const int eff_bits = n_bits & 0x7F;
    const int n_blks = d_in / block_size;
    if (!tile_norm_fp16 || eff_bits == 5 || eff_bits == 8 || n_blks > 256 || n_blks < 2) {
        return gemm_bitserial_tile16_batch(
            packed, tile_norm_fp16, x, y,
            d_out, d_in, block_size, n_bits, is_asym,
            k_tokens, 0.0f, num_threads, out_skipped_blks
        );
    }
    const int gauss_lut = (n_bits & T16C_GAUSS_BIT) != 0;
    if (eff_bits < 1 || eff_bits > 4) return -1;
    if (gauss_lut && eff_bits != 2 && eff_bits != 3) return -1;

    const int n_tiles = (d_out + 15) / 16;
    const int n_subs = d_in / 64;
    const int subs_per_blk = block_size / 64;
    const int hdr_bytes = is_asym ? 64 : 32;
    const int bytes_per_plane_blk = (block_size / 8) * 16;
    const size_t tile_blk_bytes = (size_t)hdr_bytes + (size_t)eff_bits * (size_t)bytes_per_plane_blk;
    const size_t luts_stride = (size_t)(d_in / 4) * 16;
    const int dual = x_needs_dual_plane(x, d_in, k_tokens);
    const int kc_step = dual ? 2 : 4;
    const int kc_max = (k_tokens < kc_step) ? k_tokens : kc_step;
    const int slot_cap = dual ? (kc_max * 2) : kc_max;

    int8_t* luts = (int8_t*)neon_alloc((size_t)slot_cap * luts_stride);
    float* sub_scales = (float*)neon_alloc((size_t)slot_cap * (size_t)n_subs * sizeof(float));
    float* blk_xnorm = (float*)neon_alloc((size_t)kc_max * (size_t)n_blks * sizeof(float));
    float* blk_xsum = is_asym ? (float*)neon_alloc((size_t)kc_max * (size_t)n_blks * sizeof(float)) : NULL;
    float* max_tnorm = (float*)neon_alloc((size_t)n_blks * sizeof(float));
    float* p1_acc = (float*)neon_alloc((size_t)n_tiles * (size_t)slot_cap * 16u * sizeof(float));
    float* p1_tmax = (float*)neon_alloc((size_t)n_tiles * (size_t)kc_max * sizeof(float));
    uint8_t* is_done = (uint8_t*)neon_alloc((size_t)n_tiles);
    int32_t* tile_skips = (int32_t*)neon_alloc((size_t)n_tiles * sizeof(int32_t));
    if (!luts || !sub_scales || !blk_xnorm || (is_asym && !blk_xsum) ||
        !max_tnorm || !p1_acc || !p1_tmax || !is_done || !tile_skips) {
        neon_free(luts); neon_free(sub_scales); neon_free(blk_xnorm); neon_free(blk_xsum);
        neon_free(max_tnorm); neon_free(p1_acc); neon_free(p1_tmax); neon_free(is_done); neon_free(tile_skips);
        return -1;
    }

    for (int b = 0; b < n_blks; b++) max_tnorm[b] = 0.0f;
    for (int t = 0; t < n_tiles; t++) {
        const uint16_t* tn = tile_norm_fp16 + (size_t)t * (size_t)n_blks;
        for (int b = 0; b < n_blks; b++) {
            float v = fp16_to_fp32(tn[b]);
            if (v > max_tnorm[b]) max_tnorm[b] = v;
        }
    }

#if HAS_OMP
    if (num_threads > 0) omp_set_num_threads(num_threads);
#else
    (void)num_threads;
#endif

    const int k_keep = (top_k_keep <= 0) ? 1 : ((top_k_keep > 32) ? 32 : top_k_keep);
    int64_t total_skipped = 0;

    for (int k0 = 0; k0 < k_tokens; k0 += kc_step) {
        const int kc = (k_tokens - k0 >= kc_step) ? kc_step : (k_tokens - k0);
        const int slots = dual ? (kc * 2) : kc;

        for (int kk = 0; kk < kc; kk++) {
            const float* xk = x + (size_t)(k0 + kk) * (size_t)d_in;
            if (dual) {
                const size_t shi = (size_t)(kk * 2);
                quantize_x_sub64_l1_dual_tile16(
                    xk, d_in,
                    luts + shi * luts_stride, sub_scales + shi * (size_t)n_subs,
                    luts + (shi + 1) * luts_stride, sub_scales + (shi + 1) * (size_t)n_subs
                );
            } else {
                quantize_x_sub64_l1_tile16(
                    xk, d_in,
                    luts + (size_t)kk * luts_stride,
                    sub_scales + (size_t)kk * (size_t)n_subs
                );
            }
            for (int blk = 0; blk < n_blks; blk++) {
                const float* xb = xk + (size_t)blk * (size_t)block_size;
                float n2 = 0.0f, bs = 0.0f;
                for (int i = 0; i < block_size; i++) {
                    float v = xb[i];
                    n2 += v * v;
                    bs += v;
                }
                blk_xnorm[(size_t)kk * (size_t)n_blks + (size_t)blk] = neon_sqrtf(n2);
                if (blk_xsum) blk_xsum[(size_t)kk * (size_t)n_blks + (size_t)blk] = bs;
            }
        }

        /* Sort blocks descending by peak energy contribution across kc active tokens. */
        int b_ord[256];
        float b_score[256];
        for (int b = 0; b < n_blks; b++) {
            float mx = 0.0f;
            for (int kk = 0; kk < kc; kk++) {
                float xn = blk_xnorm[(size_t)kk * (size_t)n_blks + (size_t)b];
                if (xn > mx) mx = xn;
            }
            b_ord[b] = b;
            b_score[b] = mx * max_tnorm[b];
        }
        for (int i = 1; i < n_blks; i++) {
            int bi = b_ord[i];
            float si = b_score[i];
            int j = i - 1;
            while (j >= 0 && b_score[j] < si) {
                b_ord[j + 1] = b_ord[j];
                b_score[j + 1] = b_score[j];
                j--;
            }
            b_ord[j + 1] = bi;
            b_score[j + 1] = si;
        }

        int p1_blks = (sieve_frac > 0.0f)
            ? (int)(sieve_frac * (float)n_blks + 0.999f)
            : ((n_blks >= 8) ? (n_blks / 4) : 1);
        if (p1_blks < 1) p1_blks = 1;
        if (p1_blks > n_blks) p1_blks = n_blks;

        /* Stage 1: Evaluate top p1_blks across all tiles in parallel. */
        int t;
#if HAS_OMP
        #pragma omp parallel for schedule(static)
#endif
        for (t = 0; t < n_tiles; t++) {
            const uint8_t* tptr = packed + (size_t)t * (size_t)n_blks * tile_blk_bytes;
            float acc[4][16];
            for (int s = 0; s < 4; s++) for (int r = 0; r < 16; r++) acc[s][r] = 0.0f;
            for (int step = 0; step < p1_blks; step++) {
                int blk = b_ord[step];
                t16c_accum_blk_slots(
                    tptr + (size_t)blk * tile_blk_bytes, blk, subs_per_blk, n_subs, n_blks,
                    eff_bits, gauss_lut, is_asym, dual, kc, slots,
                    tile_blk_bytes, bytes_per_plane_blk, luts_stride,
                    luts, sub_scales, blk_xsum, acc
                );
            }
            float* dst = p1_acc + (size_t)t * (size_t)slots * 16u;
            for (int s = 0; s < slots; s++) {
                memcpy(dst + (size_t)s * 16u, acc[s], 16u * sizeof(float));
            }
            int valid = ((t + 1) * 16 <= d_out) ? 16 : (d_out - t * 16);
            for (int kk = 0; kk < kc; kk++) {
                float mx = -1e30f;
                for (int r = 0; r < valid; r++) {
                    float v = dual ? (acc[kk * 2][r] + acc[kk * 2 + 1][r]) : acc[kk][r];
                    if (v > mx) mx = v;
                }
                p1_tmax[(size_t)t * (size_t)kc + (size_t)kk] = mx;
            }
        }

        /* Select seed + highest Stage-1 tiles (`<= 512` tiles total) to lock tight tau[kk]. */
        memset(is_done, 0, (size_t)n_tiles);
        int seed_prefix = (n_tiles >= 64) ? 32 : ((n_tiles >= 8) ? 2 : 0);
        for (int t0 = 0; t0 < seed_prefix; t0++) is_done[t0] = 1;
        int top_c = (sieve_frac > 0.0f) ? 512 : 256;
        if (top_c > n_tiles / 2) top_c = (n_tiles >= 4) ? 1 : n_tiles;
        for (int kk = 0; kk < kc; kk++) {
            float nth = -1e30f;
            int cand_ids[512];
            float cand_vals[512];
            int cnt = 0;
            for (int ti = 0; ti < n_tiles; ti++) {
                float v = p1_tmax[(size_t)ti * (size_t)kc + (size_t)kk];
                if (cnt < top_c) {
                    cand_ids[cnt] = ti;
                    cand_vals[cnt] = v;
                    cnt++;
                    if (cnt == top_c) {
                        float mn = cand_vals[0];
                        for (int m = 1; m < cnt; m++) if (cand_vals[m] < mn) mn = cand_vals[m];
                        nth = mn;
                    }
                } else if (v > nth) {
                    int min_i = 0;
                    float mn = cand_vals[0];
                    for (int m = 1; m < cnt; m++) {
                        if (cand_vals[m] < mn) { mn = cand_vals[m]; min_i = m; }
                    }
                    cand_ids[min_i] = ti;
                    cand_vals[min_i] = v;
                    mn = cand_vals[0];
                    for (int m = 1; m < cnt; m++) if (cand_vals[m] < mn) mn = cand_vals[m];
                    nth = mn;
                }
            }
            for (int m = 0; m < cnt; m++) is_done[cand_ids[m]] = 1;
        }

        /* Finish all remaining blocks on candidate tiles and extract exact top-k_keep threshold tau[kk]. */
        if (p1_blks < n_blks) {
#if HAS_OMP
            #pragma omp parallel for schedule(static)
#endif
            for (t = 0; t < n_tiles; t++) {
                if (!is_done[t]) continue;
                const uint8_t* tptr = packed + (size_t)t * (size_t)n_blks * tile_blk_bytes;
                float acc[4][16];
                const float* src = p1_acc + (size_t)t * (size_t)slots * 16u;
                for (int s = 0; s < slots; s++) memcpy(acc[s], src + (size_t)s * 16u, 16u * sizeof(float));
                for (int step = p1_blks; step < n_blks; step++) {
                    int blk = b_ord[step];
                    t16c_accum_blk_slots(
                        tptr + (size_t)blk * tile_blk_bytes, blk, subs_per_blk, n_subs, n_blks,
                        eff_bits, gauss_lut, is_asym, dual, kc, slots,
                        tile_blk_bytes, bytes_per_plane_blk, luts_stride,
                        luts, sub_scales, blk_xsum, acc
                    );
                }
                int base_r = t * 16;
                int valid = (base_r + 16 <= d_out) ? 16 : (d_out - base_r);
                for (int kk = 0; kk < kc; kk++) {
                    float* yk = y + (size_t)(k0 + kk) * (size_t)d_out + (size_t)base_r;
                    for (int r = 0; r < valid; r++) {
                        yk[r] = dual ? (acc[kk * 2][r] + acc[kk * 2 + 1][r]) : acc[kk][r];
                    }
                }
            }
        } else {
            for (int ti = 0; ti < n_tiles; ti++) {
                int base_r = ti * 16;
                int valid = (base_r + 16 <= d_out) ? 16 : (d_out - base_r);
                const float* src = p1_acc + (size_t)ti * (size_t)slots * 16u;
                for (int kk = 0; kk < kc; kk++) {
                    float* yk = y + (size_t)(k0 + kk) * (size_t)d_out + (size_t)base_r;
                    for (int r = 0; r < valid; r++) {
                        yk[r] = dual ? (src[(kk * 2) * 16 + r] + src[(kk * 2 + 1) * 16 + r]) : src[kk * 16 + r];
                    }
                }
            }
        }

        float tau[4];
        for (int kk = 0; kk < kc; kk++) {
            float top_vals[32];
            int top_cnt = 0;
            const float* yk = y + (size_t)(k0 + kk) * (size_t)d_out;
            for (int ti = 0; ti < n_tiles; ti++) {
                if (!is_done[ti]) continue;
                int base_r = ti * 16;
                int valid = (base_r + 16 <= d_out) ? 16 : (d_out - base_r);
                for (int r = 0; r < valid; r++) {
                    float v = yk[base_r + r];
                    if (top_cnt < k_keep) {
                        int p = top_cnt++;
                        while (p > 0 && top_vals[p - 1] < v) { top_vals[p] = top_vals[p - 1]; p--; }
                        top_vals[p] = v;
                    } else if (v > top_vals[k_keep - 1]) {
                        int p = k_keep - 1;
                        while (p > 0 && top_vals[p - 1] < v) { top_vals[p] = top_vals[p - 1]; p--; }
                        top_vals[p] = v;
                    }
                }
            }
            tau[kk] = (top_cnt > 0) ? top_vals[top_cnt - 1] : -1e30f;
        }

        memset(tile_skips, 0, (size_t)n_tiles * sizeof(int32_t));
        if (p1_blks < n_blks) {
            const int use_sieve = (sieve_frac > 0.0f) ? 1 : 0;
#if HAS_OMP
            #pragma omp parallel for schedule(static)
#endif
            for (t = 0; t < n_tiles; t++) {
                if (is_done[t]) continue;
                int base_r = t * 16;
                int valid = (base_r + 16 <= d_out) ? 16 : (d_out - base_r);
                if (use_sieve) {
                    tile_skips[t] = (int32_t)(n_blks - p1_blks);
                    for (int kk = 0; kk < kc; kk++) {
                        float* yk = y + (size_t)(k0 + kk) * (size_t)d_out + (size_t)base_r;
                        for (int r = 0; r < valid; r++) yk[r] = -1.0e9f;
                    }
                    continue;
                }
                const uint8_t* tptr = packed + (size_t)t * (size_t)n_blks * tile_blk_bytes;
                const uint16_t* tnorm = tile_norm_fp16 + (size_t)t * (size_t)n_blks;
                float acc[4][16];
                const float* src = p1_acc + (size_t)t * (size_t)slots * 16u;
                for (int s = 0; s < slots; s++) memcpy(acc[s], src + (size_t)s * 16u, 16u * sizeof(float));

                float rem_ub[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (int step = p1_blks; step < n_blks; step++) {
                    int blk = b_ord[step];
                    float wn = fp16_to_fp32(tnorm[blk]);
                    for (int kk = 0; kk < kc; kk++) {
                        rem_ub[kk] += wn * blk_xnorm[(size_t)kk * (size_t)n_blks + (size_t)blk];
                    }
                }

                int pruned = 0;
                for (int step = p1_blks; step < n_blks; step++) {
                    int can_prune = 1;
                    for (int kk = 0; kk < kc; kk++) {
                        float tile_mx = -1e30f;
                        for (int r = 0; r < valid; r++) {
                            float v = dual ? (acc[kk * 2][r] + acc[kk * 2 + 1][r]) : acc[kk][r];
                            if (v > tile_mx) tile_mx = v;
                        }
                        if (tile_mx + rem_ub[kk] * 1.02f + 0.05f >= tau[kk]) {
                            can_prune = 0;
                            break;
                        }
                    }
                    if (can_prune) {
                        tile_skips[t] = (int32_t)(n_blks - step);
                        for (int kk = 0; kk < kc; kk++) {
                            float* yk = y + (size_t)(k0 + kk) * (size_t)d_out + (size_t)base_r;
                            for (int r = 0; r < valid; r++) yk[r] = -1.0e9f;
                        }
                        pruned = 1;
                        break;
                    }
                    int blk = b_ord[step];
                    t16c_accum_blk_slots(
                        tptr + (size_t)blk * tile_blk_bytes, blk, subs_per_blk, n_subs, n_blks,
                        eff_bits, gauss_lut, is_asym, dual, kc, slots,
                        tile_blk_bytes, bytes_per_plane_blk, luts_stride,
                        luts, sub_scales, blk_xsum, acc
                    );
                    float wn = fp16_to_fp32(tnorm[blk]);
                    for (int kk = 0; kk < kc; kk++) {
                        rem_ub[kk] -= wn * blk_xnorm[(size_t)kk * (size_t)n_blks + (size_t)blk];
                        if (rem_ub[kk] < 0.0f) rem_ub[kk] = 0.0f;
                    }
                }
                if (!pruned) {
                    for (int kk = 0; kk < kc; kk++) {
                        float* yk = y + (size_t)(k0 + kk) * (size_t)d_out + (size_t)base_r;
                        for (int r = 0; r < valid; r++) {
                            yk[r] = dual ? (acc[kk * 2][r] + acc[kk * 2 + 1][r]) : acc[kk][r];
                        }
                    }
                }
            }
            for (int ti = 0; ti < n_tiles; ti++) total_skipped += tile_skips[ti];
        }
    }

    if (out_skipped_blks) *out_skipped_blks = total_skipped;
    neon_free(luts); neon_free(sub_scales); neon_free(blk_xnorm); neon_free(blk_xsum);
    neon_free(max_tnorm); neon_free(p1_acc); neon_free(p1_tmax); neon_free(is_done); neon_free(tile_skips);
    return 0;
}

NEON_API int inv_s_and_fwht_f32_c(
    float* NEON_RESTRICT x,
    const float* NEON_RESTRICT inv_s,
    int k_tokens,
    int d_in,
    int bsize
) {
    if (!x || k_tokens < 0 || d_in <= 0) return -1;
    if (k_tokens == 0) return 0;
    if (bsize != 1) {
        if (bsize < 8 || (bsize & (bsize - 1)) != 0 || (d_in % bsize) != 0) return -1;
    }

    if (inv_s != NULL) {
        for (int tok = 0; tok < k_tokens; tok++) {
            float* row = x + (size_t)tok * (size_t)d_in;
            int j = 0;
#if HAS_NEON && (defined(__aarch64__) || defined(_M_ARM64))
            for (; j + 4 <= d_in; j += 4) {
                vst1q_f32(row + j, vmulq_f32(vld1q_f32(row + j), vld1q_f32(inv_s + j)));
            }
#elif HAS_AVX2
            for (; j + 8 <= d_in; j += 8) {
                _mm256_storeu_ps(row + j, _mm256_mul_ps(_mm256_loadu_ps(row + j), _mm256_loadu_ps(inv_s + j)));
            }
#endif
            for (; j < d_in; j++) {
                row[j] *= inv_s[j];
            }
        }
    }

    if (bsize >= 8) {
        const float inv_sqrt = 1.0f / neon_sqrtf((float)bsize);
        const int n_blocks = (k_tokens * d_in) / bsize;
#if HAS_NEON && (defined(__aarch64__) || defined(_M_ARM64))
        const float32x4_t vscale = vdupq_n_f32(inv_sqrt);
#elif HAS_AVX2
        const __m256 vscale = _mm256_set1_ps(inv_sqrt);
#endif
        for (int blk = 0; blk < n_blocks; blk++) {
            float* chunk = x + (size_t)blk * (size_t)bsize;
            for (int i = 0; i < bsize; i += 8) {
                float* c = chunk + i;
                float a0 = c[0] + c[1], a1 = c[0] - c[1];
                float a2 = c[2] + c[3], a3 = c[2] - c[3];
                float a4 = c[4] + c[5], a5 = c[4] - c[5];
                float a6 = c[6] + c[7], a7 = c[6] - c[7];
                float b0 = a0 + a2, b1 = a1 + a3, b2 = a0 - a2, b3 = a1 - a3;
                float b4 = a4 + a6, b5 = a5 + a7, b6 = a4 - a6, b7 = a5 - a7;
                c[0] = b0 + b4; c[1] = b1 + b5; c[2] = b2 + b6; c[3] = b3 + b7;
                c[4] = b0 - b4; c[5] = b1 - b5; c[6] = b2 - b6; c[7] = b3 - b7;
            }
            for (int len = 8; len < bsize; len *= 2) {
                const int step = len * 2;
                for (int i = 0; i < bsize; i += step) {
                    float* left = chunk + i;
                    float* right = left + len;
#if HAS_NEON && (defined(__aarch64__) || defined(_M_ARM64))
                    for (int j = 0; j < len; j += 4) {
                        float32x4_t u = vld1q_f32(left + j);
                        float32x4_t v = vld1q_f32(right + j);
                        vst1q_f32(left + j, vaddq_f32(u, v));
                        vst1q_f32(right + j, vsubq_f32(u, v));
                    }
#elif HAS_AVX2
                    for (int j = 0; j < len; j += 8) {
                        __m256 u = _mm256_loadu_ps(left + j);
                        __m256 v = _mm256_loadu_ps(right + j);
                        _mm256_storeu_ps(left + j, _mm256_add_ps(u, v));
                        _mm256_storeu_ps(right + j, _mm256_sub_ps(u, v));
                    }
#else
                    for (int j = 0; j < len; j++) {
                        float u = left[j];
                        float v = right[j];
                        left[j] = u + v;
                        right[j] = u - v;
                    }
#endif
                }
            }
#if HAS_NEON && (defined(__aarch64__) || defined(_M_ARM64))
            for (int j = 0; j < bsize; j += 4) {
                vst1q_f32(chunk + j, vmulq_f32(vld1q_f32(chunk + j), vscale));
            }
#elif HAS_AVX2
            for (int j = 0; j < bsize; j += 8) {
                _mm256_storeu_ps(chunk + j, _mm256_mul_ps(_mm256_loadu_ps(chunk + j), vscale));
            }
#else
            for (int j = 0; j < bsize; j++) {
                chunk[j] *= inv_sqrt;
            }
#endif
        }
    }
    return 0;
}

NEON_API int int8_lowrank_add_batch_c(
    const int8_t* NEON_RESTRICT u_i8,
    const float* NEON_RESTRICT su,
    const int8_t* NEON_RESTRICT v_i8,
    const float* NEON_RESTRICT sv,
    const float* NEON_RESTRICT x_raw,
    float* NEON_RESTRICT y,
    int d_out,
    int d_in,
    int rank,
    int k_tokens,
    int num_threads
) {
    if (rank < 0 || d_out <= 0 || d_in <= 0 || k_tokens < 0) return -1;
    if (rank == 0 || k_tokens == 0) return 0;
    if (!u_i8 || !su || !v_i8 || !sv || !x_raw || !y) return -1;

    float* z = (float*)neon_alloc((size_t)k_tokens * (size_t)rank * sizeof(float));
    if (!z) return -1;

    for (int tok = 0; tok < k_tokens; tok++) {
        const float* xk = x_raw + (size_t)tok * (size_t)d_in;
        float* zk = z + (size_t)tok * (size_t)rank;
        for (int k = 0; k < rank; k++) {
            const int8_t* vrow = v_i8 + (size_t)k * (size_t)d_in;
            float acc = 0.0f;
            int j = 0;
#if HAS_NEON && (defined(__aarch64__) || defined(_M_ARM64))
            float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
            float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
            for (; j + 16 <= d_in; j += 16) {
                int8x16_t v16 = vld1q_s8(vrow + j);
                int16x8_t v0_16 = vmovl_s8(vget_low_s8(v16));
                int16x8_t v1_16 = vmovl_high_s8(v16);
                a0 = vfmaq_f32(a0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(v0_16))),  vld1q_f32(xk + j + 0));
                a1 = vfmaq_f32(a1, vcvtq_f32_s32(vmovl_high_s16(v0_16)),             vld1q_f32(xk + j + 4));
                a2 = vfmaq_f32(a2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(v1_16))),  vld1q_f32(xk + j + 8));
                a3 = vfmaq_f32(a3, vcvtq_f32_s32(vmovl_high_s16(v1_16)),             vld1q_f32(xk + j + 12));
            }
            acc = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
#elif HAS_AVX2
            __m256 va0 = _mm256_setzero_ps(), va1 = _mm256_setzero_ps();
            for (; j + 16 <= d_in; j += 16) {
                __m256 vf0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(vrow + j))));
                __m256 vf1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(vrow + j + 8))));
                va0 = _mm256_fmadd_ps(vf0, _mm256_loadu_ps(xk + j), va0);
                va1 = _mm256_fmadd_ps(vf1, _mm256_loadu_ps(xk + j + 8), va1);
            }
            for (; j + 8 <= d_in; j += 8) {
                __m256 vf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(vrow + j))));
                va0 = _mm256_fmadd_ps(vf, _mm256_loadu_ps(xk + j), va0);
            }
            acc = hsum256_ps(_mm256_add_ps(va0, va1));
#endif
            for (; j < d_in; j++) {
                acc += (float)vrow[j] * xk[j];
            }
            zk[k] = acc * sv[k];
        }
    }

#if HAS_OMP
    if (num_threads > 0) omp_set_num_threads(num_threads);
#else
    (void)num_threads;
#endif

    int row;
#if HAS_OMP
    #pragma omp parallel for schedule(static) if(d_out >= 512)
#endif
    for (row = 0; row < d_out; row++) {
        const int8_t* urow = u_i8 + (size_t)row * (size_t)rank;
        const float s_row = su[row];
        for (int k0 = 0; k0 < k_tokens; k0 += 4) {
            const int kc = (k_tokens - k0 >= 4) ? 4 : (k_tokens - k0);
            const float* zk0 = z + (size_t)(k0 + 0) * (size_t)rank;
            const float* zk1 = (kc >= 2) ? (z + (size_t)(k0 + 1) * (size_t)rank) : zk0;
            const float* zk2 = (kc >= 3) ? (z + (size_t)(k0 + 2) * (size_t)rank) : zk0;
            const float* zk3 = (kc == 4) ? (z + (size_t)(k0 + 3) * (size_t)rank) : zk0;
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            int k = 0;
#if HAS_NEON && (defined(__aarch64__) || defined(_M_ARM64))
            float32x4_t va0 = vdupq_n_f32(0.0f), va1 = vdupq_n_f32(0.0f);
            float32x4_t va2 = vdupq_n_f32(0.0f), va3 = vdupq_n_f32(0.0f);
            for (; k + 8 <= rank; k += 8) {
                int16x8_t u16 = vmovl_s8(vld1_s8(urow + k));
                float32x4_t u_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(u16)));
                float32x4_t u_hi = vcvtq_f32_s32(vmovl_high_s16(u16));
                va0 = vfmaq_f32(va0, u_lo, vld1q_f32(zk0 + k));
                va0 = vfmaq_f32(va0, u_hi, vld1q_f32(zk0 + k + 4));
                if (kc >= 2) {
                    va1 = vfmaq_f32(va1, u_lo, vld1q_f32(zk1 + k));
                    va1 = vfmaq_f32(va1, u_hi, vld1q_f32(zk1 + k + 4));
                }
                if (kc >= 3) {
                    va2 = vfmaq_f32(va2, u_lo, vld1q_f32(zk2 + k));
                    va2 = vfmaq_f32(va2, u_hi, vld1q_f32(zk2 + k + 4));
                }
                if (kc == 4) {
                    va3 = vfmaq_f32(va3, u_lo, vld1q_f32(zk3 + k));
                    va3 = vfmaq_f32(va3, u_hi, vld1q_f32(zk3 + k + 4));
                }
            }
            acc0 = vaddvq_f32(va0);
            if (kc >= 2) acc1 = vaddvq_f32(va1);
            if (kc >= 3) acc2 = vaddvq_f32(va2);
            if (kc == 4) acc3 = vaddvq_f32(va3);
#elif HAS_AVX2
            __m256 va0 = _mm256_setzero_ps(), va1 = _mm256_setzero_ps();
            __m256 va2 = _mm256_setzero_ps(), va3 = _mm256_setzero_ps();
            for (; k + 8 <= rank; k += 8) {
                __m256 uf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(urow + k))));
                va0 = _mm256_fmadd_ps(uf, _mm256_loadu_ps(zk0 + k), va0);
                if (kc >= 2) va1 = _mm256_fmadd_ps(uf, _mm256_loadu_ps(zk1 + k), va1);
                if (kc >= 3) va2 = _mm256_fmadd_ps(uf, _mm256_loadu_ps(zk2 + k), va2);
                if (kc == 4) va3 = _mm256_fmadd_ps(uf, _mm256_loadu_ps(zk3 + k), va3);
            }
            acc0 = hsum256_ps(va0);
            if (kc >= 2) acc1 = hsum256_ps(va1);
            if (kc >= 3) acc2 = hsum256_ps(va2);
            if (kc == 4) acc3 = hsum256_ps(va3);
#endif
            for (; k < rank; k++) {
                float uk = (float)urow[k];
                acc0 += uk * zk0[k];
                if (kc >= 2) acc1 += uk * zk1[k];
                if (kc >= 3) acc2 += uk * zk2[k];
                if (kc == 4) acc3 += uk * zk3[k];
            }
            y[(size_t)(k0 + 0) * (size_t)d_out + (size_t)row] += acc0 * s_row;
            if (kc >= 2) y[(size_t)(k0 + 1) * (size_t)d_out + (size_t)row] += acc1 * s_row;
            if (kc >= 3) y[(size_t)(k0 + 2) * (size_t)d_out + (size_t)row] += acc2 * s_row;
            if (kc == 4) y[(size_t)(k0 + 3) * (size_t)d_out + (size_t)row] += acc3 * s_row;
        }
    }

    neon_free(z);
    return 0;
}

static int gate_e_test_config(
    int d_out, int d_in, int bsize, int n_bits, int is_asym, int k_tokens, int test_cs_skip
) {
    const int n_tiles = (d_out + 15) / 16;
    const int n_blks = d_in / bsize;
    const int hdr = is_asym ? 64 : 32;
    const int plane_b = (bsize / 8) * 16;
    const int quintets = (bsize + 4) / 5;
    const size_t tblk = (n_bits == 5)
        ? ((size_t)32 + (size_t)quintets * (size_t)16)
        : (n_bits == 8)
            ? ((size_t)32 + (size_t)bsize * 16u)
            : ((size_t)hdr + (size_t)n_bits * (size_t)plane_b);

    uint8_t* packed = (uint8_t*)neon_alloc((size_t)n_tiles * (size_t)n_blks * tblk);
    uint16_t* norms = (uint16_t*)neon_alloc((size_t)n_tiles * (size_t)n_blks * sizeof(uint16_t));
    float* w_ref = (float*)neon_alloc((size_t)d_out * (size_t)d_in * sizeof(float));
    float* x = (float*)neon_alloc((size_t)k_tokens * (size_t)d_in * sizeof(float));
    float* y_ref = (float*)neon_alloc((size_t)k_tokens * (size_t)d_out * sizeof(float));
    float* y_got = (float*)neon_alloc((size_t)k_tokens * (size_t)d_out * sizeof(float));
    if (!packed || !norms || !w_ref || !x || !y_ref || !y_got) {
        neon_free(packed); neon_free(norms); neon_free(w_ref);
        neon_free(x); neon_free(y_ref); neon_free(y_got);
        return 0;
    }
    memset(packed, 0, (size_t)n_tiles * (size_t)n_blks * tblk);
    memset(norms, 0, (size_t)n_tiles * (size_t)n_blks * sizeof(uint16_t));
    memset(w_ref, 0, (size_t)d_out * (size_t)d_in * sizeof(float));

    uint64_t rng = 987654321ULL + (uint64_t)d_out * 31u + (uint64_t)n_bits * 7u + (uint64_t)is_asym * 99u + (uint64_t)k_tokens;
    for (int i = 0; i < k_tokens * d_in; i++) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        float v = (float)((int32_t)((rng >> 33) & 0xFFFFu) - 32768) / 32768.0f;
        x[i] = v * ((i % 19 == 0) ? 6.0f : 1.0f);
    }

    for (int t = 0; t < n_tiles; t++) {
        for (int blk = 0; blk < n_blks; blk++) {
            uint8_t* bp = packed + ((size_t)t * (size_t)n_blks + (size_t)blk) * tblk;
            uint8_t* pl = bp + hdr;
            float max_wn2 = 0.0f;
            for (int r = 0; r < 16; r++) {
                int gr = t * 16 + r;
                if (gr >= d_out) continue;
                rng = rng * 6364136223846793005ULL + 1ULL;
                float s_raw = 0.02f + 0.03f * (float)(rng & 255u) / 255.0f;
                rng = rng * 6364136223846793005ULL + 1ULL;
                float b_raw = is_asym ? (0.01f * ((float)(int32_t)((rng >> 16) & 255u) - 128.0f) / 128.0f) : 0.0f;
                uint16_t s16 = fp32_to_fp16(s_raw);
                uint16_t b16 = fp32_to_fp16(b_raw);
                write_u16_le(bp + (size_t)r * 2, s16);
                if (is_asym) write_u16_le(bp + 32 + (size_t)r * 2, b16);
                float s = fp16_to_fp32(s16);
                float b = is_asym ? fp16_to_fp32(b16) : 0.0f;
                float wn2 = 0.0f;
                if (n_bits == 5) {
                    for (int g = 0; g < quintets; g++) {
                        int code = 0, p3 = 1;
                        for (int k = 0; k < 5; k++) {
                            int col = g * 5 + k;
                            int tr = 0;
                            if (col < bsize) {
                                rng = rng * 6364136223846793005ULL + 1ULL;
                                tr = (int)((rng >> 32) % 3u) - 1;
                                float wv = s * (float)tr;
                                w_ref[(size_t)gr * (size_t)d_in + (size_t)blk * (size_t)bsize + (size_t)col] = wv;
                                wn2 += wv * wv;
                            }
                            code += (tr + 1) * p3;
                            p3 *= 3;
                        }
                        bp[32 + (size_t)g * 16 + (size_t)r] = (uint8_t)code;
                    }
                } else if (n_bits == 8) {
                    int8_t* qp8 = (int8_t*)(bp + 32);
                    for (int j = 0; j < bsize; j++) {
                        rng = rng * 6364136223846793005ULL + 1ULL;
                        int8_t qv = (int8_t)((int32_t)((rng >> 24) % 253u) - 126);
                        qp8[(size_t)j * 16u + (size_t)r] = qv;
                        float wv = s * (float)qv;
                        w_ref[(size_t)gr * (size_t)d_in + (size_t)blk * (size_t)bsize + (size_t)j] = wv;
                        wn2 += wv * wv;
                    }
                } else {
                    for (int bo = 0; bo < bsize / 8; bo++) {
                        for (int p = 0; p < n_bits; p++) {
                            rng = rng * 6364136223846793005ULL + 1ULL;
                            pl[(size_t)p * (size_t)plane_b + (size_t)bo * 16 + (size_t)r] = (uint8_t)(rng >> 24);
                        }
                    }
                    for (int j = 0; j < bsize; j++) {
                        float wv = b;
                        for (int p = 0; p < n_bits; p++) {
                            uint8_t byte_v = pl[(size_t)p * (size_t)plane_b + (size_t)(j / 8) * 16 + (size_t)r];
                            int bit = (byte_v >> (7 - (j % 8))) & 1;
                            wv += s * (float)(1u << p) * (bit ? 1.0f : -1.0f);
                        }
                        w_ref[(size_t)gr * (size_t)d_in + (size_t)blk * (size_t)bsize + (size_t)j] = wv;
                        wn2 += wv * wv;
                    }
                }
                if (wn2 > max_wn2) max_wn2 = wn2;
            }
            norms[(size_t)t * (size_t)n_blks + (size_t)blk] = fp32_to_fp16(neon_sqrtf(max_wn2));
        }
    }

    for (int kk = 0; kk < k_tokens; kk++) {
        for (int r = 0; r < d_out; r++) {
            double s = 0.0;
            for (int c = 0; c < d_in; c++) {
                s += (double)w_ref[(size_t)r * (size_t)d_in + (size_t)c] * (double)x[(size_t)kk * (size_t)d_in + (size_t)c];
            }
            y_ref[(size_t)kk * (size_t)d_out + (size_t)r] = (float)s;
        }
    }

    int rc = (k_tokens == 1)
        ? gemv_bitserial_tile16_mmap(packed, norms, x, y_got, d_out, d_in, bsize, n_bits, is_asym, 0.0f, 2, NULL)
        : gemm_bitserial_tile16_batch(packed, norms, x, y_got, d_out, d_in, bsize, n_bits, is_asym, k_tokens, 0.0f, 2, NULL);

    int ok = (rc == 0);
    if (ok) {
        for (int kk = 0; kk < k_tokens; kk++) {
            double dot = 0.0, na = 0.0, nb = 0.0;
            for (int r = 0; r < d_out; r++) {
                double vr = y_ref[(size_t)kk * (size_t)d_out + (size_t)r];
                double vg = y_got[(size_t)kk * (size_t)d_out + (size_t)r];
                dot += vr * vg;
                na += vr * vr;
                nb += vg * vg;
            }
            double denom = (double)neon_sqrtf((float)na) * (double)neon_sqrtf((float)nb) + 1e-12;
            double cos_v = dot / denom;
            if (cos_v < 0.999) {
                ok = 0;
                break;
            }
        }
    }

    if (ok && test_cs_skip) {
        int64_t skipped = -1;
        int rc_skip = gemm_bitserial_tile16_batch(
            packed, norms, x, y_got, d_out, d_in, bsize, n_bits, is_asym, k_tokens, 1e9f, 2, &skipped
        );
        const int kc_exp = x_needs_dual_plane(x, d_in, k_tokens) ? 2 : 4;
        int64_t exp_skips = (int64_t)((k_tokens + kc_exp - 1) / kc_exp) * (int64_t)n_tiles * (int64_t)n_blks;
        if (rc_skip != 0 || skipped != exp_skips) ok = 0;
        for (int i = 0; i < k_tokens * d_out && ok; i++) {
            if (neon_fabsf(y_got[i]) > 1e-7f) ok = 0;
        }
        if (gemm_bitserial_tile16_batch(NULL, NULL, x, y_got, d_out, d_in, bsize, n_bits, is_asym, 1, 0.0f, 1, NULL) != -1) ok = 0;
        if (gemm_bitserial_tile16_batch(packed, NULL, x, y_got, d_out, d_in, 32, n_bits, is_asym, 1, 0.0f, 1, NULL) != -1) ok = 0;
    }

    neon_free(packed); neon_free(norms); neon_free(w_ref);
    neon_free(x); neon_free(y_ref); neon_free(y_got);
    return ok;
}

/* ----------------------------------------------------------------------------
 * ARMv8 TBI [63:56] Flat-Pointer Helpers + Cuneiform 8D (S^7) Pattern Router
 * ---------------------------------------------------------------------------- */
static inline const uint8_t* tbi_deref_u8(uint64_t tagged_ptr) {
#if defined(__aarch64__) && defined(__linux__)
    /* Linux AArch64 EL0 configures TCR_EL1.TBI0=1 so [63:56] are ignored by MMU */
    return (const uint8_t*)(uintptr_t)tagged_ptr;
#else
    return (const uint8_t*)(uintptr_t)(tagged_ptr & 0x00FFFFFFFFFFFFFFULL);
#endif
}

NEON_API int tbi_hardware_active_c(void) {
#if defined(__aarch64__) && defined(__linux__)
    volatile uint32_t sentinel = 0x43384442u;
    uint64_t raw = (uint64_t)(uintptr_t)&sentinel;
    uint64_t tp = (((uint64_t)0xA7u) << 56) | (raw & 0x00FFFFFFFFFFFFFFULL);
    volatile const uint32_t* p = (volatile const uint32_t*)(uintptr_t)tp;
    return (*p == 0x43384442u && (uint8_t)(tp >> 56) == 0xA7u) ? 1 : 0;
#else
    return 0;
#endif
}

NEON_API uint64_t tbi_tag_ptr_c(uint64_t raw_ptr, uint8_t tag) {
    return (((uint64_t)tag) << 56) | (raw_ptr & 0x00FFFFFFFFFFFFFFULL);
}

NEON_API uint64_t tbi_untag_ptr_c(uint64_t tagged_ptr) {
    return tagged_ptr & 0x00FFFFFFFFFFFFFFULL;
}

NEON_API uint8_t tbi_extract_tag_c(uint64_t tagged_ptr) {
    return (uint8_t)(tagged_ptr >> 56);
}

NEON_API int cuneiform_s7_route_batch_c(
    const float* x,
    const float* centroids,
    int k_tokens,
    int d_in,
    int n_experts,
    int top_k,
    float temperature,
    int* out_idx,
    float* out_weights
) {
    if (!x || !centroids || !out_idx || !out_weights) return -1;
    if (k_tokens <= 0 || d_in < 8 || n_experts <= 0 || n_experts > 64) return -1;
    if (top_k <= 0 || top_k > n_experts || top_k > 8 || temperature <= 0.0f) return -1;

    const int k_blocks = d_in / 8;
    const float inv_blocks = 1.0f / (float)k_blocks;
    const float inv_temp = 1.0f / temperature;

    for (int t = 0; t < k_tokens; t++) {
        const float* xr = x + (size_t)t * (size_t)d_in;
        float sum8[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#if HAS_NEON
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        for (int b = 0; b < k_blocks; b++) {
            const float* bp = xr + (size_t)b * 8u;
            acc0 = vaddq_f32(acc0, vld1q_f32(bp));
            acc1 = vaddq_f32(acc1, vld1q_f32(bp + 4));
        }
        vst1q_f32(sum8, acc0);
        vst1q_f32(sum8 + 4, acc1);
#elif HAS_AVX2
        __m256 acc = _mm256_setzero_ps();
        for (int b = 0; b < k_blocks; b++) {
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(xr + (size_t)b * 8u));
        }
        _mm256_storeu_ps(sum8, acc);
#else
        for (int b = 0; b < k_blocks; b++) {
            const float* bp = xr + (size_t)b * 8u;
            for (int j = 0; j < 8; j++) sum8[j] += bp[j];
        }
#endif
        for (int j = 0; j < 8; j++) sum8[j] *= inv_blocks;
        float u_bar[8];
        fwht8(sum8, u_bar);

        float sq = 0.0f;
        for (int j = 0; j < 8; j++) sq += u_bar[j] * u_bar[j];
        float nrm = neon_sqrtf(sq);
        float inv_nrm = 1.0f / (nrm > 1e-9f ? nrm : 1e-9f);
        float g_hat[8];
        for (int j = 0; j < 8; j++) g_hat[j] = u_bar[j] * inv_nrm;

        float logits[64];
        for (int e = 0; e < n_experts; e++) {
            const float* ce = centroids + (size_t)e * 8u;
            float dot = 0.0f;
            for (int j = 0; j < 8; j++) dot += g_hat[j] * ce[j];
            logits[e] = dot * inv_temp;
        }

        int sel[8];
        uint8_t used[64];
        memset(used, 0, (size_t)n_experts);
        for (int k = 0; k < top_k; k++) {
            int best_e = -1;
            float best_v = -1e30f;
            for (int e = 0; e < n_experts; e++) {
                if (!used[e] && (best_e < 0 || logits[e] > best_v)) {
                    best_e = e;
                    best_v = logits[e];
                }
            }
            sel[k] = best_e;
            used[best_e] = 1u;
        }

        float max_l = logits[sel[0]];
        for (int k = 1; k < top_k; k++) {
            if (logits[sel[k]] > max_l) max_l = logits[sel[k]];
        }
        float w_unnorm[8];
        float w_sum = 0.0f;
        for (int k = 0; k < top_k; k++) {
            float wv = neon_expf(logits[sel[k]] - max_l);
            w_unnorm[k] = wv;
            w_sum += wv;
        }
        float inv_w = (w_sum > 1e-12f) ? (1.0f / w_sum) : (1.0f / (float)top_k);
        for (int k = 0; k < top_k; k++) {
            out_idx[t * top_k + k] = sel[k];
            out_weights[t * top_k + k] = (w_sum > 1e-12f) ? (w_unnorm[k] * inv_w) : inv_w;
        }
    }
    return 0;
}

NEON_API int cuneiform_bank_apply_batch_c(
    const uint64_t* expert_ptrs_tagged,
    const float* centroids,
    const float* x_raw,
    float* y,
    int d_out,
    int d_in,
    int rank,
    int n_experts,
    int top_k,
    float temperature,
    float gate_scale,
    int k_tokens,
    int num_threads,
    uint8_t* out_active_tags
) {
    if (!expert_ptrs_tagged || !centroids || !x_raw || !y) return -1;
    if (d_out <= 0 || d_in < 8 || rank <= 0 || k_tokens <= 0) return -1;
    if (n_experts <= 0 || n_experts > 64 || top_k <= 0 || top_k > n_experts || top_k > 8 || temperature <= 0.0f) return -1;

    int* idx_buf = (int*)neon_alloc((size_t)k_tokens * (size_t)top_k * sizeof(int));
    float* w_buf = (float*)neon_alloc((size_t)k_tokens * (size_t)top_k * sizeof(float));
    float* x_scaled = (float*)neon_alloc((size_t)d_in * sizeof(float));
    if (!idx_buf || !w_buf || !x_scaled) {
        neon_free(idx_buf); neon_free(w_buf); neon_free(x_scaled);
        return -1;
    }

    int rc = cuneiform_s7_route_batch_c(x_raw, centroids, k_tokens, d_in, n_experts, top_k, temperature, idx_buf, w_buf);
    if (rc != 0) {
        neon_free(idx_buf); neon_free(w_buf); neon_free(x_scaled);
        return rc;
    }

    const size_t su_bytes = (size_t)d_out * sizeof(float);
    const size_t sv_bytes = (size_t)rank * sizeof(float);
    const size_t u_bytes = (size_t)d_out * (size_t)rank;

    for (int t = 0; t < k_tokens; t++) {
        const float* xt = x_raw + (size_t)t * (size_t)d_in;
        float* yt = y + (size_t)t * (size_t)d_out;
        for (int k = 0; k < top_k; k++) {
            int e = idx_buf[t * top_k + k];
            uint64_t tp = expert_ptrs_tagged[e];
            if (out_active_tags) out_active_tags[t * top_k + k] = tbi_extract_tag_c(tp);
            if (tp == 0ULL) {
                neon_free(idx_buf); neon_free(w_buf); neon_free(x_scaled);
                return -1;
            }
            float alpha = gate_scale * w_buf[t * top_k + k];
            if (neon_fabsf(alpha) < 1e-12f) continue;

            int j = 0;
#if HAS_NEON
            float32x4_t av = vdupq_n_f32(alpha);
            for (; j + 4 <= d_in; j += 4) {
                vst1q_f32(x_scaled + j, vmulq_f32(vld1q_f32(xt + j), av));
            }
#elif HAS_AVX2
            __m256 av = _mm256_set1_ps(alpha);
            for (; j + 8 <= d_in; j += 8) {
                _mm256_storeu_ps(x_scaled + j, _mm256_mul_ps(_mm256_loadu_ps(xt + j), av));
            }
#endif
            for (; j < d_in; j++) x_scaled[j] = xt[j] * alpha;

            const uint8_t* base = tbi_deref_u8(tp);
            const float* su = (const float*)base;
            const float* sv = (const float*)(base + su_bytes);
            const int8_t* u_i8 = (const int8_t*)(base + su_bytes + sv_bytes);
            const int8_t* v_i8 = (const int8_t*)(base + su_bytes + sv_bytes + u_bytes);
            if (int8_lowrank_add_batch_c(u_i8, su, v_i8, sv, x_scaled, yt, d_out, d_in, rank, 1, num_threads) != 0) {
                neon_free(idx_buf); neon_free(w_buf); neon_free(x_scaled);
                return -1;
            }
        }
    }

    neon_free(idx_buf); neon_free(w_buf); neon_free(x_scaled);
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

    /* Gate E: Tile-16 Bit-Serial GEMV/GEMM (1..4-bit + 5-trit ternary, sym/asym, b=64/128/256, tail d_out, CS skip) */
    if (!gate_e_test_config(37, 256, 128, 1, 1, 1, 1)) return 0;
    if (!gate_e_test_config(32, 192,  64, 2, 0, 4, 0)) return 0;
    if (!gate_e_test_config(37, 256, 256, 3, 1, 5, 0)) return 0;
    if (!gate_e_test_config(16, 128, 128, 4, 0, 1, 0)) return 0;
    if (!gate_e_test_config(19, 128,  64, 5, 0, 1, 1)) return 0;
    if (!gate_e_test_config(37, 256, 128, 5, 0, 4, 0)) return 0;
    if (!gate_e_test_config(37, 512, 256, 5, 0, 5, 1)) return 0;
    if (!gate_e_test_config(37, 256,  64, 8, 0, 1, 1)) return 0;
    if (!gate_e_test_config(37, 512, 128, 8, 0, 5, 1)) return 0;
    {
        float xd[64] = {0.0f}, yd[16] = {0.0f};
        uint8_t pd[240] = {0u};
        if (gemm_bitserial_tile16_batch(pd, NULL, xd, yd, 16, 64, 64, 5, 1, 1, 0.0f, 1, NULL) != -1) return 0;
        if (gemm_bitserial_tile16_batch(pd, NULL, xd, yd, 16, 64, 64, 8, 1, 1, 0.0f, 1, NULL) != -1) return 0;
    }

    /* Gate F: Native inv_s + FWHT parity & involution */
    {
        const int k_t = 3, d_in = 256, bsz = 128;
        float* fx = (float*)neon_alloc((size_t)k_t * (size_t)d_in * sizeof(float));
        float* fx_orig = (float*)neon_alloc((size_t)k_t * (size_t)d_in * sizeof(float));
        float* finv = (float*)neon_alloc((size_t)d_in * sizeof(float));
        float* fref = (float*)neon_alloc((size_t)k_t * (size_t)d_in * sizeof(float));
        if (!fx || !fx_orig || !finv || !fref) {
            neon_free(fx); neon_free(fx_orig); neon_free(finv); neon_free(fref);
            return 0;
        }
        if (inv_s_and_fwht_f32_c(NULL, NULL, 1, 64, 64) != -1) { neon_free(fx); neon_free(fx_orig); neon_free(finv); neon_free(fref); return 0; }
        if (inv_s_and_fwht_f32_c(fx, NULL, 1, 64, 4) != -1) { neon_free(fx); neon_free(fx_orig); neon_free(finv); neon_free(fref); return 0; }
        if (inv_s_and_fwht_f32_c(fx, NULL, 1, 96, 64) != -1) { neon_free(fx); neon_free(fx_orig); neon_free(finv); neon_free(fref); return 0; }
        if (inv_s_and_fwht_f32_c(fx, NULL, 0, 64, 64) != 0) { neon_free(fx); neon_free(fx_orig); neon_free(finv); neon_free(fref); return 0; }

        uint64_t rng = 13579246801ULL;
        for (int j = 0; j < d_in; j++) {
            rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
            finv[j] = 0.5f + (float)((rng >> 32) & 255u) / 255.0f;
        }
        for (int i = 0; i < k_t * d_in; i++) {
            rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
            float v = (float)((int32_t)((rng >> 33) & 0xFFFFu) - 32768) / 16384.0f;
            fx[i] = v;
            fx_orig[i] = v;
        }
        /* bsize=1 skips FWHT, scales only */
        if (inv_s_and_fwht_f32_c(fx, finv, k_t, 67, 1) != 0) { neon_free(fx); neon_free(fx_orig); neon_free(finv); neon_free(fref); return 0; }
        for (int tok = 0; tok < k_t; tok++) {
            for (int j = 0; j < 67; j++) {
                float exp_v = fx_orig[tok * 67 + j] * finv[j];
                if (neon_fabsf(fx[tok * 67 + j] - exp_v) > 1e-5f) {
                    neon_free(fx); neon_free(fx_orig); neon_free(finv); neon_free(fref);
                    return 0;
                }
            }
        }
        /* Full bsize=128 FWHT + inv_s vs FP64 Sylvester Hadamard reference */
        for (int i = 0; i < k_t * d_in; i++) fx[i] = fx_orig[i];
        if (inv_s_and_fwht_f32_c(fx, finv, k_t, d_in, bsz) != 0) { neon_free(fx); neon_free(fx_orig); neon_free(finv); neon_free(fref); return 0; }
        double inv_sq = 1.0 / (double)neon_sqrtf((float)bsz);
        for (int tok = 0; tok < k_t; tok++) {
            for (int b = 0; b < d_in / bsz; b++) {
                int base = tok * d_in + b * bsz;
                for (int r = 0; r < bsz; r++) {
                    double acc = 0.0;
                    for (int c = 0; c < bsz; c++) {
                        uint32_t mask = (uint32_t)(r & c);
                        int par = 0;
                        while (mask) { par ^= (int)(mask & 1u); mask >>= 1; }
                        double x_eq = (double)fx_orig[base + c] * (double)finv[b * bsz + c];
                        acc += par ? -x_eq : x_eq;
                    }
                    fref[base + r] = (float)(acc * inv_sq);
                }
            }
        }
        for (int i = 0; i < k_t * d_in; i++) {
            if (neon_fabsf(fx[i] - fref[i]) > 1e-4f) {
                neon_free(fx); neon_free(fx_orig); neon_free(finv); neon_free(fref);
                return 0;
            }
        }
        /* H * H = I involution recovers scaled input */
        if (inv_s_and_fwht_f32_c(fx, NULL, k_t, d_in, bsz) != 0) { neon_free(fx); neon_free(fx_orig); neon_free(finv); neon_free(fref); return 0; }
        for (int tok = 0; tok < k_t; tok++) {
            for (int j = 0; j < d_in; j++) {
                float scaled = fx_orig[tok * d_in + j] * finv[j];
                if (neon_fabsf(fx[tok * d_in + j] - scaled) > 1e-4f) {
                    neon_free(fx); neon_free(fx_orig); neon_free(finv); neon_free(fref);
                    return 0;
                }
            }
        }
        neon_free(fx); neon_free(fx_orig); neon_free(finv); neon_free(fref);
    }

    /* Gate G: Native INT8 low-rank residual projection parity (odd dims + OMP threshold) */
    {
        const int lr_cases[2][4] = {{37, 131, 13, 5}, {600, 64, 16, 3}};
        for (int ci = 0; ci < 2; ci++) {
            const int d_o = lr_cases[ci][0];
            const int d_i = lr_cases[ci][1];
            const int rk  = lr_cases[ci][2];
            const int kt  = lr_cases[ci][3];
            int8_t* u8 = (int8_t*)neon_alloc((size_t)d_o * (size_t)rk);
            int8_t* v8 = (int8_t*)neon_alloc((size_t)rk * (size_t)d_i);
            float* su = (float*)neon_alloc((size_t)d_o * sizeof(float));
            float* sv = (float*)neon_alloc((size_t)rk * sizeof(float));
            float* lx = (float*)neon_alloc((size_t)kt * (size_t)d_i * sizeof(float));
            float* ly = (float*)neon_alloc((size_t)kt * (size_t)d_o * sizeof(float));
            float* lref = (float*)neon_alloc((size_t)kt * (size_t)d_o * sizeof(float));
            if (!u8 || !v8 || !su || !sv || !lx || !ly || !lref) {
                neon_free(u8); neon_free(v8); neon_free(su); neon_free(sv); neon_free(lx); neon_free(ly); neon_free(lref);
                return 0;
            }
            if (int8_lowrank_add_batch_c(NULL, su, v8, sv, lx, ly, d_o, d_i, rk, kt, 2) != -1) {
                neon_free(u8); neon_free(v8); neon_free(su); neon_free(sv); neon_free(lx); neon_free(ly); neon_free(lref);
                return 0;
            }
            uint64_t rng = 24681357901ULL + (uint64_t)ci * 97u;
            for (int i = 0; i < d_o * rk; i++) { rng = rng * 6364136223846793005ULL + 1ULL; u8[i] = (int8_t)((int32_t)((rng >> 32) % 251u) - 125); }
            for (int i = 0; i < rk * d_i; i++) { rng = rng * 6364136223846793005ULL + 1ULL; v8[i] = (int8_t)((int32_t)((rng >> 32) % 251u) - 125); }
            for (int i = 0; i < d_o; i++) { rng = rng * 6364136223846793005ULL + 1ULL; su[i] = 0.002f * (float)((int32_t)((rng >> 32) % 41u) - 20); }
            for (int i = 0; i < rk; i++)  { rng = rng * 6364136223846793005ULL + 1ULL; sv[i] = 0.003f * (float)((int32_t)((rng >> 32) % 41u) - 20); }
            for (int i = 0; i < kt * d_i; i++) { rng = rng * 6364136223846793005ULL + 1ULL; lx[i] = (float)((int32_t)((rng >> 33) & 1023u) - 512) / 256.0f; }
            for (int i = 0; i < kt * d_o; i++) { float init = 0.05f * (float)((i % 11) - 5); ly[i] = init; lref[i] = init; }

            if (int8_lowrank_add_batch_c(u8, su, v8, sv, lx, ly, d_o, d_i, rk, kt, 2) != 0) {
                neon_free(u8); neon_free(v8); neon_free(su); neon_free(sv); neon_free(lx); neon_free(ly); neon_free(lref);
                return 0;
            }
            for (int tok = 0; tok < kt; tok++) {
                for (int r = 0; r < d_o; r++) {
                    double acc = 0.0;
                    for (int k = 0; k < rk; k++) {
                        double vx = 0.0;
                        for (int j = 0; j < d_i; j++) vx += (double)v8[k * d_i + j] * (double)lx[tok * d_i + j];
                        acc += (double)u8[r * rk + k] * (vx * (double)sv[k]);
                    }
                    lref[tok * d_o + r] += (float)(acc * (double)su[r]);
                }
            }
            for (int i = 0; i < kt * d_o; i++) {
                if (neon_fabsf(ly[i] - lref[i]) > 1e-3f) {
                    neon_free(u8); neon_free(v8); neon_free(su); neon_free(sv); neon_free(lx); neon_free(ly); neon_free(lref);
                    return 0;
                }
            }
            neon_free(u8); neon_free(v8); neon_free(su); neon_free(sv); neon_free(lx); neon_free(ly); neon_free(lref);
        }
    }

    /* Gate H: Exhaustive 243-code unpack5 + k=2 Dual-Plane + Spiky Outlier Dual-Plane parity */
    if (!gate_e_test_config(37, 256, 128, 1, 1, 2, 1)) return 0;
    if (!gate_e_test_config(37, 256, 128, 5, 0, 2, 1)) return 0;
    {
        /* Exhaustive 0..242 quintet unpack verification */
        const int d_o = 243, d_i = 64, bsz = 64;
        const int n_tiles = (d_o + 15) / 16;
        const int quintets = (bsz + 4) / 5;
        const size_t tblk = (size_t)32 + (size_t)quintets * 16u;
        uint8_t* p5 = (uint8_t*)neon_alloc((size_t)n_tiles * tblk);
        float* x5 = (float*)neon_alloc((size_t)d_i * sizeof(float));
        float* y5_got = (float*)neon_alloc((size_t)d_o * sizeof(float));
        if (!p5 || !x5 || !y5_got) { neon_free(p5); neon_free(x5); neon_free(y5_got); return 0; }
        memset(p5, 121, (size_t)n_tiles * tblk);
        memset(x5, 0, (size_t)d_i * sizeof(float));
        x5[0] = 1.0f; x5[1] = 2.0f; x5[2] = 4.0f; x5[3] = 8.0f; x5[4] = 16.0f;
        uint16_t one_h = fp32_to_fp16(1.0f);
        for (int t = 0; t < n_tiles; t++) {
            uint8_t* bp = p5 + (size_t)t * tblk;
            for (int r = 0; r < 16; r++) {
                int gr = t * 16 + r;
                write_u16_le(bp + (size_t)r * 2u, (gr < d_o) ? one_h : 0u);
                if (gr < d_o) bp[32 + r] = (uint8_t)gr;
            }
        }
        int ok243 = (gemm_bitserial_tile16_batch(p5, NULL, x5, y5_got, d_o, d_i, bsz, 5, 0, 1, 0.0f, 1, NULL) == 0);
        for (int c = 0; c < d_o && ok243; c++) {
            int v = c;
            float exp_c = 0.0f;
            float p2 = 1.0f;
            for (int k = 0; k < 5; k++) { exp_c += (float)((v % 3) - 1) * p2; v /= 3; p2 *= 2.0f; }
            /* Dual-plane INT8 NEON achieves ~1.5e-5 rel precision (~0.00043 on |31|); wrong trit shifts >= 1.0 */
            if (neon_fabsf(y5_got[c] - exp_c) > 1e-3f) ok243 = 0;
        }
        neon_free(p5); neon_free(x5); neon_free(y5_got);
        if (!ok243) return 0;
    }
    {
        /* Spiky Outlier Dual-Plane parity (1000.0f spike on zero-contrib cols + 0.03f payload, k=2) */
        const int d_o = 32, d_i = 128, bsz = 64, kt = 2;
        const int n_blks = d_i / bsz, n_tiles = d_o / 16;
        const int quintets = (bsz + 4) / 5;
        const size_t tblk5 = (size_t)32 + (size_t)quintets * 16u;
        const size_t tblk1 = (size_t)64 + (size_t)(bsz / 8) * 16u;
        uint8_t* p5 = (uint8_t*)neon_alloc((size_t)n_tiles * (size_t)n_blks * tblk5);
        uint8_t* p1 = (uint8_t*)neon_alloc((size_t)n_tiles * (size_t)n_blks * tblk1);
        float* sx = (float*)neon_alloc((size_t)kt * (size_t)d_i * sizeof(float));
        float* sy5 = (float*)neon_alloc((size_t)kt * (size_t)d_o * sizeof(float));
        float* sy1 = (float*)neon_alloc((size_t)kt * (size_t)d_o * sizeof(float));
        float* ref5 = (float*)neon_alloc((size_t)kt * (size_t)d_o * sizeof(float));
        float* ref1 = (float*)neon_alloc((size_t)kt * (size_t)d_o * sizeof(float));
        if (!p5 || !p1 || !sx || !sy5 || !sy1 || !ref5 || !ref1) {
            neon_free(p5); neon_free(p1); neon_free(sx); neon_free(sy5); neon_free(sy1); neon_free(ref5); neon_free(ref1);
            return 0;
        }
        uint64_t rng = 31415926535ULL;
        for (int tok = 0; tok < kt; tok++) {
            for (int j = 0; j < d_i; j++) {
                int local = j % 64;
                if (local < 2) {
                    sx[tok * d_i + j] = (tok == 0) ? 1000.0f : -1000.0f;
                } else {
                    rng = rng * 6364136223846793005ULL + 1ULL;
                    sx[tok * d_i + j] = 0.04f * (float)((int32_t)((rng >> 33) & 255u) - 128) / 128.0f;
                }
            }
        }
        memset(ref5, 0, (size_t)kt * (size_t)d_o * sizeof(float));
        memset(ref1, 0, (size_t)kt * (size_t)d_o * sizeof(float));
        uint16_t s_h = fp32_to_fp16(0.25f), b_h = fp32_to_fp16(-0.02f);
        float s_f = fp16_to_fp32(s_h), b_f = fp16_to_fp32(b_h);
        for (int t = 0; t < n_tiles; t++) {
            for (int blk = 0; blk < n_blks; blk++) {
                uint8_t* bp5 = p5 + ((size_t)t * (size_t)n_blks + (size_t)blk) * tblk5;
                uint8_t* bp1 = p1 + ((size_t)t * (size_t)n_blks + (size_t)blk) * tblk1;
                for (int r = 0; r < 16; r++) {
                    int gr = t * 16 + r;
                    write_u16_le(bp5 + (size_t)r * 2u, s_h);
                    write_u16_le(bp1 + (size_t)r * 2u, s_h);
                    write_u16_le(bp1 + 32 + (size_t)r * 2u, b_h);
                    for (int g = 0; g < quintets; g++) {
                        int code = 0, p3 = 1;
                        for (int k = 0; k < 5; k++) {
                            int col = g * 5 + k;
                            int tr = 0;
                            if (col < bsz) {
                                if (col >= 2) {
                                    rng = rng * 6364136223846793005ULL + 1ULL;
                                    tr = (int)((rng >> 32) % 3u) - 1;
                                }
                                float wv = s_f * (float)tr;
                                for (int tok = 0; tok < kt; tok++) {
                                    ref5[tok * d_o + gr] += wv * sx[tok * d_i + blk * bsz + col];
                                }
                            }
                            code += (tr + 1) * p3;
                            p3 *= 3;
                        }
                        bp5[32 + (size_t)g * 16u + (size_t)r] = (uint8_t)code;
                    }
                    for (int bo = 0; bo < bsz / 8; bo++) {
                        rng = rng * 6364136223846793005ULL + 1ULL;
                        uint8_t bv = (uint8_t)(rng >> 24);
                        if (bo == 0) bv = (uint8_t)((bv & 0x3Fu) | 0x80u); /* bit0=+1, bit1=-1 so spike cancels cleanly */
                        bp1[64 + (size_t)bo * 16u + (size_t)r] = bv;
                        for (int bit = 0; bit < 8; bit++) {
                            int col = bo * 8 + bit;
                            float wv = b_f + s_f * (((bv >> (7 - bit)) & 1) ? 1.0f : -1.0f);
                            for (int tok = 0; tok < kt; tok++) {
                                ref1[tok * d_o + gr] += wv * sx[tok * d_i + blk * bsz + col];
                            }
                        }
                    }
                }
            }
        }
        int ok_dual = (gemm_bitserial_tile16_batch(p5, NULL, sx, sy5, d_o, d_i, bsz, 5, 0, kt, 0.0f, 2, NULL) == 0)
                   && (gemm_bitserial_tile16_batch(p1, NULL, sx, sy1, d_o, d_i, bsz, 1, 1, kt, 0.0f, 2, NULL) == 0);
        for (int pass = 0; pass < 2 && ok_dual; pass++) {
            const float* yg = (pass == 0) ? sy5 : sy1;
            const float* yr = (pass == 0) ? ref5 : ref1;
            for (int tok = 0; tok < kt; tok++) {
                double dot = 0.0, na = 0.0, nb = 0.0;
                for (int r = 0; r < d_o; r++) {
                    double vr = yr[tok * d_o + r], vg = yg[tok * d_o + r];
                    dot += vr * vg; na += vr * vr; nb += vg * vg;
                }
                double cos_v = dot / ((double)neon_sqrtf((float)na) * (double)neon_sqrtf((float)nb) + 1e-12);
                if (cos_v < 0.9999) ok_dual = 0;
            }
        }
        neon_free(p5); neon_free(p1); neon_free(sx); neon_free(sy5); neon_free(sy1); neon_free(ref5); neon_free(ref1);
        if (!ok_dual) return 0;
    }

    /* Gate I: ARMv8 TBI [63:56] flat pointer tagging + Cuneiform 8D (S^7) sparse expert apply parity */
    {
        uint64_t dummy_va = 0x0000123456789AB0ULL;
        uint64_t tp_test = tbi_tag_ptr_c(dummy_va, 0xD5u);
        if (tbi_extract_tag_c(tp_test) != 0xD5u) return 0;
        if (tbi_untag_ptr_c(tp_test) != dummy_va) return 0;
#if defined(__aarch64__) && defined(__linux__)
        if (!tbi_hardware_active_c()) return 0;
#endif
        const int d_o = 40, d_i = 64, rk = 5, n_exp = 4, top_k = 2, kt = 2;
        const float temp = 0.5f, gscale = 0.75f;
        float* cents = (float*)neon_alloc((size_t)n_exp * 8u * sizeof(float));
        float* cx = (float*)neon_alloc((size_t)kt * (size_t)d_i * sizeof(float));
        float* cy = (float*)neon_alloc((size_t)kt * (size_t)d_o * sizeof(float));
        float* cref = (float*)neon_alloc((size_t)kt * (size_t)d_o * sizeof(float));
        void* exp_raw[4] = {NULL, NULL, NULL, NULL};
        uint64_t exp_ptrs[4] = {0ULL, 0ULL, 0ULL, 0ULL};
        uint8_t tags[4] = {0, 0, 0, 0};
        if (!cents || !cx || !cy || !cref) {
            neon_free(cents); neon_free(cx); neon_free(cy); neon_free(cref);
            return 0;
        }
        const size_t su_b = (size_t)d_o * sizeof(float);
        const size_t sv_b = (size_t)rk * sizeof(float);
        const size_t u_b = (size_t)d_o * (size_t)rk;
        const size_t v_b = (size_t)rk * (size_t)d_i;
        const size_t exp_bytes = su_b + sv_b + u_b + v_b;
        int alloc_ok = 1;
        uint64_t rng = 9876543210123ULL;
        for (int e = 0; e < n_exp; e++) {
            float n2 = 0.0f;
            for (int j = 0; j < 8; j++) {
                float cval = (j == e) ? 1.0f : (0.15f * (float)(((e * 7 + j * 3) % 5) - 2));
                cents[e * 8 + j] = cval;
                n2 += cval * cval;
            }
            float inv_n = 1.0f / neon_sqrtf(n2);
            for (int j = 0; j < 8; j++) cents[e * 8 + j] *= inv_n;

            exp_raw[e] = neon_alloc(exp_bytes);
            if (!exp_raw[e]) { alloc_ok = 0; break; }
            uint8_t* ep = (uint8_t*)exp_raw[e];
            float* su = (float*)ep;
            float* sv = (float*)(ep + su_b);
            int8_t* ui = (int8_t*)(ep + su_b + sv_b);
            int8_t* vi = (int8_t*)(ep + su_b + sv_b + u_b);
            for (int r = 0; r < d_o; r++) { rng = rng * 6364136223846793005ULL + 1ULL; su[r] = 0.002f * (float)((int32_t)((rng >> 32) % 31u) - 15); }
            for (int k = 0; k < rk; k++) { rng = rng * 6364136223846793005ULL + 1ULL; sv[k] = 0.003f * (float)((int32_t)((rng >> 32) % 31u) - 15); }
            for (size_t i = 0; i < u_b; i++) { rng = rng * 6364136223846793005ULL + 1ULL; ui[i] = (int8_t)((int32_t)((rng >> 32) % 201u) - 100); }
            for (size_t i = 0; i < v_b; i++) { rng = rng * 6364136223846793005ULL + 1ULL; vi[i] = (int8_t)((int32_t)((rng >> 32) % 201u) - 100); }
            exp_ptrs[e] = tbi_tag_ptr_c((uint64_t)(uintptr_t)ep, (uint8_t)(0xC0u + e));
        }
        if (!alloc_ok) {
            for (int e = 0; e < n_exp; e++) neon_free(exp_raw[e]);
            neon_free(cents); neon_free(cx); neon_free(cy); neon_free(cref);
            return 0;
        }
        for (int t = 0; t < kt; t++) {
            for (int j = 0; j < d_i; j++) {
                rng = rng * 6364136223846793005ULL + 1ULL;
                cx[t * d_i + j] = (float)((int32_t)((rng >> 33) & 511u) - 256) / 256.0f + ((j % 8 == t) ? 0.5f : 0.0f);
            }
        }
        for (int i = 0; i < kt * d_o; i++) { float init = 0.02f * (float)((i % 9) - 4); cy[i] = init; cref[i] = init; }

        /* Fail-closed checks */
        int ok_c8d = (cuneiform_s7_route_batch_c(NULL, cents, kt, d_i, n_exp, top_k, temp, (int*)tags, cy) == -1)
                  && (cuneiform_bank_apply_batch_c(exp_ptrs, cents, cx, cy, d_o, 4, rk, n_exp, top_k, temp, gscale, kt, 1, tags) == -1);
        if (ok_c8d) {
            ok_c8d = (cuneiform_bank_apply_batch_c(exp_ptrs, cents, cx, cy, d_o, d_i, rk, n_exp, top_k, temp, gscale, kt, 2, tags) == 0);
        }
        for (int t = 0; t < kt && ok_c8d; t++) {
            double mean8[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            int kb = d_i / 8;
            for (int b = 0; b < kb; b++) for (int j = 0; j < 8; j++) mean8[j] += (double)cx[t * d_i + b * 8 + j];
            float mf[8], ubar[8];
            for (int j = 0; j < 8; j++) mf[j] = (float)(mean8[j] / (double)kb);
            fwht8(mf, ubar);
            double sq = 0.0;
            for (int j = 0; j < 8; j++) sq += (double)ubar[j] * (double)ubar[j];
            float inv_n = 1.0f / neon_sqrtf((float)sq);
            double sc[4];
            for (int e = 0; e < n_exp; e++) {
                double d = 0.0;
                for (int j = 0; j < 8; j++) d += (double)(ubar[j] * inv_n) * (double)cents[e * 8 + j];
                sc[e] = d / (double)temp;
            }
            int sel[2] = {-1, -1};
            for (int k = 0; k < top_k; k++) {
                for (int e = 0; e < n_exp; e++) {
                    if (k == 1 && e == sel[0]) continue;
                    if (sel[k] < 0 || sc[e] > sc[sel[k]]) sel[k] = e;
                }
                if (tags[t * top_k + k] != (uint8_t)(0xC0u + sel[k])) ok_c8d = 0;
            }
            double mx = sc[sel[0]] > sc[sel[1]] ? sc[sel[0]] : sc[sel[1]];
            double w0 = (double)neon_expf((float)(sc[sel[0]] - mx));
            double w1 = (double)neon_expf((float)(sc[sel[1]] - mx));
            double ws = w0 + w1;
            double wt[2] = {(double)gscale * (w0 / ws), (double)gscale * (w1 / ws)};
            for (int k = 0; k < top_k && ok_c8d; k++) {
                const uint8_t* ep = (const uint8_t*)exp_raw[sel[k]];
                const float* su = (const float*)ep;
                const float* sv = (const float*)(ep + su_b);
                const int8_t* ui = (const int8_t*)(ep + su_b + sv_b);
                const int8_t* vi = (const int8_t*)(ep + su_b + sv_b + u_b);
                for (int r = 0; r < d_o; r++) {
                    double acc = 0.0;
                    for (int rr = 0; rr < rk; rr++) {
                        double vx = 0.0;
                        for (int c = 0; c < d_i; c++) vx += (double)vi[rr * d_i + c] * (double)cx[t * d_i + c];
                        acc += (double)ui[r * rk + rr] * (vx * (double)sv[rr]);
                    }
                    cref[t * d_o + r] += (float)(wt[k] * (double)su[r] * acc);
                }
            }
        }
        for (int i = 0; i < kt * d_o && ok_c8d; i++) {
            if (neon_fabsf(cy[i] - cref[i]) > 1e-3f) ok_c8d = 0;
        }
        for (int e = 0; e < n_exp; e++) neon_free(exp_raw[e]);
        neon_free(cents); neon_free(cx); neon_free(cy); neon_free(cref);
        if (!ok_c8d) return 0;
    }

    /* Gate J: k=1 vs k=4 Batch-Verify IEEE-754 Bit-Exactness across ternary5, int8, and bitserial */
    {
        const int d_o = 48, d_i = 128, bsz = 64, kt = 4;
        const int n_tiles = (d_o + 15) / 16;
        const int n_blks = d_i / bsz;
        const int modes[3][2] = {{5, 0}, {8, 0}, {1, 1}};
        float* bx = (float*)neon_alloc((size_t)kt * (size_t)d_i * sizeof(float));
        float* by_batch = (float*)neon_alloc((size_t)kt * (size_t)d_o * sizeof(float));
        float* by_single = (float*)neon_alloc((size_t)kt * (size_t)d_o * sizeof(float));
        if (!bx || !by_batch || !by_single) {
            neon_free(bx); neon_free(by_batch); neon_free(by_single);
            return 0;
        }
        uint64_t rng = 0x9E3779B97F4A7C15ULL;
        for (int i = 0; i < kt * d_i; i++) {
            rng = rng * 6364136223846793005ULL + 1ULL;
            float base_v = (float)((int32_t)((rng >> 33) & 511u) - 256) / 256.0f;
            if ((i % 31) == 0) base_v *= 24.0f;
            bx[i] = base_v;
        }
        int ok_exact = 1;
        for (int m = 0; m < 3 && ok_exact; m++) {
            const int nb = modes[m][0];
            const int asym = modes[m][1];
            const size_t tblk = (nb == 5) ? ((size_t)32 + (size_t)((bsz + 4) / 5) * 16u)
                              : (nb == 8) ? ((size_t)32 + (size_t)bsz * 16u)
                              : ((size_t)(asym ? 64 : 32) + (size_t)nb * (size_t)(bsz / 8) * 16u);
            const size_t pk_bytes = (size_t)n_tiles * (size_t)n_blks * tblk;
            uint8_t* pk = (uint8_t*)neon_alloc(pk_bytes);
            if (!pk) { ok_exact = 0; break; }
            uint16_t sc_h = fp32_to_fp16(0.125f);
            uint16_t bi_h = fp32_to_fp16(-0.03125f);
            for (int t = 0; t < n_tiles; t++) {
                for (int blk = 0; blk < n_blks; blk++) {
                    uint8_t* bp = pk + ((size_t)t * (size_t)n_blks + (size_t)blk) * tblk;
                    for (int r = 0; r < 16; r++) write_u16_le(bp + (size_t)r * 2u, sc_h);
                    size_t off = 32;
                    if (nb == 1 && asym) {
                        for (int r = 0; r < 16; r++) write_u16_le(bp + 32u + (size_t)r * 2u, bi_h);
                        off = 64;
                    }
                    for (size_t z = off; z < tblk; z++) {
                        rng = rng * 6364136223846793005ULL + 1ULL;
                        uint8_t byte_v = (uint8_t)(rng >> 24);
                        if (nb == 5) byte_v = (uint8_t)(byte_v % 243u);
                        bp[z] = byte_v;
                    }
                }
            }
            if (gemm_bitserial_tile16_batch(pk, NULL, bx, by_batch, d_o, d_i, bsz, nb, asym, kt, 0.0f, 2, NULL) != 0) {
                ok_exact = 0;
            }
            for (int tok = 0; tok < kt && ok_exact; tok++) {
                if (gemm_bitserial_tile16_batch(
                        pk, NULL,
                        bx + (size_t)tok * (size_t)d_i,
                        by_single + (size_t)tok * (size_t)d_o,
                        d_o, d_i, bsz, nb, asym, 1, 0.0f, 2, NULL) != 0) {
                    ok_exact = 0;
                }
            }
            for (int i = 0; i < kt * d_o && ok_exact; i++) {
                uint32_t u_b = 0, u_s = 0;
                memcpy(&u_b, &by_batch[i], sizeof(uint32_t));
                memcpy(&u_s, &by_single[i], sizeof(uint32_t));
                if (u_b != u_s) ok_exact = 0;
            }
            neon_free(pk);
        }
        neon_free(bx); neon_free(by_batch); neon_free(by_single);
        if (!ok_exact) return 0;
    }

    /* Gate K: Non-uniform Gaussian 2-bit bit-plane weights (T16C_GAUSS_BIT) k=1 vs k=4 bitwise parity */
    {
        const int d_o = 32, d_i = 128, bsz = 64, kt = 4, nb = 2 | T16C_GAUSS_BIT;
        const int n_tiles = d_o / 16, n_blks = d_i / bsz;
        const size_t plane_b = (size_t)(bsz / 8) * 16u;
        const size_t tblk = 32u + 2u * plane_b;
        uint8_t* pk = (uint8_t*)neon_alloc((size_t)n_tiles * (size_t)n_blks * tblk);
        float* gx = (float*)neon_alloc((size_t)kt * (size_t)d_i * sizeof(float));
        float* gy_b = (float*)neon_alloc((size_t)kt * (size_t)d_o * sizeof(float));
        float* gy_s = (float*)neon_alloc((size_t)kt * (size_t)d_o * sizeof(float));
        float* gref = (float*)neon_alloc((size_t)kt * (size_t)d_o * sizeof(float));
        if (!pk || !gx || !gy_b || !gy_s || !gref) {
            neon_free(pk); neon_free(gx); neon_free(gy_b); neon_free(gy_s); neon_free(gref);
            return 0;
        }
        uint64_t rng = 0x9E3779B97F4A7C15ULL;
        for (int i = 0; i < kt * d_i; i++) {
            rng = rng * 6364136223846793005ULL + 1ULL;
            gx[i] = 0.8f * ((float)((int32_t)((rng >> 33) & 255u) - 128) / 128.0f);
        }
        memset(gref, 0, (size_t)kt * (size_t)d_o * sizeof(float));
        for (int t = 0; t < n_tiles; t++) {
            for (int blk = 0; blk < n_blks; blk++) {
                uint8_t* bp = pk + ((size_t)t * (size_t)n_blks + (size_t)blk) * tblk;
                for (int r = 0; r < 16; r++) {
                    int gr = t * 16 + r;
                    uint16_t sc_h = fp32_to_fp16(0.05f + 0.01f * (float)r);
                    float sc_f = fp16_to_fp32(sc_h);
                    write_u16_le(bp + (size_t)r * 2u, sc_h);
                    for (int p = 0; p < 2; p++) {
                        float wp = t16c_plane_weight(1, 2, p);
                        for (int bo = 0; bo < bsz / 8; bo++) {
                            rng = rng * 6364136223846793005ULL + 1ULL;
                            uint8_t bv = (uint8_t)(rng >> 24);
                            bp[32u + (size_t)p * plane_b + (size_t)bo * 16u + (size_t)r] = bv;
                            for (int bit = 0; bit < 8; bit++) {
                                float wv = sc_f * wp * (((bv >> (7 - bit)) & 1) ? 1.0f : -1.0f);
                                int col = blk * bsz + bo * 8 + bit;
                                for (int tok = 0; tok < kt; tok++) {
                                    gref[(size_t)tok * (size_t)d_o + (size_t)gr] += wv * gx[(size_t)tok * (size_t)d_i + (size_t)col];
                                }
                            }
                        }
                    }
                }
            }
        }
        int ok_g = (gemm_bitserial_tile16_batch(pk, NULL, gx, gy_b, d_o, d_i, bsz, nb, 0, kt, 0.0f, 2, NULL) == 0);
        for (int tok = 0; tok < kt && ok_g; tok++) {
            if (gemm_bitserial_tile16_batch(pk, NULL, gx + (size_t)tok * (size_t)d_i, gy_s + (size_t)tok * (size_t)d_o,
                                            d_o, d_i, bsz, nb, 0, 1, 0.0f, 2, NULL) != 0) ok_g = 0;
        }
        for (int tok = 0; tok < kt && ok_g; tok++) {
            double dot = 0.0, na = 0.0, nb2 = 0.0;
            for (int r = 0; r < d_o; r++) {
                size_t idx = (size_t)tok * (size_t)d_o + (size_t)r;
                uint32_t ub = 0, us = 0;
                memcpy(&ub, &gy_b[idx], sizeof(uint32_t));
                memcpy(&us, &gy_s[idx], sizeof(uint32_t));
                if (ub != us) ok_g = 0;
                double vr = gref[idx], vg = gy_b[idx];
                dot += vr * vg; na += vr * vr; nb2 += vg * vg;
            }
            if (dot / ((double)neon_sqrtf((float)na) * (double)neon_sqrtf((float)nb2) + 1e-12) < 0.999) ok_g = 0;
        }
        neon_free(pk); neon_free(gx); neon_free(gy_b); neon_free(gy_s); neon_free(gref);
        if (!ok_g) return 0;
    }

    /* Gate L: Block-ordered CS LM-head pruning parity & block skips (exact & sieve) */
    {
        const int d_o = 128, d_i = 512, bsz = 64, kt = 2, nb = 3;
        const int n_tiles = d_o / 16, n_blks = d_i / bsz;
        const size_t plane_b = (size_t)(bsz / 8) * 16u;
        const size_t tblk = 32u + (size_t)nb * plane_b;
        uint8_t* pk = (uint8_t*)neon_alloc((size_t)n_tiles * (size_t)n_blks * tblk);
        uint16_t* tn = (uint16_t*)neon_alloc((size_t)n_tiles * (size_t)n_blks * sizeof(uint16_t));
        float* lx = (float*)neon_alloc((size_t)kt * (size_t)d_i * sizeof(float));
        float* y_full = (float*)neon_alloc((size_t)kt * (size_t)d_o * sizeof(float));
        float* y_cs = (float*)neon_alloc((size_t)kt * (size_t)d_o * sizeof(float));
        float* y_sv = (float*)neon_alloc((size_t)kt * (size_t)d_o * sizeof(float));
        if (!pk || !tn || !lx || !y_full || !y_cs || !y_sv) {
            neon_free(pk); neon_free(tn); neon_free(lx); neon_free(y_full); neon_free(y_cs); neon_free(y_sv);
            return 0;
        }
        uint64_t rng = 0x123456789ABCDEF0ULL;
        for (int t = 0; t < n_tiles; t++) {
            for (int blk = 0; blk < n_blks; blk++) {
                uint8_t* bp = pk + ((size_t)t * (size_t)n_blks + (size_t)blk) * tblk;
                float mx_wn = 0.0f;
                for (int r = 0; r < 16; r++) {
                    int gr = t * 16 + r;
                    float sc_f = (gr == 35) ? 0.50f : ((gr < 64) ? 0.08f : 0.01f);
                    uint16_t sc_h = fp32_to_fp16(sc_f);
                    sc_f = fp16_to_fp32(sc_h);
                    write_u16_le(bp + (size_t)r * 2u, sc_h);
                    double row_ss = 0.0;
                    for (int bo = 0; bo < bsz / 8; bo++) {
                        uint8_t bv[3];
                        for (int p = 0; p < nb; p++) {
                            rng = rng * 6364136223846793005ULL + 1ULL;
                            bv[p] = (gr == 35 && blk < 2) ? 0xFFu : (uint8_t)(rng >> 24);
                            bp[32u + (size_t)p * plane_b + (size_t)bo * 16u + (size_t)r] = bv[p];
                        }
                        for (int bit = 0; bit < 8; bit++) {
                            int code = 0;
                            for (int p = 0; p < nb; p++) {
                                if ((bv[p] >> (7 - bit)) & 1) code |= (1 << p);
                            }
                            double wv = (double)sc_f * ((double)code - 3.5);
                            row_ss += wv * wv;
                        }
                    }
                    float rn = neon_sqrtf((float)row_ss);
                    if (rn > mx_wn) mx_wn = rn;
                }
                tn[(size_t)t * (size_t)n_blks + (size_t)blk] = fp32_to_fp16(mx_wn * 1.001f + 1e-4f);
            }
        }
        for (int tok = 0; tok < kt; tok++) {
            for (int j = 0; j < d_i; j++) {
                lx[(size_t)tok * (size_t)d_i + (size_t)j] = (j < 2 * bsz) ? (tok == 0 ? 1.5f : -1.5f) : (((j % 7) - 3) * 0.02f);
            }
        }
        int ok_l = (gemm_bitserial_tile16_batch(pk, NULL, lx, y_full, d_o, d_i, bsz, nb, 0, kt, 0.0f, 2, NULL) == 0);
        int64_t sk_cs = 0, sk_sv = 0;
        if (lmhead_cs_top1_tile16_batch(pk, tn, lx, y_cs, d_o, d_i, bsz, nb, 0, kt, 8, 0.0f, 2, &sk_cs) != 0) ok_l = 0;
        if (lmhead_cs_top1_tile16_batch(pk, tn, lx, y_sv, d_o, d_i, bsz, nb, 0, kt, 8, 0.25f, 2, &sk_sv) != 0) ok_l = 0;
        if (sk_cs <= 0 || sk_sv <= 0) ok_l = 0;
        for (int tok = 0; tok < kt && ok_l; tok++) {
            const float* rf = y_full + (size_t)tok * (size_t)d_o;
            const float* cs = y_cs + (size_t)tok * (size_t)d_o;
            const float* sv = y_sv + (size_t)tok * (size_t)d_o;
            int id_f = 0, id_cs = 0, id_sv = 0;
            for (int r = 1; r < d_o; r++) {
                if (rf[r] > rf[id_f]) id_f = r;
                if (cs[r] > cs[id_cs]) id_cs = r;
                if (sv[r] > sv[id_sv]) id_sv = r;
            }
            if (id_f != id_cs || id_f != id_sv) ok_l = 0;
            if (neon_fabsf(rf[id_f] - cs[id_cs]) > 1e-3f || neon_fabsf(rf[id_f] - sv[id_sv]) > 1e-3f) ok_l = 0;
        }
        neon_free(pk); neon_free(tn); neon_free(lx); neon_free(y_full); neon_free(y_cs); neon_free(y_sv);
        if (!ok_l) return 0;
    }

    /* Gate M: Sparse tile/block masking (row_tile_mask + col_blk_mask) bit-exact parity for 1b_asym & 2b_gaus */
    {
        const int d_o = 64, d_i = 256, bsz = 64, kt = 4;
        const int n_tiles = d_o / 16; /* 4 */
        const int n_blks = d_i / bsz; /* 4 */
        const uint8_t rmask[4] = { 1, 0, 1, 0 };
        const uint8_t cmask[4] = { 0, 1, 0, 1 };
        const int modes_nb[2] = { 1, 2 | T16C_GAUSS_BIT };
        const int modes_as[2] = { 1, 0 };
        for (int mi = 0; mi < 2; mi++) {
            const int nb = modes_nb[mi];
            const int asym = modes_as[mi];
            const int eff = nb & 0x7F;
            const size_t tblk = (size_t)(asym ? 64 : 32) + (size_t)eff * (size_t)(bsz / 8) * 16u;
            uint8_t* pk = (uint8_t*)neon_alloc((size_t)n_tiles * (size_t)n_blks * tblk);
            float* mx = (float*)neon_alloc((size_t)kt * (size_t)d_i * sizeof(float));
            float* mx_z = (float*)neon_alloc((size_t)kt * (size_t)d_i * sizeof(float));
            float* y_ref = (float*)neon_alloc((size_t)kt * (size_t)d_o * sizeof(float));
            float* y_sp4 = (float*)neon_alloc((size_t)kt * (size_t)d_o * sizeof(float));
            float* y_sp1 = (float*)neon_alloc((size_t)kt * (size_t)d_o * sizeof(float));
            if (!pk || !mx || !mx_z || !y_ref || !y_sp4 || !y_sp1) {
                neon_free(pk); neon_free(mx); neon_free(mx_z); neon_free(y_ref); neon_free(y_sp4); neon_free(y_sp1);
                return 0;
            }
            uint64_t rng = 0xCAFEBABE12345678ULL ^ (uint64_t)mi;
            for (size_t i = 0; i < (size_t)n_tiles * (size_t)n_blks * tblk; i++) {
                rng = rng * 6364136223846793005ULL + 1ULL;
                pk[i] = (uint8_t)(rng >> 24);
            }
            for (int t = 0; t < n_tiles; t++) {
                for (int blk = 0; blk < n_blks; blk++) {
                    uint8_t* bp = pk + ((size_t)t * (size_t)n_blks + (size_t)blk) * tblk;
                    for (int r = 0; r < 16; r++) {
                        write_u16_le(bp + (size_t)r * 2u, fp32_to_fp16(0.05f + 0.01f * (float)(r % 5)));
                        if (asym) write_u16_le(bp + 32u + (size_t)r * 2u, fp32_to_fp16(-0.02f + 0.005f * (float)(r % 3)));
                    }
                }
            }
            for (int tok = 0; tok < kt; tok++) {
                for (int j = 0; j < d_i; j++) {
                    float v = (((tok * 13 + j * 7) % 19) - 9) * 0.11f;
                    mx[(size_t)tok * (size_t)d_i + (size_t)j] = v;
                    mx_z[(size_t)tok * (size_t)d_i + (size_t)j] = cmask[j / bsz] ? v : 0.0f;
                }
            }
            int64_t sk4 = 0, sk1 = 0;
            int ok_m = (gemm_bitserial_tile16_batch(pk, NULL, mx_z, y_ref, d_o, d_i, bsz, nb, asym, kt, 0.0f, 2, NULL) == 0);
            if (gemm_bitserial_tile16_sparse_batch(pk, NULL, mx, y_sp4, d_o, d_i, bsz, nb, asym, kt, 0.0f, rmask, cmask, 2, &sk4) != 0) ok_m = 0;
            for (int tok = 0; tok < kt && ok_m; tok++) {
                int64_t sk_t = 0;
                if (gemm_bitserial_tile16_sparse_batch(pk, NULL, mx + (size_t)tok * (size_t)d_i, y_sp1 + (size_t)tok * (size_t)d_o,
                                                       d_o, d_i, bsz, nb, asym, 1, 0.0f, rmask, cmask, 2, &sk_t) != 0) ok_m = 0;
                sk1 += sk_t;
            }
            /* Expected skips per pass: 2 skipped tiles * 4 blks + 2 active tiles * 2 skipped blks = 12 */
            if (sk4 != 12 || sk1 != 48) ok_m = 0;
            for (int tok = 0; tok < kt && ok_m; tok++) {
                for (int r = 0; r < d_o; r++) {
                    float ref_v = rmask[r / 16] ? y_ref[(size_t)tok * (size_t)d_o + (size_t)r] : 0.0f;
                    float sp4_v = y_sp4[(size_t)tok * (size_t)d_o + (size_t)r];
                    float sp1_v = y_sp1[(size_t)tok * (size_t)d_o + (size_t)r];
                    if (neon_fabsf(sp4_v - ref_v) > 1e-4f || neon_fabsf(sp1_v - ref_v) > 1e-4f) ok_m = 0;
                    uint32_t u4 = 0, u1 = 0;
                    memcpy(&u4, &sp4_v, 4);
                    memcpy(&u1, &sp1_v, 4);
                    if (u4 != u1) ok_m = 0;
                }
            }
            neon_free(pk); neon_free(mx); neon_free(mx_z); neon_free(y_ref); neon_free(y_sp4); neon_free(y_sp1);
            if (!ok_m) return 0;
        }
        /* Gate M2: L1-cached tile-outer multi-group bit-exactness for k=8 (dual=1 outlier) and k=10 (dual=0) */
        const int kt_cases[2] = { 8, 10 };
        for (int ci = 0; ci < 2; ci++) {
            const int ktc = kt_cases[ci];
            const int nb = (ci == 0) ? (2 | T16C_GAUSS_BIT) : 1;
            const int asym = (ci == 0) ? 0 : 1;
            const int eff = nb & 0x7F;
            const size_t tblk = (size_t)(asym ? 64 : 32) + (size_t)eff * (size_t)(bsz / 8) * 16u;
            uint8_t* pk = (uint8_t*)neon_alloc((size_t)n_tiles * (size_t)n_blks * tblk);
            float* mx = (float*)neon_alloc((size_t)ktc * (size_t)d_i * sizeof(float));
            float* y_b = (float*)neon_alloc((size_t)ktc * (size_t)d_o * sizeof(float));
            float* y_1 = (float*)neon_alloc((size_t)ktc * (size_t)d_o * sizeof(float));
            if (!pk || !mx || !y_b || !y_1) {
                neon_free(pk); neon_free(mx); neon_free(y_b); neon_free(y_1);
                return 0;
            }
            uint64_t rng = 0xDEADBEEF00112233ULL ^ (uint64_t)ci;
            for (size_t i = 0; i < (size_t)n_tiles * (size_t)n_blks * tblk; i++) {
                rng = rng * 6364136223846793005ULL + 1ULL;
                pk[i] = (uint8_t)(rng >> 24);
            }
            for (int t = 0; t < n_tiles; t++) {
                for (int blk = 0; blk < n_blks; blk++) {
                    uint8_t* bp = pk + ((size_t)t * (size_t)n_blks + (size_t)blk) * tblk;
                    for (int r = 0; r < 16; r++) {
                        write_u16_le(bp + (size_t)r * 2u, fp32_to_fp16(0.04f + 0.01f * (float)(r % 4)));
                        if (asym) write_u16_le(bp + 32u + (size_t)r * 2u, fp32_to_fp16(-0.015f + 0.004f * (float)(r % 3)));
                    }
                }
            }
            for (int tok = 0; tok < ktc; tok++) {
                for (int j = 0; j < d_i; j++) {
                    mx[(size_t)tok * (size_t)d_i + (size_t)j] = (((tok * 17 + j * 11) % 23) - 11) * 0.08f;
                }
                if (ci == 0) {
                    /* Force dual=1 high-dynamic-range outlier on every token in an active block */
                    mx[(size_t)tok * (size_t)d_i + 3] = 28.5f + (float)tok * 0.5f;
                }
            }
            int ok_m2 = (gemm_bitserial_tile16_sparse_batch(pk, NULL, mx, y_b, d_o, d_i, bsz, nb, asym, ktc, 0.0f, rmask, cmask, 2, NULL) == 0);
            for (int tok = 0; tok < ktc && ok_m2; tok++) {
                if (gemm_bitserial_tile16_sparse_batch(pk, NULL, mx + (size_t)tok * (size_t)d_i, y_1 + (size_t)tok * (size_t)d_o,
                                                       d_o, d_i, bsz, nb, asym, 1, 0.0f, rmask, cmask, 2, NULL) != 0) ok_m2 = 0;
            }
            for (int tok = 0; tok < ktc && ok_m2; tok++) {
                for (int r = 0; r < d_o; r++) {
                    uint32_t ub = 0, u1 = 0;
                    memcpy(&ub, &y_b[(size_t)tok * (size_t)d_o + (size_t)r], 4);
                    memcpy(&u1, &y_1[(size_t)tok * (size_t)d_o + (size_t)r], 4);
                    if (ub != u1) ok_m2 = 0;
                }
            }
            neon_free(pk); neon_free(mx); neon_free(y_b); neon_free(y_1);
            if (!ok_m2) return 0;
        }
    }

    /* Gate N: Row-Major Horner AND+VCNT (T16C_ROWMAJOR_BIT) accuracy, transpose, sparse masks & k=1/4/8 bit parity */
    {
        const int prev_abits = neon_set_andvcnt_abits(0);
        const int bsizes[3] = { 64, 128, 256 };
        const int abits_cases[2] = { 4, 5 };
        int ok_n = 1;
        for (int bi = 0; bi < 3 && ok_n; bi++) {
            const int bsz = bsizes[bi];
            const int d_o = 64, d_i = 512, kt = 8;
            const int n_tiles = d_o / 16;
            const int n_blks = d_i / bsz;
            const int asym = (bi == 2) ? 1 : 0;
            const int eff = (bi == 1) ? 2 : 1;
            const int nb_col = (eff == 2) ? (2 | T16C_GAUSS_BIT) : 1;
            const int nb_rm  = nb_col | T16C_ROWMAJOR_BIT;
            const size_t plane_b = (size_t)(bsz / 8) * 16u;
            const size_t tblk = (size_t)(asym ? 64 : 32) + (size_t)eff * plane_b;
            uint8_t* pk = (uint8_t*)neon_alloc((size_t)n_tiles * (size_t)n_blks * tblk);
            float* nx = (float*)neon_alloc((size_t)kt * (size_t)d_i * sizeof(float));
            float* y_ref = (float*)neon_alloc((size_t)kt * (size_t)d_o * sizeof(float));
            float* y_rm8 = (float*)neon_alloc((size_t)kt * (size_t)d_o * sizeof(float));
            float* y_rm1 = (float*)neon_alloc((size_t)kt * (size_t)d_o * sizeof(float));
            float* y_sp  = (float*)neon_alloc((size_t)kt * (size_t)d_o * sizeof(float));
            if (!pk || !nx || !y_ref || !y_rm8 || !y_rm1 || !y_sp) {
                neon_free(pk); neon_free(nx); neon_free(y_ref); neon_free(y_rm8); neon_free(y_rm1); neon_free(y_sp);
                neon_set_andvcnt_abits(prev_abits);
                return 0;
            }
            uint64_t rng = 0x55AA33CC771199ULL ^ (uint64_t)bi;
            for (int i = 0; i < kt * d_i; i++) {
                float g = 0.0f;
                for (int u = 0; u < 4; u++) {
                    rng = rng * 6364136223846793005ULL + 1ULL;
                    g += ((float)((int32_t)((rng >> 33) & 0xFFFFu) - 32768) / 32768.0f);
                }
                nx[i] = g * 0.45f;
            }
            memset(y_ref, 0, (size_t)kt * (size_t)d_o * sizeof(float));
            for (int t = 0; t < n_tiles; t++) {
                for (int blk = 0; blk < n_blks; blk++) {
                    uint8_t* bp = pk + ((size_t)t * (size_t)n_blks + (size_t)blk) * tblk;
                    for (int r = 0; r < 16; r++) {
                        int gr = t * 16 + r;
                        uint16_t sc_h = fp32_to_fp16(0.04f + 0.01f * (float)(r % 5));
                        float sc_f = fp16_to_fp32(sc_h);
                        write_u16_le(bp + (size_t)r * 2u, sc_h);
                        float bi_f = 0.0f;
                        if (asym) {
                            uint16_t bi_h = fp32_to_fp16(-0.015f + 0.005f * (float)(r % 3));
                            bi_f = fp16_to_fp32(bi_h);
                            write_u16_le(bp + 32u + (size_t)r * 2u, bi_h);
                            for (int tok = 0; tok < kt; tok++) {
                                float sx = 0.0f;
                                for (int c = 0; c < bsz; c++) sx += nx[(size_t)tok * (size_t)d_i + (size_t)blk * (size_t)bsz + (size_t)c];
                                y_ref[(size_t)tok * (size_t)d_o + (size_t)gr] += bi_f * sx;
                            }
                        }
                        for (int p = 0; p < eff; p++) {
                            float wp = sc_f * t16c_plane_weight(eff == 2, eff, p);
                            for (int bo = 0; bo < bsz / 8; bo++) {
                                rng = rng * 6364136223846793005ULL + 1ULL;
                                uint8_t bv = (uint8_t)(rng >> 24);
                                bp[(size_t)(asym ? 64 : 32) + (size_t)p * plane_b + (size_t)bo * 16u + (size_t)r] = bv;
                                for (int bit = 0; bit < 8; bit++) {
                                    float wv = wp * (((bv >> (7 - bit)) & 1) ? 1.0f : -1.0f);
                                    int col = blk * bsz + bo * 8 + bit;
                                    for (int tok = 0; tok < kt; tok++) {
                                        y_ref[(size_t)tok * (size_t)d_o + (size_t)gr] += wv * nx[(size_t)tok * (size_t)d_i + (size_t)col];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if (t16c_transpose_col_to_rowmajor_c(pk, d_o, d_i, bsz, nb_col, asym) != 0) ok_n = 0;

            for (int ai = 0; ai < 2 && ok_n; ai++) {
                const int ab = abits_cases[ai];
                neon_set_andvcnt_abits(ab);
                if (gemm_bitserial_tile16_batch(pk, NULL, nx, y_rm8, d_o, d_i, bsz, nb_rm, asym, kt, 0.0f, 2, NULL) != 0) ok_n = 0;
                for (int tok = 0; tok < kt && ok_n; tok++) {
                    if (gemm_bitserial_tile16_batch(pk, NULL, nx + (size_t)tok * (size_t)d_i, y_rm1 + (size_t)tok * (size_t)d_o,
                                                    d_o, d_i, bsz, nb_rm, asym, 1, 0.0f, 2, NULL) != 0) ok_n = 0;
                }
                const double min_cos = (ab == 4) ? 0.985 : 0.996;
                for (int tok = 0; tok < kt && ok_n; tok++) {
                    double dot = 0.0, na = 0.0, nb2 = 0.0;
                    for (int r = 0; r < d_o; r++) {
                        size_t idx = (size_t)tok * (size_t)d_o + (size_t)r;
                        uint32_t u8 = 0, u1 = 0;
                        memcpy(&u8, &y_rm8[idx], 4);
                        memcpy(&u1, &y_rm1[idx], 4);
                        if (u8 != u1) ok_n = 0;
                        double vr = y_ref[idx], vg = y_rm8[idx];
                        dot += vr * vg; na += vr * vr; nb2 += vg * vg;
                    }
                    double c = dot / ((double)neon_sqrtf((float)na) * (double)neon_sqrtf((float)nb2) + 1e-12);
                    if (c < min_cos) ok_n = 0;
                }
            }
            /* Sparse row_tile_mask parity on active tiles */
            if (ok_n) {
                const uint8_t rmask[4] = { 1, 0, 1, 0 };
                int64_t sk = 0;
                if (gemm_bitserial_tile16_sparse_batch(pk, NULL, nx, y_sp, d_o, d_i, bsz, nb_rm, asym, kt, 0.0f, rmask, NULL, 2, &sk) != 0) ok_n = 0;
                if (sk != (int64_t)(2 * n_blks)) ok_n = 0;
                for (int tok = 0; tok < kt && ok_n; tok++) {
                    for (int r = 0; r < d_o; r++) {
                        size_t idx = (size_t)tok * (size_t)d_o + (size_t)r;
                        float want = rmask[r / 16] ? y_rm8[idx] : 0.0f;
                        if (neon_fabsf(y_sp[idx] - want) > 1e-6f) ok_n = 0;
                    }
                }
            }
            neon_free(pk); neon_free(nx); neon_free(y_ref); neon_free(y_rm8); neon_free(y_rm1); neon_free(y_sp);
        }
        neon_set_andvcnt_abits(prev_abits);
        if (!ok_n) return 0;
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
