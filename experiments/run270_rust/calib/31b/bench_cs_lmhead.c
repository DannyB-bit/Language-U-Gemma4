/*
 * Exact Cauchy-Schwarz Top-1 LM-Head Pruning Benchmark (Single-Band, 16-Band Static, 16-Band Dynamic).
 *
 * Precomputes per-row L2 norms across 16 equal channel bands over mmap'd BF16 Safetensors weights
 * and verifies 100% exact greedy Top-1 logit equivalence against full unpruned scan.
 *
 * Build (Pi 4 / Linux ARM64):
 *   gcc -O3 -fopenmp -march=armv8-a+simd -mtune=cortex-a72 bench_cs_lmhead.c -o bench_cs_lmhead -lm
 * Run selftest:
 *   ./bench_cs_lmhead --selftest
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <io.h>
#include <windows.h>
#else
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

#define N_BANDS 16

static inline float bf16_to_f32(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static inline uint16_t f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return (uint16_t)(u >> 16);
}

static double now_ms(void) {
#if defined(_WIN32)
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
#endif
}

static uint64_t read_proc_io_read_bytes(void) {
#if defined(_WIN32)
    return 0;
#else
    FILE *f = fopen("/proc/self/io", "r");
    if (!f) return 0;
    char line[256];
    uint64_t val = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "read_bytes:", 11) == 0) {
            val = (uint64_t)strtoull(line + 11, NULL, 10);
            break;
        }
    }
    fclose(f);
    return val;
#endif
}

static void read_proc_status_kb(long *vm_hwm, long *vm_rss, long *vm_swap) {
    *vm_hwm = 0;
    *vm_rss = 0;
    *vm_swap = 0;
#if !defined(_WIN32)
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmHWM:", 6) == 0) *vm_hwm = atol(line + 6);
        else if (strncmp(line, "VmRSS:", 6) == 0) *vm_rss = atol(line + 6);
        else if (strncmp(line, "VmSwap:", 7) == 0) *vm_swap = atol(line + 7);
    }
    fclose(f);
#endif
}

static inline float dot_bf16_f32(const uint16_t *w, const float *h, size_t len) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    size_t c = 0;
    for (; c + 8 <= len; c += 8) {
        uint16x8_t bw = vld1q_u16(w + c);
        uint32x4_t u0 = vshll_n_u16(vget_low_u16(bw), 16);
        uint32x4_t u1 = vshll_n_u16(vget_high_u16(bw), 16);
        float32x4_t w0 = vreinterpretq_f32_u32(u0);
        float32x4_t w1 = vreinterpretq_f32_u32(u1);
        float32x4_t h0 = vld1q_f32(h + c);
        float32x4_t h1 = vld1q_f32(h + c + 4);
        acc0 = vfmaq_f32(acc0, w0, h0);
        acc1 = vfmaq_f32(acc1, w1, h1);
    }
    float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; c < len; ++c) {
        sum += bf16_to_f32(w[c]) * h[c];
    }
    return sum;
#else
    float sum = 0.0f;
    for (size_t c = 0; c < len; ++c) {
        sum += bf16_to_f32(w[c]) * h[c];
    }
    return sum;
#endif
}

static int parse_safetensors_bf16_tensor(
    const char *json,
    size_t json_len,
    const char *tensor_name,
    size_t *out_vocab,
    size_t *out_dim,
    uint64_t *out_off0,
    uint64_t *out_off1
) {
    char pat[256];
    int plen = snprintf(pat, sizeof(pat), "\"%s\"", tensor_name);
    if (plen <= 0 || (size_t)plen >= sizeof(pat)) return -1;

    const char *p = NULL;
    for (size_t i = 0; i + (size_t)plen <= json_len; ++i) {
        if (memcmp(json + i, pat, (size_t)plen) == 0) {
            p = json + i + plen;
            break;
        }
    }
    if (!p) return -2;

    const char *end = json + json_len;
    const char *obj_start = p;
    while (obj_start < end && *obj_start != '{') obj_start++;
    if (obj_start >= end) return -3;
    const char *obj_end = obj_start;
    int depth = 0;
    while (obj_end < end) {
        if (*obj_end == '{') depth++;
        else if (*obj_end == '}') {
            depth--;
            if (depth == 0) {
                obj_end++;
                break;
            }
        }
        obj_end++;
    }
    if (depth != 0) return -4;

    const char *sh = strstr(obj_start, "\"shape\"");
    if (!sh || sh >= obj_end) return -5;
    const char *br = strchr(sh, '[');
    if (!br || br >= obj_end) return -6;
    unsigned long long s0 = 0, s1 = 0;
    if (sscanf(br, "[ %llu , %llu ]", &s0, &s1) != 2) return -7;

    const char *doff = strstr(obj_start, "\"data_offsets\"");
    if (!doff || doff >= obj_end) return -8;
    const char *dbr = strchr(doff, '[');
    if (!dbr || dbr >= obj_end) return -9;
    unsigned long long o0 = 0, o1 = 0;
    if (sscanf(dbr, "[ %llu , %llu ]", &o0, &o1) != 2) return -10;

    if (o1 <= o0 || (o1 - o0) != s0 * s1 * 2ULL) return -11;
    *out_vocab = (size_t)s0;
    *out_dim = (size_t)s1;
    *out_off0 = (uint64_t)o0;
    *out_off1 = (uint64_t)o1;
    return 0;
}

typedef struct {
    uint32_t id;
    float norm;
} RowNormPair;

typedef struct {
    size_t vocab;
    size_t dim;
    size_t band_size;
    const uint16_t *w;     /* [vocab, dim] BF16 */
    uint16_t *w_perm;      /* optional [vocab, dim] pre-permuted BF16 for contiguous band NEON */
    float *coarse_scores;  /* reusable [vocab] buffer for coarse directional sieve */
    float *row_norms;      /* [vocab] full L2 upper bound */
    float *band_norms;     /* [vocab * N_BANDS] band L2 upper bound */
    int32_t *col_perm;     /* optional [dim] static channel permutation */
    uint8_t *is_seed;      /* [vocab] seed membership flag */
    uint32_t *seed_ids;    /* [seed_count] seed row indices */
    size_t seed_count;
    float norm_min;
    float norm_p50;
    float norm_max;
} LmHeadIndex;

static int cmp_f32_asc(const void *a, const void *b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

static int cmp_row_norm_desc(const void *a, const void *b) {
    const RowNormPair *pa = (const RowNormPair *)a;
    const RowNormPair *pb = (const RowNormPair *)b;
    if (pb->norm > pa->norm) return 1;
    if (pb->norm < pa->norm) return -1;
    return (pa->id > pb->id) - (pa->id < pb->id);
}

static int cmp_u32_asc(const void *a, const void *b) {
    uint32_t ua = *(const uint32_t *)a;
    uint32_t ub = *(const uint32_t *)b;
    return (ua > ub) - (ua < ub);
}

static int32_t *load_i32_perm(const char *path, size_t dim) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    int32_t *perm = (int32_t *)malloc(dim * sizeof(int32_t));
    uint8_t *seen = (uint8_t *)calloc(dim, 1);
    if (!perm || !seen) {
        free(perm);
        free(seen);
        fclose(f);
        return NULL;
    }
    size_t got = fread(perm, sizeof(int32_t), dim, f);
    fclose(f);
    if (got != dim) {
        free(perm);
        free(seen);
        return NULL;
    }
    for (size_t i = 0; i < dim; ++i) {
        if (perm[i] < 0 || (size_t)perm[i] >= dim || seen[perm[i]]) {
            free(perm);
            free(seen);
            return NULL;
        }
        seen[perm[i]] = 1;
    }
    free(seen);
    return perm;
}

static int build_lmhead_index(const uint16_t *w, size_t vocab, size_t dim, const int32_t *col_perm_in, LmHeadIndex *idx) {
    if (!w || vocab == 0 || dim == 0 || (dim % N_BANDS) != 0) return -1;
    memset(idx, 0, sizeof(*idx));
    idx->vocab = vocab;
    idx->dim = dim;
    idx->band_size = dim / N_BANDS;
    idx->w = w;
    idx->row_norms = (float *)malloc(vocab * sizeof(float));
    idx->band_norms = (float *)malloc(vocab * N_BANDS * sizeof(float));
    idx->coarse_scores = (float *)malloc(vocab * sizeof(float));
    idx->is_seed = (uint8_t *)calloc(vocab, 1);
    if (!idx->row_norms || !idx->band_norms || !idx->coarse_scores || !idx->is_seed) {
        free(idx->row_norms);
        free(idx->band_norms);
        free(idx->coarse_scores);
        free(idx->is_seed);
        return -2;
    }

    if (col_perm_in) {
        idx->col_perm = (int32_t *)malloc(dim * sizeof(int32_t));
        idx->w_perm = (uint16_t *)malloc(vocab * dim * sizeof(uint16_t));
        if (!idx->col_perm || !idx->w_perm) {
            free(idx->row_norms);
            free(idx->band_norms);
            free(idx->coarse_scores);
            free(idx->is_seed);
            free(idx->col_perm);
            free(idx->w_perm);
            return -2;
        }
        memcpy(idx->col_perm, col_perm_in, dim * sizeof(int32_t));
    }

#if !defined(_WIN32)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t r = 0; r < (int64_t)vocab; ++r) {
        const uint16_t *row = w + (size_t)r * dim;
        uint16_t *wp_row = idx->w_perm ? (idx->w_perm + (size_t)r * dim) : NULL;
        double total_ss = 0.0;
        float *bn = idx->band_norms + (size_t)r * N_BANDS;
        if (!col_perm_in) {
            for (int b = 0; b < N_BANDS; ++b) {
                const uint16_t *seg = row + (size_t)b * idx->band_size;
                double ss = 0.0;
                for (size_t c = 0; c < idx->band_size; ++c) {
                    double v = (double)bf16_to_f32(seg[c]);
                    ss += v * v;
                }
                total_ss += ss;
                bn[b] = (float)(sqrt(ss) * 1.00005 + 1e-6);
            }
        } else {
            for (int b = 0; b < N_BANDS; ++b) {
                double ss = 0.0;
                size_t b0 = (size_t)b * idx->band_size;
                for (size_t k = 0; k < idx->band_size; ++k) {
                    uint16_t raw = row[col_perm_in[b0 + k]];
                    if (wp_row) wp_row[b0 + k] = raw;
                    double v = (double)bf16_to_f32(raw);
                    ss += v * v;
                }
                total_ss += ss;
                bn[b] = (float)(sqrt(ss) * 1.00005 + 1e-6);
            }
        }
        idx->row_norms[r] = (float)(sqrt(total_ss) * 1.00005 + 1e-6);
    }

    float *tmp = (float *)malloc(vocab * sizeof(float));
    if (tmp) {
        memcpy(tmp, idx->row_norms, vocab * sizeof(float));
        qsort(tmp, vocab, sizeof(float), cmp_f32_asc);
        idx->norm_min = tmp[0];
        idx->norm_p50 = tmp[vocab / 2];
        idx->norm_max = tmp[vocab - 1];
        free(tmp);
    }

    /* Seed candidate rows: first K_seed rows UNION top K_seed highest-L2-norm rows. */
    size_t k_seed = (vocab >= 8192) ? 512 : (vocab / 16);
    if (k_seed < 16 && vocab >= 16) k_seed = 16;
    if (k_seed > vocab) k_seed = vocab;
    RowNormPair *pairs = (RowNormPair *)malloc(vocab * sizeof(RowNormPair));
    idx->seed_ids = (uint32_t *)malloc((2 * k_seed + 1) * sizeof(uint32_t));
    if (!pairs || !idx->seed_ids) {
        free(pairs);
        free(idx->row_norms);
        free(idx->band_norms);
        free(idx->is_seed);
        free(idx->col_perm);
        free(idx->seed_ids);
        return -3;
    }
    for (size_t r = 0; r < vocab; ++r) {
        pairs[r].id = (uint32_t)r;
        pairs[r].norm = idx->row_norms[r];
    }
    qsort(pairs, vocab, sizeof(RowNormPair), cmp_row_norm_desc);
    size_t sc = 0;
    for (size_t r = 0; r < k_seed; ++r) {
        if (!idx->is_seed[r]) {
            idx->is_seed[r] = 1;
            idx->seed_ids[sc++] = (uint32_t)r;
        }
    }
    for (size_t i = 0; i < k_seed; ++i) {
        uint32_t rid = pairs[i].id;
        if (!idx->is_seed[rid]) {
            idx->is_seed[rid] = 1;
            idx->seed_ids[sc++] = rid;
        }
    }
    qsort(idx->seed_ids, sc, sizeof(uint32_t), cmp_u32_asc);
    idx->seed_count = sc;
    free(pairs);
    return 0;
}

static void free_lmhead_index(LmHeadIndex *idx) {
    free(idx->w_perm);
    free(idx->coarse_scores);
    free(idx->row_norms);
    free(idx->band_norms);
    free(idx->col_perm);
    free(idx->is_seed);
    free(idx->seed_ids);
    idx->w_perm = NULL;
    idx->coarse_scores = NULL;
    idx->row_norms = NULL;
    idx->band_norms = NULL;
    idx->col_perm = NULL;
    idx->is_seed = NULL;
    idx->seed_ids = NULL;
}

typedef enum {
    CS_MODE_FULL = 0,
    CS_MODE_SINGLE_BAND = 1,
    CS_MODE_STATIC_16BAND = 2,
    CS_MODE_DYNAMIC_16BAND = 3,
    CS_MODE_SUBSPACE_SIEVE = 4,
} CsMode;

typedef struct {
    uint32_t best_id;
    float best_logit;
    float h_norm;
    double elapsed_ms;
    uint64_t rows_early_terminated;
    uint64_t rows_full_eval;
    uint64_t col_macs_done;
    uint64_t col_macs_total;
    size_t seed_rows;
} CsResult;

static inline uint32_t f32_to_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

static inline float bits_to_f32(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static void atomic_max_best(_Atomic uint64_t *shared_best, float cand_val, uint32_t cand_id) {
    uint64_t cur = atomic_load_explicit(shared_best, memory_order_relaxed);
    for (;;) {
        float cur_val = bits_to_f32((uint32_t)(cur >> 32));
        uint32_t cur_id = (uint32_t)(cur & 0xFFFFFFFFULL);
        if (cand_val < cur_val || (cand_val == cur_val && cand_id >= cur_id)) {
            return;
        }
        uint64_t next = (((uint64_t)f32_to_bits(cand_val)) << 32) | (uint64_t)cand_id;
        if (atomic_compare_exchange_weak_explicit(
                shared_best, &cur, next, memory_order_relaxed, memory_order_relaxed)) {
            return;
        }
    }
}

static inline float dot_bf16_f32_perm(
    const uint16_t *w,
    const float *h_band,
    const int32_t *band_cols,
    size_t len
) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    size_t c = 0;
    for (; c + 8 <= len; c += 8) {
        uint16_t tmp[8] = {
            w[band_cols[c + 0]], w[band_cols[c + 1]], w[band_cols[c + 2]], w[band_cols[c + 3]],
            w[band_cols[c + 4]], w[band_cols[c + 5]], w[band_cols[c + 6]], w[band_cols[c + 7]],
        };
        uint16x8_t bw = vld1q_u16(tmp);
        float32x4_t w0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bw), 16));
        float32x4_t w1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bw), 16));
        acc0 = vfmaq_f32(acc0, w0, vld1q_f32(h_band + c));
        acc1 = vfmaq_f32(acc1, w1, vld1q_f32(h_band + c + 4));
    }
    float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; c < len; ++c) {
        sum += bf16_to_f32(w[band_cols[c]]) * h_band[c];
    }
    return sum;
#else
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    size_t c = 0;
    for (; c + 4 <= len; c += 4) {
        s0 += bf16_to_f32(w[band_cols[c + 0]]) * h_band[c + 0];
        s1 += bf16_to_f32(w[band_cols[c + 1]]) * h_band[c + 1];
        s2 += bf16_to_f32(w[band_cols[c + 2]]) * h_band[c + 2];
        s3 += bf16_to_f32(w[band_cols[c + 3]]) * h_band[c + 3];
    }
    float sum = (s0 + s1) + (s2 + s3);
    for (; c < len; ++c) {
        sum += bf16_to_f32(w[band_cols[c]]) * h_band[c];
    }
    return sum;
#endif
}

static void eval_seed_rows(
    const LmHeadIndex *idx,
    const float *h,
    _Atomic uint64_t *shared_best,
    uint64_t *out_seed_macs,
    size_t *out_seed_count
) {
    uint64_t macs = 0;
    size_t count = idx->seed_count;
#if !defined(_WIN32)
#pragma omp parallel reduction(+ : macs)
#endif
    {
        float lbest_v = -1e30f;
        uint32_t lbest_i = 0xFFFFFFFFU;
#if !defined(_WIN32)
#pragma omp for schedule(static) nowait
#endif
        for (int64_t si = 0; si < (int64_t)count; ++si) {
            uint32_t r = idx->seed_ids[si];
            float d = dot_bf16_f32(idx->w + (size_t)r * idx->dim, h, idx->dim);
            macs += (uint64_t)idx->dim;
            if (d > lbest_v || (d == lbest_v && r < lbest_i)) {
                lbest_v = d;
                lbest_i = r;
            }
        }
        if (lbest_i != 0xFFFFFFFFU) {
            atomic_max_best(shared_best, lbest_v, lbest_i);
        }
    }
    *out_seed_macs = macs;
    *out_seed_count = count;
}

static void compute_top1_mode(const LmHeadIndex *idx, const float *h, CsMode mode, CsResult *res) {
    memset(res, 0, sizeof(*res));
    float *h_perm_heap = NULL;
    float h_perm_stack[8192];
    const float *h_banded = h;
    if (idx->col_perm) {
        float *hp = (idx->dim <= 8192) ? h_perm_stack : (h_perm_heap = (float *)malloc(idx->dim * sizeof(float)));
        if (hp) {
            for (size_t c = 0; c < idx->dim; ++c) hp[c] = h[idx->col_perm[c]];
            h_banded = hp;
        }
    }

    double h_ss = 0.0;
    float h_band_norms[N_BANDS];
    for (int b = 0; b < N_BANDS; ++b) {
        const float *hb = h_banded + (size_t)b * idx->band_size;
        double bss = 0.0;
        for (size_t c = 0; c < idx->band_size; ++c) {
            double v = (double)hb[c];
            bss += v * v;
        }
        h_ss += bss;
        h_band_norms[b] = (float)(sqrt(bss) * 1.00005 + 1e-6);
    }
    float h_norm = (float)(sqrt(h_ss) * 1.00005 + 1e-6);
    res->h_norm = (float)sqrt(h_ss);
    res->col_macs_total = (uint64_t)idx->vocab * (uint64_t)idx->dim;

    int perm[N_BANDS];
    for (int i = 0; i < N_BANDS; ++i) perm[i] = i;
    if (mode == CS_MODE_DYNAMIC_16BAND) {
        /* Sort bands by descending h_band_norms[b] so residual tail tightens fastest. */
        for (int i = 0; i < N_BANDS; ++i) {
            for (int j = i + 1; j < N_BANDS; ++j) {
                if (h_band_norms[perm[j]] > h_band_norms[perm[i]]) {
                    int t = perm[i];
                    perm[i] = perm[j];
                    perm[j] = t;
                }
            }
        }
    }

    _Atomic uint64_t shared_best;
    uint64_t init_best = (((uint64_t)f32_to_bits(-1e30f)) << 32) | 0xFFFFFFFFULL;
    atomic_init(&shared_best, init_best);

    double t0 = now_ms();
    uint64_t macs_done = 0;
    uint64_t early_rows = 0;
    uint64_t full_rows = 0;

    if (mode == CS_MODE_FULL) {
        const size_t dim = idx->dim;
        const size_t vocab = idx->vocab;
#if !defined(_WIN32)
#pragma omp parallel
#endif
        {
            float lbest_v = -1e30f;
            uint32_t lbest_i = 0xFFFFFFFFU;
#if !defined(_WIN32)
#pragma omp for schedule(static) nowait
#endif
            for (int64_t r = 0; r < (int64_t)vocab; ++r) {
                float d = dot_bf16_f32(idx->w + (size_t)r * dim, h, dim);
                if (d > lbest_v || (d == lbest_v && (uint32_t)r < lbest_i)) {
                    lbest_v = d;
                    lbest_i = (uint32_t)r;
                }
            }
            atomic_max_best(&shared_best, lbest_v, lbest_i);
        }
        macs_done = res->col_macs_total;
        full_rows = (uint64_t)vocab;
    } else if (mode == CS_MODE_SUBSPACE_SIEVE) {
        /* Two-stage directional coarse-to-fine sieve over contiguous pre-permuted heavy subspace.
         * Evaluates top d_coarse channels (2 bands = 12.5% of dim) via contiguous NEON, builds an
         * O(V) 256-bin score quantile cutoff for top-K (~3.1% of vocab) candidates + seed rows,
         * and refines only surviving candidates. */
        uint64_t seed_macs = 0;
        size_t seed_cnt = 0;
        eval_seed_rows(idx, h, &shared_best, &seed_macs, &seed_cnt);
        res->seed_rows = seed_cnt;
        macs_done += seed_macs;
        full_rows += seed_cnt;

        const size_t dim = idx->dim;
        const size_t vocab = idx->vocab;
        const size_t d_coarse = idx->band_size * 2;
        const uint16_t *w_act = idx->w_perm ? idx->w_perm : idx->w;
        const float *h_act = h_banded;
        float *cscores = idx->coarse_scores;
        float g_min = 1e30f;
        float g_max = -1e30f;

#if !defined(_WIN32)
#pragma omp parallel reduction(+ : macs_done) reduction(min : g_min) reduction(max : g_max)
#endif
        {
#if !defined(_WIN32)
#pragma omp for schedule(static, 512) nowait
#endif
            for (int64_t r = 0; r < (int64_t)vocab; ++r) {
                if (idx->is_seed[r]) {
                    cscores[r] = -1e30f;
                    continue;
                }
                float c = dot_bf16_f32(w_act + (size_t)r * dim, h_act, d_coarse);
                cscores[r] = c;
                macs_done += (uint64_t)d_coarse;
                if (c < g_min) g_min = c;
                if (c > g_max) g_max = c;
            }
        }

        float tau = g_min - 1.0f;
        if (g_max > g_min + 1e-6f) {
            uint32_t hist[256] = {0};
            float inv_span = 255.0f / (g_max - g_min);
            for (size_t r = 0; r < vocab; ++r) {
                if (idx->is_seed[r]) continue;
                int b = (int)((cscores[r] - g_min) * inv_span);
                if (b < 0) b = 0;
                if (b > 255) b = 255;
                hist[b]++;
            }
            size_t k_keep = (vocab >= 8192) ? (vocab / 32) : (vocab / 8);
            if (k_keep < 64) k_keep = (vocab < 64) ? vocab : 64;
            size_t acc_cnt = 0;
            int cut_bin = 0;
            for (int b = 255; b >= 0; --b) {
                acc_cnt += hist[b];
                cut_bin = b;
                if (acc_cnt >= k_keep) break;
            }
            tau = g_min + ((float)cut_bin / 255.0f) * (g_max - g_min) - 1e-5f;
        }

        const size_t d_tail = dim - d_coarse;
#if !defined(_WIN32)
#pragma omp parallel reduction(+ : macs_done, early_rows, full_rows)
#endif
        {
            float own_best_v = -1e30f;
            uint32_t own_best_i = 0xFFFFFFFFU;
#if !defined(_WIN32)
#pragma omp for schedule(static, 256) nowait
#endif
            for (int64_t r = 0; r < (int64_t)vocab; ++r) {
                if (idx->is_seed[r]) continue;
                float c = cscores[r];
                if (c < tau) {
                    early_rows++;
                    continue;
                }
                float tot = c + dot_bf16_f32(w_act + (size_t)r * dim + d_coarse, h_act + d_coarse, d_tail);
                macs_done += (uint64_t)d_tail;
                full_rows++;
                if (tot > own_best_v || (tot == own_best_v && (uint32_t)r < own_best_i)) {
                    own_best_v = tot;
                    own_best_i = (uint32_t)r;
                }
            }
            if (own_best_i != 0xFFFFFFFFU) {
                atomic_max_best(&shared_best, own_best_v, own_best_i);
            }
        }
    } else {
        uint64_t seed_macs = 0;
        size_t seed_cnt = 0;
        eval_seed_rows(idx, h, &shared_best, &seed_macs, &seed_cnt);
        res->seed_rows = seed_cnt;
        macs_done += seed_macs;
        full_rows += seed_cnt;

        const size_t dim = idx->dim;
        const size_t bs = idx->band_size;
        const size_t vocab = idx->vocab;
        const int32_t *cperm = idx->col_perm;
        const uint16_t *w_perm = idx->w_perm;

#if !defined(_WIN32)
#pragma omp parallel reduction(+ : macs_done, early_rows, full_rows)
#endif
        {
            /* Keep shared pruning threshold separate from this thread's own (val, id) pair
             * so updating prune_thresh never pairs another thread's score with our row id. */
            float prune_thresh = bits_to_f32((uint32_t)(atomic_load_explicit(&shared_best, memory_order_relaxed) >> 32));
            float own_best_v = -1e30f;
            uint32_t own_best_i = 0xFFFFFFFFU;

#if !defined(_WIN32)
#pragma omp for schedule(static, 256) nowait
#endif
            for (int64_t r = 0; r < (int64_t)vocab; ++r) {
                if (idx->is_seed[r]) continue;
                if (((r & 63) == 0)) {
                    uint64_t cur = atomic_load_explicit(&shared_best, memory_order_relaxed);
                    float cur_best = bits_to_f32((uint32_t)(cur >> 32));
                    if (cur_best > prune_thresh) prune_thresh = cur_best;
                }

                float full_ub = idx->row_norms[r] * h_norm;
                if (full_ub + 1e-4f < prune_thresh) {
                    early_rows++;
                    continue;
                }

                if (mode == CS_MODE_SINGLE_BAND) {
                    float d = dot_bf16_f32(idx->w + (size_t)r * dim, h, dim);
                    macs_done += (uint64_t)dim;
                    full_rows++;
                    if (d > own_best_v || (d == own_best_v && (uint32_t)r < own_best_i)) {
                        own_best_v = d;
                        own_best_i = (uint32_t)r;
                    }
                    if (d >= prune_thresh - 1e-6f) {
                        if (d > prune_thresh) prune_thresh = d;
                        atomic_max_best(&shared_best, d, (uint32_t)r);
                    }
                    continue;
                }

                const float *bn = idx->band_norms + (size_t)r * N_BANDS;
                float tail_rem[N_BANDS];
                tail_rem[N_BANDS - 1] = 0.0f;
                for (int k = N_BANDS - 2; k >= 0; --k) {
                    int bnext = perm[k + 1];
                    tail_rem[k] = tail_rem[k + 1] + bn[bnext] * h_band_norms[bnext];
                }
                float total_band_ub = tail_rem[0] + bn[perm[0]] * h_band_norms[perm[0]];
                if (total_band_ub + 1e-3f + 2e-4f * total_band_ub < prune_thresh) {
                    early_rows++;
                    continue;
                }

                const uint16_t *row_w = idx->w + (size_t)r * dim;
                const uint16_t *row_wp = w_perm ? (w_perm + (size_t)r * dim) : NULL;
                float acc = 0.0f;
                int pruned = 0;
                for (int k = 0; k < N_BANDS; ++k) {
                    int b = perm[k];
                    acc += row_wp
                        ? dot_bf16_f32(row_wp + (size_t)b * bs, h_banded + (size_t)b * bs, bs)
                        : (cperm
                            ? dot_bf16_f32_perm(row_w, h_banded + (size_t)b * bs, cperm + (size_t)b * bs, bs)
                            : dot_bf16_f32(row_w + (size_t)b * bs, h + (size_t)b * bs, bs));
                    macs_done += (uint64_t)bs;
                    float rem_bound = tail_rem[k];
                    /* Conservative rounding margin proportional to |acc| + rem_bound prevents false pruning. */
                    float slack = 1e-3f + 2e-4f * (fabsf(acc) + rem_bound);
                    if (k + 1 < N_BANDS && (acc + rem_bound + slack < prune_thresh)) {
                        pruned = 1;
                        break;
                    }
                }
                if (pruned) {
                    early_rows++;
                } else {
                    full_rows++;
                    if (acc > own_best_v || (acc == own_best_v && (uint32_t)r < own_best_i)) {
                        own_best_v = acc;
                        own_best_i = (uint32_t)r;
                    }
                    if (acc >= prune_thresh - 1e-6f) {
                        if (acc > prune_thresh) prune_thresh = acc;
                        atomic_max_best(&shared_best, acc, (uint32_t)r);
                    }
                }
            }
            if (own_best_i != 0xFFFFFFFFU) {
                atomic_max_best(&shared_best, own_best_v, own_best_i);
            }
        }
    }

    double t1 = now_ms();
    free(h_perm_heap);
    uint64_t final_best = atomic_load_explicit(&shared_best, memory_order_relaxed);
    res->best_logit = bits_to_f32((uint32_t)(final_best >> 32));
    res->best_id = (uint32_t)(final_best & 0xFFFFFFFFULL);
    res->elapsed_ms = t1 - t0;
    res->rows_early_terminated = early_rows;
    res->rows_full_eval = full_rows;
    res->col_macs_done = macs_done;
}

static float *load_f32_queries(const char *path, size_t n_queries, size_t dim) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    size_t need = n_queries * dim;
    float *buf = (float *)malloc(need * sizeof(float));
    if (!buf) {
        fclose(f);
        return NULL;
    }
    size_t got = fread(buf, sizeof(float), need, f);
    fclose(f);
    if (got != need) {
        free(buf);
        return NULL;
    }
    return buf;
}

static int run_selftest(void) {
    const size_t vocab = 2048;
    const size_t dim = 256; /* 16 bands * 16 */
    uint16_t *w = (uint16_t *)malloc(vocab * dim * sizeof(uint16_t));
    float *h = (float *)malloc(dim * sizeof(float));
    if (!w || !h) {
        fprintf(stderr, "[SELFTEST] OOM\n");
        free(w);
        free(h);
        return 1;
    }

    /* Gate 1: Safetensors parser sanity */
    const char *fake_json =
        "{\"model.embed_tokens.weight\":{\"dtype\":\"BF16\",\"shape\":[2048,256],\"data_offsets\":[0,1048576]}}";
    size_t p_v = 0, p_d = 0;
    uint64_t o0 = 1, o1 = 0;
    if (parse_safetensors_bf16_tensor(fake_json, strlen(fake_json), "model.embed_tokens.weight", &p_v, &p_d, &o0, &o1) != 0 ||
        p_v != 2048 || p_d != 256 || o0 != 0 || o1 != 1048576) {
        fprintf(stderr, "[SELFTEST FAIL] Gate 1 safetensors parser\n");
        return 2;
    }

    /* Synthesize unit-normalized random rows + one planted winner aligned to heavy band */
    for (size_t r = 0; r < vocab; ++r) {
        double ss = 0.0;
        float tmp[256];
        for (size_t c = 0; c < dim; ++c) {
            float x = sinf((float)((r * 37 + c * 13) % 997) * 0.17f);
            tmp[c] = x;
            ss += (double)x * (double)x;
        }
        float inv = 1.0f / (float)sqrt(ss);
        for (size_t c = 0; c < dim; ++c) {
            w[r * dim + c] = f32_to_bf16(tmp[c] * inv);
        }
    }
    for (size_t c = 0; c < dim; ++c) {
        /* Skewed channel energy in band 11 so dynamic permutation hits highest-energy band first. */
        size_t band = c / (dim / N_BANDS);
        float scale = (band == 11) ? 18.0f : 0.35f;
        h[c] = scale * cosf((float)c * 0.23f);
    }
    const uint32_t target_row = 777;
    {
        double ss = 0.0;
        for (size_t c = 0; c < dim; ++c) ss += (double)h[c] * (double)h[c];
        float inv = 1.0f / (float)sqrt(ss);
        for (size_t c = 0; c < dim; ++c) {
            w[target_row * dim + c] = f32_to_bf16(h[c] * inv);
        }
    }

    LmHeadIndex idx;
    if (build_lmhead_index(w, vocab, dim, NULL, &idx) != 0) {
        fprintf(stderr, "[SELFTEST FAIL] Gate 2 index build\n");
        free(w);
        free(h);
        return 3;
    }
    /* Gate 3: non-degenerate norms */
    if (!(idx.norm_max > 0.9f && idx.norm_min > 0.9f && idx.norm_max < 1.05f)) {
        fprintf(stderr, "[SELFTEST FAIL] Gate 3 norm range [%.4f, %.4f]\n", idx.norm_min, idx.norm_max);
        free_lmhead_index(&idx);
        free(w);
        free(h);
        return 4;
    }

    CsResult r_full, r_sb, r_st, r_dy;
    compute_top1_mode(&idx, h, CS_MODE_FULL, &r_full);
    compute_top1_mode(&idx, h, CS_MODE_SINGLE_BAND, &r_sb);
    compute_top1_mode(&idx, h, CS_MODE_STATIC_16BAND, &r_st);
    compute_top1_mode(&idx, h, CS_MODE_DYNAMIC_16BAND, &r_dy);

    if (r_full.best_id != target_row || r_sb.best_id != target_row || r_st.best_id != target_row || r_dy.best_id != target_row) {
        fprintf(stderr, "[SELFTEST FAIL] Gate 4/5 Top-1 mismatch: full=%u sb=%u st=%u dy=%u expected=%u\n",
                r_full.best_id, r_sb.best_id, r_st.best_id, r_dy.best_id, target_row);
        free_lmhead_index(&idx);
        free(w);
        free(h);
        return 5;
    }

    double dy_saved_pct = 100.0 * (1.0 - (double)r_dy.col_macs_done / (double)r_dy.col_macs_total);
    if (dy_saved_pct < 50.0) {
        fprintf(stderr, "[SELFTEST FAIL] Gate 6 expected dynamic 16-band MAC saving >=50%% on skewed synthetic vector, got %.2f%%\n", dy_saved_pct);
        free_lmhead_index(&idx);
        free(w);
        free(h);
        return 6;
    }

    /* Static channel-RMS permutation check: place heavy band 11 into band 0 via col_perm. */
    int32_t perm_p[256];
    size_t bs = dim / N_BANDS;
    for (size_t i = 0; i < bs; ++i) perm_p[i] = (int32_t)(11 * bs + i);
    size_t wp = bs;
    for (size_t b = 0; b < N_BANDS; ++b) {
        if (b == 11) continue;
        for (size_t i = 0; i < bs; ++i) perm_p[wp++] = (int32_t)(b * bs + i);
    }
    LmHeadIndex idx_p;
    if (build_lmhead_index(w, vocab, dim, perm_p, &idx_p) != 0) {
        fprintf(stderr, "[SELFTEST FAIL] Gate 7 permuted index build\n");
        free_lmhead_index(&idx);
        free(w);
        free(h);
        return 7;
    }
    CsResult r_st_p, r_sv_p;
    compute_top1_mode(&idx_p, h, CS_MODE_STATIC_16BAND, &r_st_p);
    compute_top1_mode(&idx_p, h, CS_MODE_SUBSPACE_SIEVE, &r_sv_p);
    double st_p_saved = 100.0 * (1.0 - (double)r_st_p.col_macs_done / (double)r_st_p.col_macs_total);
    double sv_p_saved = 100.0 * (1.0 - (double)r_sv_p.col_macs_done / (double)r_sv_p.col_macs_total);
    if (r_st_p.best_id != target_row || st_p_saved < 50.0 || r_sv_p.best_id != target_row || sv_p_saved < 50.0) {
        fprintf(stderr, "[SELFTEST FAIL] Gate 7/8 static-perm/sieve mismatch st=(%u,%.2f%%) sv=(%u,%.2f%%)\n",
                r_st_p.best_id, st_p_saved, r_sv_p.best_id, sv_p_saved);
        free_lmhead_index(&idx_p);
        free_lmhead_index(&idx);
        free(w);
        free(h);
        return 7;
    }
    free_lmhead_index(&idx_p);

    printf("SELFTEST_PASS_6_GATES top1=%u logit=%.4f dy_mac_saved=%.2f%% st_perm_saved=%.2f%% sieve_saved=%.2f%% early_rows=%llu/%zu\n",
           r_dy.best_id, r_dy.best_logit, dy_saved_pct, st_p_saved, sv_p_saved,
           (unsigned long long)r_dy.rows_early_terminated, vocab);
    free_lmhead_index(&idx);
    free(w);
    free(h);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "--selftest") == 0) {
        return run_selftest();
    }
    if (argc < 6) {
        fprintf(stderr,
                "Usage:\n"
                "  %s --selftest\n"
                "  %s <label> <model.safetensors> <tensor_name> <queries.f32> <n_queries> [threads] [perm.i32]\n",
                argv[0], argv[0]);
        return 1;
    }

#if defined(_WIN32)
    fprintf(stderr, "Full mmap benchmark requires POSIX mmap (/proc/self/io); run --selftest on Windows.\n");
    return 2;
#else
    const char *label = argv[1];
    const char *sf_path = argv[2];
    const char *tensor_name = argv[3];
    const char *q_path = argv[4];
    size_t n_queries = (size_t)atoi(argv[5]);
    int threads = (argc >= 7) ? atoi(argv[6]) : 4;
    if (threads < 1) threads = 1;
    const char *perm_path = (argc >= 8 && strcmp(argv[7], "-") != 0) ? argv[7] : NULL;
    omp_set_num_threads(threads);

    int fd = open(sf_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Failed to open %s: %s\n", sf_path, strerror(errno));
        return 2;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return 3;
    }
    uint64_t hdr_len = 0;
    if (read(fd, &hdr_len, sizeof(hdr_len)) != sizeof(hdr_len) || hdr_len > (64ULL << 20)) {
        close(fd);
        return 4;
    }
    char *hdr = (char *)malloc((size_t)hdr_len + 1);
    if (!hdr) {
        close(fd);
        return 5;
    }
    size_t got = 0;
    while (got < (size_t)hdr_len) {
        ssize_t n = read(fd, hdr + got, (size_t)hdr_len - got);
        if (n <= 0) {
            free(hdr);
            close(fd);
            return 6;
        }
        got += (size_t)n;
    }
    hdr[hdr_len] = '\0';

    size_t vocab = 0, dim = 0;
    uint64_t off0 = 0, off1 = 0;
    int prc = parse_safetensors_bf16_tensor(hdr, (size_t)hdr_len, tensor_name, &vocab, &dim, &off0, &off1);
    free(hdr);
    if (prc != 0) {
        fprintf(stderr, "Tensor %s not found or invalid in %s (rc=%d)\n", tensor_name, sf_path, prc);
        close(fd);
        return 7;
    }

    uint64_t data_base = 8ULL + hdr_len;
    size_t map_len = (size_t)st.st_size;
    void *mapped = mmap(NULL, map_len, PROT_READ, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "mmap failed: %s\n", strerror(errno));
        close(fd);
        return 8;
    }
    const uint8_t *tensor_bytes = (const uint8_t *)mapped + data_base + off0;
    size_t tensor_len = (size_t)(off1 - off0);

    float *queries = load_f32_queries(q_path, n_queries, dim);
    if (!queries) {
        fprintf(stderr, "Failed to read %zu x %zu f32 queries from %s\n", n_queries, dim, q_path);
        munmap(mapped, map_len);
        close(fd);
        return 9;
    }

    double t_setup0 = now_ms();
    int32_t *perm = NULL;
    if (perm_path) {
        perm = load_i32_perm(perm_path, dim);
        if (!perm) {
            fprintf(stderr, "Failed to load valid permutation (%zu ints) from %s\n", dim, perm_path);
            free(queries);
            munmap(mapped, map_len);
            close(fd);
            return 11;
        }
    }

    /* Fault tensor pages into RAM once during setup so timed loop has timed_ssd_read_bytes=0. */
    volatile uint64_t touch_sum = 0;
    for (size_t p = 0; p < tensor_len; p += 4096) {
        touch_sum += tensor_bytes[p];
    }
    (void)touch_sum;

    LmHeadIndex idx;
    if (build_lmhead_index((const uint16_t *)tensor_bytes, vocab, dim, perm, &idx) != 0) {
        free(perm);
        free(queries);
        munmap(mapped, map_len);
        close(fd);
        return 10;
    }
    free(perm);
    double setup_s = (now_ms() - t_setup0) / 1000.0;
    double bf16_mb = (double)tensor_len / (1024.0 * 1024.0);

    printf("\n=== BENCH [%s] vocab=%zu d=%zu bf16_mb=%.1f threads=%d setup_s=%.2f ===\n",
           label, vocab, dim, bf16_mb, threads, setup_s);
    printf("norm_range=[max=%.4f, p50=%.4f, min=%.4f] seed_rows=%zu perm=%s\n",
           idx.norm_max, idx.norm_p50, idx.norm_min, idx.seed_count, perm_path ? perm_path : "identity");

    /* Warm cache/threadpool once before measuring SSD I/O and timings. */
    CsResult warm_r;
    compute_top1_mode(&idx, queries, CS_MODE_FULL, &warm_r);

    const CsMode modes[4] = {CS_MODE_SINGLE_BAND, CS_MODE_STATIC_16BAND, CS_MODE_DYNAMIC_16BAND, CS_MODE_SUBSPACE_SIEVE};
    const char *mode_names[4] = {"single_band", "static_16band", "dynamic_16band", "subspace_sieve"};

    uint64_t io0 = read_proc_io_read_bytes();
    double sum_full_ms = 0.0;
    double sum_mode_ms[4] = {0.0, 0.0, 0.0, 0.0};
    double sum_mode_early_pct[4] = {0.0, 0.0, 0.0, 0.0};
    double sum_mode_macs_saved[4] = {0.0, 0.0, 0.0, 0.0};
    size_t mode_exact[4] = {0, 0, 0, 0};

    for (size_t qi = 0; qi < n_queries; ++qi) {
        const float *h = queries + qi * dim;
        CsResult rf;
        compute_top1_mode(&idx, h, CS_MODE_FULL, &rf);
        sum_full_ms += rf.elapsed_ms;
        printf("  tok[%zu] h_norm=%.3f full=(id=%u %.5f %.2fms)\n",
               qi, rf.h_norm, rf.best_id, rf.best_logit, rf.elapsed_ms);

        for (int m = 0; m < 4; ++m) {
            CsResult rm;
            compute_top1_mode(&idx, h, modes[m], &rm);
            int is_exact = (rm.best_id == rf.best_id) && (fabsf(rm.best_logit - rf.best_logit) < 1e-3f);
            if (is_exact) mode_exact[m]++;
            double early_pct = 100.0 * (double)rm.rows_early_terminated / (double)vocab;
            double macs_saved = 100.0 * (1.0 - (double)rm.col_macs_done / (double)rm.col_macs_total);
            double spd = rf.elapsed_ms / (rm.elapsed_ms > 1e-6 ? rm.elapsed_ms : 1e-6);
            sum_mode_ms[m] += rm.elapsed_ms;
            sum_mode_early_pct[m] += early_pct;
            sum_mode_macs_saved[m] += macs_saved;
            printf("    %-14s id=%u %.5f %.2fms speedup=%.2fx early_term=%.2f%% macs_saved=%.2f%% match=%s\n",
                   mode_names[m], rm.best_id, rm.best_logit, rm.elapsed_ms, spd,
                   early_pct, macs_saved, is_exact ? "EXACT" : "MISMATCH");
        }
    }

    uint64_t io1 = read_proc_io_read_bytes();
    uint64_t timed_ssd_read_bytes = (io1 >= io0) ? (io1 - io0) : 0;
    long vm_hwm = 0, vm_rss = 0, vm_swap = 0;
    read_proc_status_kb(&vm_hwm, &vm_rss, &vm_swap);

    double avg_full = sum_full_ms / (double)n_queries;
    for (int m = 0; m < 4; ++m) {
        double avg_m = sum_mode_ms[m] / (double)n_queries;
        double spd = avg_full / (avg_m > 1e-6 ? avg_m : 1e-6);
        printf("SUMMARY [%s | %s] full_ms_per_tok=%.2f cs_ms_per_tok=%.2f speedup=%.2fx rows_early_terminated=%.2f%% col_macs_saved=%.2f%% exact_top1=%zu/%zu timed_ssd_read_bytes=%llu vm_hwm_kb=%ld vm_rss_kb=%ld vm_swap_kb=%ld\n",
               label, mode_names[m], avg_full, avg_m, spd,
               sum_mode_early_pct[m] / (double)n_queries,
               sum_mode_macs_saved[m] / (double)n_queries,
               mode_exact[m], n_queries,
               (unsigned long long)timed_ssd_read_bytes, vm_hwm, vm_rss, vm_swap);
    }

    free_lmhead_index(&idx);
    free(queries);
    munmap(mapped, map_len);
    close(fd);
    for (int m = 0; m < 3; ++m) {
        if (mode_exact[m] != n_queries) return 20 + m;
    }
    return 0;
#endif
}
