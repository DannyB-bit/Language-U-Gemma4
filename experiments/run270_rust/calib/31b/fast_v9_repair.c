/* Fast ARMv8/x86_64 OpenMP helpers for V9.0 T16C closed-form WLS repair & permutation recovery. */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

static inline uint16_t f32_to_f16_bits(float v) {
    uint32_t x;
    memcpy(&x, &v, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant = (mant | 0x800000u) >> (1 - exp);
        return (uint16_t)(sign | ((mant + 0x1000u) >> 13));
    } else if (exp >= 31) {
        return (uint16_t)(sign | 0x7BFFu);
    }
    uint32_t rounded = mant + 0x1000u;
    if (rounded & 0x800000u) {
        rounded = 0;
        exp += 1;
        if (exp >= 31) return (uint16_t)(sign | 0x7BFFu);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (rounded >> 13));
}

static inline float f16_bits_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            exp = 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FFu;
            out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &out, 4);
    return f;
}

static const float GAUSS2_LV[4] = { -3.174f, -1.000f,  1.000f,  3.174f };
static const float GAUSS3_LV[8] = { -7.430f, -5.430f, -3.270f, -1.270f,  1.270f,  3.270f,  5.430f,  7.430f };

static size_t t16c_tblk_bytes(int bsize, int n_bits, int is_asym) {
    int eff = n_bits & 0x1F;
    if (eff == 5) {
        int quintets = (bsize + 4) / 5;
        return (size_t)32 + (size_t)quintets * 16u;
    }
    size_t hdr = is_asym ? 64u : 32u;
    size_t plane_b = (size_t)(bsize / 8) * 16u;
    return hdr + (size_t)eff * plane_b;
}

/* In-place normalized 256-dim Sylvester Walsh-Hadamard transform on every 256-col block of W[rows, d_in]. */
void fwht256_rows_f32(float* W, int rows, int d_in) {
    int n_blks = d_in / 256;
    const float scale = 1.0f / 16.0f;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        float* row = W + (size_t)r * (size_t)d_in;
        for (int b = 0; b < n_blks; b++) {
            float* x = row + (size_t)b * 256u;
            for (int len = 1; len < 256; len <<= 1) {
                int step = len << 1;
                for (int i = 0; i < 256; i += step) {
                    for (int j = 0; j < len; j++) {
                        float a = x[i + j];
                        float c = x[i + j + len];
                        x[i + j] = a + c;
                        x[i + j + len] = a - c;
                    }
                }
            }
            for (int i = 0; i < 256; i++) {
                x[i] *= scale;
            }
        }
    }
}

/* Pack n_words * 64 sign bits (block-mean-centered!) from first (n_words/4) 256-dim blocks of T_rot[rows, d_in]. */
void pack_ac_bits_from_trot_f32(const float* T_rot, int rows, int d_in, int n_words, uint64_t* out_bits) {
    int blks_use = (n_words * 64 + 255) / 256;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const float* row = T_rot + (size_t)r * (size_t)d_in;
        uint64_t* dst = out_bits + (size_t)r * (size_t)n_words;
        for (int w = 0; w < n_words; w++) dst[w] = 0ULL;
        for (int b = 0; b < blks_use; b++) {
            const float* blk = row + (size_t)b * 256u;
            double s = 0.0;
            for (int j = 0; j < 256; j++) s += blk[j];
            float mean = (float)(s / 256.0);
            for (int j = 0; j < 256; j++) {
                int gbit = b * 256 + j;
                int wi = gbit >> 6;
                if (wi >= n_words) break;
                if (blk[j] >= mean) {
                    dst[wi] |= (1ULL << (gbit & 63));
                }
            }
        }
    }
}

/* Extract n_words * 64 sign bits (bit=1 iff u > 0) from packed T16C matrix (supports 1b/2b_gauss/RM/CB). */
void extract_t16c_sign_bits(
    const uint8_t* pb, int rows, int d_in, int bsize, int n_bits, int is_asym, int n_words, uint64_t* out_bits
) {
    int eff = n_bits & 0x1F;
    int is_rm = (n_bits & 0x40) != 0;
    int is_cb = (n_bits & 0x20) != 0;
    int n_tiles = (rows + 15) / 16;
    int n_blks = d_in / bsize;
    size_t hdr_b = is_asym ? 64u : 32u;
    int bpr = bsize / 8;
    size_t plane_b = (size_t)bpr * 16u;
    size_t tblk_b = t16c_tblk_bytes(bsize, n_bits, is_asym);
    int top_plane = eff - 1;
    int blks_use = (n_words * 64 + bsize - 1) / bsize;
    if (blks_use > n_blks) blks_use = n_blks;

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        int t = r / 16;
        int ro = r & 15;
        uint64_t* dst = out_bits + (size_t)r * (size_t)n_words;
        for (int w = 0; w < n_words; w++) dst[w] = 0ULL;
        for (int b = 0; b < blks_use; b++) {
            size_t t_off = is_cb
                ? ((size_t)b * (size_t)n_tiles + (size_t)t) * tblk_b
                : ((size_t)t * (size_t)n_blks + (size_t)b) * tblk_b;
            const uint8_t* pl = pb + t_off + hdr_b + (size_t)top_plane * plane_b;
            for (int bo = 0; bo < bpr; bo++) {
                uint8_t byte_v = is_rm ? pl[(size_t)ro * (size_t)bpr + (size_t)bo]
                                       : pl[(size_t)bo * 16u + (size_t)ro];
                for (int bit = 0; bit < 8; bit++) {
                    if ((byte_v >> (7 - bit)) & 1u) {
                        int gbit = b * bsize + bo * 8 + bit;
                        int wi = gbit >> 6;
                        if (wi < n_words) {
                            dst[wi] |= (1ULL << (gbit & 63));
                        }
                    }
                }
            }
        }
    }
}

/* Fast OpenMP W4A16 dequantization from packed int32 nibbles + BF16 group-32 scales -> out_f32[rows, pk_cols*8]. */
void dequant_w4a16_f32(
    const int32_t* pk,
    const uint16_t* sc_bf16,
    int rows,
    int pk_cols,
    int sc_cols,
    float* out_f32
) {
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const uint32_t* pk_r = (const uint32_t*)(pk + (size_t)r * (size_t)pk_cols);
        const uint16_t* sc_r = sc_bf16 + (size_t)r * (size_t)sc_cols;
        float* dst_r = out_f32 + (size_t)r * (size_t)(pk_cols * 8);
        for (int g = 0; g < sc_cols; g++) {
            uint32_t s_u32 = ((uint32_t)sc_r[g]) << 16;
            float sc;
            memcpy(&sc, &s_u32, 4);
            int c0 = g * 4;
            for (int k = 0; k < 4; k++) {
                uint32_t w = pk_r[c0 + k];
                float* d = dst_r + (size_t)(c0 + k) * 8u;
                d[0] = (float)((int)(w & 0xFu) - 8) * sc;
                d[1] = (float)((int)((w >> 4) & 0xFu) - 8) * sc;
                d[2] = (float)((int)((w >> 8) & 0xFu) - 8) * sc;
                d[3] = (float)((int)((w >> 12) & 0xFu) - 8) * sc;
                d[4] = (float)((int)((w >> 16) & 0xFu) - 8) * sc;
                d[5] = (float)((int)((w >> 20) & 0xFu) - 8) * sc;
                d[6] = (float)((int)((w >> 24) & 0xFu) - 8) * sc;
                d[7] = (float)((int)((w >> 28) & 0xFu) - 8) * sc;
            }
        }
    }
}

/* L2-cache-tiled exact Hamming-agreement top-2 finder over concatenated gate+up uint64 sign words [n_rows, n_words]. */
void match_mlp_perm_popcnt(
    const uint64_t* cand_bits,
    const uint64_t* target_bits,
    int n_rows,
    int n_words,
    int* out_best_j,
    int* out_best_score,
    int* out_second_score
) {
    const int total_bits = n_words * 64;
    for (int i = 0; i < n_rows; i++) {
        out_best_j[i] = 0;
        out_best_score[i] = -1;
        out_second_score[i] = -1;
    }
    const int JB = 1024;
    for (int j0 = 0; j0 < n_rows; j0 += JB) {
        int j1 = (j0 + JB <= n_rows) ? (j0 + JB) : n_rows;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n_rows; i++) {
            const uint64_t* ci = cand_bits + (size_t)i * (size_t)n_words;
            int best_j = out_best_j[i];
            int best_s = out_best_score[i];
            int sec_s = out_second_score[i];
#if defined(__aarch64__)
            if (n_words == 8) {
                const uint8_t* cb = (const uint8_t*)ci;
                uint8x16_t c0 = vld1q_u8(cb + 0);
                uint8x16_t c1 = vld1q_u8(cb + 16);
                uint8x16_t c2 = vld1q_u8(cb + 32);
                uint8x16_t c3 = vld1q_u8(cb + 48);
                for (int j = j0; j < j1; j++) {
                    const uint8_t* tb = (const uint8_t*)(target_bits + (size_t)j * 8u);
                    uint8x16_t p0 = vcntq_u8(veorq_u8(c0, vld1q_u8(tb + 0)));
                    uint8x16_t p1 = vcntq_u8(veorq_u8(c1, vld1q_u8(tb + 16)));
                    uint8x16_t p2 = vcntq_u8(veorq_u8(c2, vld1q_u8(tb + 32)));
                    uint8x16_t p3 = vcntq_u8(veorq_u8(c3, vld1q_u8(tb + 48)));
                    uint8x16_t acc = vaddq_u8(vaddq_u8(p0, p1), vaddq_u8(p2, p3));
                    int diff = (int)vaddlvq_u8(acc);
                    int s = 512 - diff;
                    if (s > best_s) {
                        sec_s = best_s;
                        best_s = s;
                        best_j = j;
                    } else if (s > sec_s) {
                        sec_s = s;
                    }
                }
            } else {
                for (int j = j0; j < j1; j++) {
                    const uint64_t* tj = target_bits + (size_t)j * (size_t)n_words;
                    int diff = 0;
                    for (int w = 0; w < n_words; w++) {
                        diff += __builtin_popcountll(ci[w] ^ tj[w]);
                    }
                    int s = total_bits - diff;
                    if (s > best_s) {
                        sec_s = best_s;
                        best_s = s;
                        best_j = j;
                    } else if (s > sec_s) {
                        sec_s = s;
                    }
                }
            }
#else
            for (int j = j0; j < j1; j++) {
                const uint64_t* tj = target_bits + (size_t)j * (size_t)n_words;
                int diff = 0;
                for (int w = 0; w < n_words; w++) {
                    diff += __builtin_popcountll(ci[w] ^ tj[w]);
                }
                int s = total_bits - diff;
                if (s > best_s) {
                    sec_s = best_s;
                    best_s = s;
                    best_j = j;
                } else if (s > sec_s) {
                    sec_s = s;
                }
            }
#endif
            out_best_j[i] = best_j;
            out_best_score[i] = best_s;
            out_second_score[i] = sec_s;
        }
    }
}

/* Solve closed-form LS FP16 (sc, bi) headers against T_rot[rows, d_in] with frozen bitplanes in-place.
 * out_stats[0..2] = {dot_sum, wq_sq_sum, wt_sq_sum}. Returns 0 on success, -1 on unsupported mode. */
int solve_and_write_wls_headers(
    uint8_t* pb,
    int rows,
    int d_in,
    int bsize,
    int n_bits,
    int is_asym,
    const float* T_rot,
    double* out_stats
) {
    int eff = n_bits & 0x1F;
    int is_g = (n_bits & 0x80) != 0;
    int is_rm = (n_bits & 0x40) != 0;
    int is_cb = (n_bits & 0x20) != 0;
    if (eff != 1 && eff != 2 && eff != 3) return -1;
    if (bsize > 256) return -1;

    int n_tiles = (rows + 15) / 16;
    int n_blks = d_in / bsize;
    size_t hdr_b = is_asym ? 64u : 32u;
    int bpr = bsize / 8;
    size_t plane_b = (size_t)bpr * 16u;
    size_t tblk_b = t16c_tblk_bytes(bsize, n_bits, is_asym);

    double dot_tot = 0.0, wq_tot = 0.0, wt_tot = 0.0;

    #pragma omp parallel for reduction(+:dot_tot,wq_tot,wt_tot) schedule(static)
    for (int t = 0; t < n_tiles; t++) {
        int r_end = ((t + 1) * 16 <= rows) ? 16 : (rows - t * 16);
        float u_blk[256];
        for (int b = 0; b < n_blks; b++) {
            size_t t_off = is_cb
                ? ((size_t)b * (size_t)n_tiles + (size_t)t) * tblk_b
                : ((size_t)t * (size_t)n_blks + (size_t)b) * tblk_b;
            uint8_t* tile_ptr = pb + t_off;

            for (int ro = 0; ro < r_end; ro++) {
                int r = t * 16 + ro;
                const float* trow = T_rot + (size_t)r * (size_t)d_in + (size_t)b * (size_t)bsize;

                for (int bo = 0; bo < bpr; bo++) {
                    uint8_t bv0 = 0, bv1 = 0, bv2 = 0;
                    if (is_rm) {
                        size_t rbo = (size_t)ro * (size_t)bpr + (size_t)bo;
                        bv0 = tile_ptr[hdr_b + rbo];
                        if (eff >= 2) bv1 = tile_ptr[hdr_b + plane_b + rbo];
                        if (eff >= 3) bv2 = tile_ptr[hdr_b + 2u * plane_b + rbo];
                    } else {
                        size_t bor = (size_t)bo * 16u + (size_t)ro;
                        bv0 = tile_ptr[hdr_b + bor];
                        if (eff >= 2) bv1 = tile_ptr[hdr_b + plane_b + bor];
                        if (eff >= 3) bv2 = tile_ptr[hdr_b + 2u * plane_b + bor];
                    }
                    for (int bit = 0; bit < 8; bit++) {
                        int sh = 7 - bit;
                        int qi = (int)((bv0 >> sh) & 1u);
                        if (eff >= 2) qi |= (int)(((bv1 >> sh) & 1u) << 1);
                        if (eff >= 3) qi |= (int)(((bv2 >> sh) & 1u) << 2);
                        float uv;
                        if (eff == 1) {
                            uv = qi ? 1.0f : -1.0f;
                        } else if (is_g && eff == 2) {
                            uv = GAUSS2_LV[qi];
                        } else if (is_g && eff == 3) {
                            uv = GAUSS3_LV[qi];
                        } else {
                            float max_l = (float)((1 << eff) - 1);
                            uv = 2.0f * (float)qi - max_l;
                        }
                        u_blk[bo * 8 + bit] = uv;
                    }
                }

                float sc = 0.0f, bi = 0.0f;
                if (is_asym) {
                    /* Exact closed-form LS for {bi + u*sc} */
                    double s_u = 0.0, s_uu = 0.0, s_t = 0.0, s_ut = 0.0;
                    for (int j = 0; j < bsize; j++) {
                        double uv = (double)u_blk[j];
                        double tv = (double)trow[j];
                        s_u += uv;
                        s_uu += uv * uv;
                        s_t += tv;
                        s_ut += uv * tv;
                    }
                    double n_f = (double)bsize;
                    double det = n_f * s_uu - s_u * s_u;
                    if (det > 1e-9) {
                        bi = (float)((s_uu * s_t - s_u * s_ut) / det);
                        sc = (float)((n_f * s_ut - s_u * s_t) / det);
                    } else {
                        bi = (float)(s_t / n_f);
                        sc = 0.0f;
                    }
                    if (sc < 0.0f) sc = 0.0f;
                    uint16_t sc16 = f32_to_f16_bits(sc);
                    uint16_t bi16 = f32_to_f16_bits(bi);
                    tile_ptr[ro * 2] = (uint8_t)(sc16 & 0xFFu);
                    tile_ptr[ro * 2 + 1] = (uint8_t)(sc16 >> 8);
                    tile_ptr[32 + ro * 2] = (uint8_t)(bi16 & 0xFFu);
                    tile_ptr[32 + ro * 2 + 1] = (uint8_t)(bi16 >> 8);
                    sc = f16_bits_to_f32(sc16);
                    bi = f16_bits_to_f32(bi16);
                } else {
                    double s_uu = 0.0, s_ut = 0.0;
                    for (int j = 0; j < bsize; j++) {
                        double uv = (double)u_blk[j];
                        s_uu += uv * uv;
                        s_ut += uv * (double)trow[j];
                    }
                    sc = (s_uu > 1e-12) ? (float)(s_ut / s_uu) : 0.0f;
                    if (sc < 0.0f) sc = 0.0f;
                    uint16_t sc16 = f32_to_f16_bits(sc);
                    tile_ptr[ro * 2] = (uint8_t)(sc16 & 0xFFu);
                    tile_ptr[ro * 2 + 1] = (uint8_t)(sc16 >> 8);
                    sc = f16_bits_to_f32(sc16);
                    bi = 0.0f;
                }

                for (int j = 0; j < bsize; j++) {
                    double wv = (double)bi + (double)sc * (double)u_blk[j];
                    double tv = (double)trow[j];
                    dot_tot += wv * tv;
                    wq_tot += wv * wv;
                    wt_tot += tv * tv;
                }
            }
        }
    }
    out_stats[0] = dot_tot;
    out_stats[1] = wq_tot;
    out_stats[2] = wt_tot;
    return 0;
}
