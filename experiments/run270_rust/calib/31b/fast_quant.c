#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#ifdef __ARM_NEON
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

static inline float round_fp16(float v) {
    return f16_bits_to_f32(f32_to_f16_bits(v));
}

/* Exact non-uniform Gaussian bit-plane grids matching t16c.rs::plane_weight & neon_stream_accelerator_v2.c */
static const float GAUSS2_LV[4]  = { -3.174f, -1.000f,  1.000f,  3.174f };
static const float GAUSS2_MID[3] = { -2.087f,  0.000f,  2.087f };
static const float GAUSS3_LV[8]  = { -7.430f, -5.430f, -3.270f, -1.270f,  1.270f,  3.270f,  5.430f,  7.430f };
static const float GAUSS3_MID[7] = { -6.430f, -4.350f, -2.270f,  0.000f,  2.270f,  4.350f,  6.430f };

static inline int quant_gauss2_idx(float u) {
    if (u < GAUSS2_MID[1]) return (u < GAUSS2_MID[0]) ? 0 : 1;
    return (u < GAUSS2_MID[2]) ? 2 : 3;
}

static inline int quant_gauss3_idx(float u) {
    if (u < GAUSS3_MID[3]) {
        if (u < GAUSS3_MID[1]) return (u < GAUSS3_MID[0]) ? 0 : 1;
        return (u < GAUSS3_MID[2]) ? 2 : 3;
    } else {
        if (u < GAUSS3_MID[5]) return (u < GAUSS3_MID[4]) ? 4 : 5;
        return (u < GAUSS3_MID[6]) ? 6 : 7;
    }
}

/* mode: 0=1b_asym, 1=tern, 2=2b_sym, 3=2b_asym, 4=3b_sym, 5=2b_gaus, 6=3b_gaus */
void rtn_quant_blocks_f32(float* W, int rows, int d_in, int bsize, int mode, int use_fp16) {
    int nblks = d_in / bsize;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        float* row = W + (size_t)r * (size_t)d_in;
        for (int b = 0; b < nblks; b++) {
            float* blk = row + (size_t)b * (size_t)bsize;
            if (mode == 0) {
                double sum = 0.0;
                for (int j = 0; j < bsize; j++) sum += blk[j];
                float bias = (float)(sum / (double)bsize);
                double abs_sum = 0.0;
                for (int j = 0; j < bsize; j++) abs_sum += fabs((double)blk[j] - bias);
                float sc = (float)(abs_sum / (double)bsize);
                /* 2 Lloyd-Max alternating updates for 1b_asym */
                for (int it = 0; it < 2; it++) {
                    double s_pos = 0.0, s_neg = 0.0;
                    int c_pos = 0, c_neg = 0;
                    for (int j = 0; j < bsize; j++) {
                        if (blk[j] >= bias) { s_pos += blk[j]; c_pos++; }
                        else { s_neg += blk[j]; c_neg++; }
                    }
                    float mp = c_pos ? (float)(s_pos / c_pos) : bias;
                    float mn = c_neg ? (float)(s_neg / c_neg) : bias;
                    bias = 0.5f * (mp + mn);
                    sc = 0.5f * (mp - mn);
                }
                if (use_fp16) { bias = round_fp16(bias); sc = round_fp16(sc); }
                for (int j = 0; j < bsize; j++) {
                    blk[j] = (blk[j] >= bias) ? (bias + sc) : (bias - sc);
                }
            } else if (mode == 1) {
                /* Ternary: exact threshold sweep + LS refinement */
                double abs_sum = 0.0;
                for (int j = 0; j < bsize; j++) abs_sum += fabs((double)blk[j]);
                float sc = 1.25f * (float)(abs_sum / (double)bsize);
                for (int it = 0; it < 4; it++) {
                    float tau = 0.5f * sc;
                    double num = 0.0;
                    int cnt = 0;
                    for (int j = 0; j < bsize; j++) {
                        float v = blk[j];
                        if (v > tau) { num += v; cnt++; }
                        else if (v < -tau) { num -= v; cnt++; }
                    }
                    sc = cnt > 0 ? (float)(num / (double)cnt) : 0.0f;
                }
                if (use_fp16) sc = round_fp16(sc);
                float tau = 0.5f * sc;
                for (int j = 0; j < bsize; j++) {
                    float v = blk[j];
                    blk[j] = (v > tau) ? sc : ((v < -tau) ? -sc : 0.0f);
                }
            } else if (mode == 2) {
                /* 2-bit symmetric: levels {-3, -1, +1, +3} * sc */
                double sq = 0.0;
                for (int j = 0; j < bsize; j++) sq += (double)blk[j] * (double)blk[j];
                float sc = 0.498f * (float)sqrt(sq / (double)bsize);
                for (int it = 0; it < 4; it++) {
                    double num = 0.0, den = 0.0;
                    float inv_s = sc > 1e-12f ? 1.0f / sc : 0.0f;
                    for (int j = 0; j < bsize; j++) {
                        float q = roundf((blk[j] * inv_s + 3.0f) * 0.5f);
                        if (q < 0.0f) q = 0.0f;
                        if (q > 3.0f) q = 3.0f;
                        float lv = 2.0f * q - 3.0f;
                        num += (double)lv * (double)blk[j];
                        den += (double)lv * (double)lv;
                    }
                    sc = den > 0.0 ? (float)(num / den) : 0.0f;
                }
                if (use_fp16) sc = round_fp16(sc);
                float inv_s = sc > 1e-12f ? 1.0f / sc : 0.0f;
                for (int j = 0; j < bsize; j++) {
                    float q = roundf((blk[j] * inv_s + 3.0f) * 0.5f);
                    if (q < 0.0f) q = 0.0f;
                    if (q > 3.0f) q = 3.0f;
                    blk[j] = sc * (2.0f * q - 3.0f);
                }
            } else if (mode == 3) {
                /* 2-bit asym: levels bias + {-3, -1, +1, +3} * sc */
                double sum = 0.0;
                for (int j = 0; j < bsize; j++) sum += blk[j];
                float bias = (float)(sum / (double)bsize);
                double sq = 0.0;
                for (int j = 0; j < bsize; j++) {
                    double d = (double)blk[j] - bias;
                    sq += d * d;
                }
                float sc = 0.498f * (float)sqrt(sq / (double)bsize);
                for (int it = 0; it < 4; it++) {
                    double num_s = 0.0, den_s = 0.0, sum_d = 0.0;
                    float inv_s = sc > 1e-12f ? 1.0f / sc : 0.0f;
                    for (int j = 0; j < bsize; j++) {
                        float q = roundf(((blk[j] - bias) * inv_s + 3.0f) * 0.5f);
                        if (q < 0.0f) q = 0.0f;
                        if (q > 3.0f) q = 3.0f;
                        float lv = 2.0f * q - 3.0f;
                        sum_d += (double)blk[j] - (double)sc * (double)lv;
                    }
                    bias = (float)(sum_d / (double)bsize);
                    for (int j = 0; j < bsize; j++) {
                        float q = roundf(((blk[j] - bias) * inv_s + 3.0f) * 0.5f);
                        if (q < 0.0f) q = 0.0f;
                        if (q > 3.0f) q = 3.0f;
                        float lv = 2.0f * q - 3.0f;
                        num_s += (double)lv * ((double)blk[j] - bias);
                        den_s += (double)lv * (double)lv;
                    }
                    sc = den_s > 0.0 ? (float)(num_s / den_s) : 0.0f;
                }
                if (use_fp16) { bias = round_fp16(bias); sc = round_fp16(sc); }
                float inv_s = sc > 1e-12f ? 1.0f / sc : 0.0f;
                for (int j = 0; j < bsize; j++) {
                    float q = roundf(((blk[j] - bias) * inv_s + 3.0f) * 0.5f);
                    if (q < 0.0f) q = 0.0f;
                    if (q > 3.0f) q = 3.0f;
                    blk[j] = bias + sc * (2.0f * q - 3.0f);
                }
            } else if (mode == 4) {
                /* 3-bit symmetric: levels {-7, -5, -3, -1, +1, +3, +5, +7} * sc */
                double sq = 0.0;
                for (int j = 0; j < bsize; j++) sq += (double)blk[j] * (double)blk[j];
                float sc = 0.293f * (float)sqrt(sq / (double)bsize);
                for (int it = 0; it < 4; it++) {
                    double num = 0.0, den = 0.0;
                    float inv_s = sc > 1e-12f ? 1.0f / sc : 0.0f;
                    for (int j = 0; j < bsize; j++) {
                        float q = roundf((blk[j] * inv_s + 7.0f) * 0.5f);
                        if (q < 0.0f) q = 0.0f;
                        if (q > 7.0f) q = 7.0f;
                        float lv = 2.0f * q - 7.0f;
                        num += (double)lv * (double)blk[j];
                        den += (double)lv * (double)lv;
                    }
                    sc = den > 0.0 ? (float)(num / den) : 0.0f;
                }
                if (use_fp16) sc = round_fp16(sc);
                float inv_s = sc > 1e-12f ? 1.0f / sc : 0.0f;
                for (int j = 0; j < bsize; j++) {
                    float q = roundf((blk[j] * inv_s + 7.0f) * 0.5f);
                    if (q < 0.0f) q = 0.0f;
                    if (q > 7.0f) q = 7.0f;
                    blk[j] = sc * (2.0f * q - 7.0f);
                }
            } else if (mode == 5 || mode == 6) {
                /* 2-bit / 3-bit non-uniform Gaussian symmetric (T16C_GAUSS_BIT) */
                float init_k = (mode == 5) ? 0.470f : 0.275f;
                double sq = 0.0;
                for (int j = 0; j < bsize; j++) sq += (double)blk[j] * (double)blk[j];
                float sc = init_k * (float)sqrt(sq / (double)bsize);
                for (int it = 0; it < 4; it++) {
                    double num = 0.0, den = 0.0;
                    float inv_s = sc > 1e-12f ? 1.0f / sc : 0.0f;
                    for (int j = 0; j < bsize; j++) {
                        float u = blk[j] * inv_s;
                        float lv = (mode == 5) ? GAUSS2_LV[quant_gauss2_idx(u)] : GAUSS3_LV[quant_gauss3_idx(u)];
                        num += (double)lv * (double)blk[j];
                        den += (double)lv * (double)lv;
                    }
                    sc = den > 0.0 ? (float)(num / den) : 0.0f;
                }
                if (use_fp16) sc = round_fp16(sc);
                float inv_s = sc > 1e-12f ? 1.0f / sc : 0.0f;
                for (int j = 0; j < bsize; j++) {
                    float u = blk[j] * inv_s;
                    float lv = (mode == 5) ? GAUSS2_LV[quant_gauss2_idx(u)] : GAUSS3_LV[quant_gauss3_idx(u)];
                    blk[j] = sc * lv;
                }
            }
        }
    }
}

/* Blockwise LS scale init + column-wise GPTQ on a contiguous chunk W_sub[rows, cw] given U[cw, cw] (upper Chol of H^{-1}). */
void gptq_quant_chunk_f32(
    float* W_sub, const float* U, int rows, int cw, int bsize, int mode, int use_fp16
) {
    int nblks = cw / bsize;
    float* E_blk = (float*)malloc((size_t)rows * (size_t)bsize * sizeof(float));
    if (!E_blk) return;

    for (int b = 0; b < nblks; b++) {
        int b0 = b * bsize;
        int b1 = b0 + bsize;
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; r++) {
            float* row = W_sub + (size_t)r * (size_t)cw;
            float* blk = row + b0;
            float* erow = E_blk + (size_t)r * (size_t)bsize;

            /* Step 1: Optimal block scale & bias from current error-corrected block */
            float sc = 0.0f, bias = 0.0f;
            if (mode == 0) {
                double sum = 0.0;
                for (int j = 0; j < bsize; j++) sum += blk[j];
                bias = (float)(sum / (double)bsize);
                for (int it = 0; it < 2; it++) {
                    double s0 = 0.0, s1 = 0.0;
                    int c0 = 0, c1 = 0;
                    for (int j = 0; j < bsize; j++) {
                        if (blk[j] >= bias) { s1 += blk[j]; c1++; }
                        else { s0 += blk[j]; c0++; }
                    }
                    float mp = c1 ? (float)(s1 / c1) : bias;
                    float mn = c0 ? (float)(s0 / c0) : bias;
                    bias = 0.5f * (mp + mn);
                    sc = 0.5f * (mp - mn);
                }
                if (use_fp16) { bias = round_fp16(bias); sc = round_fp16(sc); }
            } else if (mode == 1) {
                double abs_sum = 0.0;
                for (int j = 0; j < bsize; j++) abs_sum += fabs((double)blk[j]);
                sc = 1.25f * (float)(abs_sum / (double)bsize);
                for (int it = 0; it < 4; it++) {
                    float tau = 0.5f * sc;
                    double num = 0.0; int cnt = 0;
                    for (int j = 0; j < bsize; j++) {
                        float v = blk[j];
                        if (v > tau) { num += v; cnt++; }
                        else if (v < -tau) { num -= v; cnt++; }
                    }
                    sc = cnt > 0 ? (float)(num / (double)cnt) : 0.0f;
                }
                if (use_fp16) sc = round_fp16(sc);
            } else if (mode == 2 || mode == 4) {
                float max_lvl = (mode == 2) ? 3.0f : 7.0f;
                float init_k = (mode == 2) ? 0.498f : 0.293f;
                double sq = 0.0;
                for (int j = 0; j < bsize; j++) sq += (double)blk[j] * (double)blk[j];
                sc = init_k * (float)sqrt(sq / (double)bsize);
                for (int it = 0; it < 4; it++) {
                    double num = 0.0, den = 0.0;
                    float inv_s = sc > 1e-12f ? 1.0f / sc : 0.0f;
                    for (int j = 0; j < bsize; j++) {
                        float q = roundf((blk[j] * inv_s + max_lvl) * 0.5f);
                        if (q < 0.0f) q = 0.0f;
                        if (q > max_lvl) q = max_lvl;
                        float lv = 2.0f * q - max_lvl;
                        num += (double)lv * (double)blk[j];
                        den += (double)lv * (double)lv;
                    }
                    sc = den > 0.0 ? (float)(num / den) : 0.0f;
                }
                if (use_fp16) sc = round_fp16(sc);
            } else if (mode == 5 || mode == 6) {
                float init_k = (mode == 5) ? 0.470f : 0.275f;
                double sq = 0.0;
                for (int j = 0; j < bsize; j++) sq += (double)blk[j] * (double)blk[j];
                sc = init_k * (float)sqrt(sq / (double)bsize);
                for (int it = 0; it < 4; it++) {
                    double num = 0.0, den = 0.0;
                    float inv_s = sc > 1e-12f ? 1.0f / sc : 0.0f;
                    for (int j = 0; j < bsize; j++) {
                        float u = blk[j] * inv_s;
                        float lv = (mode == 5) ? GAUSS2_LV[quant_gauss2_idx(u)] : GAUSS3_LV[quant_gauss3_idx(u)];
                        num += (double)lv * (double)blk[j];
                        den += (double)lv * (double)lv;
                    }
                    sc = den > 0.0 ? (float)(num / den) : 0.0f;
                }
                if (use_fp16) sc = round_fp16(sc);
            }

            /* Step 2: Intrablock column-by-column GPTQ error diffusion */
            for (int k = 0; k < bsize; k++) {
                int j = b0 + k;
                float ujj = U[(size_t)j * (size_t)cw + (size_t)j];
                if (ujj < 1e-12f) ujj = 1e-12f;
                const float* urow = U + (size_t)j * (size_t)cw;
                float val = row[j];
                float qval;
                if (mode == 0) {
                    qval = (val >= bias) ? (bias + sc) : (bias - sc);
                } else if (mode == 1) {
                    float tau = 0.5f * sc;
                    qval = (val > tau) ? sc : ((val < -tau) ? -sc : 0.0f);
                } else if (mode == 5 || mode == 6) {
                    float inv_s = sc > 1e-12f ? 1.0f / sc : 0.0f;
                    float u = val * inv_s;
                    float lv = (mode == 5) ? GAUSS2_LV[quant_gauss2_idx(u)] : GAUSS3_LV[quant_gauss3_idx(u)];
                    qval = sc * lv;
                } else {
                    float max_lvl = (mode == 2) ? 3.0f : 7.0f;
                    float inv_s = sc > 1e-12f ? 1.0f / sc : 0.0f;
                    float q = roundf((val * inv_s + max_lvl) * 0.5f);
                    if (q < 0.0f) q = 0.0f;
                    if (q > max_lvl) q = max_lvl;
                    qval = sc * (2.0f * q - max_lvl);
                }
                row[j] = qval;
                float err = (val - qval) / ujj;
                erow[k] = err;
                for (int c = j + 1; c < b1; c++) {
                    row[c] -= err * urow[c];
                }
            }
        }

        /* Step 3: Lazy trailing update to remaining columns b1..cw */
        int rem = cw - b1;
        if (rem > 0) {
            #pragma omp parallel for schedule(static)
            for (int r = 0; r < rows; r++) {
                float* wrow = W_sub + (size_t)r * (size_t)cw + b1;
                const float* erow = E_blk + (size_t)r * (size_t)bsize;
#ifdef __ARM_NEON
                int k = 0;
                for (; k + 3 < bsize; k += 4) {
                    float32x4_t ve0 = vdupq_n_f32(erow[k]);
                    float32x4_t ve1 = vdupq_n_f32(erow[k+1]);
                    float32x4_t ve2 = vdupq_n_f32(erow[k+2]);
                    float32x4_t ve3 = vdupq_n_f32(erow[k+3]);
                    const float* u0 = U + (size_t)(b0 + k) * (size_t)cw + b1;
                    const float* u1 = U + (size_t)(b0 + k + 1) * (size_t)cw + b1;
                    const float* u2 = U + (size_t)(b0 + k + 2) * (size_t)cw + b1;
                    const float* u3 = U + (size_t)(b0 + k + 3) * (size_t)cw + b1;
                    int c = 0;
                    for (; c + 3 < rem; c += 4) {
                        float32x4_t vw = vld1q_f32(wrow + c);
                        vw = vmlsq_f32(vw, vld1q_f32(u0 + c), ve0);
                        vw = vmlsq_f32(vw, vld1q_f32(u1 + c), ve1);
                        vw = vmlsq_f32(vw, vld1q_f32(u2 + c), ve2);
                        vw = vmlsq_f32(vw, vld1q_f32(u3 + c), ve3);
                        vst1q_f32(wrow + c, vw);
                    }
                    for (; c < rem; c++) {
                        wrow[c] -= erow[k]*u0[c] + erow[k+1]*u1[c] + erow[k+2]*u2[c] + erow[k+3]*u3[c];
                    }
                }
                for (; k < bsize; k++) {
                    const float* uk = U + (size_t)(b0 + k) * (size_t)cw + b1;
                    float ek = erow[k];
                    for (int c = 0; c < rem; c++) wrow[c] -= ek * uk[c];
                }
#else
                for (int k = 0; k < bsize; k++) {
                    const float* uk = U + (size_t)(b0 + k) * (size_t)cw + b1;
                    float ek = erow[k];
                    for (int c = 0; c < rem; c++) wrow[c] -= ek * uk[c];
                }
#endif
            }
        }
    }
    free(E_blk);
}

size_t t16c_tblk_bytes(int bsize, int mode) {
    if (mode == 1) {
        int quintets = (bsize + 4) / 5;
        return (size_t)32 + (size_t)quintets * 16u;
    }
    int is_asym = (mode == 0 || mode == 3) ? 1 : 0;
    int n_bits = (mode == 0) ? 1 : ((mode == 2 || mode == 3 || mode == 5) ? 2 : 3);
    return (size_t)(is_asym ? 64 : 32) + (size_t)n_bits * (size_t)(bsize / 8) * 16u;
}

/* Quantize W_rot[rows, d_in] in-place AND write exact T16C tile16 bytes into out_bytes.
 * mode: 0=1b_asym, 1=tern_5trit(n_bits=5), 2=2b_sym, 4=3b_sym, 5=2b_gaus, 6=3b_gaus. */
void quant_and_pack_t16c_f32(float* W, int rows, int d_in, int bsize, int mode, uint8_t* out_bytes) {
    int nblks = d_in / bsize;
    int n_tiles = (rows + 15) / 16;
    size_t tblk_b = t16c_tblk_bytes(bsize, mode);
    int bpr = bsize / 8;
    int quintets = (bsize + 4) / 5;

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < n_tiles; t++) {
        int r_base = t * 16;
        for (int b = 0; b < nblks; b++) {
            uint8_t* dst = out_bytes + ((size_t)t * (size_t)nblks + (size_t)b) * tblk_b;
            memset(dst, 0, tblk_b);
            int r_end = (r_base + 16 <= rows) ? (r_base + 16) : rows;
            for (int r_off = 0; r_off < r_end - r_base; r_off++) {
                float* blk = W + ((size_t)(r_base + r_off) * (size_t)d_in) + (size_t)b * (size_t)bsize;
                if (mode == 0) {
                    double sum = 0.0;
                    for (int j = 0; j < bsize; j++) sum += blk[j];
                    float bias = (float)(sum / (double)bsize);
                    double abs_sum = 0.0;
                    for (int j = 0; j < bsize; j++) abs_sum += fabs((double)blk[j] - bias);
                    float sc = (float)(abs_sum / (double)bsize);
                    for (int it = 0; it < 2; it++) {
                        double s_pos = 0.0, s_neg = 0.0;
                        int c_pos = 0, c_neg = 0;
                        for (int j = 0; j < bsize; j++) {
                            if (blk[j] >= bias) { s_pos += blk[j]; c_pos++; }
                            else { s_neg += blk[j]; c_neg++; }
                        }
                        float mp = c_pos ? (float)(s_pos / c_pos) : bias;
                        float mn = c_neg ? (float)(s_neg / c_neg) : bias;
                        bias = 0.5f * (mp + mn);
                        sc = 0.5f * (mp - mn);
                    }
                    uint16_t sc16 = f32_to_f16_bits(sc);
                    uint16_t bi16 = f32_to_f16_bits(bias);
                    sc = f16_bits_to_f32(sc16);
                    bias = f16_bits_to_f32(bi16);
                    dst[r_off * 2] = (uint8_t)(sc16 & 0xFFu);
                    dst[r_off * 2 + 1] = (uint8_t)(sc16 >> 8);
                    dst[32 + r_off * 2] = (uint8_t)(bi16 & 0xFFu);
                    dst[32 + r_off * 2 + 1] = (uint8_t)(bi16 >> 8);
                    uint8_t* pl0 = dst + 64;
                    for (int bo = 0; bo < bpr; bo++) {
                        uint8_t byte_v = 0;
                        for (int bit = 0; bit < 8; bit++) {
                            int j = bo * 8 + bit;
                            if (blk[j] >= bias) {
                                byte_v |= (uint8_t)(1u << (7 - bit));
                                blk[j] = bias + sc;
                            } else {
                                blk[j] = bias - sc;
                            }
                        }
                        pl0[bo * 16 + r_off] = byte_v;
                    }
                } else if (mode == 1) {
                    double abs_sum = 0.0;
                    for (int j = 0; j < bsize; j++) abs_sum += fabs((double)blk[j]);
                    float sc = 1.25f * (float)(abs_sum / (double)bsize);
                    for (int it = 0; it < 4; it++) {
                        float tau = 0.5f * sc;
                        double num = 0.0;
                        int cnt = 0;
                        for (int j = 0; j < bsize; j++) {
                            float v = blk[j];
                            if (v > tau) { num += v; cnt++; }
                            else if (v < -tau) { num -= v; cnt++; }
                        }
                        sc = cnt > 0 ? (float)(num / (double)cnt) : 0.0f;
                    }
                    uint16_t sc16 = f32_to_f16_bits(sc);
                    sc = f16_bits_to_f32(sc16);
                    dst[r_off * 2] = (uint8_t)(sc16 & 0xFFu);
                    dst[r_off * 2 + 1] = (uint8_t)(sc16 >> 8);
                    float tau = 0.5f * sc;
                    uint8_t* pl = dst + 32;
                    for (int g = 0; g < quintets; g++) {
                        int code = 0;
                        int mul = 1;
                        for (int k = 0; k < 5; k++) {
                            int j = g * 5 + k;
                            int d = 1;
                            if (j < bsize) {
                                float v = blk[j];
                                if (v > tau) { d = 2; blk[j] = sc; }
                                else if (v < -tau) { d = 0; blk[j] = -sc; }
                                else { d = 1; blk[j] = 0.0f; }
                            }
                            code += d * mul;
                            mul *= 3;
                        }
                        pl[g * 16 + r_off] = (uint8_t)code;
                    }
                } else if (mode == 2 || mode == 4) {
                    int n_bits = (mode == 2) ? 2 : 3;
                    float max_lvl = (mode == 2) ? 3.0f : 7.0f;
                    float init_k = (mode == 2) ? 0.498f : 0.293f;
                    double sq = 0.0;
                    for (int j = 0; j < bsize; j++) sq += (double)blk[j] * (double)blk[j];
                    float sc = init_k * (float)sqrt(sq / (double)bsize);
                    for (int it = 0; it < 4; it++) {
                        double num = 0.0, den = 0.0;
                        float inv_s = sc > 1e-12f ? 1.0f / sc : 0.0f;
                        for (int j = 0; j < bsize; j++) {
                            float q = roundf((blk[j] * inv_s + max_lvl) * 0.5f);
                            if (q < 0.0f) q = 0.0f;
                            if (q > max_lvl) q = max_lvl;
                            float lv = 2.0f * q - max_lvl;
                            num += (double)lv * (double)blk[j];
                            den += (double)lv * (double)lv;
                        }
                        sc = den > 0.0 ? (float)(num / den) : 0.0f;
                    }
                    uint16_t sc16 = f32_to_f16_bits(sc);
                    sc = f16_bits_to_f32(sc16);
                    dst[r_off * 2] = (uint8_t)(sc16 & 0xFFu);
                    dst[r_off * 2 + 1] = (uint8_t)(sc16 >> 8);
                    float inv_s = sc > 1e-12f ? 1.0f / sc : 0.0f;
                    size_t plane_b = (size_t)bpr * 16u;
                    for (int bo = 0; bo < bpr; bo++) {
                        uint8_t bv[3] = {0, 0, 0};
                        for (int bit = 0; bit < 8; bit++) {
                            int j = bo * 8 + bit;
                            float qf = roundf((blk[j] * inv_s + max_lvl) * 0.5f);
                            int qi = (int)qf;
                            if (qi < 0) qi = 0;
                            if (qi > (int)max_lvl) qi = (int)max_lvl;
                            blk[j] = sc * (2.0f * (float)qi - max_lvl);
                            for (int p = 0; p < n_bits; p++) {
                                if ((qi >> p) & 1) bv[p] |= (uint8_t)(1u << (7 - bit));
                            }
                        }
                        for (int p = 0; p < n_bits; p++) {
                            dst[32u + (size_t)p * plane_b + (size_t)bo * 16u + (size_t)r_off] = bv[p];
                        }
                    }
                } else if (mode == 5 || mode == 6) {
                    int n_bits = (mode == 5) ? 2 : 3;
                    float init_k = (mode == 5) ? 0.470f : 0.275f;
                    double sq = 0.0;
                    for (int j = 0; j < bsize; j++) sq += (double)blk[j] * (double)blk[j];
                    float sc = init_k * (float)sqrt(sq / (double)bsize);
                    for (int it = 0; it < 4; it++) {
                        double num = 0.0, den = 0.0;
                        float inv_s = sc > 1e-12f ? 1.0f / sc : 0.0f;
                        for (int j = 0; j < bsize; j++) {
                            float u = blk[j] * inv_s;
                            float lv = (mode == 5) ? GAUSS2_LV[quant_gauss2_idx(u)] : GAUSS3_LV[quant_gauss3_idx(u)];
                            num += (double)lv * (double)blk[j];
                            den += (double)lv * (double)lv;
                        }
                        sc = den > 0.0 ? (float)(num / den) : 0.0f;
                    }
                    uint16_t sc16 = f32_to_f16_bits(sc);
                    sc = f16_bits_to_f32(sc16);
                    dst[r_off * 2] = (uint8_t)(sc16 & 0xFFu);
                    dst[r_off * 2 + 1] = (uint8_t)(sc16 >> 8);
                    float inv_s = sc > 1e-12f ? 1.0f / sc : 0.0f;
                    size_t plane_b = (size_t)bpr * 16u;
                    for (int bo = 0; bo < bpr; bo++) {
                        uint8_t bv[3] = {0, 0, 0};
                        for (int bit = 0; bit < 8; bit++) {
                            int j = bo * 8 + bit;
                            float u = blk[j] * inv_s;
                            int qi = (mode == 5) ? quant_gauss2_idx(u) : quant_gauss3_idx(u);
                            blk[j] = sc * ((mode == 5) ? GAUSS2_LV[qi] : GAUSS3_LV[qi]);
                            for (int p = 0; p < n_bits; p++) {
                                if ((qi >> p) & 1) bv[p] |= (uint8_t)(1u << (7 - bit));
                            }
                        }
                        for (int p = 0; p < n_bits; p++) {
                            dst[32u + (size_t)p * plane_b + (size_t)bo * 16u + (size_t)r_off] = bv[p];
                        }
                    }
                }
            }
        }
    }
}

/* Update FP16 block scales (and optional biases when is_asym != 0) inside an already-packed
 * T16C buffer after differentiable Block-Wise QAT without altering any quantized bit-planes. */
void update_t16c_headers_f32(
    uint8_t* packed_bytes,
    int rows,
    int d_in,
    int bsize,
    int mode,
    const float* sc_f32,
    const float* bi_f32
) {
    int nblks = d_in / bsize;
    int n_tiles = (rows + 15) / 16;
    size_t tblk_b = t16c_tblk_bytes(bsize, mode);
    int is_asym = (mode == 0 || mode == 3) ? 1 : 0;
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < n_tiles; t++) {
        int r_base = t * 16;
        int r_end = (r_base + 16 <= rows) ? (r_base + 16) : rows;
        for (int b = 0; b < nblks; b++) {
            uint8_t* dst = packed_bytes + ((size_t)t * (size_t)nblks + (size_t)b) * tblk_b;
            for (int r_off = 0; r_off < r_end - r_base; r_off++) {
                size_t idx = (size_t)(r_base + r_off) * (size_t)nblks + (size_t)b;
                uint16_t sc16 = f32_to_f16_bits(sc_f32[idx]);
                dst[r_off * 2] = (uint8_t)(sc16 & 0xFFu);
                dst[r_off * 2 + 1] = (uint8_t)(sc16 >> 8);
                if (is_asym && bi_f32 != NULL) {
                    uint16_t bi16 = f32_to_f16_bits(bi_f32[idx]);
                    dst[32 + r_off * 2] = (uint8_t)(bi16 & 0xFFu);
                    dst[32 + r_off * 2 + 1] = (uint8_t)(bi16 >> 8);
                }
            }
        }
    }
}
