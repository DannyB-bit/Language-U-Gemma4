#include <stdint.h>
#include <math.h>
#include <string.h>
#include <arm_neon.h>
#include <omp.h>

static inline float round_fp16(float v) {
    __fp16 h = (__fp16)v;
    return (float)h;
}

void fwht_rows_f32(float* M, int rows, int d_in, int bsize) {
    float inv_sqrt_n = 1.0f / sqrtf((float)bsize);
    int n_blks = d_in / bsize;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        float* row = M + (size_t)r * d_in;
        for (int b = 0; b < n_blks; ++b) {
            float* x = row + b * bsize;
            for (int len = 1; len < bsize; len <<= 1) {
                int step = len << 1;
                for (int i = 0; i < bsize; i += step) {
                    if (len >= 4) {
                        for (int j = 0; j < len; j += 4) {
                            float32x4_t u = vld1q_f32(x + i + j);
                            float32x4_t v = vld1q_f32(x + i + j + len);
                            vst1q_f32(x + i + j, vaddq_f32(u, v));
                            vst1q_f32(x + i + j + len, vsubq_f32(u, v));
                        }
                    } else {
                        for (int j = 0; j < len; ++j) {
                            float u = x[i + j];
                            float v = x[i + j + len];
                            x[i + j] = u + v;
                            x[i + j + len] = u - v;
                        }
                    }
                }
            }
            float32x4_t vscale = vdupq_n_f32(inv_sqrt_n);
            for (int i = 0; i < bsize; i += 4) {
                vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), vscale));
            }
        }
    }
}

/* C[M, N] = alpha * A[M, K] * B[N, K]^T + beta * C[M, N], L1-blocked K (KB=512) & 4x4 registers */
void sgemm_nt_f32(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta) {
    #pragma omp parallel for schedule(static)
    for (int m0 = 0; m0 < M; m0 += 4) {
        int mcnt = (m0 + 4 <= M) ? 4 : (M - m0);
        for (int m = m0; m < m0 + mcnt; ++m) {
            float* cr = C + (size_t)m * N;
            if (beta == 0.0f) memset(cr, 0, (size_t)N * sizeof(float));
            else if (beta != 1.0f) for (int n = 0; n < N; ++n) cr[n] *= beta;
        }
        const int KB = 512;
        if (mcnt == 4) {
            for (int k0 = 0; k0 < K; k0 += KB) {
                int klen = (k0 + KB <= K) ? KB : (K - k0);
                const float* a0 = A + (size_t)(m0 + 0) * K + k0;
                const float* a1 = A + (size_t)(m0 + 1) * K + k0;
                const float* a2 = A + (size_t)(m0 + 2) * K + k0;
                const float* a3 = A + (size_t)(m0 + 3) * K + k0;
                int n = 0;
                for (; n + 3 < N; n += 4) {
                    const float* b0 = B + (size_t)(n + 0) * K + k0;
                    const float* b1 = B + (size_t)(n + 1) * K + k0;
                    const float* b2 = B + (size_t)(n + 2) * K + k0;
                    const float* b3 = B + (size_t)(n + 3) * K + k0;
                    float32x4_t c00 = vdupq_n_f32(0.0f), c01 = vdupq_n_f32(0.0f), c02 = vdupq_n_f32(0.0f), c03 = vdupq_n_f32(0.0f);
                    float32x4_t c10 = vdupq_n_f32(0.0f), c11 = vdupq_n_f32(0.0f), c12 = vdupq_n_f32(0.0f), c13 = vdupq_n_f32(0.0f);
                    float32x4_t c20 = vdupq_n_f32(0.0f), c21 = vdupq_n_f32(0.0f), c22 = vdupq_n_f32(0.0f), c23 = vdupq_n_f32(0.0f);
                    float32x4_t c30 = vdupq_n_f32(0.0f), c31 = vdupq_n_f32(0.0f), c32 = vdupq_n_f32(0.0f), c33 = vdupq_n_f32(0.0f);
                    int k = 0;
                    for (; k + 3 < klen; k += 4) {
                        float32x4_t va0 = vld1q_f32(a0 + k), va1 = vld1q_f32(a1 + k);
                        float32x4_t va2 = vld1q_f32(a2 + k), va3 = vld1q_f32(a3 + k);
                        float32x4_t vb0 = vld1q_f32(b0 + k);
                        c00 = vfmaq_f32(c00, va0, vb0); c10 = vfmaq_f32(c10, va1, vb0);
                        c20 = vfmaq_f32(c20, va2, vb0); c30 = vfmaq_f32(c30, va3, vb0);
                        float32x4_t vb1 = vld1q_f32(b1 + k);
                        c01 = vfmaq_f32(c01, va0, vb1); c11 = vfmaq_f32(c11, va1, vb1);
                        c21 = vfmaq_f32(c21, va2, vb1); c31 = vfmaq_f32(c31, va3, vb1);
                        float32x4_t vb2 = vld1q_f32(b2 + k);
                        c02 = vfmaq_f32(c02, va0, vb2); c12 = vfmaq_f32(c12, va1, vb2);
                        c22 = vfmaq_f32(c22, va2, vb2); c32 = vfmaq_f32(c32, va3, vb2);
                        float32x4_t vb3 = vld1q_f32(b3 + k);
                        c03 = vfmaq_f32(c03, va0, vb3); c13 = vfmaq_f32(c13, va1, vb3);
                        c23 = vfmaq_f32(c23, va2, vb3); c33 = vfmaq_f32(c33, va3, vb3);
                    }
                    float t0[4], t1[4], t2[4], t3[4];
                    vst1q_f32(t0, vpaddq_f32(vpaddq_f32(c00, c01), vpaddq_f32(c02, c03)));
                    vst1q_f32(t1, vpaddq_f32(vpaddq_f32(c10, c11), vpaddq_f32(c12, c13)));
                    vst1q_f32(t2, vpaddq_f32(vpaddq_f32(c20, c21), vpaddq_f32(c22, c23)));
                    vst1q_f32(t3, vpaddq_f32(vpaddq_f32(c30, c31), vpaddq_f32(c32, c33)));
                    for (; k < klen; ++k) {
                        float f0 = a0[k], f1 = a1[k], f2 = a2[k], f3 = a3[k];
                        float g0 = b0[k], g1 = b1[k], g2 = b2[k], g3 = b3[k];
                        t0[0]+=f0*g0; t0[1]+=f0*g1; t0[2]+=f0*g2; t0[3]+=f0*g3;
                        t1[0]+=f1*g0; t1[1]+=f1*g1; t1[2]+=f1*g2; t1[3]+=f1*g3;
                        t2[0]+=f2*g0; t2[1]+=f2*g1; t2[2]+=f2*g2; t2[3]+=f2*g3;
                        t3[0]+=f3*g0; t3[1]+=f3*g1; t3[2]+=f3*g2; t3[3]+=f3*g3;
                    }
                    float* d0 = C + (size_t)(m0+0)*N + n; float* d1 = C + (size_t)(m0+1)*N + n;
                    float* d2 = C + (size_t)(m0+2)*N + n; float* d3 = C + (size_t)(m0+3)*N + n;
                    for (int j=0; j<4; ++j) {
                        d0[j] += alpha * t0[j]; d1[j] += alpha * t1[j];
                        d2[j] += alpha * t2[j]; d3[j] += alpha * t3[j];
                    }
                }
                for (; n < N; ++n) {
                    const float* bp = B + (size_t)n * K + k0;
                    float32x4_t c0=vdupq_n_f32(0), c1=vdupq_n_f32(0), c2=vdupq_n_f32(0), c3=vdupq_n_f32(0);
                    int k = 0;
                    for (; k + 3 < klen; k += 4) {
                        float32x4_t vb = vld1q_f32(bp + k);
                        c0=vfmaq_f32(c0,vld1q_f32(a0+k),vb); c1=vfmaq_f32(c1,vld1q_f32(a1+k),vb);
                        c2=vfmaq_f32(c2,vld1q_f32(a2+k),vb); c3=vfmaq_f32(c3,vld1q_f32(a3+k),vb);
                    }
                    float s0=vaddvq_f32(c0), s1=vaddvq_f32(c1), s2=vaddvq_f32(c2), s3=vaddvq_f32(c3);
                    for (; k < klen; ++k) { float b=bp[k]; s0+=a0[k]*b; s1+=a1[k]*b; s2+=a2[k]*b; s3+=a3[k]*b; }
                    C[(size_t)(m0+0)*N+n] += alpha*s0; C[(size_t)(m0+1)*N+n] += alpha*s1;
                    C[(size_t)(m0+2)*N+n] += alpha*s2; C[(size_t)(m0+3)*N+n] += alpha*s3;
                }
            }
        } else {
            for (int m = m0; m < m0 + mcnt; ++m) {
                const float* ap = A + (size_t)m * K;
                for (int n = 0; n < N; ++n) {
                    const float* bp = B + (size_t)n * K;
                    float32x4_t acc = vdupq_n_f32(0.0f);
                    int k = 0;
                    for (; k + 3 < K; k += 4) acc = vfmaq_f32(acc, vld1q_f32(ap+k), vld1q_f32(bp+k));
                    float s = vaddvq_f32(acc);
                    for (; k < K; ++k) s += ap[k] * bp[k];
                    C[(size_t)m * N + n] += alpha * s;
                }
            }
        }
    }
}

/* C[M, N] = alpha * A[M, K] * B[K, N] + beta * C[M, N] */
void sgemm_nn_f32(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta) {
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; ++m) {
        const float* ar = A + (size_t)m * K;
        float* cr = C + (size_t)m * N;
        if (beta == 0.0f) memset(cr, 0, (size_t)N * sizeof(float));
        else if (beta != 1.0f) for (int n = 0; n < N; ++n) cr[n] *= beta;
        for (int k = 0; k < K; ++k) {
            float32x4_t va = vdupq_n_f32(alpha * ar[k]);
            const float* br = B + (size_t)k * N;
            int n = 0;
            for (; n + 7 < N; n += 8) {
                vst1q_f32(cr + n,     vfmaq_f32(vld1q_f32(cr + n),     va, vld1q_f32(br + n)));
                vst1q_f32(cr + n + 4, vfmaq_f32(vld1q_f32(cr + n + 4), va, vld1q_f32(br + n + 4)));
            }
            for (; n + 3 < N; n += 4) vst1q_f32(cr + n, vfmaq_f32(vld1q_f32(cr + n), va, vld1q_f32(br + n)));
            for (; n < N; ++n) cr[n] += alpha * ar[k] * br[n];
        }
    }
}

/* C[M, N] = alpha * A[K, M]^T * B[K, N] + beta * C[M, N]
   L1-friendly tiling: MB=32, NB=16 -> tbuf[16][32] = 2 KB (fits easily in 32KB L1D) */
void sgemm_tn_f32(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta) {
    #pragma omp parallel for schedule(static)
    for (int m0 = 0; m0 < M; m0 += 32) {
        int mcnt = (m0 + 32 <= M) ? 32 : (M - m0);
        for (int m = m0; m < m0 + mcnt; ++m) {
            float* cr = C + (size_t)m * N;
            if (beta == 0.0f) memset(cr, 0, (size_t)N * sizeof(float));
            else if (beta != 1.0f) for (int n = 0; n < N; ++n) cr[n] *= beta;
        }
        for (int n0 = 0; n0 < N; n0 += 16) {
            int ncnt = (n0 + 16 <= N) ? 16 : (N - n0);
            float tbuf[16][32];
            for (int nn = 0; nn < ncnt; ++nn) {
                for (int mm = 0; mm < mcnt; ++mm) {
                    tbuf[nn][mm] = C[(size_t)(m0 + mm) * N + (n0 + nn)];
                }
            }
            for (int k = 0; k < K; ++k) {
                const float* ar = A + (size_t)k * M + m0;
                const float* br = B + (size_t)k * N + n0;
                int mm = 0;
                for (; mm + 3 < mcnt; mm += 4) {
                    float32x4_t va = vmulq_n_f32(vld1q_f32(ar + mm), alpha);
                    for (int nn = 0; nn < ncnt; ++nn) {
                        float32x4_t vc = vld1q_f32(&tbuf[nn][mm]);
                        vst1q_f32(&tbuf[nn][mm], vfmaq_n_f32(vc, va, br[nn]));
                    }
                }
                for (; mm < mcnt; ++mm) {
                    float a = alpha * ar[mm];
                    for (int nn = 0; nn < ncnt; ++nn) tbuf[nn][mm] += a * br[nn];
                }
            }
            for (int mm = 0; mm < mcnt; ++mm) {
                float* cr = C + (size_t)(m0 + mm) * N + n0;
                for (int nn = 0; nn < ncnt; ++nn) cr[nn] = tbuf[nn][mm];
            }
        }
    }
}

/* Upper Cholesky U of Hinv = (X^T X / N + damp_frac * mean_diag * I)^{-1} -> out float32 */
int build_gptq_u_f32(const float* X, int N, int cw, float damp_frac, float* U_out, double* work_n2) {
    /* First compute H_f32 = (1/N) * X^T X using unit-stride NEON sgemm_tn_f32 into U_out (temp cw*cw buffer) */
    sgemm_tn_f32(X, X, U_out, cw, cw, N, 1.0f / (float)N, 0.0f);
    double md = 0.0;
    for (int i = 0; i < cw; ++i) md += (double)U_out[(size_t)i * cw + i];
    md /= (double)cw;
    double d_add = fmax(damp_frac * md, 1e-6);

    double* L = work_n2 + (size_t)cw * cw;
    for (int i = 0; i < cw; ++i) {
        for (int j = 0; j <= i; ++j) {
            double vij = 0.5 * ((double)U_out[(size_t)(cw - 1 - i) * cw + (cw - 1 - j)] +
                                (double)U_out[(size_t)(cw - 1 - j) * cw + (cw - 1 - i)]);
            if (i == j) vij += d_add;
            L[(size_t)i * cw + j] = vij;
        }
    }
    for (int j = 0; j < cw; ++j) {
        double s = L[(size_t)j * cw + j];
        for (int k = 0; k < j; ++k) {
            double ljk = L[(size_t)j * cw + k];
            s -= ljk * ljk;
        }
        if (s <= 1e-18) s = 1e-18;
        double diag = sqrt(s);
        L[(size_t)j * cw + j] = diag;
        double inv_d = 1.0 / diag;
        #pragma omp parallel for schedule(static) if (cw - j > 32)
        for (int i = j + 1; i < cw; ++i) {
            double sum = L[(size_t)i * cw + j];
            const double* li = L + (size_t)i * cw;
            const double* lj = L + (size_t)j * cw;
            for (int k = 0; k < j; ++k) sum -= li[k] * lj[k];
            L[(size_t)i * cw + j] = sum * inv_d;
        }
    }
    double* Linv = work_n2;
    #pragma omp parallel for schedule(dynamic, 4)
    for (int col = 0; col < cw; ++col) {
        for (int i = 0; i < col; ++i) Linv[(size_t)i * cw + col] = 0.0;
        Linv[(size_t)col * cw + col] = 1.0 / L[(size_t)col * cw + col];
        for (int i = col + 1; i < cw; ++i) {
            double sum = 0.0;
            const double* li = L + (size_t)i * cw;
            for (int k = col; k < i; ++k) sum -= li[k] * Linv[(size_t)k * cw + col];
            Linv[(size_t)i * cw + col] = sum / li[i];
        }
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < cw; ++i) {
        float* ur = U_out + (size_t)i * cw;
        for (int j = 0; j < i; ++j) ur[j] = 0.0f;
        for (int j = i; j < cw; ++j) {
            ur[j] = (float)Linv[(size_t)(cw - 1 - i) * cw + (cw - 1 - j)];
        }
    }
    return 0;
}

void quant_init_blocks_f32(
    const float* W, float* sc_out, float* bi_out,
    int rows, int d_in, int bsize, int nbits, int is_asym
) {
    int n_blks = d_in / bsize;
    float max_lvl = (float)((1 << nbits) - 1);
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        const float* row = W + (size_t)r * d_in;
        float* srow = sc_out + (size_t)r * n_blks;
        float* brow = bi_out + (size_t)r * n_blks;
        for (int b = 0; b < n_blks; ++b) {
            const float* xb = row + b * bsize;
            if (nbits == 1 && is_asym) {
                float sum = 0.0f;
                for (int i = 0; i < bsize; ++i) sum += xb[i];
                float thr = sum / (float)bsize;
                float mu0 = thr, mu1 = thr;
                for (int it = 0; it < 4; ++it) {
                    float s0 = 0.0f, s1 = 0.0f;
                    int c0 = 0, c1 = 0;
                    for (int i = 0; i < bsize; ++i) {
                        float v = xb[i];
                        if (v >= thr) { s1 += v; c1++; }
                        else { s0 += v; c0++; }
                    }
                    if (c0 > 0 && c1 > 0) {
                        mu0 = s0 / (float)c0;
                        mu1 = s1 / (float)c1;
                        thr = 0.5f * (mu0 + mu1);
                    } else {
                        mu0 = thr - 1e-6f;
                        mu1 = thr + 1e-6f;
                        break;
                    }
                }
                float bi = round_fp16(0.5f * (mu0 + mu1));
                float sc = round_fp16(fmaxf(0.5f * (mu1 - mu0), 1e-6f));
                srow[b] = sc;
                brow[b] = bi;
            } else {
                float wmin = xb[0], wmax = xb[0], sum = 0.0f, s2 = 0.0f;
                for (int i = 0; i < bsize; ++i) {
                    float v = xb[i];
                    if (v < wmin) wmin = v;
                    if (v > wmax) wmax = v;
                    sum += v;
                    s2 += v * v;
                }
                float mean = sum / (float)bsize;
                float var = fmaxf(s2 / (float)bsize - mean * mean, 1e-12f);
                float sd = sqrtf(var);
                float best_err = 1e30f;
                float best_sc = 1e-6f;
                float best_bi = 0.0f;
                if (is_asym) {
                    static const float ks[6] = {0.65f, 0.75f, 0.85f, 0.93f, 1.00f, 0.0f};
                    float mid = 0.5f * (wmin + wmax);
                    float half = fmaxf(0.5f * (wmax - wmin), 1e-6f);
                    for (int ci = 0; ci < 5; ++ci) {
                        float bi_c = round_fp16(mid);
                        float sc_c = round_fp16(fmaxf((half * ks[ci]) / max_lvl, 1e-6f));
                        float inv2s = 1.0f / (2.0f * sc_c);
                        float err = 0.0f;
                        for (int i = 0; i < bsize; ++i) {
                            float fidx = roundf((xb[i] - bi_c) * inv2s + 0.5f * max_lvl);
                            if (fidx < 0.0f) fidx = 0.0f;
                            if (fidx > max_lvl) fidx = max_lvl;
                            float d = xb[i] - (bi_c + sc_c * (2.0f * fidx - max_lvl));
                            err += d * d;
                        }
                        if (err < best_err) { best_err = err; best_sc = sc_c; best_bi = bi_c; }
                    }
                    static const float sigs[4] = {2.0f, 2.5f, 3.0f, 3.5f};
                    for (int ci = 0; ci < 4; ++ci) {
                        float bi_c = round_fp16(mean);
                        float sc_c = round_fp16(fmaxf((sd * sigs[ci]) / max_lvl, 1e-6f));
                        float inv2s = 1.0f / (2.0f * sc_c);
                        float err = 0.0f;
                        for (int i = 0; i < bsize; ++i) {
                            float fidx = roundf((xb[i] - bi_c) * inv2s + 0.5f * max_lvl);
                            if (fidx < 0.0f) fidx = 0.0f;
                            if (fidx > max_lvl) fidx = max_lvl;
                            float d = xb[i] - (bi_c + sc_c * (2.0f * fidx - max_lvl));
                            err += d * d;
                        }
                        if (err < best_err) { best_err = err; best_sc = sc_c; best_bi = bi_c; }
                    }
                } else {
                    float amax = fmaxf(fabsf(wmin), fabsf(wmax));
                    static const float ps[5] = {0.72f, 0.80f, 0.88f, 0.94f, 1.00f};
                    for (int ci = 0; ci < 5; ++ci) {
                        float sc_c = round_fp16(fmaxf((amax * ps[ci]) / max_lvl, 1e-6f));
                        float inv2s = 1.0f / (2.0f * sc_c);
                        float err = 0.0f;
                        for (int i = 0; i < bsize; ++i) {
                            float fidx = roundf(xb[i] * inv2s + 0.5f * max_lvl);
                            if (fidx < 0.0f) fidx = 0.0f;
                            if (fidx > max_lvl) fidx = max_lvl;
                            float d = xb[i] - sc_c * (2.0f * fidx - max_lvl);
                            err += d * d;
                        }
                        if (err < best_err) { best_err = err; best_sc = sc_c; best_bi = 0.0f; }
                    }
                    static const float sigs[3] = {2.5f, 3.0f, 3.5f};
                    for (int ci = 0; ci < 3; ++ci) {
                        float sc_c = round_fp16(fmaxf((sd * sigs[ci]) / max_lvl, 1e-6f));
                        float inv2s = 1.0f / (2.0f * sc_c);
                        float err = 0.0f;
                        for (int i = 0; i < bsize; ++i) {
                            float fidx = roundf(xb[i] * inv2s + 0.5f * max_lvl);
                            if (fidx < 0.0f) fidx = 0.0f;
                            if (fidx > max_lvl) fidx = max_lvl;
                            float d = xb[i] - sc_c * (2.0f * fidx - max_lvl);
                            err += d * d;
                        }
                        if (err < best_err) { best_err = err; best_sc = sc_c; best_bi = 0.0f; }
                    }
                }
                srow[b] = best_sc;
                brow[b] = best_bi;
            }
        }
    }
}

void gptq_intrablock_neon(
    float* W_sub, const float* U, const float* sc_col, const float* bi_col,
    float* E_out, uint8_t* idx_out, int rows, int hblk, int b0, int bsize, int nbits, int is_asym
) {
    float max_lvl = (float)((1 << nbits) - 1);
    int b1 = b0 + bsize;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        float* row = W_sub + (size_t)r * hblk;
        float* erow = E_out + (size_t)r * bsize;
        uint8_t* irow = idx_out + (size_t)r * bsize;
        float sc = sc_col[r];
        if (sc < 1e-6f) sc = 1e-6f;
        float bi = is_asym ? bi_col[r] : 0.0f;
        for (int k = 0; k < bsize; ++k) {
            int j = b0 + k;
            float ujj = U[(size_t)j * hblk + j];
            const float* urow = U + (size_t)j * hblk;
            float val = row[j];
            float qval;
            if (nbits == 1) {
                uint8_t b = (val >= bi) ? 1 : 0;
                irow[k] = b;
                qval = b ? (bi + sc) : (bi - sc);
            } else {
                float fidx = roundf((val - bi) / (2.0f * sc) + 0.5f * max_lvl);
                if (fidx < 0.0f) fidx = 0.0f;
                if (fidx > max_lvl) fidx = max_lvl;
                irow[k] = (uint8_t)fidx;
                qval = bi + sc * (2.0f * fidx - max_lvl);
            }
            row[j] = qval;
            float err = (val - qval) / ujj;
            erow[k] = err;
            float32x4_t verr = vdupq_n_f32(err);
            int c = j + 1;
            for (; c + 3 < b1; c += 4) {
                float32x4_t vu = vld1q_f32(urow + c);
                float32x4_t vw = vld1q_f32(row + c);
                vst1q_f32(row + c, vmlsq_f32(vw, vu, verr));
            }
            for (; c < b1; ++c) row[c] -= err * urow[c];
        }
    }
}

void trailing_update_neon(float* W_sub, const float* E, const float* U, int rows, int hblk, int b0, int bsize) {
    int b1 = b0 + bsize;
    int rem = hblk - b1;
    if (rem <= 0) return;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        float* wrow = W_sub + (size_t)r * hblk + b1;
        const float* erow = E + (size_t)r * bsize;
        int k = 0;
        for (; k + 3 < bsize; k += 4) {
            float e0 = erow[k], e1 = erow[k+1], e2 = erow[k+2], e3 = erow[k+3];
            const float* u0 = U + (size_t)(b0 + k) * hblk + b1;
            const float* u1 = U + (size_t)(b0 + k + 1) * hblk + b1;
            const float* u2 = U + (size_t)(b0 + k + 2) * hblk + b1;
            const float* u3 = U + (size_t)(b0 + k + 3) * hblk + b1;
            float32x4_t ve0 = vdupq_n_f32(e0), ve1 = vdupq_n_f32(e1), ve2 = vdupq_n_f32(e2), ve3 = vdupq_n_f32(e3);
            int j = 0;
            for (; j + 3 < rem; j += 4) {
                float32x4_t vw = vld1q_f32(wrow + j);
                vw = vmlsq_f32(vw, vld1q_f32(u0 + j), ve0);
                vw = vmlsq_f32(vw, vld1q_f32(u1 + j), ve1);
                vw = vmlsq_f32(vw, vld1q_f32(u2 + j), ve2);
                vw = vmlsq_f32(vw, vld1q_f32(u3 + j), ve3);
                vst1q_f32(wrow + j, vw);
            }
            for (; j < rem; ++j) {
                wrow[j] -= e0*u0[j] + e1*u1[j] + e2*u2[j] + e3*u3[j];
            }
        }
        for (; k < bsize; ++k) {
            float ek = erow[k];
            const float* uk = U + (size_t)(b0 + k) * hblk + b1;
            for (int j = 0; j < rem; ++j) wrow[j] -= ek * uk[j];
        }
    }
}

void dequant_blocks_f32(
    const uint8_t* idx, const float* sc, const float* bi, float* out,
    int rows, int d_in, int bsize, int nbits, int is_asym
) {
    int n_blks = d_in / bsize;
    float max_lvl = (float)((1 << nbits) - 1);
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        const uint8_t* ir = idx + (size_t)r * d_in;
        const float* sr = sc + (size_t)r * n_blks;
        const float* br = bi + (size_t)r * n_blks;
        float* orow = out + (size_t)r * d_in;
        for (int b = 0; b < n_blks; ++b) {
            float s = sr[b];
            float bv = is_asym ? br[b] : 0.0f;
            const uint8_t* ib = ir + b * bsize;
            float* ob = orow + b * bsize;
            for (int i = 0; i < bsize; ++i) {
                ob[i] = bv + s * (2.0f * (float)ib[i] - max_lvl);
            }
        }
    }
}

void unpack_qat_rows_f32(const int32_t* pk, const uint16_t* sc_bf16, float* out, int rows, int d_in) {
    int n_u32 = d_in / 8;
    int n_grp = d_in / 32;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        const uint32_t* p = (const uint32_t*)(pk + (size_t)r * n_u32);
        const uint16_t* s = sc_bf16 + (size_t)r * n_grp;
        float* o = out + (size_t)r * d_in;
        for (int g = 0; g < n_grp; ++g) {
            uint32_t sb = ((uint32_t)s[g]) << 16;
            float sc;
            memcpy(&sc, &sb, 4);
            for (int w = 0; w < 4; ++w) {
                uint32_t word = p[g * 4 + w];
                for (int i = 0; i < 8; ++i) {
                    int nib = (int)((word >> (4 * i)) & 0xFU) - 8;
                    o[g * 32 + w * 8 + i] = (float)nib * sc;
                }
            }
        }
    }
}


void pack_tile16_bytes_c(
    const float* sc, const float* bi, const uint8_t* idx, uint8_t* out_bytes,
    int d_out, int d_in, int block_size, int n_bits, int is_asym
) {
    int n_tiles = (d_out + 15) / 16;
    int n_blks = d_in / block_size;
    int bo_cnt = block_size / 8;
    int per_blk_bytes = (is_asym ? 64 : 32) + n_bits * bo_cnt * 16;

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < n_tiles; ++t) {
        int r0 = t * 16;
        uint8_t* tbase = out_bytes + (size_t)t * n_blks * per_blk_bytes;
        for (int b = 0; b < n_blks; ++b) {
            uint8_t* bptr = tbase + (size_t)b * per_blk_bytes;
            /* scales FP16: 16 entries (32 bytes) */
            uint16_t* sc16 = (uint16_t*)bptr;
            for (int rr = 0; rr < 16; ++rr) {
                int r = r0 + rr;
                __fp16 hv = (r < d_out) ? (__fp16)sc[(size_t)r * n_blks + b] : (__fp16)0.0f;
                memcpy(&sc16[rr], &hv, 2);
            }
            bptr += 32;
            if (is_asym) {
                uint16_t* bi16 = (uint16_t*)bptr;
                for (int rr = 0; rr < 16; ++rr) {
                    int r = r0 + rr;
                    __fp16 hv = (r < d_out) ? (__fp16)bi[(size_t)r * n_blks + b] : (__fp16)0.0f;
                    memcpy(&bi16[rr], &hv, 2);
                }
                bptr += 32;
            }
            for (int p = 0; p < n_bits; ++p) {
                for (int bo = 0; bo < bo_cnt; ++bo) {
                    int col0 = b * block_size + bo * 8;
                    for (int rr = 0; rr < 16; ++rr) {
                        int r = r0 + rr;
                        uint8_t byte_val = 0;
                        if (r < d_out) {
                            const uint8_t* ir = idx + (size_t)r * d_in + col0;
                            byte_val = (uint8_t)(
                                (((ir[0] >> p) & 1U) << 7) |
                                (((ir[1] >> p) & 1U) << 6) |
                                (((ir[2] >> p) & 1U) << 5) |
                                (((ir[3] >> p) & 1U) << 4) |
                                (((ir[4] >> p) & 1U) << 3) |
                                (((ir[5] >> p) & 1U) << 2) |
                                (((ir[6] >> p) & 1U) << 1) |
                                (((ir[7] >> p) & 1U) << 0)
                            );
                        }
                        *bptr++ = byte_val;
                    }
                }
            }
        }
    }
}

void sub_inplace_f32(float* A, const float* B, size_t n) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) A[i] -= B[i];
}
