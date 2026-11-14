#include <arm_neon.h>
#include <omp.h>
#include <stddef.h>

void fast_sgemm_nt_f32(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                       int M, int N, int K, float alpha, float beta) {
    if (beta == 0.0f) {
        #pragma omp parallel for schedule(static)
        for (int m = 0; m < M; m++) {
            float* c = C + (size_t)m * N;
            for (int n = 0; n < N; n++) c[n] = 0.0f;
        }
    } else if (beta != 1.0f) {
        #pragma omp parallel for schedule(static)
        for (int m = 0; m < M; m++) {
            float* c = C + (size_t)m * N;
            for (int n = 0; n < N; n++) c[n] *= beta;
        }
    }
    const int MB = 64;
    const int KB = 512;
    #pragma omp parallel
    {
        for (int m0 = 0; m0 < M; m0 += MB) {
            int mend = (m0 + MB <= M) ? (m0 + MB) : M;
            for (int k0 = 0; k0 < K; k0 += KB) {
                int kend = (k0 + KB <= K) ? (k0 + KB) : K;
                #pragma omp for schedule(dynamic, 4) nowait
                for (int n0 = 0; n0 < N; n0 += 4) {
                    int nr = (n0 + 4 <= N) ? 4 : (N - n0);
                    if (nr == 4) {
                        const float* b0 = B + (size_t)(n0 + 0) * K;
                        const float* b1 = B + (size_t)(n0 + 1) * K;
                        const float* b2 = B + (size_t)(n0 + 2) * K;
                        const float* b3 = B + (size_t)(n0 + 3) * K;
                        int m = m0;
                        for (; m + 4 <= mend; m += 4) {
                            const float* a0 = A + (size_t)(m + 0) * K;
                            const float* a1 = A + (size_t)(m + 1) * K;
                            const float* a2 = A + (size_t)(m + 2) * K;
                            const float* a3 = A + (size_t)(m + 3) * K;
                            float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0), c02 = vdupq_n_f32(0), c03 = vdupq_n_f32(0);
                            float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0), c12 = vdupq_n_f32(0), c13 = vdupq_n_f32(0);
                            float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0), c22 = vdupq_n_f32(0), c23 = vdupq_n_f32(0);
                            float32x4_t c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0), c32 = vdupq_n_f32(0), c33 = vdupq_n_f32(0);
                            int k = k0;
                            for (; k + 4 <= kend; k += 4) {
                                float32x4_t w0 = vld1q_f32(b0 + k);
                                float32x4_t w1 = vld1q_f32(b1 + k);
                                float32x4_t w2 = vld1q_f32(b2 + k);
                                float32x4_t w3 = vld1q_f32(b3 + k);
                                float32x4_t x0 = vld1q_f32(a0 + k);
                                c00 = vfmaq_f32(c00, x0, w0); c01 = vfmaq_f32(c01, x0, w1);
                                c02 = vfmaq_f32(c02, x0, w2); c03 = vfmaq_f32(c03, x0, w3);
                                float32x4_t x1 = vld1q_f32(a1 + k);
                                c10 = vfmaq_f32(c10, x1, w0); c11 = vfmaq_f32(c11, x1, w1);
                                c12 = vfmaq_f32(c12, x1, w2); c13 = vfmaq_f32(c13, x1, w3);
                                float32x4_t x2 = vld1q_f32(a2 + k);
                                c20 = vfmaq_f32(c20, x2, w0); c21 = vfmaq_f32(c21, x2, w1);
                                c22 = vfmaq_f32(c22, x2, w2); c23 = vfmaq_f32(c23, x2, w3);
                                float32x4_t x3 = vld1q_f32(a3 + k);
                                c30 = vfmaq_f32(c30, x3, w0); c31 = vfmaq_f32(c31, x3, w1);
                                c32 = vfmaq_f32(c32, x3, w2); c33 = vfmaq_f32(c33, x3, w3);
                            }
                            float s[4][4] = {
                                {vaddvq_f32(c00), vaddvq_f32(c01), vaddvq_f32(c02), vaddvq_f32(c03)},
                                {vaddvq_f32(c10), vaddvq_f32(c11), vaddvq_f32(c12), vaddvq_f32(c13)},
                                {vaddvq_f32(c20), vaddvq_f32(c21), vaddvq_f32(c22), vaddvq_f32(c23)},
                                {vaddvq_f32(c30), vaddvq_f32(c31), vaddvq_f32(c32), vaddvq_f32(c33)},
                            };
                            for (; k < kend; k++) {
                                float w0 = b0[k], w1 = b1[k], w2 = b2[k], w3 = b3[k];
                                float x0 = a0[k], x1 = a1[k], x2 = a2[k], x3 = a3[k];
                                s[0][0]+=x0*w0; s[0][1]+=x0*w1; s[0][2]+=x0*w2; s[0][3]+=x0*w3;
                                s[1][0]+=x1*w0; s[1][1]+=x1*w1; s[1][2]+=x1*w2; s[1][3]+=x1*w3;
                                s[2][0]+=x2*w0; s[2][1]+=x2*w1; s[2][2]+=x2*w2; s[2][3]+=x2*w3;
                                s[3][0]+=x3*w0; s[3][1]+=x3*w1; s[3][2]+=x3*w2; s[3][3]+=x3*w3;
                            }
                            for (int r = 0; r < 4; r++) {
                                float* cr = C + (size_t)(m + r) * N + n0;
                                cr[0] += alpha * s[r][0]; cr[1] += alpha * s[r][1];
                                cr[2] += alpha * s[r][2]; cr[3] += alpha * s[r][3];
                            }
                        }
                        for (; m < mend; m++) {
                            const float* a0 = A + (size_t)m * K;
                            float32x4_t c0 = vdupq_n_f32(0), c1 = vdupq_n_f32(0), c2 = vdupq_n_f32(0), c3 = vdupq_n_f32(0);
                            int k = k0;
                            for (; k + 4 <= kend; k += 4) {
                                float32x4_t x0 = vld1q_f32(a0 + k);
                                c0 = vfmaq_f32(c0, x0, vld1q_f32(b0 + k));
                                c1 = vfmaq_f32(c1, x0, vld1q_f32(b1 + k));
                                c2 = vfmaq_f32(c2, x0, vld1q_f32(b2 + k));
                                c3 = vfmaq_f32(c3, x0, vld1q_f32(b3 + k));
                            }
                            float s0=vaddvq_f32(c0), s1=vaddvq_f32(c1), s2=vaddvq_f32(c2), s3=vaddvq_f32(c3);
                            for (; k < kend; k++) { float x=a0[k]; s0+=x*b0[k]; s1+=x*b1[k]; s2+=x*b2[k]; s3+=x*b3[k]; }
                            float* cr = C + (size_t)m * N + n0;
                            cr[0]+=alpha*s0; cr[1]+=alpha*s1; cr[2]+=alpha*s2; cr[3]+=alpha*s3;
                        }
                    } else {
                        for (int r = 0; r < nr; r++) {
                            const float* br = B + (size_t)(n0 + r) * K;
                            for (int m = m0; m < mend; m++) {
                                const float* am = A + (size_t)m * K;
                                float32x4_t acc = vdupq_n_f32(0);
                                int k = k0;
                                for (; k + 4 <= kend; k += 4) acc = vfmaq_f32(acc, vld1q_f32(am + k), vld1q_f32(br + k));
                                float s = vaddvq_f32(acc);
                                for (; k < kend; k++) s += am[k] * br[k];
                                C[(size_t)m * N + n0 + r] += alpha * s;
                            }
                        }
                    }
                }
                #pragma omp barrier
            }
        }
    }
}
