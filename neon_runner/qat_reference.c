/* Offline correctness reference for compressed-tensors symmetric group-32 INT4.
 * No candidate LUT, approximate activation quantization, skipping, or codebook.
 * The decoded BF16 weights and FP32 accumulators match checkpoint_io.py's
 * independent NumPy reference up to FP32 summation order.
 */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <limits.h>

static float from_bf16(uint16_t bits) {
    uint32_t word = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &word, sizeof(value));
    return value;
}

static float rounded_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

int qat_group32_project(const uint8_t *packed, const uint8_t *scales,
                        const float *inputs, float *output,
                        int rows, int columns, int count, int threads) {
    int row;
    if (!packed || !scales || !inputs || !output || rows <= 0 || columns <= 0 ||
        columns % 32 || count <= 0 || count > 64 || threads <= 0) return 0;
#ifndef _OPENMP
    (void)threads;
#endif
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (row = 0; row < rows; row++) {
        float total[64] = {0};
        const uint8_t *codes = packed + (size_t)row * (size_t)(columns / 2);
        const uint8_t *row_scales = scales + (size_t)row * (size_t)(columns / 32) * 2;
        int group, token;
        for (group = 0; group < columns / 32; group++) {
            float weights[32];
            const uint8_t *sp = row_scales + (size_t)group * 2;
            float scale = from_bf16((uint16_t)(sp[0] | ((uint16_t)sp[1] << 8)));
            const uint8_t *cp = codes + (size_t)group * 16;
            int j;
            for (j = 0; j < 16; j++) {
                weights[2*j] = rounded_bf16((float)((int)(cp[j] & 15) - 8) * scale);
                weights[2*j+1] = rounded_bf16((float)((int)(cp[j] >> 4) - 8) * scale);
            }
            for (token = 0; token < count; token++) {
                const float *x = inputs + (size_t)token * (size_t)columns + (size_t)group * 32;
                float part = 0;
#ifdef _OPENMP
#pragma omp simd reduction(+:part)
#endif
                for (j = 0; j < 32; j++) part += weights[j] * x[j];
                total[token] += part;
            }
        }
        for (token = 0; token < count; token++) output[(size_t)token * (size_t)rows + (size_t)row] = total[token];
    }
    return 1;
}
