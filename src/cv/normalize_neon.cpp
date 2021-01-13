#include "normalize_neon.h"

#include <math.h>

#if defined (USE_NEON) and __ARM_NEON
#include "arm_neon.h"

namespace va_cv {

/**
 * for bgr/rgb hwc
 */
void NormalizeNeon::mean_stddev_neon_hwc_bgr(float* src, int stride, float* mean, float* stddev){

    // calculate mean
    float32_t b_sum_f32 = 0.0f;
    float32_t g_sum_f32 = 0.0f;
    float32_t r_sum_f32 = 0.0f;

    int num_32x4 = stride / 4;
    int remain = stride % 4;
    int i = 0;
    for (i = 0; i < num_32x4; i++) {
        float32x4x3_t bgr_fp32x4x3 = vld3q_f32(src + i * 4 * 3);
#if __aarch64__
        b_sum_f32 += vaddvq_f32(bgr_fp32x4x3.val[0]);
        g_sum_f32 += vaddvq_f32(bgr_fp32x4x3.val[1]);
        r_sum_f32 += vaddvq_f32(bgr_fp32x4x3.val[2]);
#else
        float32x4_t b_val = bgr_fp32x4x3.val[0];
        float32x4_t g_val = bgr_fp32x4x3.val[1];
        float32x4_t r_val = bgr_fp32x4x3.val[2];
        for (int i = 0; i < 4; i++) {
            b_sum_f32 += b_val[i];
            g_sum_f32 += g_val[i];
            r_sum_f32 += r_val[i];
        }
#endif // __aarch64__
    }
    if (remain > 0) {
        for (int j = 0; j < remain; j++) {
            b_sum_f32 += *(src + (i * 4 + j) * 3);
            g_sum_f32 += *(src + (i * 4 + j) * 3 + 1);
            r_sum_f32 += *(src + (i * 4 + j) * 3 + 2);
        }
    }
    float b_mean = b_sum_f32 / stride;
    float g_mean = g_sum_f32 / stride;
    float r_mean = r_sum_f32 / stride;
    mean[0] = b_mean;
    mean[1] = g_mean;
    mean[2] = r_mean;

    // calculate standard deviation
    float32_t b_stddev_f32 = 0.0f;
    float32_t g_stddev_f32 = 0.0f;
    float32_t r_stddev_f32 = 0.0f;
    float32_t b_stddev_f32_mean = 0.0f;
    float32_t g_stddev_f32_mean = 0.0f;
    float32_t r_stddev_f32_mean = 0.0f;
    float32x4_t b_mean_float32x4 = vdupq_n_f32(b_mean);
    float32x4_t g_mean_float32x4 = vdupq_n_f32(g_mean);
    float32x4_t r_mean_float32x4 = vdupq_n_f32(r_mean);
    i = 0;
    for (i = 0; i < num_32x4; i++) {
        float32x4x3_t bgr_fp32x4x3 = vld3q_f32(src + i*4*3);
        // b
        float32x4_t b_float32x4 = bgr_fp32x4x3.val[0];
        // sub mean
        b_float32x4 = vsubq_f32(b_float32x4, b_mean_float32x4);
        // power2
        b_float32x4 = vmulq_f32(b_float32x4, b_float32x4);

        // g
        float32x4_t g_float32x4 = bgr_fp32x4x3.val[1];
        g_float32x4 = vsubq_f32(g_float32x4, g_mean_float32x4);
        g_float32x4 = vmulq_f32(g_float32x4, g_float32x4);

        // r
        float32x4_t r_float32x4 = bgr_fp32x4x3.val[2];
        r_float32x4 = vsubq_f32(r_float32x4, r_mean_float32x4);
        r_float32x4 = vmulq_f32(r_float32x4, r_float32x4);

        // mean
#if __aarch64__
        b_stddev_f32 = vaddvq_f32(b_float32x4);
        g_stddev_f32 = vaddvq_f32(g_float32x4);
        r_stddev_f32 = vaddvq_f32(r_float32x4);
#else
        for (int i = 0; i < 4; i++) {
            b_stddev_f32 += b_float32x4[i];
            g_stddev_f32 += g_float32x4[i];
            r_stddev_f32 += r_float32x4[i];
        }
#endif // __aarch64__

        b_stddev_f32_mean += b_stddev_f32 / stride;
        g_stddev_f32_mean += g_stddev_f32 / stride;
        r_stddev_f32_mean += r_stddev_f32 / stride;
    }
    if (remain > 0) {
        for (int j = 0; j < remain; j++) {
            float b_f32 = *(src + (i * 4 + j) * 3);
            b_f32 -= mean[0];
            b_f32 = b_f32 * b_f32;
            b_stddev_f32_mean += b_f32 / stride;

            float g_f32 = *(src + (i * 4 + j) * 3 + 1);
            g_f32 -= mean[1];
            g_f32 = g_f32 * g_f32;
            g_stddev_f32_mean += g_f32 / stride;

            float r_f32 = *(src + (i * 4 + j) * 3 + 2);
            r_f32 -= mean[2];
            r_f32 = r_f32 * r_f32;
            r_stddev_f32_mean += r_f32 / stride;
        }
    }
    stddev[0] = sqrt(b_stddev_f32_mean);
    stddev[1] = sqrt(g_stddev_f32_mean);
    stddev[2] = sqrt(r_stddev_f32_mean);
}

/**
 * for gray and chw
 */
void NormalizeNeon::mean_stddev_neon_chw(float* src, int stride, int c, float* mean, float* stddev){

    // calculate mean
    int num_32x4 = stride / 4;
    int remain = stride % 4;
    for (int k = 0; k < c; k++) {
        float channel_sum = 0.0f;
        float* channel_ofs = src + k * stride;
        int i = 0;
        for (i = 0; i < num_32x4; i++) {
            float32x4_t channel_fp32x4 = vld1q_f32(channel_ofs + i * 4);
#if __aarch64__
            channel_sum += vaddvq_f32(channel_fp32x4);
#else
            for (int i = 0; i < 4; i++) {
                channel_sum += channel_fp32x4[i];
            }
#endif // __aarch64__
        }
        if (remain > 0) {
            for (int j = 1; j <= remain; j++) {
                channel_sum += *(channel_ofs + i * 4 + j);
            }
        }
        mean[k] = channel_sum / stride;
    }

    // calculate standard deviation
    for (int k = 0; k < c; k++) {
        float32x4_t channel_mean_float32x4 = vdupq_n_f32(mean[k]);
        float channel_stddev_f32_mean = 0.0f;
        float* channel_ofs = src + k * stride;
        int i = 0;
        for (i = 0; i < num_32x4; i++) {
            float32x4_t channel_fp32x4 = vld1q_f32(channel_ofs + i*4);
            channel_fp32x4 = vsubq_f32(channel_fp32x4, channel_mean_float32x4);
            channel_fp32x4 = vmulq_f32(channel_fp32x4, channel_fp32x4);
            float channel_stddev_f32 = 0.0f;
#if __aarch64__
            channel_stddev_f32 = vaddvq_f32(channel_fp32x4);
#else
            for (int i = 0; i < 4; i++) {
                channel_stddev_f32 += channel_fp32x4[i];
            }
#endif // __aarch64__
            channel_stddev_f32_mean += channel_stddev_f32 / stride;
        }
        if (remain > 0) {
            for (int j = 1; j <= remain; j++) {
                float pixel = *(channel_ofs + i * 4 + j);
                channel_stddev_f32_mean += pow(pixel - mean[k], 2) / stride;
            }
        }
        stddev[k] = sqrt(channel_stddev_f32_mean);
    }
}

/**
 * for bgr/rgb hwc
 */
void NormalizeNeon::normalize_neon_hwc_bgr(float* src, float* dst, int stride, float* mean, float* stddev){

    float32x4_t b_mean_float32x4 = vdupq_n_f32(mean[0]);
    float32x4_t g_mean_float32x4 = vdupq_n_f32(mean[1]);
    float32x4_t r_mean_float32x4 = vdupq_n_f32(mean[2]);
    float32x4_t b_stddev_float32x4 = vdupq_n_f32(stddev[0] + 1e-6);
    float32x4_t g_stddev_float32x4 = vdupq_n_f32(stddev[1] + 1e-6);
    float32x4_t r_stddev_float32x4 = vdupq_n_f32(stddev[2] + 1e-6);

    int num_32x4 = stride / 4;
    int remain = stride % 4;
    float32x4x3_t bgr_fp32x4x3;
    float32x4_t b_float32x4;
    float32x4_t g_float32x4;
    float32x4_t r_float32x4;
    int i = 0;
    for (i = 0; i < num_32x4; i++) {
        bgr_fp32x4x3 = vld3q_f32(src + i * 4 * 3);

        b_float32x4 = bgr_fp32x4x3.val[0];
        b_float32x4 = vsubq_f32(b_float32x4, b_mean_float32x4);

        g_float32x4 = bgr_fp32x4x3.val[1];
        g_float32x4 = vsubq_f32(g_float32x4, g_mean_float32x4);

        r_float32x4 = bgr_fp32x4x3.val[2];
        r_float32x4 = vsubq_f32(r_float32x4, r_mean_float32x4);

#if __aarch64__
        b_float32x4 = vdivq_f32(b_float32x4, b_stddev_float32x4);
        g_float32x4 = vdivq_f32(g_float32x4, g_stddev_float32x4);
        r_float32x4 = vdivq_f32(r_float32x4, r_stddev_float32x4);
#else
        for (int i = 0; i < 4; i++) {
            b_float32x4[i] = b_float32x4[i] / (stddev[0] + 1e-6);
            g_float32x4[i] = g_float32x4[i] / (stddev[1] + 1e-6);
            r_float32x4[i] = r_float32x4[i] / (stddev[2] + 1e-6);
        }
#endif // __aarch64__

        bgr_fp32x4x3.val[0] = b_float32x4;
        bgr_fp32x4x3.val[1] = g_float32x4;
        bgr_fp32x4x3.val[2] = r_float32x4;
        vst3q_f32(dst + i * 4 * 3, bgr_fp32x4x3);
    }

    if (remain > 0) {
        float* src_ofs = src + i * 4 * 3;
        float* dst_ofs = dst + i * 4 * 3;
        for (int j = 0; j < remain; j++) {
            for (int k = 0; k < 3; k++) {
                float pixel = *(src_ofs + j * 3 + k);
                pixel = (pixel - mean[k]) / (stddev[k] + 1e-6);
                *(dst_ofs + j * 3 + k) = pixel;
            }
        }
    }
}

/**
 * for gray and chw
 */
void NormalizeNeon::normalize_neon_chw(float* src, float* dst, int stride, int c, float* mean, float* stddev){

    int num_32x4 = stride / 4;
    int remain = stride % 4;
    for (int k = 0; k < c; k++) {
        float32x4_t channel_mean_float32x4 = vdupq_n_f32(mean[k]);
        float32x4_t channel_stddev_float32x4 = vdupq_n_f32(stddev[k] + 1e-6);

        float* channel_ofs = src + k*stride;
        float* dst_ofs = dst + k*stride;

        int i = 0;
        for (i = 0; i < num_32x4; i++) {
            float32x4_t channel_fp32x4 = vld1q_f32(channel_ofs + i*4);
            channel_fp32x4 = vsubq_f32(channel_fp32x4, channel_mean_float32x4);
#if __aarch64__
            channel_fp32x4 = vdivq_f32(channel_fp32x4, channel_stddev_float32x4);
#else
            for (int i = 0; i < 4; i++) {
                channel_fp32x4[i] = channel_fp32x4[i] / (stddev[k] + 1e-6);
            }
#endif // __aarch64__

            vst1q_f32(dst_ofs + i*4, channel_fp32x4);
        }

        if (remain > 0) {
            for (int j = 0; j < remain; j++) {
                float pixel = *(channel_ofs + i*4 + j);
                pixel = (pixel - mean[k]) / (stddev[k] + 1e-6);
                *(dst_ofs + i*4 + j) = pixel;
            }
        }
    }
}

}

#endif