//
// Created by b1xian on 2020-10-29.
//
#include <arm_neon.h>
#include <iostream>
#include <math.h>

/*
 * for bgr/rgb hwc layout
 */
static void mean_std_dev_fp32_bgr_hwc(float* src, int w, int h, int c, float* mean, float* stddev) {

    // 计算均值
    int stride = w * h;
    float32_t b_sum_f32 = 0.0f;
    float32_t g_sum_f32 = 0.0f;
    float32_t r_sum_f32 = 0.0f;

    int num_32x4 = (stride % 4 == 0) ? (stride / 4) : (stride / 4 + 1);
    for (int i = 0; i < num_32x4; i++) {
        float32x4x3_t bgr_fp32x4x3 = vld3q_f32(src + i*4*3);
        b_sum_f32 += vaddvq_f32(bgr_fp32x4x3.val[0]);
        g_sum_f32 += vaddvq_f32(bgr_fp32x4x3.val[1]);
        r_sum_f32 += vaddvq_f32(bgr_fp32x4x3.val[2]);
    }
    float b_mean = b_sum_f32/stride;
    float g_mean = g_sum_f32/stride;
    float r_mean = r_sum_f32/stride;
    mean[0] = b_mean;
    mean[1] = g_mean;
    mean[2] = r_mean;

    // 计算方差
    float32_t b_stddev_f32 = 0.0f;
    float32_t g_stddev_f32 = 0.0f;
    float32_t r_stddev_f32 = 0.0f;
    float32_t b_stddev_f32_mean = 0.0f;
    float32_t g_stddev_f32_mean = 0.0f;
    float32_t r_stddev_f32_mean = 0.0f;
    float32x4_t b_mean_float32x4 = vdupq_n_f32(b_mean);
    float32x4_t g_mean_float32x4 = vdupq_n_f32(g_mean);
    float32x4_t r_mean_float32x4 = vdupq_n_f32(r_mean);
    for (int i = 0; i < num_32x4; i++) {
        float32x4x3_t bgr_fp32x4x3 = vld3q_f32(src + i*4*3);
        // b
        float32x4_t b_float32x4 = bgr_fp32x4x3.val[0];
        // 减均值
        b_float32x4 = vsubq_f32(b_float32x4, b_mean_float32x4);
        // 平方
        b_float32x4 = vmulq_f32(b_float32x4, b_float32x4);
        // 累加后求平均
        b_stddev_f32 = vaddvq_f32(b_float32x4);
        b_stddev_f32_mean += b_stddev_f32 / stride;

        // g
        float32x4_t g_float32x4 = bgr_fp32x4x3.val[1];
        // 减均值
        g_float32x4 = vsubq_f32(g_float32x4, g_mean_float32x4);
        // 平方
        g_float32x4 = vmulq_f32(g_float32x4, g_float32x4);
        // 累加后求平均
        g_stddev_f32 = vaddvq_f32(g_float32x4);
        g_stddev_f32_mean += g_stddev_f32 / stride;

        // r
        float32x4_t r_float32x4 = bgr_fp32x4x3.val[2];
        r_float32x4 = vsubq_f32(r_float32x4, r_mean_float32x4);
        r_float32x4 = vmulq_f32(r_float32x4, r_float32x4);
        r_stddev_f32 = vaddvq_f32(r_float32x4);
        r_stddev_f32_mean += r_stddev_f32 / stride;


    }
    stddev[0] = sqrt(b_stddev_f32_mean);
    stddev[1] = sqrt(g_stddev_f32_mean);
    stddev[2] = sqrt(r_stddev_f32_mean);
}

/*
 *  for gray and bgr/rgb chw layout
 */
static void mean_std_dev_fp32_bgr_chw(float* src, int w, int h, int c, float* mean, float* stddev) {

    int stride = w * h;

    // 计算均值
    int num_32x4 = stride / 4;
    int remain = stride % 4;
    for (int k = 0; k < c; k++) {
        float channel_sum = 0.0f;
        float* channel_ofs = src + k*stride;
        int i = 0;
        for (i = 0; i < num_32x4; i++) {
            float32x4_t channel_fp32x4 = vld1q_f32(channel_ofs + i*4);
            channel_sum += vaddvq_f32(channel_fp32x4);
        }
        for (int j = 1; j <= remain; j++) {
            channel_sum += *(channel_ofs + i*4 + j);
        }
        mean[k] = channel_sum / stride;
    }

    for (int k = 0; k < c; k++) {
        float32x4_t channel_mean_float32x4 = vdupq_n_f32(mean[k]);
        float channel_stddev_f32_mean = 0.0f;
        float* channel_ofs = src + k*stride;
        int i = 0;
        for (i = 0; i < num_32x4; i++) {
            float32x4_t channel_fp32x4 = vld1q_f32(channel_ofs + i*4);
            channel_fp32x4 = vsubq_f32(channel_fp32x4, channel_mean_float32x4);
            channel_fp32x4 = vmulq_f32(channel_fp32x4, channel_fp32x4);
            float channel_stddev_f32 = vaddvq_f32(channel_fp32x4);
            channel_stddev_f32_mean += channel_stddev_f32 / stride;
        }
        if (remain > 0) {
            for (int j = 1; j <= remain; j++) {
                float pixel = *(channel_ofs + i*4 + j);
                channel_stddev_f32_mean += pow(pixel - mean[k], 2) / stride;
            }
        }
        stddev[k] = sqrt(channel_stddev_f32_mean);
    }
}


static void normalize_fp32_bgr_hwc (float* src, float* dst, int w, int h, int c, float* mean, float* stddev) {

    int stride = w * h;

    float32x4_t b_mean_float32x4 = vdupq_n_f32(mean[0]);
    float32x4_t g_mean_float32x4 = vdupq_n_f32(mean[1]);
    float32x4_t r_mean_float32x4 = vdupq_n_f32(mean[2]);
    float32x4_t b_stddev_float32x4 = vdupq_n_f32(stddev[0] + 1e-6);
    float32x4_t g_stddev_float32x4 = vdupq_n_f32(stddev[1] + 1e-6);
    float32x4_t r_stddev_float32x4 = vdupq_n_f32(stddev[2] + 1e-6);

    int num_32x4 = (stride % 4 == 0) ? (stride / 4) : (stride / 4 + 1);
    float32x4x3_t bgr_fp32x4x3;
    float32x4_t b_float32x4;
    float32x4_t g_float32x4;
    float32x4_t r_float32x4;
    for (int i = 0; i < num_32x4; i++) {
        bgr_fp32x4x3 = vld3q_f32(src + i*4*3);

        b_float32x4 = bgr_fp32x4x3.val[0];
        b_float32x4 = vsubq_f32(b_float32x4, b_mean_float32x4);
        b_float32x4 = vdivq_f32(b_float32x4, b_stddev_float32x4);

        g_float32x4 = bgr_fp32x4x3.val[1];
        g_float32x4 = vsubq_f32(g_float32x4, g_mean_float32x4);
        g_float32x4 = vdivq_f32(g_float32x4, g_stddev_float32x4);

        r_float32x4 = bgr_fp32x4x3.val[2];
        r_float32x4 = vsubq_f32(r_float32x4, r_mean_float32x4);
        r_float32x4 = vdivq_f32(r_float32x4, r_stddev_float32x4);

        bgr_fp32x4x3.val[0] = b_float32x4;
        bgr_fp32x4x3.val[1] = g_float32x4;
        bgr_fp32x4x3.val[2] = r_float32x4;
        vst3q_f32(dst+i*4*3, bgr_fp32x4x3);
    }
}


static void normalize_fp32_bgr_chw (float* src, float* dst, int w, int h, int c, float* mean, float* stddev) {

    // TODO c == len(mean) && c == len(stddev)
    int stride = w * h;

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
            channel_fp32x4 = vdivq_f32(channel_fp32x4, channel_stddev_float32x4);
            vst1q_f32(dst_ofs + i*4, channel_fp32x4);
        }

        if (remain > 0) {
            for (int j = 1; j <= remain; j++) {
                float pixel = *(channel_ofs + i*4 + j);
                pixel = (pixel - mean[k]) / (stddev[k] + 1e-6);
                *(dst_ofs + i*4 + j) = pixel;
            }
        }
    }

}