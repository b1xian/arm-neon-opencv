#include "normalize_naive.h"

#include <math.h>

namespace va_cv {

void NormalizeNaive::mean_stddev_naive_hwc_bgr(float* src, int stride, float* mean, float* stddev){

    float b_sum_f32 = 0.0f;
    float g_sum_f32 = 0.0f;
    float r_sum_f32 = 0.0f;

    for (int i = 0; i < stride; i++) {
        b_sum_f32 += *(src + i * 3);
        g_sum_f32 += *(src + i * 3 + 1);
        r_sum_f32 += *(src + i * 3 + 2);
    }

    mean[0] = b_sum_f32 / stride;
    mean[1] = g_sum_f32 / stride;
    mean[2] = r_sum_f32 / stride;

    float b_stddev_f32_mean = 0.0f;
    float g_stddev_f32_mean = 0.0f;
    float r_stddev_f32_mean = 0.0f;
    for (int i = 0; i < stride; i++) {
        float b_f32 = *(src + i * 3);
        // sub mean
        b_f32 -= mean[0];
        // power2
        b_f32 = b_f32 * b_f32;
        b_stddev_f32_mean += b_f32 / stride;

        float g_f32 = *(src + i * 3 + 1);
        g_f32 -= mean[1];
        g_f32 = g_f32 * g_f32;
        g_stddev_f32_mean += g_f32 / stride;

        float r_f32 = *(src + i * 3 + 2);
        r_f32 -= mean[2];
        r_f32 = r_f32 * r_f32;
        r_stddev_f32_mean += r_f32 / stride;
    }

    stddev[0] = sqrt(b_stddev_f32_mean);
    stddev[1] = sqrt(g_stddev_f32_mean);
    stddev[2] = sqrt(r_stddev_f32_mean);
}

void NormalizeNaive::mean_stddev_naive_chw(float* src, int stride, int c, float* mean, float* stddev){

    for (int k = 0; k < c; k++) {
        float channel_sum = 0.0f;
        float* channel_ofs = src + k * stride;
        for (int i = 0; i < stride; i++) {
            channel_sum += *(channel_ofs + i);
        }
        mean[k] = channel_sum / stride;
    }

    for (int k = 0; k < c; k++) {
        float channel_stddev_f32_mean = 0.0f;
        float* channel_ofs = src + k * stride;
        for (int i = 0; i < stride; i++) {
            float pixel = *(channel_ofs + i);
            pixel -= mean[k];
            pixel = pixel * pixel;
            channel_stddev_f32_mean += pixel / stride;
        }
        stddev[k] = sqrt(channel_stddev_f32_mean);
    }
}

void NormalizeNaive::normalize_naive_hwc_bgr(float* src, float* dst, int stride, float* mean, float* stddev){
    for (int i = 0; i < stride; i++) {
        *(dst + i * 3)     = (*(src + i * 3) - mean[0]) / (stddev[0] + 1e-6);
        *(dst + i * 3 + 1) = (*(src + i * 3 + 1) - mean[1]) / (stddev[1] + 1e-6);
        *(dst + i * 3 + 2) = (*(src + i * 3 + 2) - mean[2]) / (stddev[2] + 1e-6);
    }
}

void NormalizeNaive::normalize_naive_chw(float* src, float* dst, int stride, int c, float* mean, float* stddev){
    for (int k = 0; k < c; k++) {
        float* src_channel_ofs = src + k * stride;
        float* dst_channel_ofs = dst + k * stride;
        for (int i = 0; i < stride; i++) {
            *(dst_channel_ofs + i) = (*(src_channel_ofs + i) - mean[k]) / (stddev[k] + 1e-6);
        }
    }
}

}