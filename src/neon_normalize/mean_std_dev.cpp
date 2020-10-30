//
// Created by b1xian on 2020-10-29.
//
#include <arm_neon.h>
#include <iostream>


static void mean_std_dev_u8_hwc(uint8_t* src, float* mean, float* std) {
    // 计算均值,方差
    // hwc计算
    uint32x4x3_t bgr_sum;
    uint32x4_t b_sum;
    uint32x4_t g_sum;
    uint32x4_t r_sum;




    // chw计算
}
