//
// Created by b1xian on 2020-10-29.
//

#include <arm_neon.h>
#include <iostream>
#include <stdlib.h>

static void u8_2_fp32(uint8_t* src, float* dst, int len) {
    int num8x16 = len / 16;

    uint8x16_t uint8X16;
    uint8x8_t uint8X8_low;
    uint8x8_t uint8X8_high;
    uint16x8_t uint16X8_low;
    uint16x4_t uint16X4_low;
    uint32x4_t uint32X4_low;
    float32x4_t float32X4_low;
    uint16x4_t uint16X4_high;
    uint32x4_t uint32X4_high;
    float32x4_t float32X4_high;

    for (int i = 0; i < num8x16; i++) {
        // 取16个像素
        uint8x16_t uint8X16 = vld1q_u8(src + i * 16);
        uint8x8_t uint8X8_low = vget_low_u8(uint8X16);

        uint16X8_low = vmovl_u8(uint8X8_low);
        uint16X4_low = vget_low_u16(uint16X8_low);
        uint32X4_low = vmovl_u16(uint16X4_low);
        float32X4_low = vcvtq_f32_u32(uint32X4_low);
        uint16X4_high = vget_high_u16(uint16X8_low);
        uint32X4_high = vmovl_u16(uint16X4_high);
        float32X4_high = vcvtq_f32_u32(uint32X4_high);

        vst1q_f32(dst + i * 16, float32X4_low);
        vst1q_f32(dst + i * 16 + 4, float32X4_high);

        uint8X8_high = vget_high_u8(uint8X16);
        uint16X8_low = vmovl_u8(uint8X8_high);
        uint16X4_low = vget_low_u16(uint16X8_low);
        uint32X4_low = vmovl_u16(uint16X4_low);
        float32X4_low = vcvtq_f32_u32(uint32X4_low);
        uint16X4_high = vget_high_u16(uint16X8_low);
        uint32X4_high = vmovl_u16(uint16X4_high);
        float32X4_high = vcvtq_f32_u32(uint32X4_high);

        vst1q_f32(dst + i * 16 + 8, float32X4_low);
        vst1q_f32(dst + i * 16 + 12, float32X4_high);
    }
}

static void fp32_2_u8(float* src, uint8_t* dst, int len) {
    int num8x16 = len / 16;

    uint8x16_t uint8X16;
    uint8x8_t uint8X8_low;
    uint8x8_t uint8X8_high;
    uint16x8_t uint16X8_low;
    uint16x4_t uint16X4_low;
    uint32x4_t uint32X4_low;
    float32x4_t float32X4_low;
    uint16x4_t uint16X4_high;
    uint32x4_t uint32X4_high;
    float32x4_t float32X4_high;

    for (int i = 0; i < num8x16; i++) {
        // 取16个像素
        float32x4_t fp32x4_0 = vld1q_f32(src + i*16);
        float32x4_t fp32x4_1 = vld1q_f32(src + i*16 + 4);
        float32x4_t fp32x4_2 = vld1q_f32(src + i*16 + 8);
        float32x4_t fp32x4_3 = vld1q_f32(src + i*16 + 12);

        // float32->uint32
        uint32x4_t u32x4_0 = vcvtq_u32_f32(fp32x4_0);
        uint32x4_t u32x4_1 = vcvtq_u32_f32(fp32x4_1);
        uint32x4_t u32x4_2 = vcvtq_u32_f32(fp32x4_2);
        uint32x4_t u32x4_3 = vcvtq_u32_f32(fp32x4_3);

        // u32->u16
        uint16x4_t u16x4_0 = vmovn_u32(u32x4_0);
        uint16x4_t u16x4_1 = vmovn_u32(u32x4_1);
        uint16x4_t u16x4_2 = vmovn_u32(u32x4_2);
        uint16x4_t u16x4_3 = vmovn_u32(u32x4_3);

        // u16 combine
        uint16x8_t u16x8_0 = vcombine_u16(u16x4_0, u16x4_1);
        uint16x8_t u16x8_1 = vcombine_u16(u16x4_2, u16x4_3);

        // u16->u8
        uint8x8_t u8x8_0 = vmovn_u16(u16x8_0);
        uint8x8_t u8x8_1 = vmovn_u16(u16x8_1);

        uint8x16_t u8x16 = vcombine_u8(u8x8_0, u8x8_1);
        vst1q_u8(dst + i * 16, u8x16);
    }
}


static void u8_2_fp16(uint8_t* src, __fp16* dst, int len) {
    int num8x16 = (len % 16 == 0) ? (len / 16) : (len / 16 + 1);

    for (int i = 0; i < num8x16; i++) {
        // 取8个像素
        uint8x16_t uint8X16 = vld1q_u8(src + i*16);

        uint8x8_t uint8X8_low = vget_low_u8(uint8X16);
        uint16x8_t uint16X8_low = vmovl_u8(uint8X8_low);
        float16x8_t float16X8_low = vcvtq_f32_u32(uint16X8_low);
        vst1q_f16(dst+i*16, float16X8_low);


        uint8x8_t uint8X8_high = vget_low_u8(uint8X16);
        uint16x8_t uint16X8_high = vmovl_u8(uint8X8_high);
        float16x8_t float16X8_high = vcvtq_f32_u32(uint16X8_high);
        vst1q_f16(dst+i*16 + 8, float16X8_high);
    }
}