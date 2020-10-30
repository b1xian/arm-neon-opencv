//
// Created by b1xian on 2020-10-29.
//

#include <arm_neon.h>
#include <iostream>
#include <stdlib.h>

static void u8_2_fp32(uint8_t* src, float* dst, int len) {
    int num8x16 = (len % 16 == 0) ? (len / 16) : (len / 16 + 1);

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
        // 取8个像素
        uint8x16_t uint8X16 = vld1q_u8(src + i*16);
        uint8x8_t uint8X8_low = vget_low_u8(uint8X16);

        uint16X8_low = vmovl_u8(uint8X8_low);
        uint16X4_low = vget_low_u16(uint16X8_low);
        uint32X4_low = vmovl_u16(uint16X4_low);
        float32X4_low = vcvtq_f32_u32(uint32X4_low);
        uint16X4_high = vget_high_u16(uint16X8_low);
        uint32X4_high = vmovl_u16(uint16X4_high);
        float32X4_high = vcvtq_f32_u32(uint32X4_high);

        vst1q_f32(dst+i*16, float32X4_low);
        vst1q_f32(dst+i*16+4, float32X4_high);

        uint8X8_high = vget_high_u8(uint8X16);
        uint16X8_low = vmovl_u8(uint8X8_high);
        uint16X4_low = vget_low_u16(uint16X8_low);
        uint32X4_low = vmovl_u16(uint16X4_low);
        float32X4_low = vcvtq_f32_u32(uint32X4_low);
        uint16X4_high = vget_high_u16(uint16X8_low);
        uint32X4_high = vmovl_u16(uint16X4_high);
        float32X4_high = vcvtq_f32_u32(uint32X4_high);

        vst1q_f32(dst+i*16+8, float32X4_low);
        vst1q_f32(dst+i*16+12, float32X4_high);
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