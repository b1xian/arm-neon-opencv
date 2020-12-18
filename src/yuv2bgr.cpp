//
// Created by b1xian on 2020-10-14.
//

#include <iostream>
#include <time.h>

#include <arm_neon.h>
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>

#include "../vision/common/tensor.h"
#include "../vision/common/tensor_converter.h"


using namespace std;

void nv_to_bgr_naive(const unsigned char* src,
                       unsigned char* dst,
                       int srcw,
                       int srch,
                       int x_num,
                       int y_num) {
    std::cout << "naive" << std::endl;
    // nv21 x = 0, y = 1
    // nv12 x = 1, y = 0
    int y_h = srch;
    int wout = srcw * 3;
    const unsigned char* y = src;
    const unsigned char* vu = src + y_h * srcw;

    unsigned char* zerobuf = new unsigned char[srcw];
    unsigned char* writebuf = new unsigned char[wout];
    memset(zerobuf, 0, sizeof(uint8_t) * srcw);

    int i = 0;
#pragma omp parallel for
    for (i = 0; i < y_h; i += 2) {
        const unsigned char* ptr_y1 = y + i * srcw;
        const unsigned char* ptr_y2 = ptr_y1 + srcw;
        const unsigned char* ptr_vu = vu + (i / 2) * srcw;
        unsigned char* ptr_bgr1 = dst + i * wout;
        unsigned char* ptr_bgr2 = ptr_bgr1 + wout;
        if (i + 2 > y_h) {
            ptr_y2 = zerobuf;
            ptr_bgr2 = writebuf;
        }
        for (int j = 0; j < srcw; j += 2) {
            unsigned char _y0 = ptr_y1[0];
            unsigned char _y1 = ptr_y1[1];
            unsigned char _v = ptr_vu[x_num];
            unsigned char _u = ptr_vu[y_num];
            unsigned char _y0_1 = ptr_y2[0];
            unsigned char _y1_1 = ptr_y2[1];

            int ra = floor((179 * (_v - 128)) >> 7);
            int ga = floor((44 * (_u - 128) + 91 * (_v - 128)) >> 7);
            int ba = floor((227 * (_u - 128)) >> 7);

            int r = _y0 + ra;
            int g = _y0 - ga;
            int b = _y0 + ba;

            int r1 = _y1 + ra;
            int g1 = _y1 - ga;
            int b1 = _y1 + ba;

            r = r < 0 ? 0 : (r > 255) ? 255 : r;
            g = g < 0 ? 0 : (g > 255) ? 255 : g;
            b = b < 0 ? 0 : (b > 255) ? 255 : b;

            r1 = r1 < 0 ? 0 : (r1 > 255) ? 255 : r1;
            g1 = g1 < 0 ? 0 : (g1 > 255) ? 255 : g1;
            b1 = b1 < 0 ? 0 : (b1 > 255) ? 255 : b1;

            *ptr_bgr1++ = b;
            *ptr_bgr1++ = g;
            *ptr_bgr1++ = r;

            int r2 = _y0_1 + ra;
            int g2 = _y0_1 - ga;
            int b2 = _y0_1 + ba;

            int r3 = _y1_1 + ra;
            int g3 = _y1_1 - ga;
            int b3 = _y1_1 + ba;

            r2 = r2 < 0 ? 0 : (r2 > 255) ? 255 : r2;
            g2 = g2 < 0 ? 0 : (g2 > 255) ? 255 : g2;
            b2 = b2 < 0 ? 0 : (b2 > 255) ? 255 : b2;

            r3 = r3 < 0 ? 0 : (r3 > 255) ? 255 : r3;
            g3 = g3 < 0 ? 0 : (g3 > 255) ? 255 : g3;
            b3 = b3 < 0 ? 0 : (b3 > 255) ? 255 : b3;

            *ptr_bgr1++ = b1;
            *ptr_bgr1++ = g1;
            *ptr_bgr1++ = r1;

            *ptr_bgr2++ = b2;
            *ptr_bgr2++ = g2;
            *ptr_bgr2++ = r2;

            ptr_y1 += 2;
            ptr_y2 += 2;
            ptr_vu += 2;

            *ptr_bgr2++ = b3;
            *ptr_bgr2++ = g3;
            *ptr_bgr2++ = r3;
        }
    }
    delete[] zerobuf;
    delete[] writebuf;
}


void nv_to_bgr(const uint8_t* src,
               uint8_t* dst,
               int srcw,
               int srch,
               int x_num,
               int y_num) {
    // nv21 x = 0, y = 1
    // nv12 x = 1, y = 0
    int y_h = srch;
    int wout = srcw * 3;
    const uint8_t* y = src;
    const uint8_t* vu = src + y_h * srcw;

    int16x8_t bias = vdupq_n_s16(128);
    int16x8_t ga = vdupq_n_s16(44);
    int16x8_t ra = vdupq_n_s16(179);
    int16x8_t ba = vdupq_n_s16(227);
    int16x8_t gb = vdupq_n_s16(91);
    int16x8_t zero = vdupq_n_s16(0);
    int16x8_t max = vdupq_n_s16(255);

    uint8_t* zerobuf = new uint8_t[srcw];
    uint8_t* writebuf = new uint8_t[wout];
    memset(zerobuf, 0, sizeof(uint8_t) * srcw);

    int i = 0;
#pragma omp parallel for
    for (i = 0; i < y_h; i += 2) {
        // 每次操作两行的Y数据
        const uint8_t* ptr_y1 = y + i * srcw;
        const uint8_t* ptr_y2 = ptr_y1 + srcw;
        // 每次操作两行的UV数据
        const uint8_t* ptr_vu = vu + (i / 2) * srcw;
        // 每次输出bgr的两行
        uint8_t* ptr_bgr1 = dst + i * wout;
        uint8_t* ptr_bgr2 = ptr_bgr1 + wout;
        if (i + 2 > y_h) {
            ptr_y2 = zerobuf;
            ptr_bgr2 = writebuf;
        }
        int j = 0;
        // 每行操作的步长是16，
        for (; j < srcw - 15; j += 16) {
            // 两个uint8x8_t共保存前一行16个元素的Y数值
            uint8x8x2_t y1 = vld2_u8(ptr_y1);
            // d0 = y0y2y4y6...y14; d1 = y1y3y5...y15
            // 两个uint8x8_t共保存后两行共32个元素的UV数值
            uint8x8x2_t vu = vld2_u8(ptr_vu);
            // d0 = v0v1v2v3v4v5...v7 d1 = u0u1u2...u7

            // 两个uint8x8_t共保存后一行16个元素的Y数值
            uint8x8x2_t y2 = vld2_u8(ptr_y2);

            uint16x8_t v = vmovl_u8(vu.val[x_num]);
            uint16x8_t u = vmovl_u8(vu.val[y_num]);
            int16x8_t v_s = vreinterpretq_s16_u16(v);
            int16x8_t u_s = vreinterpretq_s16_u16(u);
            // UV 减去 128
            int16x8_t v_bias = vsubq_s16(v_s, bias);
            int16x8_t u_bias = vsubq_s16(u_s, bias);

            // G = Y - 0.34414*(U-128) - 0.71414*(V-128);
            int16x8_t g0 = vmulq_s16(ga, u_bias);
            // R = Y + 1.402*(V-128);
            int16x8_t r0 = vmulq_s16(ra, v_bias);
            // B = Y + 1.772*(U-128);
            int16x8_t b0 = vmulq_s16(ba, u_bias);

            g0 = vmlaq_s16(g0, gb, v_bias);

            int16x8_t y1_0_8 = vreinterpretq_s16_u16(vmovl_u8(y1.val[0]));
            int16x8_t y1_1_8 = vreinterpretq_s16_u16(vmovl_u8(y1.val[1]));

            int16x8_t y2_0_8 = vreinterpretq_s16_u16(vmovl_u8(y2.val[0]));
            int16x8_t y2_1_8 = vreinterpretq_s16_u16(vmovl_u8(y2.val[1]));

            int16x8_t r0_bias = vshrq_n_s16(r0, 7);  // r0 / 128
            int16x8_t b0_bias = vshrq_n_s16(b0, 7);
            int16x8_t g0_bias = vshrq_n_s16(g0, 7);

            int16x8_t r0_1 = vaddq_s16(y1_0_8, r0_bias);
            int16x8_t b0_1 = vaddq_s16(y1_0_8, b0_bias);
            int16x8_t g0_1 = vsubq_s16(y1_0_8, g0_bias);  // g0_1 = y1_0_8 - g0_1

            int16x8_t r0_2 = vaddq_s16(y1_1_8, r0_bias);
            int16x8_t b0_2 = vaddq_s16(y1_1_8, b0_bias);
            int16x8_t g0_2 = vsubq_s16(y1_1_8, g0_bias);

            r0_1 = vmaxq_s16(r0_1, zero);
            b0_1 = vmaxq_s16(b0_1, zero);
            g0_1 = vmaxq_s16(g0_1, zero);

            r0_2 = vmaxq_s16(r0_2, zero);
            b0_2 = vmaxq_s16(b0_2, zero);
            g0_2 = vmaxq_s16(g0_2, zero);

            r0_1 = vminq_s16(r0_1, max);
            b0_1 = vminq_s16(b0_1, max);
            g0_1 = vminq_s16(g0_1, max);

            r0_2 = vminq_s16(r0_2, max);
            b0_2 = vminq_s16(b0_2, max);
            g0_2 = vminq_s16(g0_2, max);

            uint8x8_t r00 = vreinterpret_u8_s8(vmovn_s16(r0_1));
            uint8x8_t b00 = vreinterpret_u8_s8(vmovn_s16(b0_1));
            uint8x8_t g00 = vreinterpret_u8_s8(vmovn_s16(g0_1));

            uint8x8_t r01 = vreinterpret_u8_s8(vmovn_s16(r0_2));
            uint8x8_t b01 = vreinterpret_u8_s8(vmovn_s16(b0_2));
            uint8x8_t g01 = vreinterpret_u8_s8(vmovn_s16(g0_2));

            // ------
            int16x8_t r1_1 = vaddq_s16(y2_0_8, r0_bias);
            int16x8_t b1_1 = vaddq_s16(y2_0_8, b0_bias);
            int16x8_t g1_1 = vsubq_s16(y2_0_8, g0_bias);  // g0_1 = y1_0_8 - g0_1

            int16x8_t r1_2 = vaddq_s16(y2_1_8, r0_bias);
            int16x8_t b1_2 = vaddq_s16(y2_1_8, b0_bias);
            int16x8_t g1_2 = vsubq_s16(y2_1_8, g0_bias);

            uint8x8x2_t r00_0 = vtrn_u8(r00, r01);  // 014589  236710
            uint8x8x2_t b00_0 = vtrn_u8(b00, b01);
            uint8x8x2_t g00_0 = vtrn_u8(g00, g01);

            r1_1 = vmaxq_s16(r1_1, zero);
            b1_1 = vmaxq_s16(b1_1, zero);
            g1_1 = vmaxq_s16(g1_1, zero);

            r1_2 = vmaxq_s16(r1_2, zero);
            b1_2 = vmaxq_s16(b1_2, zero);
            g1_2 = vmaxq_s16(g1_2, zero);

            uint16x4_t r0_16 = vreinterpret_u16_u8(r00_0.val[0]);
            uint16x4_t r1_16 = vreinterpret_u16_u8(r00_0.val[1]);

            uint16x4_t b0_16 = vreinterpret_u16_u8(b00_0.val[0]);
            uint16x4_t b1_16 = vreinterpret_u16_u8(b00_0.val[1]);

            uint16x4_t g0_16 = vreinterpret_u16_u8(g00_0.val[0]);
            uint16x4_t g1_16 = vreinterpret_u16_u8(g00_0.val[1]);

            uint16x4x2_t r00_1 = vtrn_u16(r0_16, r1_16);  // 012389 456710
            uint16x4x2_t b00_1 = vtrn_u16(b0_16, b1_16);
            uint16x4x2_t g00_1 = vtrn_u16(g0_16, g1_16);

            r1_1 = vminq_s16(r1_1, max);
            b1_1 = vminq_s16(b1_1, max);
            g1_1 = vminq_s16(g1_1, max);

            r1_2 = vminq_s16(r1_2, max);
            b1_2 = vminq_s16(b1_2, max);
            g1_2 = vminq_s16(g1_2, max);

            uint32x2_t r0_32 = vreinterpret_u32_u16(r00_1.val[0]);
            uint32x2_t r1_32 = vreinterpret_u32_u16(r00_1.val[1]);

            uint32x2_t b0_32 = vreinterpret_u32_u16(b00_1.val[0]);
            uint32x2_t b1_32 = vreinterpret_u32_u16(b00_1.val[1]);

            uint32x2_t g0_32 = vreinterpret_u32_u16(g00_1.val[0]);
            uint32x2_t g1_32 = vreinterpret_u32_u16(g00_1.val[1]);

            uint32x2x2_t r00_2 = vtrn_u32(r0_32, r1_32);  // 01234567 8910
            uint32x2x2_t b00_2 = vtrn_u32(b0_32, b1_32);
            uint32x2x2_t g00_2 = vtrn_u32(g0_32, g1_32);

            r00 = vreinterpret_u8_s8(vmovn_s16(r1_1));
            b00 = vreinterpret_u8_s8(vmovn_s16(b1_1));
            g00 = vreinterpret_u8_s8(vmovn_s16(g1_1));

            r01 = vreinterpret_u8_s8(vmovn_s16(r1_2));
            b01 = vreinterpret_u8_s8(vmovn_s16(b1_2));
            g01 = vreinterpret_u8_s8(vmovn_s16(g1_2));

            uint8x8_t r0_8 = vreinterpret_u8_u32(r00_2.val[0]);
            uint8x8_t b0_8 = vreinterpret_u8_u32(b00_2.val[0]);
            uint8x8_t g0_8 = vreinterpret_u8_u32(g00_2.val[0]);

            uint8x8_t r1_8 = vreinterpret_u8_u32(r00_2.val[1]);
            uint8x8_t b1_8 = vreinterpret_u8_u32(b00_2.val[1]);
            uint8x8_t g1_8 = vreinterpret_u8_u32(g00_2.val[1]);

            uint8x8x3_t v_bgr;
            v_bgr.val[0] = b0_8;
            v_bgr.val[1] = g0_8;
            v_bgr.val[2] = r0_8;

            r00_0 = vtrn_u8(r00, r01);  // 014589  236710
            b00_0 = vtrn_u8(b00, b01);
            g00_0 = vtrn_u8(g00, g01);

            vst3_u8(ptr_bgr1, v_bgr);

            r0_16 = vreinterpret_u16_u8(r00_0.val[0]);
            r1_16 = vreinterpret_u16_u8(r00_0.val[1]);

            b0_16 = vreinterpret_u16_u8(b00_0.val[0]);
            b1_16 = vreinterpret_u16_u8(b00_0.val[1]);

            g0_16 = vreinterpret_u16_u8(g00_0.val[0]);
            g1_16 = vreinterpret_u16_u8(g00_0.val[1]);

            ptr_bgr1 += 24;
            uint8x8x3_t v_bgr1;
            v_bgr1.val[0] = b1_8;
            v_bgr1.val[1] = g1_8;
            v_bgr1.val[2] = r1_8;

            r00_1 = vtrn_u16(r0_16, r1_16);  // 012389 456710
            b00_1 = vtrn_u16(b0_16, b1_16);
            g00_1 = vtrn_u16(g0_16, g1_16);

            vst3_u8(ptr_bgr1, v_bgr1);

            r0_32 = vreinterpret_u32_u16(r00_1.val[0]);
            r1_32 = vreinterpret_u32_u16(r00_1.val[1]);

            b0_32 = vreinterpret_u32_u16(b00_1.val[0]);
            b1_32 = vreinterpret_u32_u16(b00_1.val[1]);

            g0_32 = vreinterpret_u32_u16(g00_1.val[0]);
            g1_32 = vreinterpret_u32_u16(g00_1.val[1]);

            ptr_bgr1 += 24;

            r00_2 = vtrn_u32(r0_32, r1_32);  // 01234567 8910
            b00_2 = vtrn_u32(b0_32, b1_32);
            g00_2 = vtrn_u32(g0_32, g1_32);

            ptr_vu += 16;
            ptr_y1 += 16;
            ptr_y2 += 16;

            r0_8 = vreinterpret_u8_u32(r00_2.val[0]);
            b0_8 = vreinterpret_u8_u32(b00_2.val[0]);
            g0_8 = vreinterpret_u8_u32(g00_2.val[0]);

            r1_8 = vreinterpret_u8_u32(r00_2.val[1]);
            b1_8 = vreinterpret_u8_u32(b00_2.val[1]);
            g1_8 = vreinterpret_u8_u32(g00_2.val[1]);

            v_bgr.val[0] = b0_8;
            v_bgr.val[1] = g0_8;
            v_bgr.val[2] = r0_8;

            v_bgr1.val[0] = b1_8;
            v_bgr1.val[1] = g1_8;
            v_bgr1.val[2] = r1_8;

            vst3_u8(ptr_bgr2, v_bgr);
            vst3_u8(ptr_bgr2 + 24, v_bgr1);

            ptr_bgr2 += 48;
        }
        // two data  每行处理不完的数据，不用neon处理
        for (; j < srcw; j += 2) {
            uint8_t _y0 = ptr_y1[0];
            uint8_t _y1 = ptr_y1[1];
            uint8_t _v = ptr_vu[x_num];
            uint8_t _u = ptr_vu[y_num];
            uint8_t _y0_1 = ptr_y2[0];
            uint8_t _y1_1 = ptr_y2[1];

            int ra = floor((179 * (_v - 128)) >> 7);
            int ga = floor((44 * (_u - 128) + 91 * (_v - 128)) >> 7);
            int ba = floor((227 * (_u - 128)) >> 7);

            int r = _y0 + ra;
            int g = _y0 - ga;
            int b = _y0 + ba;

            int r1 = _y1 + ra;
            int g1 = _y1 - ga;
            int b1 = _y1 + ba;

            r = r < 0 ? 0 : (r > 255) ? 255 : r;
            g = g < 0 ? 0 : (g > 255) ? 255 : g;
            b = b < 0 ? 0 : (b > 255) ? 255 : b;

            r1 = r1 < 0 ? 0 : (r1 > 255) ? 255 : r1;
            g1 = g1 < 0 ? 0 : (g1 > 255) ? 255 : g1;
            b1 = b1 < 0 ? 0 : (b1 > 255) ? 255 : b1;

            *ptr_bgr1++ = b;
            *ptr_bgr1++ = g;
            *ptr_bgr1++ = r;

            int r2 = _y0_1 + ra;
            int g2 = _y0_1 - ga;
            int b2 = _y0_1 + ba;

            int r3 = _y1_1 + ra;
            int g3 = _y1_1 - ga;
            int b3 = _y1_1 + ba;

            r2 = r2 < 0 ? 0 : (r2 > 255) ? 255 : r2;
            g2 = g2 < 0 ? 0 : (g2 > 255) ? 255 : g2;
            b2 = b2 < 0 ? 0 : (b2 > 255) ? 255 : b2;

            r3 = r3 < 0 ? 0 : (r3 > 255) ? 255 : r3;
            g3 = g3 < 0 ? 0 : (g3 > 255) ? 255 : g3;
            b3 = b3 < 0 ? 0 : (b3 > 255) ? 255 : b3;

            *ptr_bgr1++ = b1;
            *ptr_bgr1++ = g1;
            *ptr_bgr1++ = r1;

            *ptr_bgr2++ = b2;
            *ptr_bgr2++ = g2;
            *ptr_bgr2++ = r2;

            ptr_y1 += 2;
            ptr_y2 += 2;
            ptr_vu += 2;

            *ptr_bgr2++ = b3;
            *ptr_bgr2++ = g3;
            *ptr_bgr2++ = r3;
        }
    }
    delete[] zerobuf;
    delete[] writebuf;
}


const unsigned int R2YI = 4899;
const unsigned int G2YI = 9617;
const unsigned int B2YI = 1868;
const unsigned int B2UI = 9241;
const unsigned int R2VI = 11682;
const unsigned char ITUR_BT_602_CY = 37;
const unsigned char ITUR_BT_602_CUB = 65;
const unsigned char ITUR_BT_602_CUG = 13;
const unsigned char ITUR_BT_602_CVG = 26;
const unsigned char ITUR_BT_602_CVR = 51;
const unsigned char ITUR_BT_602_SHIFT = 5;

void cvt_color_yuv2bgr_nv21(unsigned char *src, unsigned char *dst, int w, int h) {
    if (src == nullptr || dst == nullptr) {
        return;
    }

    if (w % 2 != 0 || h % 2 != 0) {
        return;
    }

    const int stride = w;

    // Y通道的数据个数
    int y_stride = w * h;

    // y通道首地址
    unsigned char *y1 = src;
    // vu通道首地址
    unsigned char *vu1 = y1 + y_stride;

    int8x8_t cvr = vdup_n_s8 (ITUR_BT_602_CVR);
    int8x8_t cvg = vdup_n_s8 (-ITUR_BT_602_CVG);
    int8x8_t cug = vdup_n_s8 (-ITUR_BT_602_CUG);
    int8x8_t cub = vdup_n_s8 (ITUR_BT_602_CUB);
    int16x8_t cy = vdupq_n_s16 (ITUR_BT_602_CY);
    uint8x8_t uvoffset = vdup_n_u8 (128);
    uint8x8_t yoffset = vdup_n_u8 (16);
    int16x8_t round_offset = vdupq_n_s16(1 << (ITUR_BT_602_SHIFT - 1));

    int16x8_t zero = vdupq_n_s16(0);
    int16x8_t max = vdupq_n_s16(255);

    for (int j = 0; j < h; j += 2) {
        // rgb第一行首地址
        unsigned char *row1 = dst + j * w * 3;

        // rgb第二行首地址
        unsigned char *row2 = row1 + w * 3;

        // y通道第二行首地址
        const unsigned char *y2 = y1 + stride;

        // 对于y通道的每一行，128位寄存器每次操作16个元素，共需要w/16个操作指令
        // 对于bgr数据，16个元素*3通道，偏移值为48
        for (int i = 0; i < w / 16; i += 1, row1 += 48, row2 += 48) {
            // 对于y通道一行，每次取16个元素的起始位置
            const unsigned char* ly1 = y1 + i * 16;
            const unsigned char* ly2 = y2 + i * 16;
            // VU通道取16个元素，交替取到2个uint8x8_t结构体中，对应y1和y2
            uint8x8x2_t vvu = vld2_u8(vu1 + i * 16);
            // nv21是VU交叉存储
            uint8x8_t vv = vvu.val[0];
            uint8x8_t vu = vvu.val[1];
            // y通道每行取8个元素到寄存器 todo 这里不懂，交替给到2个结构体
            uint8x8x2_t vy1 = vld2_u8(ly1);
            uint8x8x2_t vy2 = vld2_u8(ly2);
            // U V 分别减去128
            uint16x8_t vu16 = vsubl_u8(vu, uvoffset); // widen subtract
            uint16x8_t vv16 = vsubl_u8(vv, uvoffset);

            // vreinterpretq_s16_u16:uint16->sint16
            // vqmovn_s16:截取sint16为                                                             int8
            int8x8_t svu = vqmovn_s16(vreinterpretq_s16_u16(vu16)); // convert to signed integer
            int8x8_t svv = vqmovn_s16(vreinterpretq_s16_u16(vv16));
            // U 乘以 13
            int16x8_t gu = vmull_s8(svu, cug);
            // V 乘以 26
            int16x8_t gv = vmull_s8(svv, cvg);
            // gu + gv
            int16x8_t guv = vaddq_s16(gu, gv);

            // V 乘以 51
            int16x8_t ruv = vmull_s8(svv, cvr);
            // U 乘以 65
            int16x8_t buv = vmull_s8(svu, cub);

            // ruv + (1<<4)
            ruv = vaddq_s16(ruv, round_offset);
            // buv + (1<<4)
            buv = vaddq_s16(buv, round_offset);
            // guv + (1<<4)
            guv = vaddq_s16(guv, round_offset);

            // y通道每行分别减去16
            uint16x8x2_t vy116, vy216;
            vy116.val[0] = vsubl_u8(vy1.val[0], yoffset);
            vy116.val[1] = vsubl_u8(vy1.val[1], yoffset);
            vy216.val[0] = vsubl_u8(vy2.val[0], yoffset);
            vy216.val[1] = vsubl_u8(vy2.val[1], yoffset);
            // 转为有符号的int16
            int16x8x2_t svy1, svy2;
            svy1.val[0] = vreinterpretq_s16_u16(vy116.val[0]);
            svy1.val[1] = vreinterpretq_s16_u16(vy116.val[1]);
            svy2.val[0] = vreinterpretq_s16_u16(vy216.val[0]);
            svy2.val[1] = vreinterpretq_s16_u16(vy216.val[1]);
            // 乘以37
            int16x8x2_t vy1_2, vy2_2;
            vy1_2.val[0] = vmulq_s16(svy1.val[0], cy);
            vy1_2.val[1] = vmulq_s16(svy1.val[1], cy);
            vy2_2.val[0] = vmulq_s16(svy2.val[0], cy);
            vy2_2.val[1] = vmulq_s16(svy2.val[1], cy);

            //
            int16x8x2_t vb1_2, vg1_2, vr1_2, vb2_2, vg2_2, vr2_2;
            vb1_2.val[0] = vaddq_s16(vy1_2.val[0], buv);
            vg1_2.val[0] = vaddq_s16(vy1_2.val[0], guv);
            vr1_2.val[0] = vaddq_s16(vy1_2.val[0], ruv);

            vb1_2.val[1] = vaddq_s16(vy1_2.val[1], buv);
            vg1_2.val[1] = vaddq_s16(vy1_2.val[1], guv);
            vr1_2.val[1] = vaddq_s16(vy1_2.val[1], ruv);

            vb2_2.val[0] = vaddq_s16(vy2_2.val[0], buv);
            vg2_2.val[0] = vaddq_s16(vy2_2.val[0], guv);
            vr2_2.val[0] = vaddq_s16(vy2_2.val[0], ruv);

            vb2_2.val[1] = vaddq_s16(vy2_2.val[1], buv);
            vg2_2.val[1] = vaddq_s16(vy2_2.val[1], guv);
            vr2_2.val[1] = vaddq_s16(vy2_2.val[1], ruv);


            // 转为无符号int16
            uint16x8x2_t svb1_2, svg1_2, svr1_2, svb2_2, svg2_2, svr2_2;
            svb1_2.val[0] = vreinterpretq_u16_s16(vqabsq_s16(vb1_2.val[0]));
            svb1_2.val[1] = vreinterpretq_u16_s16(vqabsq_s16(vb1_2.val[1]));
            svg1_2.val[0] = vreinterpretq_u16_s16(vqabsq_s16(vg1_2.val[0]));
            svg1_2.val[1] = vreinterpretq_u16_s16(vqabsq_s16(vg1_2.val[1]));
            svr1_2.val[0] = vreinterpretq_u16_s16(vqabsq_s16(vr1_2.val[0]));
            svr1_2.val[1] = vreinterpretq_u16_s16(vqabsq_s16(vr1_2.val[1]));
            svb2_2.val[0] = vreinterpretq_u16_s16(vqabsq_s16(vb2_2.val[0]));
            svb2_2.val[1] = vreinterpretq_u16_s16(vqabsq_s16(vb2_2.val[1]));
            svg2_2.val[0] = vreinterpretq_u16_s16(vqabsq_s16(vg2_2.val[0]));
            svg2_2.val[1] = vreinterpretq_u16_s16(vqabsq_s16(vg2_2.val[1]));
            svr2_2.val[0] = vreinterpretq_u16_s16(vqabsq_s16(vr2_2.val[0]));
            svr2_2.val[1] = vreinterpretq_u16_s16(vqabsq_s16(vr2_2.val[1]));

            // vqshrn_n_u16:右移5并截断为一半
            uint8x8x2_t v8b1_2, v8g1_2, v8r1_2, v8b2_2, v8g2_2, v8r2_2;
            v8b1_2.val[0] = vqshrn_n_u16(svb1_2.val[0], ITUR_BT_602_SHIFT);
            v8b1_2.val[1] = vqshrn_n_u16(svb1_2.val[1], ITUR_BT_602_SHIFT);
            v8g1_2.val[0] = vqshrn_n_u16(svg1_2.val[0], ITUR_BT_602_SHIFT);
            v8g1_2.val[1] = vqshrn_n_u16(svg1_2.val[1], ITUR_BT_602_SHIFT);
            v8r1_2.val[0] = vqshrn_n_u16(svr1_2.val[0], ITUR_BT_602_SHIFT);
            v8r1_2.val[1] = vqshrn_n_u16(svr1_2.val[1], ITUR_BT_602_SHIFT);

            v8b2_2.val[0] = vqshrn_n_u16(svb2_2.val[0], ITUR_BT_602_SHIFT);
            v8b2_2.val[1] = vqshrn_n_u16(svb2_2.val[1], ITUR_BT_602_SHIFT);
            v8g2_2.val[0] = vqshrn_n_u16(svg2_2.val[0], ITUR_BT_602_SHIFT);
            v8g2_2.val[1] = vqshrn_n_u16(svg2_2.val[1], ITUR_BT_602_SHIFT);
            v8r2_2.val[0] = vqshrn_n_u16(svr2_2.val[0], ITUR_BT_602_SHIFT);
            v8r2_2.val[1] = vqshrn_n_u16(svr2_2.val[1], ITUR_BT_602_SHIFT);

            uint8x8_t vzero = vdup_n_u8(0);
            // 高位放全0
            uint8x16_t v8b1_11 = vcombine_u8(vzero, v8b1_2.val[0]);
            uint8x16_t v8b1_12 = vcombine_u8(vzero, v8b1_2.val[1]);
            uint8x16_t v8g1_11 = vcombine_u8(vzero, v8g1_2.val[0]);
            uint8x16_t v8g1_12 = vcombine_u8(vzero, v8g1_2.val[1]);
            uint8x16_t v8r1_11 = vcombine_u8(vzero, v8r1_2.val[0]);
            uint8x16_t v8r1_12 = vcombine_u8(vzero, v8r1_2.val[1]);

            uint8x16_t v8b2_11 = vcombine_u8(vzero, v8b2_2.val[0]);
            uint8x16_t v8b2_12 = vcombine_u8(vzero, v8b2_2.val[1]);
            uint8x16_t v8g2_11 = vcombine_u8(vzero, v8g2_2.val[0]);
            uint8x16_t v8g2_12 = vcombine_u8(vzero, v8g2_2.val[1]);
            uint8x16_t v8r2_11 = vcombine_u8(vzero, v8r2_2.val[0]);
            uint8x16_t v8r2_12 = vcombine_u8(vzero, v8r2_2.val[1]);

#if __aarch64__
            // 合并两个结构体，取它们的低位交替放置到新的结构体中
            uint8x16_t v8b1_1 = vzip2q_u8(v8b1_11, v8b1_12);
            uint8x16_t v8g1_1 = vzip2q_u8(v8g1_11, v8g1_12);
            uint8x16_t v8r1_1 = vzip2q_u8(v8r1_11, v8r1_12);

            uint8x16_t v8b2_1 = vzip2q_u8(v8b2_11, v8b2_12);
            uint8x16_t v8g2_1 = vzip2q_u8(v8g2_11, v8g2_12);
            uint8x16_t v8r2_1 = vzip2q_u8(v8r2_11, v8r2_12);
#else
            uint8x16_t v8b1_1 = vzipq_u8(v8b1_11, v8b1_12).val[1];
            uint8x16_t v8g1_1 = vzipq_u8(v8g1_11, v8g1_12).val[1];
            uint8x16_t v8r1_1 = vzipq_u8(v8r1_11, v8r1_12).val[1];

            uint8x16_t v8b2_1 = vzipq_u8(v8b2_11, v8b2_12).val[1];
            uint8x16_t v8g2_1 = vzipq_u8(v8g2_11, v8g2_12).val[1];
            uint8x16_t v8r2_1 = vzipq_u8(v8r2_11, v8r2_12).val[1];
#endif // __aarch64__

            uint8x16x3_t bgr1, bgr2;
            bgr1.val[0] = v8b1_1;
            bgr1.val[1] = v8g1_1;
            bgr1.val[2] = v8r1_1;

            bgr2.val[0] = v8b2_1;
            bgr2.val[1] = v8g2_1;
            bgr2.val[2] = v8r2_1;

            vst3q_u8(row1, bgr1);
            vst3q_u8(row2, bgr2);
        }

        y1 += stride * 2;
        vu1 += w;
    }
}

void bgr2nv21(unsigned char *src, unsigned char *dst, int width, int height) {
    if (src == nullptr || dst == nullptr) {
        return;
    }

    if (width % 2 != 0 || height % 2 != 0) {
        return;
    }

    static unsigned short shift = 14;
    static unsigned int coeffs[5] = {B2YI, G2YI, R2YI, B2UI, R2VI};
    static unsigned int offset = 128 << shift;

    unsigned char *y_plane = dst;
    unsigned char *vu_plane = dst + width * height;

    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; ++c) {
            int Y = (unsigned int) (src[0] * coeffs[0] + src[1] * coeffs[1] + src[2] * coeffs[2]) >> shift;
            *y_plane++ = (unsigned char) Y;

            if (r % 2 == 0 && c % 2 == 0) {
                int U = (unsigned int) ((src[0] - Y) * coeffs[3] + offset) >> shift;
                int V = (unsigned int) ((src[2] - Y) * coeffs[4] + offset) >> shift;

                vu_plane[0] = (unsigned char) V;
                vu_plane[1] = (unsigned char) U;
                vu_plane += 2;
            }
            src += 3;
        }
    }
}


int main() {
    cv::Mat img = cv::imread("res/akiyo_qcif.jpg");

    int h = img.rows;
    int w = img.cols;
    int c = img.channels();

    // prepare yuv data
    unsigned char* nv21_mem = (unsigned char*)malloc(w*h*3/2*sizeof(unsigned char));
    bgr2nv21(img.data, nv21_mem, w, h);
    cv::Mat yuv_img(h * 3 / 2, w, CV_8UC1);
    memcpy(yuv_img.data, nv21_mem, w*h*3/2*sizeof(unsigned char));

    // opencv yuv_nv21 to bgr
    cv::Mat opencv_bgr_img(h, w, CV_8UC3);
    uint8_t* opencv_bgr_nv21_mem = (uint8_t*)malloc(w*h*c*sizeof(uint8_t));
    clock_t start_time = clock();
    cv::cvtColor(yuv_img, opencv_bgr_img, CV_YUV2BGR_I420);
    std::cout << "opencv_cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
    cv::imwrite("output/yuv_nv21_to_bgr_opencv.jpg", opencv_bgr_img);

    // paddle yuv_nv21 to bgr
    uint8_t* paddle_bgr_nv21_mem = (uint8_t*)malloc(w*h*c*sizeof(uint8_t));
    // nv21 x = 0, y = 1
    start_time = clock();
    nv_to_bgr((uint8_t*)nv21_mem, paddle_bgr_nv21_mem, w, h, 0, 1);
    std::cout << "paddle_neon_cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;

    cv::Mat paddle_bgr_from_nv21_neon_img(h, w, CV_8UC3);
    memcpy(paddle_bgr_from_nv21_neon_img.data, (unsigned char*)paddle_bgr_nv21_mem, w*h*3*sizeof(unsigned char));
    cv::imwrite("output/yuv_nv21_to_bgr_neon.jpg", paddle_bgr_from_nv21_neon_img);

    // naive yuv_nv21 to bgr
    unsigned char* naive_bgr_nv21_mem = (unsigned char*)malloc(w*h*c*sizeof(unsigned char));
    // nv21 x = 0, y = 1
    start_time = clock();
    nv_to_bgr_naive((uint8_t*)nv21_mem, naive_bgr_nv21_mem, w, h, 0, 1);
    std::cout << "naive_cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;

    cv::Mat naive_bgr_from_nv21_img(h, w, CV_8UC3);
    memcpy(naive_bgr_from_nv21_img.data, naive_bgr_nv21_mem, w*h*3*sizeof(unsigned char));
    cv::imwrite("output/yuv_nv21_to_bgr_naive.jpg", naive_bgr_from_nv21_img);


//    unsigned char* our_bgr_nv21_mem = (unsigned char*)malloc(w*h*c*sizeof(unsigned char));
//    start_time = clock();
//    cvt_color_yuv2bgr_nv21(nv21_mem, our_bgr_nv21_mem, w, h);
//    std::cout << "our_neon_cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
//    cv::Mat our_bgr_neon_nv21_img(h, w, CV_8UC3);
//    memcpy(our_bgr_neon_nv21_img.data, our_bgr_nv21_mem, w*h*c*sizeof(unsigned char));
//    cv::imwrite("output/yuv_nv21_to_bgr_neon_ours.jpg", our_bgr_neon_nv21_img);


    return 0;
}