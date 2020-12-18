#include "cvt_color.h"

#include <math.h>

#ifdef USE_OPENCV
#include "opencv2/opencv.hpp"
#endif

#if defined (USE_NEON) and __ARM_NEON
#include "arm_neon.h"
#endif

#include "../common/tensor_converter.h"
#include "cv.h"

namespace va_cv {

using namespace vision;

void CvtColor::cvt_color(const Tensor& src, Tensor& dst, int code) {
#if defined (USE_NEON) and __ARM_NEON
    cvt_color_neon(src, dst, code);
#elif defined (USE_SSE)
    cvt_color_sse(src, dst, code);
#elif defined (USE_OPENCV)
    cvt_color_opencv(src, dst, code);
#else
    cvt_color_naive(src, dst, code);
#endif // USE_OPENCV
}

void CvtColor::cvt_color_opencv(const Tensor& src, Tensor& dst, int code) {
#ifdef USE_OPENCV
    const auto& mat_src = vision::TensorConverter::convert_to<cv::Mat>(src);
    cv::Mat mat_dst;
    cv::cvtColor(mat_src, mat_dst, code);
    dst = vision::TensorConverter::convert_from<cv::Mat>(mat_dst, true);
#else
    cvt_color_naive(src, dst, code);
#endif
}

void CvtColor::nv_to_bgr_naive(const unsigned char* src,
                                     unsigned char* dst,
                                     int srcw,
                                     int srch,
                                     int x_num,
                                     int y_num) {
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

void CvtColor::cvt_color_naive(const Tensor& src, Tensor& dst, int code) {

    if (code != COLOR_YUV2BGR_NV21 && code != COLOR_YUV2BGR_NV12) {
        cvt_color_opencv(src, dst, code);
        return;
    }

    int x_num = 0;
    int y_num = 1;
    if (code == COLOR_YUV2RGB_NV12) {
        x_num = 1;
        y_num = 0;
    }

    int bgr_w = src.w;
    int bgr_h = src.h / 3 * 2;
    int bgr_c = 3;
    dst.create(bgr_w, bgr_h, bgr_c, NHWC, INT8);

    nv_to_bgr_naive((unsigned char*)src.data, (unsigned char*)dst.data, bgr_w, bgr_h, x_num, y_num);
}

void CvtColor::cvt_color_sse(const Tensor& src, Tensor& dst, int code) {
    // todo:
}

#if defined (USE_NEON) and __ARM_NEON
void CvtColor::cvt_color_neon(const Tensor& src, Tensor& dst, int code) {

    if (code != COLOR_YUV2BGR_NV21 && code != COLOR_YUV2BGR_NV12) {
        cvt_color_opencv(src, dst, code);
        return;
    }
    int x_num = 0;
    int y_num = 1;
    if (code == COLOR_YUV2RGB_NV12) {
        x_num = 1;
        y_num = 0;
    }
    int bgr_w = src.w;
    int bgr_h = src.h / 3 * 2;
    int bgr_c = 3;
    dst.create(bgr_w, bgr_h, bgr_c, NHWC, INT8);

    nv_to_bgr_neon((uint8_t*)src.data, (uint8_t*)dst.data, bgr_w, bgr_h, x_num, y_num);
}

void CvtColor::nv_to_bgr_neon(const uint8_t* src, uint8_t* dst,
                              int srcw, int srch,
                              int x_num, int y_num) {
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
        const uint8_t* ptr_y1 = y + i * srcw;
        const uint8_t* ptr_y2 = ptr_y1 + srcw;
        const uint8_t* ptr_vu = vu + (i / 2) * srcw;
        uint8_t* ptr_bgr1 = dst + i * wout;
        uint8_t* ptr_bgr2 = ptr_bgr1 + wout;
        if (i + 2 > y_h) {
            ptr_y2 = zerobuf;
            ptr_bgr2 = writebuf;
        }
        int j = 0;
        for (; j < srcw - 15; j += 16) {
            uint8x8x2_t y1 = vld2_u8(ptr_y1);  // d8 = y0y2y4y6...y14 d9 =
            // y1y3y5...y15
            uint8x8x2_t vu = vld2_u8(ptr_vu);  // d0 = v0v1v2v3v4v5...v7 d1 = u0u1u2...u7

            uint8x8x2_t y2 = vld2_u8(ptr_y2);

            uint16x8_t v = vmovl_u8(vu.val[x_num]);
            uint16x8_t u = vmovl_u8(vu.val[y_num]);
            int16x8_t v_s = vreinterpretq_s16_u16(v);
            int16x8_t u_s = vreinterpretq_s16_u16(u);
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
        // two data
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
#endif

} // namespace va_cv