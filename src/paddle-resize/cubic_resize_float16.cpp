//
// Created by b1xian on 2020-10-26.
//
#include <math.h>

#include <iostream>
#include "arm_neon.h"

#include "../vision/tensor.h"
#include "../vision/tensor_converter.h"
#include "opencv2/opencv.hpp"

using namespace std;

static void du_interpolate_cubic_fp16(float fx, __fp16 *coeffs)
{
    const float A = -0.5f;

    float fx0 = fx + 1;
    float fx1 = fx;
    float fx2 = 1 - fx;
    // float fx3 = 2 - fx;

    coeffs[0] = (__fp16)(A * fx0 * fx0 * fx0 - 5 * A * fx0 * fx0 + 8 * A * fx0 - 4 * A);
    coeffs[1] = (__fp16)((A + 2) * fx1 * fx1 * fx1 - (A + 3) * fx1 * fx1 + 1);
    coeffs[2] = (__fp16)((A + 2) * fx2 * fx2 * fx2 - (A + 3) * fx2 * fx2 + 1);
    coeffs[3] = (__fp16)(1.f - coeffs[0] - coeffs[1] - coeffs[2]);

}

static void du_cubic_coeffs_fp16(int w, int outw, int *xofs, __fp16 *alpha)
{
    double scale = (double)w / outw;

    for (int dx = 0; dx < outw; dx++)
    {
        float fx = (float)((dx + 0.5) * scale - 0.5);
        int sx = static_cast<int>(floor(fx));
        fx -= sx;

        du_interpolate_cubic_fp16(fx, alpha + dx * 4);

        if (sx <= -1)
        {
            sx = 1;
            alpha[dx * 4 + 0] = (__fp16)(1.f - alpha[dx * 4 + 3]);
            alpha[dx * 4 + 1] = (__fp16)alpha[dx * 4 + 3];
            alpha[dx * 4 + 2] = (__fp16)0.f;
            alpha[dx * 4 + 3] = (__fp16)0.f;
        }
        if (sx == 0)
        {
            sx = 1;
            alpha[dx * 4 + 0] = (__fp16)(alpha[dx * 4 + 0] + alpha[dx * 4 + 1]);
            alpha[dx * 4 + 1] = (__fp16)alpha[dx * 4 + 2];
            alpha[dx * 4 + 2] = (__fp16)alpha[dx * 4 + 3];
            alpha[dx * 4 + 3] = (__fp16)0.f;
        }
        if (sx == w - 2)
        {
            sx = w - 3;
            alpha[dx * 4 + 3] = (__fp16)(alpha[dx * 4 + 2] + alpha[dx * 4 + 3]);
            alpha[dx * 4 + 2] = (__fp16)alpha[dx * 4 + 1];
            alpha[dx * 4 + 1] = (__fp16)alpha[dx * 4 + 0];
            alpha[dx * 4 + 0] = (__fp16)0.f;
        }
        if (sx >= w - 1)
        {
            sx = w - 3;
            alpha[dx * 4 + 3] = (__fp16)(1.f - alpha[dx * 4 + 0]);
            alpha[dx * 4 + 2] = (__fp16)(alpha[dx * 4 + 0]);
            alpha[dx * 4 + 1] = (__fp16)0.f;
            alpha[dx * 4 + 0] = (__fp16)0.f;
        }

        xofs[dx] = sx;
    }
}
static void du_resize_bicubic_image_fp16(__fp16* src, int srcw, int srch, __fp16* dst, int outw, int outh,
                                 __fp16* alpha, int* xofs, __fp16* beta, int* yofs)
{
    int w = outw;
    int h = outh;

    // loop body
    __fp16* rowsbuf0 = (__fp16*)malloc(sizeof(__fp16) * w * 4);
    __fp16* rowsbuf1 = (__fp16*)malloc(sizeof(__fp16) * w * 4);
    __fp16* rowsbuf2 = (__fp16*)malloc(sizeof(__fp16) * w * 4);
    __fp16* rowsbuf3 = (__fp16*)malloc(sizeof(__fp16) * w * 4);
    __fp16* rows0 = rowsbuf0;
    __fp16* rows1 = rowsbuf1;
    __fp16* rows2 = rowsbuf2;
    __fp16* rows3 = rowsbuf3;

    int prev_sy1 = -3;

    for (int dy = 0; dy < h; dy++)
    {
        int sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            __fp16* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows2;
            rows2 = rows3;
            rows3 = rows0_old;
            __fp16* S3 = src + (sy+2)*srcw;

            const __fp16* alphap = alpha;
            __fp16* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * 8;
                const __fp16* S3p = S3 + sx;

                float16x4_t _a0123 = vld1_f16(alphap);

                float16x8_t _S30 = vld1q_f16(S3p - 8);
                float16x8_t _S31 = vld1q_f16(S3p + 0);
                float16x8_t _S32 = vld1q_f16(S3p + 8);
                float16x8_t _S33 = vld1q_f16(S3p + 16);
                float16x8_t _rows3 = vmulq_lane_f16(_S30, _a0123, 0);
                _rows3 = vfmaq_lane_f16(_rows3, _S31, _a0123, 1);
                _rows3 = vfmaq_lane_f16(_rows3, _S32, _a0123, 2);
                _rows3 = vfmaq_lane_f16(_rows3, _S33, _a0123, 3);
                vst1q_f16(rows3p + dx * 8, _rows3);

                alphap += 4;
            }
        }
        else if (sy == prev_sy1 + 2)
        {
            // hresize two rows
            __fp16* rows0_old = rows0;
            __fp16* rows1_old = rows1;
            rows0 = rows2;
            rows1 = rows3;
            rows2 = rows0_old;
            rows3 = rows1_old;
            __fp16* S2 = src + (sy+1)*srcw;
            __fp16* S3 = src + (sy+2)*srcw;

            const __fp16* alphap = alpha;
            __fp16* rows2p = rows2;
            __fp16* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * 8;
                const __fp16* S2p = S2 + sx;
                const __fp16* S3p = S3 + sx;

                float16x4_t _a0123 = vld1_f16(alphap);

                float16x8_t _S20 = vld1q_f16(S2p - 8);
                float16x8_t _S21 = vld1q_f16(S2p + 0);
                float16x8_t _S22 = vld1q_f16(S2p + 8);
                float16x8_t _S23 = vld1q_f16(S2p + 16);
                float16x8_t _S30 = vld1q_f16(S3p - 8);
                float16x8_t _S31 = vld1q_f16(S3p + 0);
                float16x8_t _S32 = vld1q_f16(S3p + 8);
                float16x8_t _S33 = vld1q_f16(S3p + 16);
                float16x8_t _rows2 = vmulq_lane_f16(_S20, _a0123, 0);
                float16x8_t _rows3 = vmulq_lane_f16(_S30, _a0123, 0);
                _rows2 = vfmaq_lane_f16(_rows2, _S21, _a0123, 1);
                _rows3 = vfmaq_lane_f16(_rows3, _S31, _a0123, 1);
                _rows2 = vfmaq_lane_f16(_rows2, _S22, _a0123, 2);
                _rows3 = vfmaq_lane_f16(_rows3, _S32, _a0123, 2);
                _rows2 = vfmaq_lane_f16(_rows2, _S23, _a0123, 3);
                _rows3 = vfmaq_lane_f16(_rows3, _S33, _a0123, 3);
                vst1q_f16(rows2p + dx * 8, _rows2);
                vst1q_f16(rows3p + dx * 8, _rows3);

                alphap += 4;
            }
        }
        else if (sy == prev_sy1 + 3)
        {
            // hresize three rows
            __fp16* rows0_old = rows0;
            __fp16* rows1_old = rows1;
            __fp16* rows2_old = rows2;
            rows0 = rows3;
            rows1 = rows0_old;
            rows2 = rows1_old;
            rows3 = rows2_old;
            __fp16* S1 = src + (sy)*srcw;
            __fp16* S2 = src + (sy+1)*srcw;
            __fp16* S3 = src + (sy+2)*srcw;

            const __fp16* alphap = alpha;
            __fp16* rows1p = rows1;
            __fp16* rows2p = rows2;
            __fp16* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * 8;
                const __fp16* S1p = S1 + sx;
                const __fp16* S2p = S2 + sx;
                const __fp16* S3p = S3 + sx;

                float16x4_t _a0123 = vld1_f16(alphap);

                float16x8_t _S10 = vld1q_f16(S1p - 8);
                float16x8_t _S11 = vld1q_f16(S1p + 0);
                float16x8_t _S12 = vld1q_f16(S1p + 8);
                float16x8_t _S13 = vld1q_f16(S1p + 16);
                float16x8_t _S20 = vld1q_f16(S2p - 8);
                float16x8_t _S21 = vld1q_f16(S2p + 0);
                float16x8_t _S22 = vld1q_f16(S2p + 8);
                float16x8_t _S23 = vld1q_f16(S2p + 16);
                float16x8_t _S30 = vld1q_f16(S3p - 8);
                float16x8_t _S31 = vld1q_f16(S3p + 0);
                float16x8_t _S32 = vld1q_f16(S3p + 8);
                float16x8_t _S33 = vld1q_f16(S3p + 16);
                float16x8_t _rows1 = vmulq_lane_f16(_S10, _a0123, 0);
                float16x8_t _rows2 = vmulq_lane_f16(_S20, _a0123, 0);
                float16x8_t _rows3 = vmulq_lane_f16(_S30, _a0123, 0);
                _rows1 = vfmaq_lane_f16(_rows1, _S11, _a0123, 1);
                _rows2 = vfmaq_lane_f16(_rows2, _S21, _a0123, 1);
                _rows3 = vfmaq_lane_f16(_rows3, _S31, _a0123, 1);
                _rows1 = vfmaq_lane_f16(_rows1, _S12, _a0123, 2);
                _rows2 = vfmaq_lane_f16(_rows2, _S22, _a0123, 2);
                _rows3 = vfmaq_lane_f16(_rows3, _S32, _a0123, 2);
                _rows1 = vfmaq_lane_f16(_rows1, _S13, _a0123, 3);
                _rows2 = vfmaq_lane_f16(_rows2, _S23, _a0123, 3);
                _rows3 = vfmaq_lane_f16(_rows3, _S33, _a0123, 3);
                vst1q_f16(rows1p + dx * 8, _rows1);
                vst1q_f16(rows2p + dx * 8, _rows2);
                vst1q_f16(rows3p + dx * 8, _rows3);

                alphap += 4;
            }
        }
        else
        {
            // hresize four rows
            __fp16* S0 = src + (sy-1)*srcw;
            __fp16* S1 = src + (sy)*srcw;
            __fp16* S2 = src + (sy+1)*srcw;
            __fp16* S3 = src + (sy+2)*srcw;

            const __fp16* alphap = alpha;
            __fp16* rows0p = rows0;
            __fp16* rows1p = rows1;
            __fp16* rows2p = rows2;
            __fp16* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * 8;
                const __fp16* S0p = S0 + sx;
                const __fp16* S1p = S1 + sx;
                const __fp16* S2p = S2 + sx;
                const __fp16* S3p = S3 + sx;

                float16x4_t _a0123 = vld1_f16(alphap);

                float16x8_t _S00 = vld1q_f16(S0p - 8);
                float16x8_t _S01 = vld1q_f16(S0p + 0);
                float16x8_t _S02 = vld1q_f16(S0p + 8);
                float16x8_t _S03 = vld1q_f16(S0p + 16);
                float16x8_t _S10 = vld1q_f16(S1p - 8);
                float16x8_t _S11 = vld1q_f16(S1p + 0);
                float16x8_t _S12 = vld1q_f16(S1p + 8);
                float16x8_t _S13 = vld1q_f16(S1p + 16);
                float16x8_t _S20 = vld1q_f16(S2p - 8);
                float16x8_t _S21 = vld1q_f16(S2p + 0);
                float16x8_t _S22 = vld1q_f16(S2p + 8);
                float16x8_t _S23 = vld1q_f16(S2p + 16);
                float16x8_t _S30 = vld1q_f16(S3p - 8);
                float16x8_t _S31 = vld1q_f16(S3p + 0);
                float16x8_t _S32 = vld1q_f16(S3p + 8);
                float16x8_t _S33 = vld1q_f16(S3p + 16);
                float16x8_t _rows0 = vmulq_lane_f16(_S00, _a0123, 0);
                float16x8_t _rows1 = vmulq_lane_f16(_S10, _a0123, 0);
                float16x8_t _rows2 = vmulq_lane_f16(_S20, _a0123, 0);
                float16x8_t _rows3 = vmulq_lane_f16(_S30, _a0123, 0);
                _rows0 = vfmaq_lane_f16(_rows0, _S01, _a0123, 1);
                _rows1 = vfmaq_lane_f16(_rows1, _S11, _a0123, 1);
                _rows2 = vfmaq_lane_f16(_rows2, _S21, _a0123, 1);
                _rows3 = vfmaq_lane_f16(_rows3, _S31, _a0123, 1);
                _rows0 = vfmaq_lane_f16(_rows0, _S02, _a0123, 2);
                _rows1 = vfmaq_lane_f16(_rows1, _S12, _a0123, 2);
                _rows2 = vfmaq_lane_f16(_rows2, _S22, _a0123, 2);
                _rows3 = vfmaq_lane_f16(_rows3, _S32, _a0123, 2);
                _rows0 = vfmaq_lane_f16(_rows0, _S03, _a0123, 3);
                _rows1 = vfmaq_lane_f16(_rows1, _S13, _a0123, 3);
                _rows2 = vfmaq_lane_f16(_rows2, _S23, _a0123, 3);
                _rows3 = vfmaq_lane_f16(_rows3, _S33, _a0123, 3);
                vst1q_f16(rows0p + dx * 8, _rows0);
                vst1q_f16(rows1p + dx * 8, _rows1);
                vst1q_f16(rows2p + dx * 8, _rows2);
                vst1q_f16(rows3p + dx * 8, _rows3);

                alphap += 4;
            }
        }

        prev_sy1 = sy;

        // vresize
        float16x4_t _b0123 = vld1_f16(beta);

        __fp16* rows0p = rows0;
        __fp16* rows1p = rows1;
        __fp16* rows2p = rows2;
        __fp16* rows3p = rows3;
        __fp16* Dp = dst + dy*w;

        for (int dx = 0; dx < w; dx++)
        {
            float16x8_t _rows0 = vld1q_f16(rows0p);
            float16x8_t _rows1 = vld1q_f16(rows1p);
            float16x8_t _rows2 = vld1q_f16(rows2p);
            float16x8_t _rows3 = vld1q_f16(rows3p);
            float16x8_t _D = vmulq_lane_f16(_rows0, _b0123, 0);
            _D = vfmaq_lane_f16(_D, _rows1, _b0123, 1);
            _D = vfmaq_lane_f16(_D, _rows2, _b0123, 2);
            _D = vfmaq_lane_f16(_D, _rows3, _b0123, 3);
            vst1q_f16(Dp, _D);

            Dp += 8;
            rows0p += 8;
            rows1p += 8;
            rows2p += 8;
            rows3p += 8;
        }

        beta += 4;
    }
}

static void du_resize_bicubic_fp16(__fp16* src, int srcw, int srch, __fp16* dst, int outw, int outh) {
    int* buf = new int[outw + outh + outw * 4 + outh * 4];

    int* xofs = buf;        //new int[outw];
    int* yofs = buf + outw; //new int[outh];

    __fp16* alpha = (__fp16*)(buf + outw + outh);           //new __fp16[outw * 4];
    __fp16* beta = (__fp16*)(buf + outw + outh + outw * 4); //new __fp16[outh * 4];

    du_cubic_coeffs_fp16(srcw, outw, xofs, alpha);
    du_cubic_coeffs_fp16(srch, outh, yofs, beta);

    int dst_stride = outw * outh;
    int src_stride = srcw * srch;
    for (int i = 0; i < 3; i++) {
        __fp16* src_channel_data = src + src_stride * i;
        __fp16* dst_channel_data = dst + dst_stride * i;

        du_resize_bicubic_image_fp16(src_channel_data, srcw, srch, dst_channel_data, outw, outh, alpha, xofs, beta, yofs);
    }
//    std::ofstream fout;
//    fout.open("./output/coeffs_x.txt");
//
//    // write coeffs
//    for (int i = 0; i < outw; i++) {
//        __fp16* pix_row_coef = alpha + i*4;
//        fout << *pix_row_coef << "," << *(pix_row_coef+1) << "," << *(pix_row_coef+2) << "," << *(pix_row_coef+3) << endl;
//    }
//    fout.close();

//    std::ofstream fout1;
//    fout.open("./output/coeffs_y.txt");
//    for (int i = 0; i < outh; i++) {
//        __fp16* pix_row_coef = beta + i*4;
//        fout1 << *pix_row_coef << "," << *(pix_row_coef+1) << "," << *(pix_row_coef+2) << "," << *(pix_row_coef+3) << endl;
//    }
//    fout1.close();

    delete[] buf;
}
