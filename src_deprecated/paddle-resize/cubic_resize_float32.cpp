//
// Created by b1xian on 2020-10-26.
//
#include <math.h>

#include <iostream>
#include "arm_neon.h"

#include "../../vision/common/tensor.h"
#include "../../vision/common/tensor_converter.h"
#include "opencv2/opencv.hpp"

using namespace std;
static void interpolate_cubic(float fx, float* coeffs)
{
    const float A = -0.5f;

    float fx0 = fx + 1;
    float fx1 = fx;
    float fx2 = 1 - fx;
    // float fx3 = 2 - fx;
    // a0X的横坐标权重分别为W(1+u)，W(u)，W(1-u)，W(2-u)
    /*        (a+2)|x|**3 - (a+3)|x|**2 + 1;    for |x| <= 1
     * w(x) = a|x|**3 - 5a|x|**2 + 8a|x| - 4a;  for 1 < |x| < 2
     *        0                                 otherwise
     */
    // x = 1 + u
    coeffs[0] = A * fx0 * fx0 * fx0 - 5 * A * fx0 * fx0 + 8 * A * fx0 - 4 * A;
    // x = u
    coeffs[1] = (A + 2) * fx1 * fx1 * fx1 - (A + 3) * fx1 * fx1 + 1;
    // x = 1 - u
    coeffs[2] = (A + 2) * fx2 * fx2 * fx2 - (A + 3) * fx2 * fx2 + 1;
    // x = 2 - u
    coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
}

static void cubic_coeffs(int w, int outw, int* xofs, float* alpha)
{
    double scale = (double)w / outw;

    // 计算系数
    for (int dx = 0; dx < outw; dx++)
    {
        // srcx的小数部分
        float fx = (float)((dx + 0.5) * scale - 0.5);
        // srcx的整数部分
        int sx = static_cast<int>(floor(fx));
        fx -= sx;

        // dst的每个点有四个coeffs
        interpolate_cubic(fx, alpha + dx * 4);

        // 边界情况
        if (sx <= -1)
        {
            sx = 1;
            alpha[dx * 4 + 0] = 1.f - alpha[dx * 4 + 3];
            alpha[dx * 4 + 1] = alpha[dx * 4 + 3];
            alpha[dx * 4 + 2] = 0.f;
            alpha[dx * 4 + 3] = 0.f;
        }
        if (sx == 0)
        {
            sx = 1;
            alpha[dx * 4 + 0] = alpha[dx * 4 + 0] + alpha[dx * 4 + 1];
            alpha[dx * 4 + 1] = alpha[dx * 4 + 2];
            alpha[dx * 4 + 2] = alpha[dx * 4 + 3];
            alpha[dx * 4 + 3] = 0.f;
        }
        if (sx == w - 2)
        {
            sx = w - 3;
            alpha[dx * 4 + 3] = alpha[dx * 4 + 2] + alpha[dx * 4 + 3];
            alpha[dx * 4 + 2] = alpha[dx * 4 + 1];
            alpha[dx * 4 + 1] = alpha[dx * 4 + 0];
            alpha[dx * 4 + 0] = 0.f;
        }
        if (sx >= w - 1)
        {
            sx = w - 3;
            alpha[dx * 4 + 3] = 1.f - alpha[dx * 4 + 0];
            alpha[dx * 4 + 2] = alpha[dx * 4 + 0];
            alpha[dx * 4 + 1] = 0.f;
            alpha[dx * 4 + 0] = 0.f;
        }

        xofs[dx] = sx;
    }
}
static void du_resize_bicubic_image_fp32(float* src, int srcw, int srch, float* dst, int outw, int outh,
                                      float* alpha, int* xofs, float* beta, int* yofs){
    int w = outw;
    int h = outh;

    // loop body
    float* rowsbuf0 = (float*)malloc(sizeof(float) * w * 4);
    float* rowsbuf1 = (float*)malloc(sizeof(float) * w * 4);
    float* rowsbuf2 = (float*)malloc(sizeof(float) * w * 4);
    float* rowsbuf3 = (float*)malloc(sizeof(float) * w * 4);

    float* rows0 = rowsbuf0;
    float* rows1 = rowsbuf1;
    float* rows2 = rowsbuf2;
    float* rows3 = rowsbuf3;

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
            float* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows2;
            rows2 = rows3;
            rows3 = rows0_old;
            float* S3 = src + (sy+2)*srcw;

            const float* alphap = alpha;
            float* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * 4;
                const float* S3p = S3 + sx;

                float32x2_t _a01 = vld1_f32(alphap);
                float32x2_t _a23 = vld1_f32(alphap + 2);

                float32x4_t _S30 = vld1q_f32(S3p - 4);
                float32x4_t _S31 = vld1q_f32(S3p + 0);
                float32x4_t _S32 = vld1q_f32(S3p + 4);
                float32x4_t _S33 = vld1q_f32(S3p + 8);
                float32x4_t _rows3 = vmulq_lane_f32(_S30, _a01, 0);
                _rows3 = vfmaq_lane_f32(_rows3, _S31, _a01, 1);
                _rows3 = vfmaq_lane_f32(_rows3, _S32, _a23, 0);
                _rows3 = vfmaq_lane_f32(_rows3, _S33, _a23, 1);
                vst1q_f32(rows3p + dx * 4, _rows3);

                alphap += 4;
            }
        }
        else if (sy == prev_sy1 + 2)
        {
            // hresize two rows
            float* rows0_old = rows0;
            float* rows1_old = rows1;
            rows0 = rows2;
            rows1 = rows3;
            rows2 = rows0_old;
            rows3 = rows1_old;
            float* S2 = src + (sy+1)*srcw;
            float* S3 = src + (sy+2)*srcw;

            const float* alphap = alpha;
            float* rows2p = rows2;
            float* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * 4;
                const float* S2p = S2 + sx;
                const float* S3p = S3 + sx;

                float32x2_t _a01 = vld1_f32(alphap);
                float32x2_t _a23 = vld1_f32(alphap + 2);

                float32x4_t _S20 = vld1q_f32(S2p - 4);
                float32x4_t _S21 = vld1q_f32(S2p + 0);
                float32x4_t _S22 = vld1q_f32(S2p + 4);
                float32x4_t _S23 = vld1q_f32(S2p + 8);
                float32x4_t _S30 = vld1q_f32(S3p - 4);
                float32x4_t _S31 = vld1q_f32(S3p + 0);
                float32x4_t _S32 = vld1q_f32(S3p + 4);
                float32x4_t _S33 = vld1q_f32(S3p + 8);
                float32x4_t _rows2 = vmulq_lane_f32(_S20, _a01, 0);
                float32x4_t _rows3 = vmulq_lane_f32(_S30, _a01, 0);
                _rows2 = vfmaq_lane_f32(_rows2, _S21, _a01, 1);
                _rows3 = vfmaq_lane_f32(_rows3, _S31, _a01, 1);
                _rows2 = vfmaq_lane_f32(_rows2, _S22, _a23, 0);
                _rows3 = vfmaq_lane_f32(_rows3, _S32, _a23, 0);
                _rows2 = vfmaq_lane_f32(_rows2, _S23, _a23, 1);
                _rows3 = vfmaq_lane_f32(_rows3, _S33, _a23, 1);
                vst1q_f32(rows2p + dx * 4, _rows2);
                vst1q_f32(rows3p + dx * 4, _rows3);

                alphap += 4;
            }
        }
        else if (sy == prev_sy1 + 3)
        {
            // hresize three rows
            float* rows0_old = rows0;
            float* rows1_old = rows1;
            float* rows2_old = rows2;
            rows0 = rows3;
            rows1 = rows0_old;
            rows2 = rows1_old;
            rows3 = rows2_old;
            float* S1 = src + (sy)*srcw;
            float* S2 = src + (sy+1)*srcw;
            float* S3 = src + (sy+2)*srcw;

            const float* alphap = alpha;
            float* rows1p = rows1;
            float* rows2p = rows2;
            float* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * 4;
                const float* S1p = S1 + sx;
                const float* S2p = S2 + sx;
                const float* S3p = S3 + sx;

                float32x2_t _a01 = vld1_f32(alphap);
                float32x2_t _a23 = vld1_f32(alphap + 2);

                float32x4_t _S10 = vld1q_f32(S1p - 4);
                float32x4_t _S11 = vld1q_f32(S1p + 0);
                float32x4_t _S12 = vld1q_f32(S1p + 4);
                float32x4_t _S13 = vld1q_f32(S1p + 8);
                float32x4_t _S20 = vld1q_f32(S2p - 4);
                float32x4_t _S21 = vld1q_f32(S2p + 0);
                float32x4_t _S22 = vld1q_f32(S2p + 4);
                float32x4_t _S23 = vld1q_f32(S2p + 8);
                float32x4_t _S30 = vld1q_f32(S3p - 4);
                float32x4_t _S31 = vld1q_f32(S3p + 0);
                float32x4_t _S32 = vld1q_f32(S3p + 4);
                float32x4_t _S33 = vld1q_f32(S3p + 8);

                float32x4_t _rows1 = vmulq_lane_f32(_S10, _a01, 0);
                float32x4_t _rows2 = vmulq_lane_f32(_S20, _a01, 0);
                float32x4_t _rows3 = vmulq_lane_f32(_S30, _a01, 0);
                _rows1 = vfmaq_lane_f32(_rows1, _S11, _a01, 1);
                _rows2 = vfmaq_lane_f32(_rows2, _S21, _a01, 1);
                _rows3 = vfmaq_lane_f32(_rows3, _S31, _a01, 1);
                _rows1 = vfmaq_lane_f32(_rows1, _S12, _a23, 0);
                _rows2 = vfmaq_lane_f32(_rows2, _S22, _a23, 0);
                _rows3 = vfmaq_lane_f32(_rows3, _S32, _a23, 0);
                _rows1 = vfmaq_lane_f32(_rows1, _S13, _a23, 1);
                _rows2 = vfmaq_lane_f32(_rows2, _S23, _a23, 1);
                _rows3 = vfmaq_lane_f32(_rows3, _S33, _a23, 1);

                vst1q_f32(rows1p + dx * 4, _rows1);
                vst1q_f32(rows2p + dx * 4, _rows2);
                vst1q_f32(rows3p + dx * 4, _rows3);

                alphap += 4;
            }
        }
        else {
            // hresize four rows
            float* S0 = src + (sy-1)*srcw;
            float* S1 = src + (sy)*srcw;
            float* S2 = src + (sy+1)*srcw;
            float* S3 = src + (sy+2)*srcw;

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;
            float* rows2p = rows2;
            float* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                // 每行每次取4个元素
                int sx = xofs[dx] * 4;
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;
                const float* S2p = S2 + sx;
                const float* S3p = S3 + sx;

                // 取该像素的最邻近行的系数（四个系数）
                float32x2_t _a01 = vld1_f32(alphap);
                float32x2_t _a23 = vld1_f32(alphap + 2);

                // TODO check the generated assembly on armv7
                float32x4_t _S00 = vld1q_f32(S0p - 4);
                float32x4_t _S01 = vld1q_f32(S0p + 0);
                float32x4_t _S02 = vld1q_f32(S0p + 4);
                float32x4_t _S03 = vld1q_f32(S0p + 8);
                float32x4_t _S10 = vld1q_f32(S1p - 4);
                float32x4_t _S11 = vld1q_f32(S1p + 0);
                float32x4_t _S12 = vld1q_f32(S1p + 4);
                float32x4_t _S13 = vld1q_f32(S1p + 8);
                float32x4_t _S20 = vld1q_f32(S2p - 4);
                float32x4_t _S21 = vld1q_f32(S2p + 0);
                float32x4_t _S22 = vld1q_f32(S2p + 4);
                float32x4_t _S23 = vld1q_f32(S2p + 8);
                float32x4_t _S30 = vld1q_f32(S3p - 4);
                float32x4_t _S31 = vld1q_f32(S3p + 0);
                float32x4_t _S32 = vld1q_f32(S3p + 4);
                float32x4_t _S33 = vld1q_f32(S3p + 8);
                // 最邻近第一行
                float32x4_t _rows0 = vmulq_lane_f32(_S00, _a01, 0);
                float32x4_t _rows1 = vmulq_lane_f32(_S10, _a01, 0);
                float32x4_t _rows2 = vmulq_lane_f32(_S20, _a01, 0);
                float32x4_t _rows3 = vmulq_lane_f32(_S30, _a01, 0);
                // 最邻近第二行
                _rows0 = vfmaq_lane_f32(_rows0, _S01, _a01, 1);
                _rows1 = vfmaq_lane_f32(_rows1, _S11, _a01, 1);
                _rows2 = vfmaq_lane_f32(_rows2, _S21, _a01, 1);
                _rows3 = vfmaq_lane_f32(_rows3, _S31, _a01, 1);
                // 最邻近第三行
                _rows0 = vfmaq_lane_f32(_rows0, _S02, _a23, 0);
                _rows1 = vfmaq_lane_f32(_rows1, _S12, _a23, 0);
                _rows2 = vfmaq_lane_f32(_rows2, _S22, _a23, 0);
                _rows3 = vfmaq_lane_f32(_rows3, _S32, _a23, 0);
                // 最邻近第四行
                _rows0 = vfmaq_lane_f32(_rows0, _S03, _a23, 1);
                _rows1 = vfmaq_lane_f32(_rows1, _S13, _a23, 1);
                _rows2 = vfmaq_lane_f32(_rows2, _S23, _a23, 1);
                _rows3 = vfmaq_lane_f32(_rows3, _S33, _a23, 1);
                vst1q_f32(rows0p + dx * 4, _rows0);
                vst1q_f32(rows1p + dx * 4, _rows1);
                vst1q_f32(rows2p + dx * 4, _rows2);
                vst1q_f32(rows3p + dx * 4, _rows3);

                alphap += 4;
            }
        }

        prev_sy1 = sy;

        // vresize  列resize
        float32x2_t _b01 = vld1_f32(beta);
        float32x2_t _b23 = vld1_f32(beta + 2);

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* rows2p = rows2;
        float* rows3p = rows3;
        float* Dp = dst + dy*w;


        for (int dx = 0; dx < w; dx++)
        {
            float32x4_t _rows0 = vld1q_f32(rows0p);
            float32x4_t _rows1 = vld1q_f32(rows1p);
            float32x4_t _rows2 = vld1q_f32(rows2p);
            float32x4_t _rows3 = vld1q_f32(rows3p);
            // 列系数
            float32x4_t _D = vmulq_lane_f32(_rows0, _b01, 0);
            _D = vfmaq_lane_f32(_D, _rows1, _b01, 1);
            _D = vfmaq_lane_f32(_D, _rows2, _b23, 0);
            _D = vfmaq_lane_f32(_D, _rows3, _b23, 1);
            vst1q_f32(Dp, _D);

            Dp += 4;
            rows0p += 4;
            rows1p += 4;
            rows2p += 4;
            rows3p += 4;
        }

        beta += 4;
    }
}

static void du_resize_bicubic(float* src, int srcw, int srch, float* dst, int outw, int outh) {
    int* buf = new int[outw + outh + outw * 4 + outh * 4];

    int* xofs = buf;        //new int[outw];
    int* yofs = buf + outw; //new int[outh];

    float* alpha = (float*)(buf + outw + outh);           //new float[outw * 4];
    float* beta = (float*)(buf + outw + outh + outw * 4); //new float[outh * 4];

    cubic_coeffs(srcw, outw, xofs, alpha);
    cubic_coeffs(srch, outh, yofs, beta);

    int dst_stride = outw * outh;
    int src_stride = srcw * srch;
    for (int i = 0; i < 3; i++) {
        float* src_channel_data = src + src_stride * i;
        float* dst_channel_data = dst + dst_stride * i;

        du_resize_bicubic_image_fp32(src_channel_data, srcw, srch, dst_channel_data, outw, outh, alpha, xofs, beta, yofs);
    }

    delete[] buf;
}

