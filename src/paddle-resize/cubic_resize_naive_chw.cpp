//
// Created by v_guojinlong on 2020-10-30.
//

// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <math.h>

#include <iostream>


static void du_interpolate_cubic_naive(float fx, float* coeffs)
{
    const float A = -0.75f;

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

static void du_cubic_coeffs_naive(int w, int outw, int *xofs, float *alpha)
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
        du_interpolate_cubic_naive(fx, alpha + dx * 4);

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

static void du_resize_bicubic_image_one_channel_naive(float* src, int srcw, int srch, float* dst, int outw, int outh,
                                          float* alpha, int* xofs, float* beta, int* yofs)
{
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
                int sx = xofs[dx];
                const float* S3p = S3 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];
                rows3p[dx] = S3p[-1] * a0 + S3p[0] * a1 + S3p[1] * a2 + S3p[2] * a3;

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
                int sx = xofs[dx];
                const float* S2p = S2 + sx;
                const float* S3p = S3 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];
                rows2p[dx] = S2p[-1] * a0 + S2p[0] * a1 + S2p[1] * a2 + S2p[2] * a3;
                rows3p[dx] = S3p[-1] * a0 + S3p[0] * a1 + S3p[1] * a2 + S3p[2] * a3;

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
                int sx = xofs[dx];
                const float* S1p = S1 + sx;
                const float* S2p = S2 + sx;
                const float* S3p = S3 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];
                rows1p[dx] = S1p[-1] * a0 + S1p[0] * a1 + S1p[1] * a2 + S1p[2] * a3;
                rows2p[dx] = S2p[-1] * a0 + S2p[0] * a1 + S2p[1] * a2 + S2p[2] * a3;
                rows3p[dx] = S3p[-1] * a0 + S3p[0] * a1 + S3p[1] * a2 + S3p[2] * a3;

                alphap += 4;
            }
        }
        else
        {
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
                int sx = xofs[dx];
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;
                const float* S2p = S2 + sx;
                const float* S3p = S3 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];
                rows0p[dx] = S0p[-1] * a0 + S0p[0] * a1 + S0p[1] * a2 + S0p[2] * a3;
                rows1p[dx] = S1p[-1] * a0 + S1p[0] * a1 + S1p[1] * a2 + S1p[2] * a3;
                rows2p[dx] = S2p[-1] * a0 + S2p[0] * a1 + S2p[1] * a2 + S2p[2] * a3;
                rows3p[dx] = S3p[-1] * a0 + S3p[0] * a1 + S3p[1] * a2 + S3p[2] * a3;

                alphap += 4;
            }
        }

        prev_sy1 = sy;

        // vresize
        float b0 = beta[0];
        float b1 = beta[1];
        float b2 = beta[2];
        float b3 = beta[3];

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* rows2p = rows2;
        float* rows3p = rows3;
        float* Dp = dst + dy*w;
        for (int dx = 0; dx < w; dx++)
        {
            //             D[x] = rows0[x]*b0 + rows1[x]*b1 + rows2[x]*b2 + rows3[x]*b3;
            *Dp++ = *rows0p++ * b0 + *rows1p++ * b1 + *rows2p++ * b2 + *rows3p++ * b3;
        }

        beta += 4;
    }
}

static void du_chw_resize_bicubic_naive(float* src, int srcw, int srch, float* dst, int outw, int outh) {
    int* buf = new int[outw + outh + outw * 4 + outh * 4];

    int* xofs = buf;        //new int[outw];
    int* yofs = buf + outw; //new int[outh];

    float* alpha = (float*)(buf + outw + outh);           //new float[outw * 4];
    float* beta = (float*)(buf + outw + outh + outw * 4); //new float[outh * 4];

    du_cubic_coeffs_naive(srcw, outw, xofs, alpha);
    du_cubic_coeffs_naive(srch, outh, yofs, beta);


    int dst_stride = outw * outh;
    int src_stride = srcw * srch;
    // TODO 可以使用多线程并行执行三通道
    #pragma omp parallel for
    for (int i = 0; i < 3; i++) {
        float* src_channel_data = src + src_stride * i;
        float* dst_channel_data = dst + dst_stride * i;
        du_resize_bicubic_image_one_channel_naive(src_channel_data, srcw, srch, dst_channel_data, outw, outh, alpha, xofs, beta, yofs);
    }

    delete[] buf;
}
static void du_resize_bicubic_image_naive_three_channel(float* src, int srcw, int srch, float* dst, int outw, int outh,
                                                        float* alpha, int* xofs, float* beta, int* yofs)
{
    int w = outw;
    int h = outh;

    // loop body
    // 每次操作一行
    float* rowsbuf0 = (float*)malloc(sizeof(float) * w * 4 * 3);
    float* rowsbuf1 = (float*)malloc(sizeof(float) * w * 4 * 3);
    float* rowsbuf2 = (float*)malloc(sizeof(float) * w * 4 * 3);
    float* rowsbuf3 = (float*)malloc(sizeof(float) * w * 4 * 3);
    float* rows0 = rowsbuf0;
    float* rows1 = rowsbuf1;
    float* rows2 = rowsbuf2;
    float* rows3 = rowsbuf3;

    int prev_sy1 = -3;

    for (int dy = 0; dy < h; dy++) {
        int sy = yofs[dy];

        if (sy == prev_sy1) {
            // reuse all rows
        } else if (sy == prev_sy1 + 1) {
            // hresize one row
            float* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows2;
            rows2 = rows3;
            rows3 = rows0_old;
            float* S3 = src + (sy+2)*srcw*3;

            const float* alphap = alpha;
            float* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S3p = S3 + sx*3;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];

                for (int i = 0; i < 3; i++){
                    rows3p[dx*3 +i] = S3p[-1*3+i] * a0 + S3p[0*3+i] * a1 + S3p[1*3+i] * a2 + S3p[2*3+i] * a3;
                }

                alphap += 4;
            }
        } else if (sy == prev_sy1 + 2) {
            // hresize two rows
            float* rows0_old = rows0;
            float* rows1_old = rows1;
            rows0 = rows2;
            rows1 = rows3;
            rows2 = rows0_old;
            rows3 = rows1_old;
            float* S2 = src + (sy+1)*srcw * 3;
            float* S3 = src + (sy+2)*srcw * 3;

            const float* alphap = alpha;
            float* rows2p = rows2;
            float* rows3p = rows3;
            for (int dx = 0; dx < w; dx++) {
                int sx = xofs[dx];
                const float* S2p = S2 + sx*3;
                const float* S3p = S3 + sx*3;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];

                for (int i = 0; i < 3; i++){
                    rows2p[dx*3 +i] = S2p[-1*3+i] * a0 + S2p[0*3+i] * a1 + S2p[1*3+i] * a2 + S2p[2*3+i] * a3;
                    rows3p[dx*3 +i] = S3p[-1*3+i] * a0 + S3p[0*3+i] * a1 + S3p[1*3+i] * a2 + S3p[2*3+i] * a3;
                }

                alphap += 4;
            }
        } else if (sy == prev_sy1 + 3) {
            // hresize three rows
            float* rows0_old = rows0;
            float* rows1_old = rows1;
            float* rows2_old = rows2;
            rows0 = rows3;
            rows1 = rows0_old;
            rows2 = rows1_old;
            rows3 = rows2_old;
            float* S1 = src + (sy)*srcw * 3;
            float* S2 = src + (sy+1)*srcw * 3;
            float* S3 = src + (sy+2)*srcw * 3;

            const float* alphap = alpha;
            float* rows1p = rows1;
            float* rows2p = rows2;
            float* rows3p = rows3;
            for (int dx = 0; dx < w; dx++) {
                int sx = xofs[dx];
                const float* S1p = S1 + sx*3;
                const float* S2p = S2 + sx*3;
                const float* S3p = S3 + sx*3;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];

                for (int i = 0; i < 3; i++){
                    rows1p[dx*3 +i] = S1p[-1*3+i] * a0 + S1p[0*3+i] * a1 + S1p[1*3+i] * a2 + S1p[2*3+i] * a3;
                    rows2p[dx*3 +i] = S2p[-1*3+i] * a0 + S2p[0*3+i] * a1 + S2p[1*3+i] * a2 + S2p[2*3+i] * a3;
                    rows3p[dx*3 +i] = S3p[-1*3+i] * a0 + S3p[0*3+i] * a1 + S3p[1*3+i] * a2 + S3p[2*3+i] * a3;
                }

                alphap += 4;
            }
        } else {
            // hresize four rows
            float* S0 = src + (sy-1)*srcw * 3;
            float* S1 = src + (sy)*srcw * 3;
            float* S2 = src + (sy+1)*srcw * 3;
            float* S3 = src + (sy+2)*srcw * 3;

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;
            float* rows2p = rows2;
            float* rows3p = rows3;
            for (int dx = 0; dx < w; dx++) {
                int sx = xofs[dx];
                const float* S0p = S0 + sx*3;
                const float* S1p = S1 + sx*3;
                const float* S2p = S2 + sx*3;
                const float* S3p = S3 + sx*3;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];
                for (int i = 0; i < 3; i++){
                    rows0p[dx*3 +i] = S0p[-1*3+i] * a0 + S0p[0*3+i] * a1 + S0p[1*3+i] * a2 + S0p[2*3+i] * a3;
                    rows1p[dx*3 +i] = S1p[-1*3+i] * a0 + S1p[0*3+i] * a1 + S1p[1*3+i] * a2 + S1p[2*3+i] * a3;
                    rows2p[dx*3 +i] = S2p[-1*3+i] * a0 + S2p[0*3+i] * a1 + S2p[1*3+i] * a2 + S2p[2*3+i] * a3;
                    rows3p[dx*3 +i] = S3p[-1*3+i] * a0 + S3p[0*3+i] * a1 + S3p[1*3+i] * a2 + S3p[2*3+i] * a3;
                }

                alphap += 4;
            }
        }

        prev_sy1 = sy;

        // vresize
        float b0 = beta[0];
        float b1 = beta[1];
        float b2 = beta[2];
        float b3 = beta[3];

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* rows2p = rows2;
        float* rows3p = rows3;
        float* Dp = dst + dy*w*3;
        for (int dx = 0; dx < w; dx++) {
            //             D[x] = rows0[x]*b0 + rows1[x]*b1 + rows2[x]*b2 + rows3[x]*b3;
//            *Dp++ = *rows0p++ * b0 + *rows1p++ * b1 + *rows2p++ * b2 + *rows3p++ * b3;
            // b
            *(Dp) = *rows0p * b0 + *rows1p * b1 + *rows2p * b2 + *rows3p * b3;
            // g
            *(Dp+1) = *(rows0p+1) * b0 + *(rows1p+1) * b1 + *(rows2p+1) * b2 + *(rows3p+1) * b3;
            // r
            *(Dp+2) = *(rows0p+2) * b0 + *(rows1p+2) * b1 + *(rows2p+2) * b2 + *(rows3p+2) * b3;

            Dp+=3;
            rows0p+=3;
            rows1p+=3;
            rows2p+=3;
            rows3p+=3;
        }

        beta += 4;
    }
}

static void du_hwc_resize_bicubic_naive(float* src, int srcw, int srch, float* dst, int outw, int outh) {
    int* buf = new int[outw + outh + outw * 4 + outh * 4];

    int* xofs = buf;        //new int[outw];
    int* yofs = buf + outw; //new int[outh];

    float* alpha = (float*)(buf + outw + outh);           //new float[outw * 4];
    float* beta = (float*)(buf + outw + outh + outw * 4); //new float[outh * 4];

    du_cubic_coeffs_naive(srcw, outw, xofs, alpha);
    du_cubic_coeffs_naive(srch, outh, yofs, beta);

    du_resize_bicubic_image_naive_three_channel(src, srcw, srch, dst, outw, outh, alpha, xofs, beta, yofs);

    delete[] buf;
}
