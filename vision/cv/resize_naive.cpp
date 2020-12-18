#include "resize_naive.h"

#include <cstdlib>
#include <math.h>

#include "../common/macro.h"

namespace va_cv {

void ResizeNaive::resize_naive_inter_linear_u8(const char* src,
                                          int w_in,
                                          int h_in,
                                          int c,
                                          char* dst,
                                          int w_out,
                                          int h_out) {
    float scale_x = static_cast<float>(w_in) / w_out;
    float scale_y = static_cast<float>(h_in) / h_out;

    for (int dy = 0; dy < h_out; dy++) {
        float fy = static_cast<float>((dy + 0.5) * scale_y - 0.5);
        int sy = floor(fy);
        fy -= sy;
        if (sy < 0) {
            sy = 0;
            fy = 0.f;
        }
        if (sy >= h_in - 1) {
            sy = h_in - 2;
            fy = 1.f;
        }

        short cbufy[2];
        cbufy[0] = SATURATE_CAST_SHORT((1.f - fy) * 2048);
        cbufy[1] = SATURATE_CAST_SHORT(2048 * fy);

        for (int dx = 0; dx < w_out; dx++) {
            float fx = static_cast<float>((dx + 0.5) * scale_x - 0.5);
            int sx = floor(fx);
            fx -= sx;

            if (sx < 0) {
                sx = 0;
                fx = 0.f;
            }
            if (sx >= w_in - 1) {
                sx = w_in - 2;
                fx = 1.f;
            }

            short cbufx[2];
            cbufx[0] = SATURATE_CAST_SHORT((1.f - fx) * 2048);
            cbufx[1] = SATURATE_CAST_SHORT(2048 * fx);

            int lt_ofs = (sy * w_in + sx) * c;
            int rt_ofs = (sy * w_in + sx + 1) * c;
            int lb_ofs = ((sy + 1) * w_in + sx) * c;
            int rb_ofs = ((sy + 1) * w_in + sx + 1) * c;
            int dst_ofs = (dy * w_out + dx) * c;
            for (int k = 0; k < c; k++) {
                *(dst + dst_ofs + k) = (*(src + lt_ofs + k) * cbufx[0] * cbufy[0] +
                                        *(src + lb_ofs + k) * cbufx[0] * cbufy[1] +
                                        *(src + rt_ofs + k) * cbufx[1] * cbufy[0] +
                                        *(src + rb_ofs + k) * cbufx[1] * cbufy[1]) >> 22;
            }
        }
    }
}

void ResizeNaive::resize_naive_inter_linear_fp32(const float* src,
                                            int w_in,
                                            int h_in,
                                            int c,
                                            float* dst,
                                            int w_out,
                                            int h_out) {
    float scale_x = static_cast<float>(w_in) / w_out;
    float scale_y = static_cast<float>(h_in) / h_out;

    for (int dy = 0; dy < h_out; dy++) {
        float fy = static_cast<float>((dy + 0.5) * scale_y - 0.5);
        int sy = floor(fy);
        fy -= sy;
        if (sy < 0) {
            sy = 0;
            fy = 0.f;
        }
        if (sy >= h_in - 1) {
            sy = h_in - 2;
            fy = 1.f;
        }

        float cbufy[2];
        cbufy[0] = 1.f - fy;
        cbufy[1] = fy;

        for (int dx = 0; dx < w_out; dx++) {
            float fx = static_cast<float>((dx + 0.5) * scale_x - 0.5);
            int sx = floor(fx);
            fx -= sx;

            if (sx < 0) {
                sx = 0;
                fx = 0.f;
            }
            if (sx >= w_in - 1) {
                sx = w_in - 2;
                fx = 1.f;
            }

            float cbufx[2];
            cbufx[0] = 1.f - fx;
            cbufx[1] = fx;

            int lt_ofs = (sy * w_in + sx) * c;
            int rt_ofs = (sy * w_in + sx + 1) * c;
            int lb_ofs = ((sy + 1) * w_in + sx) * c;
            int rb_ofs = ((sy + 1) * w_in + sx + 1) * c;
            int dst_ofs = (dy * w_out + dx) * c;
            for (int k = 0; k < c; k++) {
                *(dst + dst_ofs + k) = *(src + lt_ofs + k) * cbufx[0] * cbufy[0] +
                                       *(src + lb_ofs + k) * cbufx[0] * cbufy[1] +
                                       *(src + rt_ofs + k) * cbufx[1] * cbufy[0] +
                                       *(src + rb_ofs + k) * cbufx[1] * cbufy[1];
            }
        }
    }
}

void ResizeNaive::cubic_interpolate_naive(float fx, float* coeffs) {
    const float A = -0.75f;

    float fx0 = fx + 1;
    float fx1 = fx;
    float fx2 = 1 - fx;

    coeffs[0] = A * fx0 * fx0 * fx0 - 5 * A * fx0 * fx0 + 8 * A * fx0 - 4 * A;
    coeffs[1] = (A + 2) * fx1 * fx1 * fx1 - (A + 3) * fx1 * fx1 + 1;
    coeffs[2] = (A + 2) * fx2 * fx2 * fx2 - (A + 3) * fx2 * fx2 + 1;
    coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
}

void ResizeNaive::cubic_coeffs_naive(int w_in, int w_out, int *xofs, float *alpha) {
    double scale = (double)w_in / w_out;

    // 计算系数
    for (int dx = 0; dx < w_out; dx++) {
        float fx = (float)((dx + 0.5) * scale - 0.5);
        int sx = static_cast<int>(floor(fx));
        fx -= sx;

        cubic_interpolate_naive(fx, alpha + dx * 4);

        if (sx <= -1) {
            sx = 1;
            alpha[dx * 4 + 0] = 1.f - alpha[dx * 4 + 3];
            alpha[dx * 4 + 1] = alpha[dx * 4 + 3];
            alpha[dx * 4 + 2] = 0.f;
            alpha[dx * 4 + 3] = 0.f;
        }
        if (sx == 0) {
            sx = 1;
            alpha[dx * 4 + 0] = alpha[dx * 4 + 0] + alpha[dx * 4 + 1];
            alpha[dx * 4 + 1] = alpha[dx * 4 + 2];
            alpha[dx * 4 + 2] = alpha[dx * 4 + 3];
            alpha[dx * 4 + 3] = 0.f;
        }
        if (sx == w_in - 2) {
            sx = w_in - 3;
            alpha[dx * 4 + 3] = alpha[dx * 4 + 2] + alpha[dx * 4 + 3];
            alpha[dx * 4 + 2] = alpha[dx * 4 + 1];
            alpha[dx * 4 + 1] = alpha[dx * 4 + 0];
            alpha[dx * 4 + 0] = 0.f;
        }
        if (sx >= w_in - 1) {
            sx = w_in - 3;
            alpha[dx * 4 + 3] = 1.f - alpha[dx * 4 + 0];
            alpha[dx * 4 + 2] = alpha[dx * 4 + 0];
            alpha[dx * 4 + 1] = 0.f;
            alpha[dx * 4 + 0] = 0.f;
        }

        xofs[dx] = sx;
    }
}

void ResizeNaive::resize_naive_inter_cubic_fp32_three_channel(float* src, int srcw, int srch, float* dst, int outw, int outh,
                                                 float* alpha, int* xofs, float* beta, int* yofs) {
    int w = outw;
    int h = outh;

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
            *(Dp) = *rows0p * b0 + *rows1p * b1 + *rows2p * b2 + *rows3p * b3;
            *(Dp+1) = *(rows0p+1) * b0 + *(rows1p+1) * b1 + *(rows2p+1) * b2 + *(rows3p+1) * b3;
            *(Dp+2) = *(rows0p+2) * b0 + *(rows1p+2) * b1 + *(rows2p+2) * b2 + *(rows3p+2) * b3;

            Dp+=3;
            rows0p+=3;
            rows1p+=3;
            rows2p+=3;
            rows3p+=3;
        }

        beta += 4;
    }
    free(rowsbuf0);
    free(rowsbuf1);
    free(rowsbuf2);
    free(rowsbuf3);
}

void ResizeNaive::resize_naive_inter_cubic_fp32_one_channel(float* src, int srcw, int srch, float* dst, int outw, int outh,
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
        } else if (sy == prev_sy1 + 2) {
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
            for (int dx = 0; dx < w; dx++) {
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
        } else if (sy == prev_sy1 + 3) {
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
            for (int dx = 0; dx < w; dx++) {
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
        } else {
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
            for (int dx = 0; dx < w; dx++) {
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
        for (int dx = 0; dx < w; dx++) {
            *Dp++ = *rows0p++ * b0 + *rows1p++ * b1 + *rows2p++ * b2 + *rows3p++ * b3;
        }
        beta += 4;
    }
    free(rowsbuf0);
    free(rowsbuf1);
    free(rowsbuf2);
    free(rowsbuf3);
}

void ResizeNaive::resize_naive_inter_cubic_fp32_hwc(float* src, int w_in, int h_in, float* dst, int w_out, int h_out) {
    int* buf = new int[w_out + h_out + w_out * 4 + h_out * 4];

    int* xofs = buf;        //new int[w_out];
    int* yofs = buf + w_out; //new int[h_out];

    float* alpha = (float*)(buf + h_out + h_out);           //new float[w_out * 4];
    float* beta = (float*)(buf + h_out + h_out + w_out * 4); //new float[h_out * 4];

    cubic_coeffs_naive(w_in, w_out, xofs, alpha);
    cubic_coeffs_naive(h_in, h_out, yofs, beta);

    resize_naive_inter_cubic_fp32_three_channel(src, w_in, h_in, dst, w_out, h_out, alpha, xofs, beta, yofs);
    delete[] buf;
}

void ResizeNaive::resize_naive_inter_cubic_fp32_chw(float* src, int w_in, int h_in, int c, float* dst, int w_out, int h_out) {
    int* buf = new int[w_out + h_out + w_out * 4 + h_out * 4];

    int* xofs = buf;        //new int[w_out];
    int* yofs = buf + w_out; //new int[h_out];

    float* alpha = (float*)(buf + h_out + h_out);           //new float[w_out * 4];
    float* beta = (float*)(buf + h_out + h_out + w_out * 4); //new float[h_out * 4];

    cubic_coeffs_naive(w_in, w_out, xofs, alpha);
    cubic_coeffs_naive(h_in, h_out, yofs, beta);


    int dst_stride = w_out * h_out;
    int src_stride = w_in * h_in;
#pragma omp parallel for
    for (int i = 0; i < c; i++) {
        float* src_channel_data = src + src_stride * i;
        float* dst_channel_data = dst + dst_stride * i;
        resize_naive_inter_cubic_fp32_one_channel(src_channel_data, w_in, h_in, dst_channel_data, w_out, h_out, alpha, xofs, beta, yofs);
    }
    delete[] buf;
}


}