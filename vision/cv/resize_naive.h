#ifndef VISION_RESIZE_NAIVE_H
#define VISION_RESIZE_NAIVE_H

namespace va_cv {

class ResizeNaive {

public:
    static void resize_naive_inter_linear_u8(const char* src, int w_in, int h_in, int c,
                                             char* dst, int w_out, int h_out);

    static void resize_naive_inter_linear_fp32(const float* src, int w_in, int h_in, int c,
                                               float* dst, int w_out, int h_out);

    static void cubic_interpolate_naive(float fx, float* coeffs);

    static void cubic_coeffs_naive(int w_in, int w_out, int *xofs, float *alpha);

    static void resize_naive_inter_cubic_fp32_three_channel(float* src, int srcw, int srch,
                                                            float* dst, int outw, int outh,
                                                            float* alpha, int* xofs, float* beta, int* yofs);

    static void resize_naive_inter_cubic_fp32_one_channel(float* src, int srcw, int srch,
                                                            float* dst, int outw, int outh,
                                                            float* alpha, int* xofs, float* beta, int* yofs);

    static void resize_naive_inter_cubic_fp32_hwc(float* src, int w_in, int h_in,
                                                    float* dst, int w_out, int h_out);

    static void resize_naive_inter_cubic_fp32_chw(float* src, int w_in, int h_in, int c,
                                                    float* dst, int w_out, int h_out);

};

}

#endif //VISION_RESIZE_NAIVE_H
