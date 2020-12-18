#include "warp_affine_naive.h"

#include <math.h>

#include "../common/macro.h"

namespace va_cv {

void WarpAffineNaive::warp_affine_naive_hwc_u8(char* src, int w_in, int h_in, int c,
                                               char* dst, int w_out, int h_out, float* m) {

    int resize_coef_scale = 2048;

    float fx = 0.f;
    float fy = 0.f;
    int sx = 0;
    int sy = 0;
    short cbufy[2];
    short cbufx[2];

    for (int dy = 0; dy < h_out; dy++) {
        for (int dx = 0; dx < w_out; dx++) {
            fx = static_cast<float>(m[0] * dx + m[1] * dy + m[2]);
            fy = static_cast<float>(m[3] * dx + m[4] * dy + m[5]);

            sy = floor(fy);
            fy -= sy;
            if (sy < 0 || sy >= (h_in - 1)) {
                continue;
            }
            cbufy[0] = SATURATE_CAST_SHORT((1.f - fy) * resize_coef_scale);
            cbufy[1] = SATURATE_CAST_SHORT(resize_coef_scale - cbufy[0]);

            sx = floor(fx);
            fx -= sx;
            if (sx < 0 || sx >= (w_in - 1)) {
                continue;
            }

            cbufx[0] = SATURATE_CAST_SHORT((1.f - fx) * resize_coef_scale);
            cbufx[1] = SATURATE_CAST_SHORT(resize_coef_scale - cbufx[0]);

            int lt_ofs = (sy * w_in + sx) * c;
            int rt_ofs = (sy * w_in + sx + 1) * c;
            int lb_ofs = ((sy + 1) * w_in + sx) * c;
            int rb_ofs = ((sy + 1) * w_in + sx + 1) * c;
            int dst_ofs = (dy * w_out + dx) * c;

            for (int k = 0; k < c; k++) {
                *(dst + dst_ofs + k) =
                        (*(src + lt_ofs + k) * cbufx[0] * cbufy[0] +
                         *(src + lb_ofs + k) * cbufx[0] * cbufy[1] +
                         *(src + rt_ofs + k) * cbufx[1] * cbufy[0] +
                         *(src + rb_ofs + k) * cbufx[1] * cbufy[1]) >> 22;
            }
        }
    }
}

void WarpAffineNaive::warp_affine_naive_hwc_fp32(float* src, int w_in, int h_in, int c,
                                                float* dst, int w_out, int h_out, float* m) {

    float fx = 0.f;
    float fy = 0.f;
    int sx = 0;
    int sy = 0;
    float cbufy[2];
    float cbufx[2];

    for (int dy = 0; dy < h_out; dy++) {
        for (int dx = 0; dx < w_out; dx++) {
            fx = static_cast<float>(m[0] * dx + m[1] * dy + m[2]);
            fy = static_cast<float>(m[3] * dx + m[4] * dy + m[5]);
            sy = floor(fy);
            fy -= sy;
            if (sy < 0 || sy >= (h_in - 1)) {
                continue;
            }
            cbufy[0] = 1.f - fy;
            cbufy[1] = fy;

            sx = floor(fx);
            fx -= sx;
            if (sx < 0 || sx >= (w_in - 1)) {
                continue;
            }

            cbufx[0] = 1.f - fx;
            cbufx[1] = fx;

            int lt_ofs = (sy * w_in + sx) * c;
            int rt_ofs = (sy * w_in + sx + 1) * c;
            int lb_ofs = ((sy + 1) * w_in + sx) * c;
            int rb_ofs = ((sy + 1) * w_in + sx + 1) * c;
            int dst_ofs = (dy * w_out + dx) * c;

            for (int k = 0; k < c; k++) {
                *(dst + dst_ofs + k) =
                         *(src + lt_ofs + k) * cbufx[0] * cbufy[0] +
                         *(src + lb_ofs + k) * cbufx[0] * cbufy[1] +
                         *(src + rt_ofs + k) * cbufx[1] * cbufy[0] +
                         *(src + rb_ofs + k) * cbufx[1] * cbufy[1];
            }
        }
    }
}

}