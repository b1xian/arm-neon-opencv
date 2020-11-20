//
// Created by b1xian on 2020-11-11.
//
#include <arm_neon.h>
#include <iostream>
#include <math.h>

#define SATURATE_CAST_SHORT(X)                                               \
  (int16_t)::std::min(                                                       \
      ::std::max(static_cast<int>(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), \
      SHRT_MAX);

static void warp_affine_naive_hwc(uint8_t* src,
                                  int w_in,
                                  int h_in,
                                  int c,
                                   uint8_t* dst,
                                  int w_out,
                                  int h_out,
                                  float* m) {

    int resize_coef_scale = 2048;

    float fx = 0.f;
    float fy = 0.f;
    int sx = 0;
    int sy = 0;
    int16_t cbufy[2];
    int16_t cbufx[2];

    for (int dy = 0; dy < h_out; dy++) {
        for (int dx = 0; dx < w_out; dx++) {
            fx = static_cast<float>(m[0]*dx + m[1]*dy + m[2]);
            fy = static_cast<float>(m[3]*dx + m[4]*dy + m[5]);

            sy  = floor(fy);
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
