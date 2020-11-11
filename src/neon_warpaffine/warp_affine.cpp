//
// Created by b1xian on 2020-11-11.
//
#include <arm_neon.h>
#include <iostream>
#include <math.h>

static void warp_affine_naive(const uint8_t* src,
                             int w_in,
                             int h_in,
                             uint8_t* dst,
                             int w_out,
                             int h_out,
                             float* m) {

#define SATURATE_CAST_SHORT(X)                                               \
  (int16_t)::std::min(                                                       \
      ::std::max(static_cast<int>(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), \
      SHRT_MAX);

    int resize_coef_scale = 2048;

    float fx = 0.f;
    float fy = 0.f;
    int sx = 0;
    int sy = 0;

    for (int dy = 0; dy < h_out; dy++) {
        for (int dx = 0; dx < w_out; dx++) {
            fx = static_cast<float>(m[0]*dx + m[1]*dy + m[2]);
            sx = floor(fx);
            fx -= sx;
            if (sx < 0 || sx >= w_in) {
                continue;
            }
            float a0 = SATURATE_CAST_SHORT((1.f - fx) * resize_coef_scale);
            float a1 = SATURATE_CAST_SHORT(fx * resize_coef_scale);
            int16_t a0i = SATURATE_CAST_SHORT(a0);
            int16_t a1i = SATURATE_CAST_SHORT(a1);

            fy = static_cast<float>(m[3]*dx + m[4]*dy + m[5]);
            sy = floor(fy);
            fy -= sy;
            if (sy < 0 || sy >= h_in) {
                continue;
            }
            float b0 = (1.f - fy) * resize_coef_scale;
            float b1 = fy * resize_coef_scale;
            int16_t b0i = SATURATE_CAST_SHORT(b0);
            int16_t b1i = SATURATE_CAST_SHORT(b1);
            *(dst + dy*w_out + dx) = (*(src + sy*w_in + sx) * a0i * b0i +
                                    *(src + (sy+1)*w_in + sx) * a0i * b1i +
                                    *(src + sy*w_in + sx+1) * a1i * b0i +
                                    *(src + (sy+1)*w_in + sx+1) * a1i * b1i ) >> 22;
        }
    }

}