#ifndef VISION_WARP_AFFINE_NAIVE_H
#define VISION_WARP_AFFINE_NAIVE_H

namespace va_cv {

class WarpAffineNaive {

public:

    static void warp_affine_naive_hwc_u8(char* src, int w_in, int h_in, int c,
                                         char* dst, int w_out, int h_out, float* m);

    static void warp_affine_naive_hwc_fp32(float* src, int w_in, int h_in, int c,
                                        float* dst, int w_out, int h_out, float* m);

};

}

#endif //VISION_WARP_AFFINE_NAIVE_H
