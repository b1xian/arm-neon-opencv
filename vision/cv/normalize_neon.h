#ifndef VISION_NORMALIZE_NEON_H
#define VISION_NORMALIZE_NEON_H

namespace va_cv {

class NormalizeNeon {

public:

    static void mean_stddev_neon_hwc_bgr(float* src, int stride, float* mean, float* stddev);

    static void mean_stddev_neon_chw(float* src, int stride, int c, float* mean, float* stddev);

    static void normalize_neon_hwc_bgr(float* src, float* dst, int stride, float* mean, float* stddev);

    static void normalize_neon_chw(float* src, float* dst, int stride, int c, float* mean, float* stddev);

};

}

#endif //VISION_NORMALIZE_NEON_H
