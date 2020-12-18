#ifndef VISION_NORMALIZE_NAIVE_H
#define VISION_NORMALIZE_NAIVE_H

namespace va_cv {

class NormalizeNaive {

public:

    static void mean_stddev_naive_hwc_bgr(float *src, int stride, float *mean, float *stddev);

    static void mean_stddev_naive_chw(float* src, int stride, int c, float* mean, float* stddev);

    static void normalize_naive_hwc_bgr(float* src, float* dst, int stride, float* mean, float* stddev);

    static void normalize_naive_chw(float* src, float* dst, int stride, int c, float* mean, float* stddev);

};

}

#endif //VISION_NORMALIZE_NAIVE_H
