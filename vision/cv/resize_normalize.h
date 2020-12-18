#ifndef VISION_RESIZE_NORMALIZE_H
#define VISION_RESIZE_NORMALIZE_H

#include "../common/tensor.h"
#include "cv.h"

namespace va_cv {

class ResizeNormalize {
public:
    static void resize_normalize(const vision::Tensor& src, vision::Tensor& dst,
                                 VSize dsize, double fx = 0, double fy = 0,
                                 int interpolation = INTER_LINEAR,
                                 const vision::Tensor& mean = vision::Tensor(),
                                 const vision::Tensor& stddev = vision::Tensor());

private:
    static void resize_normalize_opencv(const vision::Tensor& src, vision::Tensor& dst,
                                        VSize dsize, double fx = 0, double fy = 0,
                                        int interpolation = INTER_LINEAR,
                                        const vision::Tensor& mean = vision::Tensor(),
                                        const vision::Tensor& stddev = vision::Tensor());

    static void resize_normalize_naive(const vision::Tensor& src, vision::Tensor& dst,
                                       VSize dsize, double fx = 0, double fy = 0,
                                       int interpolation = INTER_LINEAR,
                                       const vision::Tensor& mean = vision::Tensor(),
                                       const vision::Tensor& stddev = vision::Tensor());

    static void resize_normalize_sse(const vision::Tensor& src, vision::Tensor& dst,
                                     VSize dsize, double fx = 0, double fy = 0,
                                     int interpolation = INTER_LINEAR,
                                     const vision::Tensor& mean = vision::Tensor(),
                                     const vision::Tensor& stddev = vision::Tensor());

    static void resize_normalize_neon(const vision::Tensor& src, vision::Tensor& dst,
                                      VSize dsize, double fx = 0, double fy = 0,
                                      int interpolation = INTER_LINEAR,
                                      const vision::Tensor& mean = vision::Tensor(),
                                      const vision::Tensor& stddev = vision::Tensor());
};

} // namespace va_cv

#endif //VISION_RESIZE_NORMALIZE_H
