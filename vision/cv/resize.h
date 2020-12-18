#ifndef VISION_RESIZE_H
#define VISION_RESIZE_H

#include "cv.h"
#include "../common/tensor.h"

namespace va_cv {

class Resize {
public:
    static void resize(const vision::Tensor& src, vision::Tensor& dst,
                       VSize dsize, double fx = 0, double fy = 0,
                       int interpolation = INTER_LINEAR);

private:
    static void resize_opencv(const vision::Tensor& src, vision::Tensor& dst,
                              VSize dsize, double fx = 0, double fy = 0,
                              int interpolation = INTER_LINEAR);

    static void resize_naive(const vision::Tensor& src, vision::Tensor& dst,
                             VSize dsize, double fx = 0, double fy = 0,
                             int interpolation = INTER_LINEAR);

    static void resize_sse(const vision::Tensor& src, vision::Tensor& dst,
                           VSize dsize, double fx = 0, double fy = 0,
                           int interpolation = INTER_LINEAR);

    static void resize_neon(const vision::Tensor& src, vision::Tensor& dst,
                            VSize dsize, double fx = 0, double fy = 0,
                            int interpolation = INTER_LINEAR);
};

} // namespace vision

#endif //VISION_RESIZE_H
