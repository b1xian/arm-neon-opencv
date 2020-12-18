#ifndef VISION_NORMALIZE_H
#define VISION_NORMALIZE_H

#include "../common/tensor.h"

namespace va_cv {

class Normalize {
public:
    static void normalize(const vision::Tensor& src, vision::Tensor& dst,
                          const vision::Tensor& mean, const vision::Tensor& stddev);

private:
    static void normalize_opencv(const vision::Tensor& src, vision::Tensor& dst,
                                 const vision::Tensor& mean, const vision::Tensor& stddev);

    static void normalize_naive(const vision::Tensor& src, vision::Tensor& dst,
                                const vision::Tensor& mean, const vision::Tensor& stddev);

    static void normalize_sse(const vision::Tensor& src, vision::Tensor& dst,
                              const vision::Tensor& mean, const vision::Tensor& stddev);

    static void normalize_neon(const vision::Tensor& src, vision::Tensor& dst,
                               const vision::Tensor& mean, const vision::Tensor& stddev);
};

} // namespace va_cv

#endif //VISION_NORMALIZE_H
