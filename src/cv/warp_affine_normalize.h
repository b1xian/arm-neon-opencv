#ifndef VISION_WARP_AFFINE_NORMALIZE_H
#define VISION_WARP_AFFINE_NORMALIZE_H

#include "../common/tensor.h"
#include "cv.h"

namespace va_cv {

class WarpAffineNormalize {
public:
    static void warp_affine_normalize(const vision::Tensor& src, vision::Tensor& dst,
                                      const vision::Tensor& M, VSize dsize,
                                      int flags = INTER_LINEAR,
                                      int borderMode = BORDER_CONSTANT,
                                      const VScalar& borderValue = VScalar(),
                                      const vision::Tensor& mean = vision::Tensor(),
                                      const vision::Tensor& stddev = vision::Tensor());

    static void warp_affine_normalize(const vision::Tensor& src, vision::Tensor& dst,
                                      float scale, float rot, VSize dsize,
                                      const VScalar& aux_param = VScalar(),
                                      int flags = INTER_LINEAR,
                                      int borderMode = BORDER_CONSTANT,
                                      const VScalar& borderValue = VScalar(),
                                      const vision::Tensor& mean = vision::Tensor(),
                                      const vision::Tensor& stddev = vision::Tensor());

private:
    static void warp_affine_normalize_opencv(const vision::Tensor& src, vision::Tensor& dst,
                                             const vision::Tensor& M, VSize dsize,
                                             int flags = INTER_LINEAR,
                                             int borderMode = BORDER_CONSTANT,
                                             const VScalar& borderValue = VScalar(),
                                             const vision::Tensor& mean = vision::Tensor(),
                                             const vision::Tensor& stddev = vision::Tensor());

    static void warp_affine_normalize_naive(const vision::Tensor& src, vision::Tensor& dst,
                                            const vision::Tensor& M, VSize dsize,
                                            int flags = INTER_LINEAR,
                                            int borderMode = BORDER_CONSTANT,
                                            const VScalar& borderValue = VScalar(),
                                            const vision::Tensor& mean = vision::Tensor(),
                                            const vision::Tensor& stddev = vision::Tensor());

    static void warp_affine_normalize_sse(const vision::Tensor& src, vision::Tensor& dst,
                                          const vision::Tensor& M, VSize dsize,
                                          int flags = INTER_LINEAR,
                                          int borderMode = BORDER_CONSTANT,
                                          const VScalar& borderValue = VScalar(),
                                          const vision::Tensor& mean = vision::Tensor(),
                                          const vision::Tensor& stddev = vision::Tensor());

    static void warp_affine_normalize_neon(const vision::Tensor& src, vision::Tensor& dst,
                                           const vision::Tensor& M, VSize dsize,
                                           int flags = INTER_LINEAR,
                                           int borderMode = BORDER_CONSTANT,
                                           const VScalar& borderValue = VScalar(),
                                           const vision::Tensor& mean = vision::Tensor(),
                                           const vision::Tensor& stddev = vision::Tensor());

    static void warp_affine_normalize_opencv(const vision::Tensor& src, vision::Tensor& dst,
                                             float scale, float rot, VSize dsize,
                                             const VScalar& aux_param = VScalar(),
                                             int flags = INTER_LINEAR,
                                             int borderMode = BORDER_CONSTANT,
                                             const VScalar& borderValue = VScalar(),
                                             const vision::Tensor& mean = vision::Tensor(),
                                             const vision::Tensor& stddev = vision::Tensor());
};

} // namespace va_cv

#endif //VISION_WARP_AFFINE_NORMALIZE_H
