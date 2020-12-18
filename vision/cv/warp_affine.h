#ifndef VISION_WARP_AFFINE_H
#define VISION_WARP_AFFINE_H

#include "cv.h"
#include "../common/tensor.h"

namespace va_cv {

class WarpAffine {
public:
    static void warp_affine(const vision::Tensor& src, vision::Tensor& dst,
                            const vision::Tensor& M, VSize dsize,
                            int flags = INTER_LINEAR,
                            int borderMode = BORDER_CONSTANT,
                            const VScalar& borderValue = VScalar());

    static void warp_affine(const vision::Tensor& src, vision::Tensor& dst,
                            float scale, float rot, VSize dsize,
                            const VScalar& aux_param = VScalar(),
                            int flags = INTER_LINEAR,
                            int borderMode = BORDER_CONSTANT,
                            const VScalar& borderValue = VScalar());

private:
    static void warp_affine_opencv(const vision::Tensor& src, vision::Tensor& dst,
                                   const vision::Tensor& M, VSize dsize,
                                   int flags = INTER_LINEAR,
                                   int borderMode = BORDER_CONSTANT,
                                   const VScalar& borderValue = VScalar());

    static vision::Tensor get_rotation_matrix_2D(const vision::VPoint& point,
                                                 float angle, float scale);

    static void warp_affine_naive(const vision::Tensor& src, vision::Tensor& dst,
                                  const vision::Tensor& M, VSize dsize,
                                  int flags = INTER_LINEAR,
                                  int borderMode = BORDER_CONSTANT,
                                  const VScalar& borderValue = VScalar());

    static void warp_affine_sse(const vision::Tensor& src, vision::Tensor& dst,
                                const vision::Tensor& M, VSize dsize,
                                int flags = INTER_LINEAR,
                                int borderMode = BORDER_CONSTANT,
                                const VScalar& borderValue = VScalar());

    static void warp_affine_neon(const vision::Tensor& src, vision::Tensor& dst,
                                 const vision::Tensor& M, VSize dsize,
                                 int flags = INTER_LINEAR,
                                 int borderMode = BORDER_CONSTANT,
                                 const VScalar& borderValue = VScalar());

    static void warp_affine_opencv(const vision::Tensor& src, vision::Tensor& dst,
                                   float scale, float rot, VSize dsize,
                                   const VScalar& aux_param = VScalar(),
                                   int flags = INTER_LINEAR,
                                   int borderMode = BORDER_CONSTANT,
                                   const VScalar& borderValue = VScalar());

    static void warp_affine_naive(const vision::Tensor& src, vision::Tensor& dst,
                                   float scale, float rot, VSize dsize,
                                   const VScalar& aux_param = VScalar(),
                                   int flags = INTER_LINEAR,
                                   int borderMode = BORDER_CONSTANT,
                                   const VScalar& borderValue = VScalar());
};

} // namespace vision

#endif //VISION_WARP_AFFINE_H
