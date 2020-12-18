#ifndef VISION_CROP_H
#define VISION_CROP_H

#include "../common/tensor.h"
#include "../common/vision_structs.h"

namespace va_cv {

class Crop {
public:
    static void crop(const vision::Tensor& src, vision::Tensor& dst, const vision::VRect& rect);

private:
    static void crop_opencv(const vision::Tensor& src, vision::Tensor& dst, const vision::VRect& rect);
    static void crop_naive(const vision::Tensor& src, vision::Tensor& dst, const vision::VRect& rect);
    static void crop_sse(const vision::Tensor& src, vision::Tensor& dst, const vision::VRect& rect);
    static void crop_neon(const vision::Tensor& src, vision::Tensor& dst, const vision::VRect& rect);
    static void crop_neon_hwc_rgb_ir(const vision::Tensor& src, vision::Tensor& dst, const vision::VRect& rect);
    static void crop_neon_chw_rgb(const vision::Tensor& src, vision::Tensor& dst, const vision::VRect& rect);
};

} // namespace va_cv

#endif //VISION_CROP_H
