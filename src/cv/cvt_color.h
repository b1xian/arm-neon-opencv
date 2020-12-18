#ifndef VISION_CVT_COLOR_H
#define VISION_CVT_COLOR_H

#include "../common/tensor.h"

namespace va_cv {

class CvtColor {
public:
    static void cvt_color(const vision::Tensor& src, vision::Tensor& dst, int code);

private:
    static void cvt_color_opencv(const vision::Tensor& src, vision::Tensor& dst, int code);
    static void cvt_color_naive(const vision::Tensor& src, vision::Tensor& dst, int code);
    static void cvt_color_sse(const vision::Tensor& src, vision::Tensor& dst, int code);
#if defined (USE_NEON) and __ARM_NEON
    static void cvt_color_neon(const vision::Tensor& src, vision::Tensor& dst, int code);
    static void nv_to_bgr_neon(const uint8_t* src, uint8_t* dst, int srcw, int srch, int x_num, int y_num);
#endif
    static void nv_to_bgr_naive(const unsigned char* src, unsigned char* dst, int srcw, int srch, int x_num, int y_num);
};

} // namespace va_cv

#endif //VISION_CVT_COLOR_H
