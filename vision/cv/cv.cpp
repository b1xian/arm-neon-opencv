#include "cv.h"
#include "crop.h"
#include "cvt_color.h"
#include "imencode.h"
#include "match_template.h"
#include "normalize.h"
#include "resize.h"
#include "resize_normalize.h"
#include "warp_affine.h"
#include "warp_affine_normalize.h"

namespace va_cv {

using namespace vision;

void resize( const Tensor& src, Tensor& dst,
             VSize dsize, double fx, double fy,
             int interpolation) {
    Resize::resize(src, dst, dsize, fx, fy, interpolation);
}

void cvt_color(const Tensor& src, Tensor& dst, int code) {
    CvtColor::cvt_color(src, dst, code);
}

void normalize(const Tensor& src, Tensor& dst,
               const Tensor& mean, const Tensor& stddev) {
    Normalize::normalize(src, dst, mean, stddev);
}

void warp_affine( const Tensor& src, Tensor& dst,
                   const Tensor& M, VSize dsize, int flags,
                   int borderMode, const VScalar& borderValue) {
    WarpAffine::warp_affine(src, dst, M, dsize, flags, borderMode, borderValue);
}

void warp_affine( const Tensor& src, Tensor& dst,
                  float scale, float rot, VSize dsize,
                  const VScalar& aux_param, int flags,
                  int borderMode, const VScalar& borderValue) {
    WarpAffine::warp_affine(src, dst, scale, rot, dsize, aux_param, flags, borderMode, borderValue);
}

void resize_normalize(const vision::Tensor& src, vision::Tensor& dst,
                      VSize dsize, double fx, double fy,
                      int interpolation,
                      const vision::Tensor& mean,
                      const vision::Tensor& stddev) {
    ResizeNormalize::resize_normalize(src, dst, dsize, fx, fy, interpolation, mean, stddev);
}

void warp_affine_normalize(const vision::Tensor& src, vision::Tensor& dst,
                           const vision::Tensor& M, VSize dsize,
                           int flags,
                           int borderMode,
                           const VScalar& borderValue,
                           const vision::Tensor& mean,
                           const vision::Tensor& stddev) {
    WarpAffineNormalize::warp_affine_normalize(src, dst, M, dsize, flags, borderMode, borderValue, mean, stddev);
}

void warp_affine_normalize(const vision::Tensor& src, vision::Tensor& dst,
                           float scale, float rot, VSize dsize,
                           const VScalar& aux_param,
                           int flags, int borderMode,
                           const VScalar& borderValue,
                           const vision::Tensor& mean,
                           const vision::Tensor& stddev) {
    WarpAffineNormalize::warp_affine_normalize(src, dst, scale, rot, dsize, aux_param,
            flags, borderMode, borderValue, mean, stddev);
}

void crop(const vision::Tensor& src, vision::Tensor& dst, const vision::VRect& rect) {
    Crop::crop(src, dst, rect);
}

void match_template(const vision::Tensor& src, const vision::Tensor& target,
                    vision::Tensor& result, int method) {
    MatchTemplate::match_template(src, target, result, method);
}

void minMaxIdx(const vision::Tensor& src, double* minVal, double* maxVal,
               int* minIdx, int* maxIdx, const vision::Tensor& mask) {
    MatchTemplate::minMaxIdx(src, minVal, maxVal, minIdx, maxIdx, mask);
}

void imencode(const vision::Tensor& src, std::vector<unsigned char>& buf, const char* format) {
    ImEncode::imencode(src, buf, format);
}

} // namespace va_cv