#include "resize.h"

#ifdef USE_OPENCV
#include "opencv2/opencv.hpp"
#endif

#if defined (USE_NEON) and __ARM_NEON
#include "arm_neon.h"
#include "resize_neon.h"
#endif

#include "resize_naive.h"
#include "../common/tensor_converter.h"

namespace va_cv {

using namespace vision;

void Resize::resize(const Tensor& src, Tensor& dst,
                    VSize dsize, double fx, double fy,
                    int interpolation) {
#if defined (USE_NEON) and __ARM_NEON
    resize_opencv(src, dst, dsize, fx, fy, interpolation);
#else
    resize_naive(src, dst, dsize, fx, fy, interpolation);
#endif // USE_NEON
}

void Resize::resize_opencv(const Tensor& src, Tensor& dst,
                           VSize dsize, double fx, double fy,
                           int interpolation) {
#ifdef USE_OPENCV
    const auto& mat_src = vision::TensorConverter::convert_to<cv::Mat>(src);
    cv::Mat mat_dst;
    cv::resize(mat_src, mat_dst, cv::Size(dsize.w, dsize.h), fx, fy, interpolation);
    dst = vision::TensorConverter::convert_from<cv::Mat>(mat_dst, true);
#else
    resize_naive(src, dst, dsize, fx, fy, interpolation);
#endif
}

void Resize::resize_naive(const Tensor& src, Tensor& dst,
                          VSize dsize, double fx, double fy,
                          int interpolation) {

    if (interpolation != INTER_LINEAR && interpolation != INTER_CUBIC) {
        resize_opencv(src, dst, dsize, fx, fy, interpolation);
        return;
    }

    int w_in  = src.w;
    int h_in  = src.h;
    int c     = src.c;
    int w_out = dsize.w;
    int h_out = dsize.h;
    dst.create(w_out, h_out, src.c, src.dtype, src.layout);

    if (w_out == w_in && h_out == h_in) {
        memcpy(dst.data, src.data, sizeof(uint8_t) * w_in * h_in * c);
        return;
    }

    if (interpolation == INTER_LINEAR && (src.dtype == INT8 || src.dtype == FP32)) {
        if (src.layout == NHWC) {
            if (src.dtype == vision::DType::INT8) {
                ResizeNaive::resize_naive_inter_linear_u8((char*)src.data, w_in, h_in, c,
                                                          (char*)dst.data, w_out, h_out);
            } else if (src.dtype == vision::DType::FP32) {
                ResizeNaive::resize_naive_inter_linear_fp32((float*)src.data, w_in, h_in, c,
                                                            (float*)dst.data, w_out, h_out);
            }
        } else {
            int src_stride = w_in * h_in;
            int dst_stride = w_out * h_out;
            for (int i = 0; i < c; i++) {
                if (src.dtype == INT8) {
                    char* src_channel_data = (char*)src.data + src_stride * i;
                    char* dst_channel_data = (char*)dst.data + dst_stride * i;
                    ResizeNaive::resize_naive_inter_linear_u8(src_channel_data, w_in, h_in, 1,
                                                              dst_channel_data, w_out, h_out);
                } else if (src.dtype == FP32) {
                    float* src_channel_data = (float*)src.data + src_stride * i;
                    float* dst_channel_data = (float*)dst.data + dst_stride * i;
                    ResizeNaive::resize_naive_inter_linear_fp32(src_channel_data, w_in, h_in, 1,
                                                              dst_channel_data, w_out, h_out);
                }
            }
        }
    } else if (interpolation == INTER_CUBIC && src.dtype == FP32) {
        if (src.layout == NHWC) {
            ResizeNaive::resize_naive_inter_cubic_fp32_hwc((float*)src.data, w_in, h_in,
                                                           (float*)dst.data, w_out, h_out);
        } else {
            ResizeNaive::resize_naive_inter_cubic_fp32_chw((float*)src.data, w_in, h_in, c,
                                                           (float*)dst.data, w_out, h_out);
        }
    } else {
        resize_opencv(src, dst, dsize, fx, fy, interpolation);
    }
}

#if defined (USE_NEON) and __ARM_NEON
void Resize::resize_sse(const Tensor& src, Tensor& dst,
                        VSize dsize, double fx, double fy,
                        int interpolation) {
    // todo:
}

void Resize::resize_neon(const Tensor& src, Tensor& dst,
                         VSize dsize, double fx, double fy,
                         int interpolation) {

    if (interpolation != INTER_LINEAR || src.dtype != INT8) {
        if (src.layout == NCHW) {
            resize_naive(src, dst, dsize, fx, fy, interpolation);
        } else {
            resize_opencv(src, dst, dsize, fx, fy, interpolation);
        }
        return;
    }

    int w_in  = src.w;
    int h_in  = src.h;
    int c     = src.c;
    int w_out = dsize.w;
    int h_out = dsize.h;
    dst.create(w_out, h_out, c, src.dtype, src.layout);

    if (w_out == w_in && h_out == h_in) {
        memcpy(dst.data, src.data, sizeof(uint8_t) * w_in * h_in * c);
        return;
    }

    if (src.layout == NHWC) {
        ResizeNeon::resize_neon_inter_linear_three_channel((uint8_t*)src.data, w_in * 3, h_in,
                                                           (uint8_t*)dst.data, w_out * 3, h_out);
    } else {
        int src_stride = w_in * h_in;
        int dst_stride = w_out * h_out;
        for (int i = 0; i < 3; i++) {
            uint8_t* src_channel_data = (uint8_t*)src.data + src_stride * i;
            uint8_t* dst_channel_data = (uint8_t*)dst.data + dst_stride * i;
            ResizeNeon::resize_neon_inter_linear_one_channel(src_channel_data, w_in, h_in,
                                                             dst_channel_data, w_out, h_out);
        }
    }
}
#endif


} // namespace va_cv