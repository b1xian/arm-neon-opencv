#include "normalize.h"

#include <stdexcept>

#ifdef USE_OPENCV
#include "opencv2/opencv.hpp"
#endif

#if defined (USE_NEON) and __ARM_NEON
#include "arm_neon.h"
#include "normalize_neon.h"
#endif

#include "normalize_naive.h"
#include "../common/tensor_converter.h"

namespace va_cv {

using namespace vision;

void Normalize::normalize(const Tensor& src, Tensor& dst,
               const Tensor& mean, const Tensor& stddev) {

#if defined (USE_NEON) and __ARM_NEON
    normalize_neon(src, dst, mean, stddev);
#else
    normalize_naive(src, dst, mean, stddev);
#endif // USE_NEON
}

void Normalize::normalize_opencv(const Tensor& src, Tensor& dst,
                                 const Tensor& mean, const Tensor& stddev) {
#ifdef USE_OPENCV
    const auto& mat_src = vision::TensorConverter::convert_to<cv::Mat>(src);
    cv::Mat mat_dst;
    if (mean.empty() && stddev.empty()) {
        cv::Mat mat_mean;
        cv::Mat mat_stddev;
        cv::meanStdDev(mat_src, mat_mean, mat_stddev);

        if (src.c == 1) {
            auto m = *((double*)mat_mean.data);
            auto s = *((double*)mat_stddev.data);
            mat_dst = (mat_src - m) / (s + 1e-6);
        } else {
            std::vector<cv::Mat> mats;
            cv::split(mat_src, mats);
            int c = 0;
            for (auto& mat : mats) {
                auto m = ((double *)mat_mean.data)[c];
                auto s = ((double *)mat_stddev.data)[c];
                mat = (mat - m) / (s + 1e-6);
                c++;
            }
            cv::merge(mats, mat_dst);
        }
    } else {
        if (static_cast<int>(mean.size()) != src.c) {
            throw std::runtime_error("The input mean or stddev channels is not matched with tensor dims, in resize_normalize_opencv()");
        }

        auto* mean_data = (float*)mean.data;
        auto* stddev_data = (float*)stddev.data;

        if (src.c == 1) {
            mat_dst = (mat_src - mean_data[0]) / (stddev_data[0] + 1e-6);
        } else {
            std::vector<cv::Mat> mats;
            cv::split(mat_src, mats);
            int c = 0;
            for (auto& mat : mats) {
                mat = (mat - mean_data[c]) / (stddev_data[c] + 1e-6);
                c++;
            }
            cv::merge(mats, mat_dst);
        }
    }
    dst = vision::TensorConverter::convert_from<cv::Mat>(mat_dst, true);
#else
    normalize_naive(src, dst, mean, stddev);
#endif
}

void Normalize::normalize_naive(const Tensor& src, Tensor& dst,
                                const Tensor& mean, const Tensor& stddev) {
    int w       = src.w;
    int h       = src.h;
    int c       = src.c;
    int stride  = src.stride;
    dst.create(w, h, c, FP32, src.layout);

    Tensor src_fp32 = src;
    if (src.dtype != FP32) {
        src_fp32 = src_fp32.change_dtype(FP32);
    }
    Tensor mean_tensor;
    Tensor stddev_tensor;
    if (mean.empty() && stddev.empty()) {
        mean_tensor.create(c, 1, 1, FP32, NCHW);
        stddev_tensor.create(c, 1, 1, FP32, NCHW);
        if (src.layout == NHWC) {
            NormalizeNaive::mean_stddev_naive_hwc_bgr((float*)src_fp32.data, src_fp32.stride,
                                                      (float*)mean_tensor.data, (float*)stddev_tensor.data);
        } else {

            NormalizeNaive::mean_stddev_naive_chw((float*)src_fp32.data, src_fp32.stride, c,
                                                  (float*)mean_tensor.data, (float*)stddev_tensor.data);
        }
    } else {
        mean_tensor = mean;
        stddev_tensor = stddev;
    }

    if (src.layout == NHWC) {
        NormalizeNaive::normalize_naive_hwc_bgr((float*)src_fp32.data, (float*)dst.data, stride,
                                                (float*)mean_tensor.data, (float*)stddev_tensor.data);
    } else {
        NormalizeNaive::normalize_naive_chw((float*)src_fp32.data, (float*)dst.data, stride, c,
                                            (float*)mean_tensor.data, (float*)stddev_tensor.data);
    }
}

void Normalize::normalize_sse(const Tensor& src, Tensor& dst,
                              const Tensor& mean, const Tensor& stddev) {
    // todo：
}

#if defined (USE_NEON) and __ARM_NEON
void Normalize::normalize_neon(const Tensor& src, Tensor& dst,
                               const Tensor& mean, const Tensor& stddev) {
    int w       = src.w;
    int h       = src.h;
    int c       = src.c;
    int stride  = src.stride;
    dst.create(w, h, c, FP32, src.layout);

    Tensor src_fp32 = src;
    if (src.dtype != FP32) {
        src_fp32 = src_fp32.change_dtype(FP32);
    }
    Tensor mean_tensor;
    Tensor stddev_tensor;
    if (mean.empty() && stddev.empty()) {
        mean_tensor.create(c, 1, 1, FP32, NCHW);
        stddev_tensor.create(c, 1, 1, FP32, NCHW);
        if (src.layout == NHWC) {
            NormalizeNeon::mean_stddev_neon_hwc_bgr((float*)src_fp32.data, src_fp32.stride,
                                                      (float*)mean_tensor.data, (float*)stddev_tensor.data);
        } else {

            NormalizeNeon::mean_stddev_neon_chw((float*)src_fp32.data, src_fp32.stride, c,
                                                  (float*)mean_tensor.data, (float*)stddev_tensor.data);
        }
    } else {
        mean_tensor = mean;
        stddev_tensor = stddev;
    }

    if (src.layout == NHWC) {
        NormalizeNeon::normalize_neon_hwc_bgr((float*)src_fp32.data, (float*)dst.data, stride,
                                                (float*)mean_tensor.data, (float*)stddev_tensor.data);
    } else {
        NormalizeNeon::normalize_neon_chw((float*)src_fp32.data, (float*)dst.data, stride, c,
                                            (float*)mean_tensor.data, (float*)stddev_tensor.data);
    }
}
#endif

} // namespace va_cv