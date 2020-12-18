#include "resize_normalize.h"

#include <stdexcept>

#ifdef USE_OPENCV
#include "opencv2/opencv.hpp"
#endif

#include "../common/tensor_converter.h"

namespace va_cv {

using namespace vision;

void ResizeNormalize::resize_normalize(const vision::Tensor& src, vision::Tensor& dst,
                                       VSize dsize, double fx, double fy,
                                       int interpolation,
                                       const vision::Tensor& mean,
                                       const vision::Tensor& stddev) {
#ifdef USE_OPENCV
    resize_normalize_opencv(src, dst, dsize, fx, fy, interpolation, mean, stddev);
#else
#if defined (USE_NEON) and __ARM_NEON
    resize_normalize_neon(src, dst, dsize, fx, fy, interpolation, mean, stddev);
#elif defined (USE_SSE)
    resize_normalize_sse(src, dst, dsize, fx, fy, interpolation, mean, stddev);
#else
    resize_normalize_naive(src, dst, dsize, fx, fy, interpolation, mean, stddev);
#endif
#endif // USE_OPENCV
}

void ResizeNormalize::resize_normalize_opencv(const vision::Tensor& src, vision::Tensor& dst,
                                              VSize dsize, double fx, double fy,
                                              int interpolation,
                                              const vision::Tensor& mean,
                                              const vision::Tensor& stddev) {
#ifdef USE_OPENCV
    const auto& mat_src = vision::TensorConverter::convert_to<cv::Mat>(src);
    cv::Mat mat_resized;
    cv::resize(mat_src, mat_resized, cv::Size(cvRound(dsize.w), cvRound(dsize.h)), fx, fy, interpolation);

    cv::Mat mat_resized_f;
    if (src.dtype != FP32) {
        if (src.c == 1) {
            mat_resized.convertTo(mat_resized_f, CV_32FC1);
        } else if (src.c == 3){
            mat_resized.convertTo(mat_resized_f, CV_32FC3);
        } else {
            // not supported!
            throw std::runtime_error("The tensor channel number is not supported in resize_normalize_opencv()");
        }
    } else {
        mat_resized_f = mat_resized;
    }

    cv::Mat mat_dst;
    if (mean.empty() || stddev.empty()) {
        cv::Mat mat_mean;
        cv::Mat mat_stddev;
        cv::meanStdDev(mat_resized_f, mat_mean, mat_stddev);

        if (src.c == 1) {
            auto m = *((double*)mat_mean.data);
            auto s = *((double*)mat_stddev.data);
            mat_dst = (mat_resized_f - m) / (s + 1e-6);
            VLOGD(TAG, "image w=%d h=%d mean=%lf", mat_resized_f.cols, mat_resized_f.rows, *((double*)mat_mean.data));
        } else {
            std::vector<cv::Mat> mats;
            cv::split(mat_resized_f, mats);
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
            mat_dst = (mat_resized_f - mean_data[0]) / (stddev_data[0] + 1e-6);
        } else {
            std::vector<cv::Mat> mats;
            cv::split(mat_resized_f, mats);
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
    resize_normalize_naive(src, dst, dsize, fx, fy, interpolation, mean, stddev);
#endif
}

void ResizeNormalize::resize_normalize_naive(const vision::Tensor& src, vision::Tensor& dst,
                                             VSize dsize, double fx, double fy,
                                             int interpolation,
                                             const vision::Tensor& mean,
                                             const vision::Tensor& stddev) {
    // todo:
}

void ResizeNormalize::resize_normalize_sse(const vision::Tensor& src, vision::Tensor& dst,
                                           VSize dsize, double fx, double fy,
                                           int interpolation,
                                           const vision::Tensor& mean,
                                           const vision::Tensor& stddev) {
    // todo:
}

void ResizeNormalize::resize_normalize_neon(const vision::Tensor& src, vision::Tensor& dst,
                                            VSize dsize, double fx, double fy,
                                            int interpolation,
                                            const vision::Tensor& mean,
                                            const vision::Tensor& stddev) {
    // todo:
}

} // namespace va_cv