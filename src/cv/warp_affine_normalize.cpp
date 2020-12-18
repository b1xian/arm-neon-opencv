#include "warp_affine_normalize.h"

#ifdef USE_OPENCV
#include "opencv2/opencv.hpp"
#endif

#include "../common/tensor_converter.h"

namespace va_cv {

using namespace vision;

void WarpAffineNormalize::warp_affine_normalize(const vision::Tensor& src, vision::Tensor& dst,
                                                const vision::Tensor& M, VSize dsize,
                                                int flags,
                                                int borderMode,
                                                const VScalar& borderValue,
                                                const vision::Tensor& mean,
                                                const vision::Tensor& stddev) {
#ifdef USE_OPENCV
    warp_affine_normalize_opencv(src, dst, M, dsize, flags, borderMode, borderValue, mean, stddev);
#else
#if defined (USE_NEON) and __ARM_NEON
    warp_affine_normalize_neon(src, dst, M, dsize, flags, borderMode, borderValue, mean, stddev);
#elif defined (USE_SSE)
    warp_affine_normalize_sse(src, dst, M, dsize, flags, borderMode, borderValue, mean, stddev);
#else
    warp_affine_normalize_naive(src, dst, M, dsize, flags, borderMode, borderValue, mean, stddev);
#endif
#endif // USE_OPENCV
}

void WarpAffineNormalize::warp_affine_normalize(const vision::Tensor& src, vision::Tensor& dst,
                                                float scale, float rot, VSize dsize,
                                                const VScalar& aux_param,
                                                int flags, int borderMode,
                                                const VScalar& borderValue,
                                                const vision::Tensor& mean,
                                                const vision::Tensor& stddev) {
#ifdef USE_OPENCV
    warp_affine_normalize_opencv(src, dst, scale, rot, dsize, aux_param, flags, borderMode, borderValue, mean, stddev);
#else
    // todo:
#endif
}

void WarpAffineNormalize::warp_affine_normalize_opencv(const vision::Tensor& src, vision::Tensor& dst,
                                                       const vision::Tensor& M, va_cv::VSize dsize, int flags,
                                                       int borderMode, const va_cv::VScalar& borderValue,
                                                       const vision::Tensor& mean, const vision::Tensor& stddev) {
#ifdef USE_OPENCV
    cv::Mat mat_src = vision::TensorConverter::convert_to<cv::Mat>(src);
    cv::Mat mat_M = vision::TensorConverter::convert_to<cv::Mat>(M);
    cv::Scalar sca_border(borderValue.v0, borderValue.v1, borderValue.v2, borderValue.v3);
    cv::Mat mat_warp;
    cv::warpAffine(mat_src, mat_warp, mat_M, cv::Size(dsize.w, dsize.h), flags, borderMode, sca_border);

    cv::Mat mat_warp_f;
    if (src.dtype != FP32) {
        if (src.c == 1) {
            mat_warp.convertTo(mat_warp_f, CV_32FC1);
        } else if (src.c == 3){
            mat_warp.convertTo(mat_warp_f, CV_32FC3);
        } else {
            // not supported!
            throw std::runtime_error("The tensor channel number is not supported in warp_affine_normalize_opencv()");
        }
    } else {
        mat_warp_f = mat_warp;
    }

    cv::Mat mat_dst;
    if (mean.empty() && stddev.empty()) {
        cv::Scalar scalar_mean;
        cv::Scalar scalar_stddev;
        cv::meanStdDev(mat_warp_f, scalar_mean, scalar_stddev);
        float m = scalar_mean.val[0];
        float s = scalar_stddev.val[0];
        mat_dst = (mat_warp_f - m) / (s + 1e-6);
    } else {
        if (static_cast<int>(mean.size()) != src.dims) {
            throw std::runtime_error("The input mean or stddev channels is not matched with tensor dims, in warp_affine_normalize_opencv()");
        }

        auto* mean_data = (float*)mean.data;
        auto* stddev_data = (float*)stddev.data;

        if (src.dims == 1) {
            mat_dst = (mat_warp_f - mean_data[0]) / (stddev_data[0] + 1e-6);
        } else {
            std::vector<cv::Mat> mats;
            cv::split(mat_warp_f, mats);
            int c = 0;
            for (auto& mat : mats) {
                mat = (mat - mean_data[c]) / (stddev_data[0] + 1e-6);
                c++;
            }
            cv::merge(mats, mat_dst);
        }
    }

    dst = vision::TensorConverter::convert_from<cv::Mat>(mat_dst, true);
#else
    warp_affine_normalize_naive(src, dst, M, dsize, flags, borderMode, borderValue);
#endif
}

void WarpAffineNormalize::warp_affine_normalize_opencv(const vision::Tensor& src, vision::Tensor& dst,
                                                       float scale, float rot, VSize dsize,
                                                       const VScalar& aux_param,
                                                       int flags, int borderMode,
                                                       const VScalar& borderValue,
                                                       const vision::Tensor& mean,
                                                       const vision::Tensor& stddev) {
#ifdef USE_OPENCV
    cv::Mat mat_src = vision::TensorConverter::convert_to<cv::Mat>(src);

    // get rotation matrix
    cv::Mat rot_mat = cv::getRotationMatrix2D(cv::Point(0, 0), rot, scale);
    // 修正
    rot_mat.at<double>(0, 2) = aux_param.v2 - rot_mat.at<double>(0, 0) * aux_param.v0 -
                               rot_mat.at<double>(0, 1) * aux_param.v1;
    rot_mat.at<double>(1, 2) = aux_param.v3 - rot_mat.at<double>(1, 0) * aux_param.v0 -
                               rot_mat.at<double>(1, 1) * aux_param.v1;

    cv::Mat mat_warp;
    cv::Scalar sca_border(borderValue.v0, borderValue.v1, borderValue.v2, borderValue.v3);
    cv::warpAffine(mat_src, mat_warp, rot_mat, cv::Size(dsize.w, dsize.h), flags, borderMode, sca_border);

    cv::Mat mat_warp_f;
    if (src.dtype != FP32) {
        if (src.c == 1) {
            mat_warp.convertTo(mat_warp_f, CV_32FC1);
        } else if (src.c == 3){
            mat_warp.convertTo(mat_warp_f, CV_32FC3);
        } else {
            // not supported!
            throw std::runtime_error("The tensor channel number is not supported in warp_affine_normalize_opencv()");
        }
    } else {
        mat_warp_f = mat_warp;
    }

    cv::Mat mat_dst;
    if (mean.empty() && stddev.empty()) {
        cv::Mat mat_mean;
        cv::Mat mat_stddev;
        cv::meanStdDev(mat_warp_f, mat_mean, mat_stddev);
        if (src.c == 1) {
            auto m = *((double*)mat_mean.data);
            auto s = *((double*)mat_stddev.data);
            mat_dst = (mat_warp_f - m) / (s + 1e-6);
        } else {
            std::vector<cv::Mat> mats;
            cv::split(mat_warp_f, mats);
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
        if (static_cast<int>(mean.size()) != src.dims) {
            throw std::runtime_error("The input mean or stddev channels is not matched with tensor dims, in warp_affine_normalize_opencv()");
        }

        auto* mean_data = (float*)mean.data;
        auto* stddev_data = (float*)stddev.data;

        if (src.dims == 1) {
            mat_dst = (mat_warp_f - mean_data[0]) / (stddev_data[0] + 1e-6);
        } else {
            std::vector<cv::Mat> mats;
            cv::split(mat_warp_f, mats);
            int c = 0;
            for (auto& mat : mats) {
                mat = (mat - mean_data[c]) / (stddev_data[c] + 1e-6);
                c++;
            }
            cv::merge(mats, mat_dst);
        }
    }

    dst = vision::TensorConverter::convert_from<cv::Mat>(mat_dst, true);
#endif
}

void WarpAffineNormalize::warp_affine_normalize_naive(const vision::Tensor& src, vision::Tensor& dst,
                                                      const vision::Tensor& M, va_cv::VSize dsize, int flags,
                                                      int borderMode, const va_cv::VScalar& borderValue,
                                                      const vision::Tensor& mean, const vision::Tensor& stddev) {
    // todo:
}

void WarpAffineNormalize::warp_affine_normalize_sse(const vision::Tensor& src, vision::Tensor& dst,
                                                    const vision::Tensor& M, va_cv::VSize dsize, int flags,
                                                    int borderMode, const va_cv::VScalar& borderValue,
                                                    const vision::Tensor& mean, const vision::Tensor& stddev) {
    // todo:
}

void WarpAffineNormalize::warp_affine_normalize_neon(const vision::Tensor& src, vision::Tensor& dst,
                                                     const vision::Tensor& M, va_cv::VSize dsize, int flags,
                                                     int borderMode, const va_cv::VScalar& borderValue,
                                                     const vision::Tensor& mean, const vision::Tensor& stddev) {
    // todo:
}

} // namespace va_cv