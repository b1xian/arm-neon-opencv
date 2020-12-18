#include "test_resize.h"

#include <string>

#include <opencv2/opencv.hpp>

#include "../../../common/tensor_converter.h"
#include "../../../cv/cv.h"
#include "../../../util/image_util.h"
#include "../../../util/perf_util.h"

using namespace vision;
using namespace va_cv;

static std::string test_img_2560x1440 = "./res/2560x1440.jpeg";

namespace vacv {

static cv::Size cv_size_320x180(320, 180);

static VSize vacv_size_320x180(320, 180);


std::vector<double> TestResize::test_resize_bilinear_hwc_u8_320x180() {
    cv::Mat src_mat = cv::imread(test_img_2560x1440);
    cv::Mat cv_resize320x180_mat;
    double opencv_duration;
    {
        TIME_PERF(opencv_duration);
        cv::resize(src_mat, cv_resize320x180_mat, cv_size_320x180, 0, 0, CV_INTER_LINEAR);
    }

    Tensor src_tensor = TensorConverter::convert_from<cv::Mat>(src_mat);
    Tensor vacv_resize_bilinear_hwc_u8_tensor;
    double vacv_duration;
    {
        TIME_PERF(vacv_duration);
        va_cv::resize(src_tensor, vacv_resize_bilinear_hwc_u8_tensor, vacv_size_320x180, va_cv::INTER_NEAREST);
    }

    float cosine_distance = ImageUtil::compare_image_data((char*)cv_resize320x180_mat.data,
                                                            (char*)vacv_resize_bilinear_hwc_u8_tensor.data,
                                                            int(vacv_resize_bilinear_hwc_u8_tensor.size()));

    std::vector<double> profile_details;
    profile_details.push_back(opencv_duration);
    profile_details.push_back(vacv_duration);
    profile_details.push_back(static_cast<double>(cosine_distance));
    profile_details.push_back(1);
    return profile_details;
}

std::vector<double> TestResize::test_resize_bilinear_chw_u8_320x180() {
    cv::Mat src_mat = cv::imread(test_img_2560x1440);
    cv::Mat cv_resize320x180_mat;
    double opencv_duration;
    {
        TIME_PERF(opencv_duration);
        cv::resize(src_mat, cv_resize320x180_mat, cv_size_320x180, 0, 0, CV_INTER_LINEAR);
    }

    Tensor src_tensor = TensorConverter::convert_from<cv::Mat>(src_mat);
    Tensor src_tensor_chw = src_tensor.change_layout(NCHW);
    Tensor vacv_resize_bilinear_chw_u8_tensor;
    double vacv_duration;
    {
        TIME_PERF(vacv_duration);
        va_cv::resize(src_tensor_chw, vacv_resize_bilinear_chw_u8_tensor, vacv_size_320x180, va_cv::INTER_NEAREST);
    }
    Tensor vacv_resize_bilinear_hwc_u8_tensor = vacv_resize_bilinear_chw_u8_tensor.change_layout(NHWC);

    float cosine_distance = ImageUtil::compare_image_data((char*)cv_resize320x180_mat.data,
                                                           (char*)vacv_resize_bilinear_hwc_u8_tensor.data,
                                                           int(vacv_resize_bilinear_hwc_u8_tensor.size()));

    std::vector<double> profile_details;
    profile_details.push_back(opencv_duration);
    profile_details.push_back(vacv_duration);
    profile_details.push_back(static_cast<double>(cosine_distance));
    profile_details.push_back(1);
    return profile_details;
}

std::vector<double> TestResize::test_resize_bilinear_hwc_fp32_320x180() {
    cv::Mat src_mat = cv::imread(test_img_2560x1440);
    cv::Mat src_mat_fp32;
    src_mat.convertTo(src_mat_fp32, CV_32FC(src_mat.channels()));
    cv::Mat cv_resize320x180_mat;
    double opencv_duration;
    {
        TIME_PERF(opencv_duration);
        cv::resize(src_mat_fp32, cv_resize320x180_mat, cv_size_320x180, 0, 0, CV_INTER_LINEAR);
    }

    Tensor src_tensor = TensorConverter::convert_from<cv::Mat>(src_mat_fp32);
    Tensor vacv_resize_bilinear_hwc_fp32_tensor;
    double vacv_duration;
    {
        TIME_PERF(vacv_duration);
        va_cv::resize(src_tensor, vacv_resize_bilinear_hwc_fp32_tensor, vacv_size_320x180, va_cv::INTER_NEAREST);
    }

    float cosine_distance = ImageUtil::compare_image_data((float*)cv_resize320x180_mat.data,
                                                           (float*)vacv_resize_bilinear_hwc_fp32_tensor.data,
                                                           int(vacv_resize_bilinear_hwc_fp32_tensor.size()));

    std::vector<double> profile_details;
    profile_details.push_back(opencv_duration);
    profile_details.push_back(vacv_duration);
    profile_details.push_back(static_cast<double>(cosine_distance));
    profile_details.push_back(1);
    return profile_details;
}

std::vector<double> TestResize::test_resize_bilinear_chw_fp32_320x180() {
    cv::Mat src_mat = cv::imread(test_img_2560x1440);
    cv::Mat src_mat_fp32;
    src_mat.convertTo(src_mat_fp32, CV_32FC(src_mat.channels()));
    cv::Mat cv_resize320x180_mat;
    double opencv_duration;
    {
        TIME_PERF(opencv_duration);
        cv::resize(src_mat_fp32, cv_resize320x180_mat, cv_size_320x180, 0, 0, CV_INTER_LINEAR);
    }

    Tensor src_tensor = TensorConverter::convert_from<cv::Mat>(src_mat_fp32);
    Tensor src_tensor_chw = src_tensor.change_layout(NCHW);
    Tensor vacv_resize_bilinear_chw_fp32_tensor;
    double vacv_duration;
    {
        TIME_PERF(vacv_duration);
        va_cv::resize(src_tensor_chw, vacv_resize_bilinear_chw_fp32_tensor, vacv_size_320x180, va_cv::INTER_NEAREST);
    }
    Tensor vacv_resize_bilinear_hwc_fp32_tensor = vacv_resize_bilinear_chw_fp32_tensor.change_layout(NHWC);
    float cosine_distance = ImageUtil::compare_image_data((float*)cv_resize320x180_mat.data,
                                                           (float*)vacv_resize_bilinear_hwc_fp32_tensor.data,
                                                           int(vacv_resize_bilinear_hwc_fp32_tensor.size()));

    std::vector<double> profile_details;
    profile_details.push_back(opencv_duration);
    profile_details.push_back(vacv_duration);
    profile_details.push_back(static_cast<double>(cosine_distance));
    profile_details.push_back(1);
    return profile_details;
}

std::vector<double> TestResize::test_resize_cubic_hwc_fp32_320x180() {
    cv::Mat src_mat = cv::imread(test_img_2560x1440);
    cv::Mat cv_resize320x180_mat;
    cv::Mat src_mat_fp32;
    src_mat.convertTo(src_mat_fp32, CV_32FC(src_mat.channels()));
    double opencv_duration;
    {
        TIME_PERF(opencv_duration);
        cv::resize(src_mat_fp32, cv_resize320x180_mat, cv_size_320x180, 0, 0, CV_INTER_CUBIC);
    }

    Tensor src_tensor = TensorConverter::convert_from<cv::Mat>(src_mat);
    Tensor vacv_resize_bilinear_hwc_fp32_tensor;
    Tensor src_tensor_fp32 = src_tensor.change_dtype(FP32);
    double vacv_duration;
    {
        TIME_PERF(vacv_duration);
        va_cv::resize(src_tensor_fp32, vacv_resize_bilinear_hwc_fp32_tensor, vacv_size_320x180, va_cv::INTER_CUBIC);
    }

    float cosine_distance = ImageUtil::compare_image_data((float*)cv_resize320x180_mat.data,
                                                           (float*)vacv_resize_bilinear_hwc_fp32_tensor.data,
                                                           int(vacv_resize_bilinear_hwc_fp32_tensor.size()));

    std::vector<double> profile_details;
    profile_details.push_back(opencv_duration);
    profile_details.push_back(vacv_duration);
    profile_details.push_back(static_cast<double>(cosine_distance));
    profile_details.push_back(1);
    return profile_details;
}

std::vector<double> TestResize::test_resize_cubic_chw_fp32_320x180() {
    cv::Mat src_mat = cv::imread(test_img_2560x1440);
    cv::Mat cv_resize320x180_mat;
    double opencv_duration;
    cv::Mat src_mat_fp32;
    src_mat.convertTo(src_mat_fp32, CV_32FC(src_mat.channels()));
    {
        TIME_PERF(opencv_duration);
        cv::resize(src_mat_fp32, cv_resize320x180_mat, cv_size_320x180, 0, 0, CV_INTER_CUBIC);
    }

    Tensor src_tensor = TensorConverter::convert_from<cv::Mat>(src_mat);
    Tensor vacv_resize_bilinear_chw_fp32_tensor;
    Tensor src_tensor_chw = src_tensor.change_layout(NCHW);
    double vacv_duration;
    Tensor src_tensor_fp32 = src_tensor_chw.change_dtype(FP32);
    {
        TIME_PERF(vacv_duration);
        va_cv::resize(src_tensor_fp32, vacv_resize_bilinear_chw_fp32_tensor, vacv_size_320x180, va_cv::INTER_CUBIC);
    }
    Tensor vacv_resize_bilinear_hwc_fp32_tensor = vacv_resize_bilinear_chw_fp32_tensor.change_layout(NHWC);
    float cosine_distance = ImageUtil::compare_image_data((float*)cv_resize320x180_mat.data,
                                                           (float*)vacv_resize_bilinear_hwc_fp32_tensor.data,
                                                           int(vacv_resize_bilinear_hwc_fp32_tensor.size()));

    std::vector<double> profile_details;
    profile_details.push_back(opencv_duration);
    profile_details.push_back(vacv_duration);
    profile_details.push_back(static_cast<double>(cosine_distance));
    profile_details.push_back(1);
    return profile_details;
}


} // namespace vacv