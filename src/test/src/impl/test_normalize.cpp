#include "test_normalize.h"

#include <opencv2/opencv.hpp>

#include "../../../common/tensor_converter.h"
#include "../../../cv/cv.h"
#include "../../../util/image_util.h"
#include "../../../util/perf_util.h"

using namespace vision;
using namespace va_cv;

static std::string test_img_176x144   = "./res/176x144.jpg";
static std::string test_img_284x214   = "./res/284x214.jpg";
static std::string test_img_640x360   = "./res/640x360.jpg";
static std::string test_img_1280x720  = "./res/1280x720.jpg";
static std::string test_img_1920x1080 = "./res/1920x1080.jpeg";
static std::string test_img_2560x1440 = "./res/2560x1440.jpeg";

namespace vacv {

std::vector<double> TestNormalize::test_normalize_hwc(std::string img_path) {
    cv::Mat src_mat = cv::imread(img_path);
    cv::Mat src_mat_fp32;
    cv::Mat src_mat_normalized;
    double opencv_duration;
    {
        TIME_PERF(opencv_duration);
        cv::Mat mat_mean;
        cv::Mat mat_stddev;
        cv::meanStdDev(src_mat, mat_mean, mat_stddev);

        src_mat.convertTo(src_mat_fp32, CV_32FC(src_mat.channels()));
        if (src_mat.channels() == 1) {
            auto m = *((double*)mat_mean.data);
            auto s = *((double*)mat_stddev.data);
            src_mat_normalized = (src_mat_fp32 - m) / (s + 1e-6);
        } else {
            std::vector<cv::Mat> mats;
            cv::split(src_mat_fp32, mats);
            int c = 0;
            for (auto& mat : mats) {
                auto m = ((double *)mat_mean.data)[c];
                auto s = ((double *)mat_stddev.data)[c];
                mat = (mat - m) / (s + 1e-6);
                c++;
            }
            cv::merge(mats, src_mat_normalized);
        }
    }

    Tensor src_tensor = TensorConverter::convert_from<cv::Mat>(src_mat);
    Tensor src_tensor_fp32;
    Tensor src_tensor_normalized;
    double vacv_duration;
    {
        TIME_PERF(vacv_duration);
        src_tensor_fp32 = src_tensor.change_dtype(FP32);
        va_cv::normalize(src_tensor_fp32, src_tensor_normalized);
    }

    float cosine_distance = ImageUtil::compare_image_data((float*)src_mat_normalized.data,
                                                            (float*)src_tensor_normalized.data,
                                                            int(src_tensor_normalized.size()));

    std::vector<double> profile_details;
    profile_details.push_back(opencv_duration);
    profile_details.push_back(vacv_duration);
    profile_details.push_back(static_cast<double>(cosine_distance));
    profile_details.push_back(1);
    return profile_details;
}

std::vector<double> TestNormalize::test_normalize_chw(std::string img_path) {
    cv::Mat src_mat = cv::imread(img_path);
    cv::Mat src_mat_fp32;
    cv::Mat src_mat_normalized;
    double opencv_duration;
    {
        TIME_PERF(opencv_duration);
        cv::Mat mat_mean;
        cv::Mat mat_stddev;
        cv::meanStdDev(src_mat, mat_mean, mat_stddev);

        src_mat.convertTo(src_mat_fp32, CV_32FC(src_mat.channels()));
        if (src_mat.channels() == 1) {
            auto m = *((double*)mat_mean.data);
            auto s = *((double*)mat_stddev.data);
            src_mat_normalized = (src_mat_fp32 - m) / (s + 1e-6);
        } else {
            std::vector<cv::Mat> mats;
            cv::split(src_mat_fp32, mats);
            int c = 0;
            for (auto& mat : mats) {
                auto m = ((double *)mat_mean.data)[c];
                auto s = ((double *)mat_stddev.data)[c];
                mat = (mat - m) / (s + 1e-6);
                c++;
            }
            cv::merge(mats, src_mat_normalized);
        }
    }

    Tensor src_tensor = TensorConverter::convert_from<cv::Mat>(src_mat);
    Tensor src_tensor_chw = src_tensor.change_layout(NCHW);
    Tensor src_tensor_fp32;
    Tensor src_tensor_normalized;
    double vacv_duration;
    {
        TIME_PERF(vacv_duration);
        src_tensor_fp32 = src_tensor_chw.change_dtype(FP32);
        va_cv::normalize(src_tensor_fp32, src_tensor_normalized);
    }

    Tensor src_tensor_normalized_hwc = src_tensor_normalized.change_layout(NHWC);

    float cosine_distance = ImageUtil::compare_image_data((float*)src_mat_normalized.data,
                                                           (float*)src_tensor_normalized_hwc.data,
                                                           int(src_tensor_normalized_hwc.size()));

    std::vector<double> profile_details;
    profile_details.push_back(opencv_duration);
    profile_details.push_back(vacv_duration);
    profile_details.push_back(static_cast<double>(cosine_distance));
    profile_details.push_back(1);
    return profile_details;
}

std::vector<double> TestNormalize::test_normalize_hwc_176x144() {
    return test_normalize_hwc(test_img_176x144);
}

std::vector<double> TestNormalize::test_normalize_chw_176x144() {
    return test_normalize_chw(test_img_176x144);
}

std::vector<double> TestNormalize::test_normalize_hwc_284x214() {
    return test_normalize_hwc(test_img_284x214);
}

std::vector<double> TestNormalize::test_normalize_chw_284x214() {
    return test_normalize_chw(test_img_284x214);
}


} // namespace vacv