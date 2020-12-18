#include "test_change_dtype.h"

#include <string>

#include <opencv2/opencv.hpp>

#include "../../../common/tensor_converter.h"
#include "../../../cv/cv.h"
#include "../../../util/image_util.h"
#include "../../../util/perf_util.h"

using namespace vision;
using namespace va_cv;

static std::string test_img_176x144   = "./res/176x144.jpg";
static std::string test_img_640x360   = "./res/640x360.jpg";
static std::string test_img_1280x720  = "./res/1280x720.jpg";
static std::string test_img_1920x1080 = "./res/1920x1080.jpeg";
static std::string test_img_2560x1440 = "./res/2560x1440.jpeg";

namespace vacv {

std::vector<double> TestChangeDtype::test_change_dtype_u8_to_fp32_176x144() {
    cv::Mat src_mat = cv::imread(test_img_176x144);
    cv::Mat src_mat_fp32;
    double opencv_duration;
    {
        TIME_PERF(opencv_duration);
        src_mat.convertTo(src_mat_fp32, CV_32FC(src_mat.channels()));
    }

    Tensor src_tensor = TensorConverter::convert_from<cv::Mat>(src_mat);
    Tensor src_tensor_fp32;
    double vacv_duration;
    {
        TIME_PERF(vacv_duration);
        src_tensor_fp32 = src_tensor.change_dtype(FP32);
    }

    float cosine_distance = ImageUtil::compare_image_data((float*)src_mat_fp32.data,
                                                            (float*)src_tensor_fp32.data,
                                                            int(src_tensor_fp32.size()));

    std::vector<double> profile_details;
    profile_details.push_back(opencv_duration);
    profile_details.push_back(vacv_duration);
    profile_details.push_back(static_cast<double>(cosine_distance));
    profile_details.push_back(1);
    return profile_details;
}

std::vector<double> TestChangeDtype::test_change_dtype_fp32_to_u8_176x144() {
    cv::Mat src_mat = cv::imread(test_img_176x144);
    cv::Mat src_mat_fp32;
    src_mat.convertTo(src_mat_fp32, CV_32FC(src_mat.channels()));
    cv::Mat src_mat_u8;

    double opencv_duration;
    {
        TIME_PERF(opencv_duration);
        src_mat_fp32.convertTo(src_mat_u8, CV_8UC(src_mat.channels()));
    }

    Tensor src_tensor = TensorConverter::convert_from<cv::Mat>(src_mat);
    Tensor src_tensor_fp32;
    src_tensor_fp32 = src_tensor.change_dtype(FP32);
    Tensor src_tensor_u8;
    double vacv_duration;
    {
        TIME_PERF(vacv_duration);
        src_tensor_u8 = src_tensor_fp32.change_dtype(INT8);
    }

    float cosine_distance = ImageUtil::compare_image_data((char*)src_mat_u8.data,
                                                           (char*)src_tensor_u8.data,
                                                           int(src_tensor_u8.size()));

    std::vector<double> profile_details;
    profile_details.push_back(opencv_duration);
    profile_details.push_back(vacv_duration);
    profile_details.push_back(static_cast<double>(cosine_distance));
    profile_details.push_back(1);
    return profile_details;
}


} // namespace vacv