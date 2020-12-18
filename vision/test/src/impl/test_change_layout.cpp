#include "test_change_layout.h"

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

std::vector<double> TestChangeLayout::test_change_layout_hwc_to_chw_u8_176x144() {
    cv::Mat src_mat = cv::imread(test_img_176x144);
    double opencv_duration;
    {
        TIME_PERF(opencv_duration);
    }

    Tensor src_tensor = TensorConverter::convert_from<cv::Mat>(src_mat);
    Tensor src_tensor_chw;
    double vacv_duration;
    {
        TIME_PERF(vacv_duration);
        src_tensor_chw = src_tensor.change_layout(NCHW);
    }
    Tensor src_tensor_hwc;
    src_tensor_hwc = src_tensor_chw.change_layout(NHWC);
    float cosine_distance = ImageUtil::compare_image_data((char*)src_mat.data,
                                                            (char*)src_tensor_hwc.data,
                                                            int(src_tensor_hwc.size()));

    std::vector<double> profile_details;
    profile_details.push_back(opencv_duration);
    profile_details.push_back(vacv_duration);
    profile_details.push_back(static_cast<double>(cosine_distance));
    profile_details.push_back(1);
    return profile_details;
}

std::vector<double> TestChangeLayout::test_change_layout_hwc_to_chw_fp32_176x144() {
    cv::Mat src_mat = cv::imread(test_img_176x144);
    cv::Mat src_mat_fp32;
    src_mat.convertTo(src_mat_fp32, CV_32FC(src_mat.channels()));

    double opencv_duration;
    {
        TIME_PERF(opencv_duration);
    }

    Tensor src_tensor = TensorConverter::convert_from<cv::Mat>(src_mat);
    Tensor src_tensor_fp32;
    src_tensor_fp32 = src_tensor.change_dtype(FP32);
    Tensor src_tensor_fp32_chw;
    double vacv_duration;
    {
        TIME_PERF(vacv_duration);
        src_tensor_fp32_chw = src_tensor_fp32.change_layout(NCHW);
    }
    Tensor src_tensor_fp32_hwc;
    src_tensor_fp32_hwc = src_tensor_fp32_chw.change_layout(NHWC);

    float cosine_distance = ImageUtil::compare_image_data((float*)src_mat_fp32.data,
                                                           (float*)src_tensor_fp32_hwc.data,
                                                           int(src_tensor_fp32_hwc.size()));

    std::vector<double> profile_details;
    profile_details.push_back(opencv_duration);
    profile_details.push_back(vacv_duration);
    profile_details.push_back(static_cast<double>(cosine_distance));
    profile_details.push_back(1);
    return profile_details;
}


} // namespace vacv