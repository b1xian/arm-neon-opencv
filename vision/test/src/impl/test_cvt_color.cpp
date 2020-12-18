#include "test_cvt_color.h"

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

std::vector<double> TestCvtColor::test_nv21_to_bgr(std::string img_path) {
    cv::Mat src_mat = cv::imread(img_path);
    int h = src_mat.rows;
    int w = src_mat.cols;
    // to nv21
    cv::Mat yuv_img(h * 3 / 2, w, CV_8UC1);
    ImageUtil::bgr2nv21((unsigned char*)src_mat.data, (unsigned char*)yuv_img.data, w, h);

    cv::Mat yuv_to_bgr_mat;

    double opencv_duration;
    {
        TIME_PERF(opencv_duration);
        cv::cvtColor(yuv_img, yuv_to_bgr_mat, CV_YUV2BGR_I420);
    }

    Tensor yuv_tensor = TensorConverter::convert_from<cv::Mat>(yuv_img);
    Tensor yuv_to_bgr_tensor;
    double vacv_duration;
    {
        TIME_PERF(vacv_duration);
        va_cv::cvt_color(yuv_tensor, yuv_to_bgr_tensor, COLOR_YUV2BGR_NV21);
    }

    float cosine_distance = ImageUtil::compare_image_data((unsigned char*)src_mat.data,
                                                            (unsigned char*)yuv_to_bgr_tensor.data,
                                                            int(yuv_to_bgr_tensor.size()));

    std::vector<double> profile_details;
    profile_details.push_back(opencv_duration);
    profile_details.push_back(vacv_duration);
    profile_details.push_back(static_cast<double>(cosine_distance));
    profile_details.push_back(1);
    return profile_details;
}

std::vector<double> TestCvtColor::test_nv21_to_bgr_176x144() {
    return test_nv21_to_bgr(test_img_176x144);
}

std::vector<double> TestCvtColor::test_nv21_to_bgr_640x360() {
    return test_nv21_to_bgr(test_img_640x360);
}

std::vector<double> TestCvtColor::test_nv21_to_bgr_1280x720() {
    return test_nv21_to_bgr(test_img_1280x720);
}

std::vector<double> TestCvtColor::test_nv21_to_bgr_1920x1080() {
    return test_nv21_to_bgr(test_img_1920x1080);
}

std::vector<double> TestCvtColor::test_nv21_to_bgr_2560x1440() {
    return test_nv21_to_bgr(test_img_2560x1440);
}

} // namespace vacv