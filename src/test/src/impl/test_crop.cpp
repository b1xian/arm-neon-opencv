#include "test_crop.h"

#include <string>

#include <opencv2/opencv.hpp>

#include "../../../common/tensor_converter.h"
#include "../../../cv/cv.h"
#include "../../../util/image_util.h"
#include "../../../util/perf_util.h"

using namespace vision;
using namespace va_cv;

static std::string test_img_2560x1440 = "./res/2560x1440.jpeg";

static VRect v_rect320x180(0, 0, 320, 180);
static VRect v_rect640x360(0, 0, 640, 360);
static VRect v_rect1280x720(0, 0, 1280, 720);
static VRect v_rect1920x1080(0, 0, 1920, 1080);

static cv::Rect cv_rect320x180(cvRound(v_rect320x180.left), cvRound(v_rect320x180.top),
                        cvRound(v_rect320x180.right - v_rect320x180.left),
                        cvRound(v_rect320x180.bottom - v_rect320x180.top));
static cv::Rect cv_rect640x360(cvRound(v_rect640x360.left), cvRound(v_rect640x360.top),
                        cvRound(v_rect640x360.right  - v_rect640x360.left),
                        cvRound(v_rect640x360.bottom - v_rect640x360.top));
static cv::Rect cv_rect1280x720(cvRound(v_rect1280x720.left), cvRound(v_rect1280x720.top),
                         cvRound(v_rect1280x720.right  - v_rect1280x720.left),
                         cvRound(v_rect1280x720.bottom - v_rect1280x720.top));
static cv::Rect cv_rect1920x1080(cvRound(v_rect1920x1080.left), cvRound(v_rect1920x1080.top),
                          cvRound(v_rect1920x1080.right  - v_rect1920x1080.left),
                          cvRound(v_rect1920x1080.bottom - v_rect1920x1080.top));

namespace vacv {

std::vector<double> TestCrop::test_crop320x180() {
    cv::Mat src_mat = cv::imread(test_img_2560x1440);
    cv::Mat cv_crop320x180_mat;
    double opencv_duration;
    {
        TIME_PERF(opencv_duration);
        cv_crop320x180_mat = src_mat(cv_rect320x180).clone();
    }

    Tensor src_tensor = TensorConverter::convert_from<cv::Mat>(src_mat);
    Tensor vacv_crop320x180_tensor;
    double vacv_duration;
    {
        TIME_PERF(vacv_duration);
        va_cv::crop(src_tensor, vacv_crop320x180_tensor, v_rect320x180);
    }

    float cosine_distance = ImageUtil::compare_image_data((char*)cv_crop320x180_mat.data,
                                                            (char*)vacv_crop320x180_tensor.data,
                                                            int(vacv_crop320x180_tensor.size()));

    std::vector<double> profile_details;
    profile_details.push_back(opencv_duration);
    profile_details.push_back(vacv_duration);
    profile_details.push_back(static_cast<double>(cosine_distance));
    profile_details.push_back(1);
    return profile_details;
}

std::vector<double> TestCrop::test_crop640x360() {

    cv::Mat src_mat = cv::imread(test_img_2560x1440);
    cv::Mat cv_crop640x360_mat;
    double opencv_duration;
    {
        TIME_PERF(opencv_duration);
        cv_crop640x360_mat = src_mat(cv_rect640x360).clone();
    }

    Tensor src_tensor = TensorConverter::convert_from<cv::Mat>(src_mat);
    Tensor vacv_crop640x360_tensor;
    double vacv_duration;
    {
        TIME_PERF(vacv_duration);
        va_cv::crop(src_tensor, vacv_crop640x360_tensor, v_rect640x360);
    }

    float cosine_distance = ImageUtil::compare_image_data((char*)cv_crop640x360_mat.data,
                                                            (char*)vacv_crop640x360_tensor.data,
                                                           int(vacv_crop640x360_tensor.size()));

    std::vector<double> profile_details;
    profile_details.push_back(opencv_duration);
    profile_details.push_back(vacv_duration);
    profile_details.push_back(static_cast<double>(cosine_distance));
    profile_details.push_back(1);

    return profile_details;
}

std::vector<double>  TestCrop::test_crop1280x720() {
    cv::Mat src_mat = cv::imread(test_img_2560x1440);
    cv::Mat cv_crop1280x720_mat;
    double opencv_duration;
    {
        TIME_PERF(opencv_duration);
        cv_crop1280x720_mat = src_mat(cv_rect1280x720).clone();
    }

    Tensor src_tensor = TensorConverter::convert_from<cv::Mat>(src_mat);
    Tensor vacv_crop1280x720_tensor;
    double vacv_duration;
    {
        TIME_PERF(vacv_duration);
        va_cv::crop(src_tensor, vacv_crop1280x720_tensor, v_rect1280x720);
    }

    float cosine_distance = ImageUtil::compare_image_data((char*)cv_crop1280x720_mat.data,
                                                            (char*)vacv_crop1280x720_tensor.data,
                                                           int(vacv_crop1280x720_tensor.size()));

    std::vector<double> profile_details;
    profile_details.push_back(opencv_duration);
    profile_details.push_back(vacv_duration);
    profile_details.push_back(static_cast<double>(cosine_distance));
    profile_details.push_back(1);

    return profile_details;
}

std::vector<double>  TestCrop::test_crop1920x1080() {
    cv::Mat src_mat = cv::imread(test_img_2560x1440);
    cv::Mat cv_crop1920x1080_mat;
    double opencv_duration;
    {
        TIME_PERF(opencv_duration);
        cv_crop1920x1080_mat = src_mat(cv_rect1920x1080).clone();
    }

    Tensor src_tensor = TensorConverter::convert_from<cv::Mat>(src_mat);
    Tensor vacv_crop1920x1080_tensor;
    double vacv_duration;
    {
        TIME_PERF(vacv_duration);
        va_cv::crop(src_tensor, vacv_crop1920x1080_tensor, v_rect1920x1080);
    }

    float cosine_distance = ImageUtil::compare_image_data((char*)cv_crop1920x1080_mat.data,
                                                            (char*)vacv_crop1920x1080_tensor.data,
                                                           int(vacv_crop1920x1080_tensor.size()));

    std::vector<double> profile_details;
    profile_details.push_back(opencv_duration);
    profile_details.push_back(vacv_duration);
    profile_details.push_back(static_cast<double>(cosine_distance));
    profile_details.push_back(1);

    return profile_details;
}


} // namespace vacv