#include "test_crop.h"

#include <string>

#include <opencv2/opencv.hpp>

#include "../../../common/tensor_converter.h"
#include "../../../util/image_util.h"
#include "../../../util/perf_util.h"

using namespace vision;
using namespace va_cv;

static std::string test_img_2560x1440 = "./res/2560x1440.jpeg";

static VRect v_rect320x180(0, 0, 320, 180);
static VRect v_rect640x360(0, 0, 640, 360);
static VRect v_rect1280x720(0, 0, 1280, 720);
static VRect v_rect1920x1080(0, 0, 1920, 1080);
static VRect v_rect_5x5(0, 0, 5, 5);

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

static cv::Rect cv_rect_5x5(cvRound(v_rect_5x5.left), cvRound(v_rect_5x5.top),
                            cvRound(v_rect_5x5.right  - v_rect_5x5.left),
                            cvRound(v_rect_5x5.bottom - v_rect_5x5.top));

namespace vacv {

    static const char* TAG = "TestCrop";


    std::vector<double> TestCrop::test_crop(cv::Rect& cv_rect, vision::VRect& va_rect,
                                            vision::DLayout layout, vision::DType dtype) {
        cv::Mat src_mat = cv::imread(test_img_2560x1440);
        if (dtype == FP32) {
            src_mat.convertTo(src_mat, CV_32FC(src_mat.channels()));
        }
        cv::Mat cv_crop_mat;
        double opencv_duration = 0.;
        {
            TIME_PERF(opencv_duration);
            cv_crop_mat = src_mat(cv_rect).clone();
        }

        Tensor src_tensor = TensorConverter::convert_from<cv::Mat>(src_mat);
        src_tensor = src_tensor.change_layout(layout);
        src_tensor = src_tensor.change_dtype(dtype);

        Tensor vacv_crop_tensor;
        double vacv_duration = 0.;
        {
            TIME_PERF(vacv_duration);
            va_cv::crop(src_tensor, vacv_crop_tensor, va_rect);
        }

        if (layout == NCHW) {
            vacv_crop_tensor = vacv_crop_tensor.change_layout(NHWC);
        }

        float cosine_distance = 0.f;
        if (dtype == INT8) {
            cosine_distance = ImageUtil::compare_image_data((char*)cv_crop_mat.data,
                                                             (char*)vacv_crop_tensor.data,
                                                             int(vacv_crop_tensor.size()));
        } else if (dtype == FP32) {
            cosine_distance = ImageUtil::compare_image_data((float*)cv_crop_mat.data,
                                                             (float*)vacv_crop_tensor.data,
                                                             int(vacv_crop_tensor.size()));
        }

        std::vector<double> profile_details;
        profile_details.push_back(opencv_duration);
        profile_details.push_back(vacv_duration);
        profile_details.push_back(static_cast<double>(cosine_distance));
        profile_details.push_back(1);
        return profile_details;
    }

    std::vector<double> TestCrop::test_crop_hwc_5x5() {
        return test_crop(cv_rect_5x5, v_rect_5x5, NHWC, INT8);
    }

    std::vector<double> TestCrop::test_crop_hwc_5x5_FP32() {
        return test_crop(cv_rect_5x5, v_rect_5x5, NHWC, FP32);
    }

    std::vector<double> TestCrop::test_crop_hwc_320x180() {
        return test_crop(cv_rect320x180, v_rect320x180, NHWC, INT8);
    }

    std::vector<double> TestCrop::test_crop_hwc_640x360() {
        return test_crop(cv_rect640x360, v_rect640x360, NHWC, INT8);
    }

    std::vector<double> TestCrop::test_crop_hwc_1280x720() {
        return test_crop(cv_rect1280x720, v_rect1280x720, NHWC, INT8);
    }

    std::vector<double> TestCrop::test_crop_hwc_1920x1080() {
        return test_crop(cv_rect1920x1080, v_rect1920x1080, NHWC, INT8);
    }

    std::vector<double> TestCrop::test_crop_chw_320x180() {
        return test_crop(cv_rect320x180, v_rect320x180, NCHW, INT8);
    }

    std::vector<double> TestCrop::test_crop_chw_320x180_FP32() {
        return test_crop(cv_rect320x180, v_rect320x180, NCHW, FP32);
    }

    std::vector<double> TestCrop::test_crop_chw_640x360() {
        return test_crop(cv_rect640x360, v_rect640x360, NCHW, INT8);
    }

    std::vector<double> TestCrop::test_crop_chw_5x5() {
        return test_crop(cv_rect_5x5, v_rect_5x5, NCHW, INT8);
    }

    std::vector<double> TestCrop::test_crop_chw_5x5_FP32() {
        return test_crop(cv_rect_5x5, v_rect_5x5, NCHW, FP32);
    }

} // namespace vacv