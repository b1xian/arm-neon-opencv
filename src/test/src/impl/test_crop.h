#ifndef VISION_TEST_CROP_H
#define VISION_TEST_CROP_H

#include <vector>

#include <opencv2/opencv.hpp>

#include "../../../cv/cv.h"

namespace vacv {

    class TestCrop {

    public:

        static std::vector<double> test_crop(cv::Rect& cv_rect, vision::VRect& va_rect,
                                             vision::DLayout layout, vision::DType dtype);

        static std::vector<double> test_crop_hwc_320x180();

        static std::vector<double> test_crop_hwc_640x360();

        static std::vector<double> test_crop_hwc_1280x720();

        static std::vector<double> test_crop_hwc_1920x1080();

        static std::vector<double> test_crop_hwc_5x5();

        static std::vector<double> test_crop_hwc_5x5_FP32();

        static std::vector<double> test_crop_chw_320x180();

        static std::vector<double> test_crop_chw_5x5();

        static std::vector<double> test_crop_chw_5x5_FP32();


    };

} // namespace vacv

#endif //VISION_TEST_CROP_H
