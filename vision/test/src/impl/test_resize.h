#ifndef VISION_TEST_RESIZE_H
#define VISION_TEST_RESIZE_H

#include <vector>

namespace vacv {

class TestResize {

public:

    static std::vector<double> test_resize_bilinear_hwc_u8_320x180();

    static std::vector<double> test_resize_bilinear_chw_u8_320x180();

    static std::vector<double> test_resize_bilinear_hwc_fp32_320x180();

    static std::vector<double> test_resize_bilinear_chw_fp32_320x180();

    static std::vector<double> test_resize_cubic_hwc_fp32_320x180();

    static std::vector<double> test_resize_cubic_chw_fp32_320x180();

};

} // namespace vacv

#endif //VISION_TEST_RESIZE_H
