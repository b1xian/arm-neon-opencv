#ifndef VISION_TEST_CHANGE_LAYOUT_H
#define VISION_TEST_CHANGE_LAYOUT_H

#include <vector>

namespace vacv {

class TestChangeLayout {

public:

    static std::vector<double> test_change_layout_hwc_to_chw_u8_176x144();
    static std::vector<double> test_change_layout_hwc_to_chw_fp32_176x144();
    static std::vector<double> test_change_layout_chw_to_hwc_u8_176x144();
    static std::vector<double> test_change_layout_chw_to_hwc_fp32_176x144();

};

} // namespace vacv

#endif //VISION_TEST_CHANGE_LAYOUT_H
