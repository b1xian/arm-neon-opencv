#ifndef VISION_TEST_CVT_COLOR
#define VISION_TEST_CVT_COLOR

#include <string>
#include <vector>

namespace vacv {

class TestCvtColor {

public:

    static std::vector<double> test_nv21_to_bgr(std::string img_path);

    static std::vector<double> test_nv21_to_bgr_176x144();

    static std::vector<double> test_nv21_to_bgr_640x360();

    static std::vector<double> test_nv21_to_bgr_1280x720();

    static std::vector<double> test_nv21_to_bgr_1920x1080();

    static std::vector<double> test_nv21_to_bgr_2560x1440();

};

} // namespace vacv

#endif //VISION_TEST_CVT_COLOR
