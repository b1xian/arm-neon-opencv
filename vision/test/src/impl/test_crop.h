#ifndef VISION_TEST_CROP_H
#define VISION_TEST_CROP_H

#include <vector>

namespace vacv {

class TestCrop {

public:

    static std::vector<double> test_crop320x180();

    static std::vector<double> test_crop640x360();

    static std::vector<double> test_crop1280x720();

    static std::vector<double> test_crop1920x1080();

};

} // namespace vacv

#endif //VISION_TEST_CROP_H
