#ifndef VISION_TEST_NORMALIZE_H
#define VISION_TEST_NORMALIZE_H

#include <string>
#include <vector>

namespace vacv {

class TestNormalize {

public:

    static std::vector<double> test_normalize_hwc(std::string img_path);

    static std::vector<double> test_normalize_chw(std::string img_path);

    static std::vector<double> test_normalize_hwc_176x144();

    static std::vector<double> test_normalize_chw_176x144();

    static std::vector<double> test_normalize_hwc_284x214();

    static std::vector<double> test_normalize_chw_284x214();

};

} // namespace vacv

#endif //VISION_TEST_NORMALIZE_H
