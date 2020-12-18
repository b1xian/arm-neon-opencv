#ifndef VISION_TEST_WARP_AFFINE_H
#define VISION_TEST_WARP_AFFINE_H

#include <string>
#include <vector>

namespace vacv {

class TestWarpAffine {

public:

    static std::vector<double> test_warp_affine_hwc_u8();

    static std::vector<double> test_warp_affine_hwc_fp32();

    static std::vector<double> test_warp_affine_chw_u8();

    static std::vector<double> test_warp_affine_chw_fp32();

    static std::vector<double> test_get_rotation_matrix_hwc_u8();

    static std::vector<double> test_get_rotation_matrix_chw_u8();

    static std::vector<double> test_get_rotation_matrix_hwc_fp32();

};

} // namespace vacv

#endif //VISION_TEST_WARP_AFFINE_H
