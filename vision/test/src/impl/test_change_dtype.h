#ifndef VISION_TEST_CHANGE_DTYPE_H
#define VISION_TEST_CHANGE_DTYPE_H

#include <vector>

namespace vacv {

class TestChangeDtype {

public:

    static std::vector<double> test_change_dtype_u8_to_fp32_176x144();
    static std::vector<double> test_change_dtype_fp32_to_u8_176x144();

};

} // namespace vacv

#endif //VISION_TEST_CHANGE_DTYPE_H
