#ifndef VISION_RESIZE_NEON_H
#define VISION_RESIZE_NEON_H

#include <cstdint>

namespace va_cv {

class ResizeNeon {

public:

    static void resize_neon_inter_linear_one_channel(const uint8_t* src, int w_in, int h_in,
                                                     uint8_t* dst, int w_out, int h_out);

    static void resize_neon_inter_linear_three_channel(const uint8_t* src, int w_in, int h_in,
                                                       uint8_t* dst, int w_out, int h_out);

};

}
#endif //VISION_RESIZE_NEON_H
