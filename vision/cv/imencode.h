#ifndef VISION_IMENCODE_H
#define VISION_IMENCODE_H

#include <vector>

#include "../common/tensor.h"

namespace va_cv {

class ImEncode {
public:
    static void imencode(const vision::Tensor& src, std::vector<unsigned char>& buf, const char* format);
};

} // namespace va_cv

#endif //VISION_IMENCODE_H
