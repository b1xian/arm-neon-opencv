#ifndef VISION_TENSOR_CONVERTER_H
#define VISION_TENSOR_CONVERTER_H

#include "tensor.h"

namespace vision {

class TensorConverter {
public:
    template <typename T>
    static T convert_to(const Tensor& tensor, bool copy = false);

    template <typename T>
    static Tensor convert_from(const T& mat, bool copy = false);

};

} // namespace vision

#endif //VISION_TENSOR_CONVERTER_H
