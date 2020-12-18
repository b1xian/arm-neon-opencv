#include "imencode.h"

#ifdef USE_OPENCV
#include "opencv2/opencv.hpp"
#endif

#include "../common/tensor_converter.h"

namespace va_cv {

void ImEncode::imencode(const vision::Tensor& src, std::vector<unsigned char>& buf, const char* format) {
#ifdef USE_OPENCV
    const auto& mat_src = vision::TensorConverter::convert_to<cv::Mat>(src);
    cv::imencode(format, mat_src, buf);
#endif // USE_OPENCV
}

} // namespace va_cv