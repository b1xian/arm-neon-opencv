#include <stdexcept>

#include "opencv2/opencv.hpp"

#include "tensor_converter.h"

namespace vision {

static const char* TAG = "TensorConverter";

template <typename T>
Tensor TensorConverter::convert_from(const T& mat, bool copy) {
    return Tensor();
}

template <>
cv::Mat TensorConverter::convert_to<cv::Mat>(const Tensor& tensor, bool copy) {
    if (tensor.empty()) {
        return cv::Mat();
    }

    int w = tensor.w;
    int h = tensor.h;
    int c = tensor.c;
    auto dtype = tensor.dtype;

    int mat_type;
    if (dtype == FP32) {
        mat_type = CV_32FC(c);
    } else if (dtype == FP16) {
        mat_type = CV_16UC(c);
    } else if (dtype == INT8) {
        mat_type = CV_8UC(c);
    } else if (dtype == FP64) {
        mat_type = CV_64FC(c);
    } else {
        throw std::runtime_error("TensorConverter exception when converted to cv::Mat, dtype not supported!");
    }

    if (copy) {
        cv::Mat mat(h, w, mat_type);
        memcpy(mat.data, tensor.data, tensor.len());
        return mat;
    }
    return cv::Mat(h, w, mat_type, tensor.data);
}

template <>
Tensor TensorConverter::convert_from<cv::Mat>(const cv::Mat& mat, bool copy) {
    if (mat.empty()) {
        return Tensor();
    }

    int w = mat.cols;
    int h = mat.rows;
    int c = mat.channels();
    auto depth = mat.depth();
    DType dtype = INT8;
    switch (depth) {
        case CV_8U:
        case CV_8S:
            dtype = INT8;
            break;
        case CV_16U:
        case CV_16S:
            dtype = FP16;
            break;
        case CV_32S:
        case CV_32F:
            dtype = FP32;
            break;
        case CV_64F:
            dtype = FP64;
            break;
        default:
            throw std::runtime_error("TensorConverter exception when converted from cv::Mat, depth not supported!");
    }

    if (copy) {
        Tensor t(w, h, c, dtype, NHWC);
        memcpy(t.data, mat.data, t.len());
        return t;
    }
    return Tensor(w, h, c, mat.data, dtype, NHWC);
}


} // namespace vision