#include "test_warp_affine.h"

#include <opencv2/opencv.hpp>

#include "../../../common/tensor_converter.h"
#include "../../../cv/cv.h"
#include "../../../util/image_util.h"
#include "../../../util/perf_util.h"

using namespace vision;

using namespace va_cv;

static std::string test_img_1280x720  = "./res/1280x720.jpg";
static std::string test_img_1280x720_grey  = "./res/1280x720_grey.jpg";


int flags = INTER_LINEAR;
int borderMode = BORDER_CONSTANT;
cv::Scalar sca_border(0, 0, 0, 0);
VScalar vsca_border;

namespace vacv {

std::vector<double> TestWarpAffine::test_warp_affine_hwc_u8() {
    cv::Mat src_mat = cv::imread(test_img_1280x720);
    cv::Mat mat_warpped;

    int out_h = 240;
    int out_w = 240;
    float* m = new float[6]{0.849158f, 0.012257f, -474.827f,
                            -0.01225f, 0.849158f, -379.18f};
    cv::Mat mat_M(2, 3, CV_32FC1);
    memcpy(mat_M.data, m, sizeof(float) * 6);
    Tensor tensor_M(3, 2, 1, NCHW, FP32);
    memcpy(tensor_M.data, m, sizeof(float) * 6);
    delete[] m;

    double opencv_duration;
    {
        TIME_PERF(opencv_duration);
        cv::warpAffine(src_mat, mat_warpped, mat_M, cv::Size(out_w, out_h), flags, borderMode, sca_border);
    }

    Tensor src_tensor = TensorConverter::convert_from<cv::Mat>(src_mat);
    Tensor tensor_warpped;
    double vacv_duration;
    {
        TIME_PERF(vacv_duration);
        va_cv::warp_affine(src_tensor, tensor_warpped, tensor_M, VSize(out_w, out_h));
    }

    float cosine_distance = ImageUtil::compare_image_data((char*)mat_warpped.data,
                                                            (char*)tensor_warpped.data,
                                                            int(tensor_warpped.size()));

    std::vector<double> profile_details;
    profile_details.push_back(opencv_duration);
    profile_details.push_back(vacv_duration);
    profile_details.push_back(static_cast<double>(cosine_distance));
    profile_details.push_back(1);
    return profile_details;
}

std::vector<double> TestWarpAffine::test_warp_affine_chw_u8() {
    cv::Mat src_mat = cv::imread(test_img_1280x720);
    cv::Mat mat_warpped;

    int out_h = 240;
    int out_w = 240;
    float* m = new float[6]{0.849158f, 0.012257f, -474.827f,
                            -0.01225f, 0.849158f, -379.18f};
    cv::Mat mat_M(2, 3, CV_32FC1);
    memcpy(mat_M.data, m, sizeof(float) * 6);
    Tensor tensor_M(3, 2, 1, NCHW, FP32);
    memcpy(tensor_M.data, m, sizeof(float) * 6);
    delete[] m;

    double opencv_duration;
    {
        TIME_PERF(opencv_duration);
        cv::warpAffine(src_mat, mat_warpped, mat_M, cv::Size(out_w, out_h), flags, borderMode, sca_border);
    }

    Tensor src_tensor = TensorConverter::convert_from<cv::Mat>(src_mat);
    Tensor src_tensor_chw = src_tensor.change_layout(NCHW);
    Tensor tensor_warpped_chw;
    double vacv_duration;
    {
        TIME_PERF(vacv_duration);
        va_cv::warp_affine(src_tensor_chw, tensor_warpped_chw, tensor_M, VSize(out_w, out_h));
    }
    Tensor tensor_warpped_hwc = tensor_warpped_chw.change_layout(NHWC);

    float cosine_distance = ImageUtil::compare_image_data((char*)mat_warpped.data,
                                                           (char*)tensor_warpped_hwc.data,
                                                           int(tensor_warpped_hwc.size()));

    std::vector<double> profile_details;
    profile_details.push_back(opencv_duration);
    profile_details.push_back(vacv_duration);
    profile_details.push_back(static_cast<double>(cosine_distance));
    profile_details.push_back(1);
    return profile_details;
}

std::vector<double> TestWarpAffine::test_warp_affine_chw_fp32() {
    cv::Mat src_mat = cv::imread(test_img_1280x720);
    cv::Mat src_mat_fp32;
    src_mat.convertTo(src_mat_fp32, CV_32FC(src_mat.channels()));
    cv::Mat mat_warpped;

    int out_h = 240;
    int out_w = 240;
    float* m = new float[6]{0.849158f, 0.012257f, -474.827f,
                            -0.01225f, 0.849158f, -379.18f};
    cv::Mat mat_M(2, 3, CV_32FC1);
    memcpy(mat_M.data, m, sizeof(float) * 6);
    Tensor tensor_M(3, 2, 1, NCHW, FP32);
    memcpy(tensor_M.data, m, sizeof(float) * 6);
    delete[] m;

    double opencv_duration;
    {
        TIME_PERF(opencv_duration);
        cv::warpAffine(src_mat_fp32, mat_warpped, mat_M, cv::Size(out_w, out_h), flags, borderMode, sca_border);
    }

    Tensor src_tensor = TensorConverter::convert_from<cv::Mat>(src_mat_fp32);
    Tensor src_tensor_chw = src_tensor.change_layout(NCHW);
    Tensor tensor_warpped;
    double vacv_duration;
    {
        TIME_PERF(vacv_duration);
        va_cv::warp_affine(src_tensor_chw, tensor_warpped, tensor_M, VSize(out_w, out_h));
    }
    Tensor tensor_warpped_hwc = tensor_warpped.change_layout(NHWC);

    float cosine_distance = ImageUtil::compare_image_data((float*)mat_warpped.data,
                                                           (float*)tensor_warpped_hwc.data,
                                                           int(tensor_warpped_hwc.size()));

    std::vector<double> profile_details;
    profile_details.push_back(opencv_duration);
    profile_details.push_back(vacv_duration);
    profile_details.push_back(static_cast<double>(cosine_distance));
    profile_details.push_back(1);
    return profile_details;
}

std::vector<double> TestWarpAffine::test_warp_affine_hwc_fp32() {
    cv::Mat src_mat = cv::imread(test_img_1280x720);
    cv::Mat src_mat_fp32;
    src_mat.convertTo(src_mat_fp32, CV_32FC(src_mat.channels()));
    cv::Mat mat_warpped;

    int out_h = 240;
    int out_w = 240;
    float* m = new float[6]{0.849158f, 0.012257f, -474.827f,
                            -0.01225f, 0.849158f, -379.18f};
    cv::Mat mat_M(2, 3, CV_32FC1);
    memcpy(mat_M.data, m, sizeof(float) * 6);
    Tensor tensor_M(3, 2, 1, NCHW, FP32);
    memcpy(tensor_M.data, m, sizeof(float) * 6);
    delete[] m;

    double opencv_duration;
    {
        TIME_PERF(opencv_duration);
        cv::warpAffine(src_mat_fp32, mat_warpped, mat_M, cv::Size(out_w, out_h), flags, borderMode, sca_border);
    }

    Tensor src_tensor = TensorConverter::convert_from<cv::Mat>(src_mat);
    Tensor src_tensor_fp32;
    src_tensor_fp32 = src_tensor.change_dtype(FP32);
    Tensor tensor_warpped;
    double vacv_duration;
    {
        TIME_PERF(vacv_duration);
        va_cv::warp_affine(src_tensor_fp32, tensor_warpped, tensor_M, VSize(out_w, out_h));
    }

    float cosine_distance = ImageUtil::compare_image_data((float*)mat_warpped.data,
                                                           (float*)tensor_warpped.data,
                                                           int(tensor_warpped.size()));

    std::vector<double> profile_details;
    profile_details.push_back(opencv_duration);
    profile_details.push_back(vacv_duration);
    profile_details.push_back(static_cast<double>(cosine_distance));
    profile_details.push_back(1);
    return profile_details;
}

std::vector<double> TestWarpAffine::test_get_rotation_matrix_hwc_u8() {
    cv::Mat src_mat = cv::imread(test_img_1280x720_grey);

    VSize dsize(140, 210);
    float scale = 1.073914;
    float rot_angle = -3.314525;
    VScalar aux_param;
    aux_param.v0 = 738.518372f;
    aux_param.v1 = 537.672852f;
    aux_param.v2 = 204.766998f;
    aux_param.v3 = 73.329681f;

    cv::Mat mat_warpped;
    double opencv_duration;
    {
        TIME_PERF(opencv_duration);
        cv::Mat rot_mat = cv::getRotationMatrix2D(cv::Point(0, 0), rot_angle, scale);
        rot_mat.at<double>(0, 2) = aux_param.v2 - rot_mat.at<double>(0, 0) * aux_param.v0 -
                                   rot_mat.at<double>(0, 1) * aux_param.v1;
        rot_mat.at<double>(1, 2) = aux_param.v3 - rot_mat.at<double>(1, 0) * aux_param.v0 -
                                   rot_mat.at<double>(1, 1) * aux_param.v1;

        cv::warpAffine(src_mat, mat_warpped, rot_mat, cv::Size(dsize.w, dsize.h), flags, borderMode, sca_border);
    }

    Tensor src_tensor = TensorConverter::convert_from<cv::Mat>(src_mat);
    Tensor tensor_warpped;
    double vacv_duration;
    {
        TIME_PERF(vacv_duration);
        va_cv::warp_affine(src_tensor, tensor_warpped, scale, rot_angle, dsize,
                           aux_param, flags, borderMode, vsca_border);
    }

    float cosine_distance = ImageUtil::compare_image_data((char*)mat_warpped.data,
                                                           (char*)tensor_warpped.data,
                                                           int(tensor_warpped.size()));

    std::vector<double> profile_details;
    profile_details.push_back(opencv_duration);
    profile_details.push_back(vacv_duration);
    profile_details.push_back(static_cast<double>(cosine_distance));
    profile_details.push_back(1);
    return profile_details;
}

std::vector<double> TestWarpAffine::test_get_rotation_matrix_chw_u8() {
    cv::Mat src_mat = cv::imread(test_img_1280x720_grey, 0);

    VSize dsize(140, 210);
    float scale = 1.073914;
    float rot_angle = -3.314525;
    VScalar aux_param;
    aux_param.v0 = 738.518372f;
    aux_param.v1 = 537.672852f;
    aux_param.v2 = 204.766998f;
    aux_param.v3 = 73.329681f;

    cv::Mat mat_warpped;
    double opencv_duration;
    {
        TIME_PERF(opencv_duration);
        cv::Mat rot_mat = cv::getRotationMatrix2D(cv::Point(0, 0), rot_angle, scale);
        rot_mat.at<double>(0, 2) = aux_param.v2 - rot_mat.at<double>(0, 0) * aux_param.v0 -
                                   rot_mat.at<double>(0, 1) * aux_param.v1;
        rot_mat.at<double>(1, 2) = aux_param.v3 - rot_mat.at<double>(1, 0) * aux_param.v0 -
                                   rot_mat.at<double>(1, 1) * aux_param.v1;
        cv::warpAffine(src_mat, mat_warpped, rot_mat, cv::Size(dsize.w, dsize.h), flags, borderMode, sca_border);
    }

    Tensor src_tensor = TensorConverter::convert_from<cv::Mat>(src_mat);
    Tensor src_tensor_chw = src_tensor.change_layout(NCHW);
    src_tensor_chw.layout = NCHW;
    Tensor tensor_warpped;
    double vacv_duration;
    {
        TIME_PERF(vacv_duration);
        va_cv::warp_affine(src_tensor_chw, tensor_warpped, scale, rot_angle, dsize,
                           aux_param, flags, borderMode, vsca_border);
    }
    Tensor tensor_warpped_hwc = tensor_warpped.change_layout(NHWC);
    float cosine_distance = ImageUtil::compare_image_data((char*)mat_warpped.data,
                                                           (char*)tensor_warpped_hwc.data,
                                                           int(tensor_warpped_hwc.size()));

    std::vector<double> profile_details;
    profile_details.push_back(opencv_duration);
    profile_details.push_back(vacv_duration);
    profile_details.push_back(static_cast<double>(cosine_distance));
    profile_details.push_back(1);
    return profile_details;
}

std::vector<double> TestWarpAffine::test_get_rotation_matrix_hwc_fp32() {
    cv::Mat src_mat = cv::imread(test_img_1280x720_grey, 0);
    cv::Mat src_mat_fp32;
    src_mat.convertTo(src_mat_fp32, CV_32FC(src_mat.channels()));

    VSize dsize(140, 210);
    float scale = 1.073914;
    float rot_angle = -3.314525;
    VScalar aux_param;
    aux_param.v0 = 738.518372f;
    aux_param.v1 = 537.672852f;
    aux_param.v2 = 204.766998f;
    aux_param.v3 = 73.329681f;

    cv::Mat mat_warpped;
    double opencv_duration;
    {
        TIME_PERF(opencv_duration);
        cv::Mat rot_mat = cv::getRotationMatrix2D(cv::Point(0, 0), rot_angle, scale);
        rot_mat.at<double>(0, 2) = aux_param.v2 - rot_mat.at<double>(0, 0) * aux_param.v0 -
                                   rot_mat.at<double>(0, 1) * aux_param.v1;
        rot_mat.at<double>(1, 2) = aux_param.v3 - rot_mat.at<double>(1, 0) * aux_param.v0 -
                                   rot_mat.at<double>(1, 1) * aux_param.v1;
        cv::warpAffine(src_mat_fp32, mat_warpped, rot_mat, cv::Size(dsize.w, dsize.h), flags, borderMode, sca_border);
    }

    Tensor src_tensor = TensorConverter::convert_from<cv::Mat>(src_mat);
    Tensor src_tensor_fp32;
    src_tensor_fp32 = src_tensor.change_dtype(FP32);
    Tensor tensor_warpped;
    double vacv_duration;
    {
        TIME_PERF(vacv_duration);
        va_cv::warp_affine(src_tensor_fp32, tensor_warpped, scale, rot_angle, dsize,
                           aux_param, flags, borderMode, vsca_border);
    }

    float cosine_distance = ImageUtil::compare_image_data((float*)mat_warpped.data,
                                                           (float*)tensor_warpped.data,
                                                           int(tensor_warpped.size()));

    std::vector<double> profile_details;
    profile_details.push_back(opencv_duration);
    profile_details.push_back(vacv_duration);
    profile_details.push_back(static_cast<double>(cosine_distance));
    profile_details.push_back(1);
    return profile_details;
}

} // namespace vacv