//
// Created by b1xian on 2020-10-29.
//

#include <arm_neon.h>
#include <iostream>
#include <stdlib.h>

#include <opencv2/opencv.hpp>

#include "../vision/common/tensor.h"
#include "../vision/common/tensor_converter.h"

#include "neon_normalize/layout_change.cpp"
#include "neon_normalize/dtype_change.cpp"
#include "neon_normalize/mean_std_dev.cpp"


using namespace std;
using namespace cv;
using namespace vision;


int main () {


    Mat src_mat = imread("res/lakers25601440.jpeg");
    int h = src_mat.rows;
    int w = src_mat.cols;
    int c = src_mat.channels();

    clock_t start_time = clock();
    cv::Mat mat_mean;
    cv::Mat mat_stddev;
    cv::meanStdDev(src_mat, mat_mean, mat_stddev);

    std::cout << "mean:" << ((double*)mat_mean.data)[0] << "," << ((double*)mat_mean.data)[1] << "," << ((double*)mat_mean.data)[2] << std::endl;
    std::cout << "stddev:"  << ((double*)mat_stddev.data)[0] << "," << ((double*)mat_stddev.data)[1] << "," << ((double*)mat_stddev.data)[2] << std::endl;

    cv::Mat src_mat_f;
    cv::Mat mat_dst;
    vector<Mat> bgr_mats;
    src_mat.convertTo(src_mat_f, CV_32FC3);
    cv::split(src_mat_f, bgr_mats);
    int k = 0;
    for (auto& mat : bgr_mats) {
        auto m = ((double *)mat_mean.data)[k];
        auto s = ((double *)mat_stddev.data)[k];
        mat = (mat - m) / (s + 1e-6);
        k++;
    }
    cv::merge(bgr_mats, mat_dst);
    std::cout << "opencv meanStdDev : " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s"  << std::endl;
    // --------------------------------------------------------------------------------------------

    Tensor u8_tensor = TensorConverter::convert_from<cv::Mat>(src_mat, true);
    float* fp32_data = (float*)malloc(sizeof(float)*h*w*c);
    u8_2_fp32((uint8_t*)src_mat.data, fp32_data, h*w*c);
    start_time = clock();
    float* mean = (float*)malloc(sizeof(float)*c);
    float* stddev = (float*)malloc(sizeof(float)*c);
    mean_std_dev_fp32_bgr_hwc(fp32_data, w * h, mean, stddev);

    std::cout << "mean:"  << mean[0] << "," << mean[1] << "," << mean[2] << std::endl;
    std::cout << "stddev:"  << stddev[0] << "," << stddev[1] << "," << stddev[2] << std::endl;

    float* fp32_data_normalize = (float*)malloc(sizeof(float)*h*w*c);
    normalize_fp32_bgr_hwc(fp32_data, fp32_data_normalize, w * h, mean, stddev);
    std::cout << "neon hwc meanStdDev : " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
    // --------------------------------------------------------------------------------------------



    float* fp32_data_chw = (float*)malloc(sizeof(float)*h*w*c);
    start_time = clock();
    float* mean1 = (float*)malloc(sizeof(float)*c);
    float* stddev1 = (float*)malloc(sizeof(float)*c);
    hwc_2_chw_neon_fp32(fp32_data, fp32_data_chw, w, h, c);
    mean_std_dev_fp32_bgr_chw(fp32_data_chw, w * h, c, mean1, stddev1);

    std::cout << "mean:"  << mean1[0] << "," << mean1[1] << "," << mean1[2] << std::endl;
    std::cout << "stddev" << stddev1[0] << "," << stddev1[1] << "," << stddev1[2] << std::endl;

    float* fp32_data_normalize_chw = (float*)malloc(sizeof(float)*h*w*c);
    normalize_fp32_bgr_chw(fp32_data_chw, fp32_data_normalize_chw, w * h, c, mean1, stddev1);
    float* fp32_data_normalize_chw_hwc = (float*)malloc(sizeof(float)*h*w*c);
    chw_2_hwc_neon_fp32(fp32_data_normalize_chw, fp32_data_normalize_chw_hwc, w, h, c);
    std::cout << "neon chw meanStdDev : " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
    // --------------------------------------------------------------------------------------------

    return 0;
}