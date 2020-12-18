//
// Created by b1xian on 2020-10-14.
//

#include <iostream>
#include <fstream>
#include <time.h>

#include <arm_neon.h>
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>

#include "../vision/common/tensor.h"
#include "../vision/common/tensor_converter.h"

#include "paddle-resize/cubic_resize_float16.cpp"
#include "paddle-resize/cubic_resize_float32.cpp"
#include "paddle-resize/cubic_resize_naive_chw.cpp"
#include "neon_normalize/layout_change.cpp"

using namespace std;

int main() {
    cv::Mat matSrc = cv::imread("res/face1280720.jpg", 1);
    int h = matSrc.rows;
    int w = matSrc.cols;
    int c = matSrc.channels();
    int dst_h = 360;
    int dst_w = 640;
    cv::Mat matDst1(dst_h, dst_w, CV_8UC(c));
    cv::Mat matDst2(dst_h, dst_w, CV_8UC(c));
    cv::Mat matDst3(dst_h, dst_w, CV_8UC(c));

    clock_t start_time = clock();
    cv::resize(matSrc, matDst1, matDst1.size(), 0, 0, CV_INTER_CUBIC);
    std::cout << "bicubic_opencv_cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
    cv::imwrite("output/bicubic_opencv.jpg", matDst1);

    start_time = clock();
    cv::resize(matSrc, matDst2, matDst1.size(), 0, 0, CV_INTER_LINEAR);
    std::cout << "bilinear_opencv_cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
    cv::imwrite("output/bilinear_opencv.jpg", matDst2);

    // bicubic使用三通道分别计算
    cv::Mat mat_src_fp32;
    matSrc.convertTo(mat_src_fp32, CV_32FC3);

    // float32 bicubic
    start_time = clock();
    vision::Tensor tensor = vision::TensorConverter::convert_from<cv::Mat>(mat_src_fp32, true);
    float* dst_data_fp32_hwc = (float*)malloc(sizeof(float) * dst_h*dst_w*c);
    du_hwc_resize_bicubic_naive((float*)tensor.data, w, h, dst_data_fp32_hwc, dst_w, dst_h);
    vision::Tensor resize_tensor(dst_w, dst_h, c, vision::DLayout::NHWC, vision::DType::FP32);
    resize_tensor.data = dst_data_fp32_hwc;
    std::cout << "bicubic_naive_fp32_cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
    cv::Mat bicubic_resize_img = vision::TensorConverter::convert_to<cv::Mat>(resize_tensor, true);
    cv::imwrite("output/bicubic_naive_fp32.jpg", bicubic_resize_img);

    // float16 bicubic
//    cv::Mat mat_src_fp16;
//    matSrc.convertTo(mat_src_fp16, CV_16SC3);
//    vision::Tensor tensor_fp16 = vision::TensorConverter::convert_from<cv::Mat>(mat_src_fp16, true);
//    tensor_fp16 = tensor_fp16.change_layout(vision::DLayout::NCHW);
//    __fp16* src_data_fp16 = (__fp16*)tensor_fp16.data;
//    __fp16* dst_data_fp16 = (__fp16*)malloc(sizeof(__fp16) * dst_h*dst_w*c);
//    start_time = clock();
//    du_resize_bicubic_fp16(src_data_fp16, w, h, dst_data_fp16, dst_w, dst_h);
//    std::cout << "bicubic_neon_fp16: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
//    vision::Tensor resize_tensor_fp16(dst_w, dst_h, c, vision::DLayout::NCHW, vision::DType::FP16);
//    resize_tensor_fp16.data = dst_data_fp16;
//    resize_tensor_fp16 = resize_tensor_fp16.change_layout(vision::DLayout::NHWC);
//
//    cv::Mat bicubic_resize_img_fp16 = vision::TensorConverter::convert_to<cv::Mat>(resize_tensor_fp16, true);
//    cv::imwrite("output/bicubic_neon_fp16.jpg", bicubic_resize_img_fp16);

    return 0;
}

