//
// Created by b1xian on 2020-10-12.
//
#include <arm_neon.h>
#include <iostream>
#include <stdlib.h>

#include <opencv2/opencv.hpp>

#include "vision/tensor.h"
#include "vision/tensor_converter.h"

#include "neon_normalize/layout_change.cpp"
#include "neon_normalize/dtype_change.cpp"

using namespace std;
using namespace cv;
using namespace vision;

int main() {
    Mat src_mat = imread("res/lakers.jpeg");
    int h = src_mat.rows;
    int w = src_mat.cols;
    int c = src_mat.channels();

    clock_t start_time = clock();
    Mat src_mat_fp32_opencv;
    src_mat.convertTo(src_mat_fp32_opencv, CV_32FC3);
    std::cout << "opencv_u82fp32_cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
    imwrite("output/lakers_fp32_opencv.jpg", src_mat_fp32_opencv);

    vision::Tensor int8_tensor = vision::TensorConverter::convert_from<cv::Mat>(src_mat, true);
    start_time = clock();
    vision::Tensor fp32_tensor = int8_tensor.change_dtype(FP32);
    std::cout << "naive_u82fp32_cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;

    start_time = clock();
    Mat src_mat_fp32_neon(src_mat.size(), CV_32FC3);
    u8_2_fp32((uint8_t*)src_mat.data, (float*)src_mat_fp32_neon.data, h*w*c);
    std::cout << "neon_u82fp32_cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
    imwrite("output/lakers_fp32_neon.jpg", src_mat_fp32_neon);


//    // opencv
//    start_time = clock();
//    std::vector<Mat> bgr_mats;
//    cv::split(src_mat_fp32_opencv, bgr_mats);
//    std::cout << "opencv_hwc2chw_cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
//    start_time = clock();
//    cv::merge(bgr_mats, src_mat);
//    std::cout << "opencv_chw2hwc_cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
//
    vision::Tensor hwc_tensor = vision::TensorConverter::convert_from<cv::Mat>(src_mat_fp32_opencv, true);
//
//    // naive
//    start_time = clock();
//    vision::Tensor naive_chw_tensor = hwc_tensor.change_layout(DLayout::NCHW);
//    std::cout << "naive_hwc2chw_cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
//    start_time = clock();
//    vision::Tensor naive_write_hwc_tensor = naive_chw_tensor.change_layout(DLayout::NHWC);
//    std::cout << "naive_chw2hwc_cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
//
//    // neon
//    start_time = clock();
//    float* hwc_data = (float*)hwc_tensor.data;
//    float* chw_data = (float*)malloc(sizeof(float) * h*w*c);
//    hwc_2_chw_neon_fp32(hwc_data, chw_data, w, h, c);
//    vision::Tensor neon_chw_tensor(w,h,c,NCHW,FP32);
//    neon_chw_tensor.data = chw_data;
//    std::cout << "neon_hwc2chw_cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;

//
//    start_time = clock();
//    float* hwc_data_2 = (float*)malloc(sizeof(float) * h*w*c);
//    chw_2_hwc_neon_fp32(chw_data, hwc_data_2, w, h, c);
//    vision::Tensor neon_write_hwc_tensor(w,h,c,NHWC,FP32);
//    neon_write_hwc_tensor.data = hwc_data_2;
//    std::cout << "neon_chw2hwc_cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
//
//    Mat hwc_mat = vision::TensorConverter::convert_to<cv::Mat>(neon_write_hwc_tensor, true);
//    imwrite("output/lakers_hwc_neon.jpg", hwc_mat);

    return 0;
}


int main1() {
    Mat src_mat = imread("res/lakers.jpeg");
    clock_t start_time = clock();
    std::vector<Mat> bgr_mats;
    cv::split(src_mat, bgr_mats);
    std::cout << "opencv_hwc2chw_cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;

    start_time = clock();
    cv::merge(bgr_mats, src_mat);
    std::cout << "opencv_chw2hwc_cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;

    vision::Tensor hwc_tensor = vision::TensorConverter::convert_from<cv::Mat>(src_mat, true);
    int h = hwc_tensor.h;
    int w = hwc_tensor.w;
    int c = hwc_tensor.c;

    // naive;
    start_time = clock();
    vision::Tensor naive_chw_tensor = hwc_tensor.change_layout(DLayout::NCHW);
    std::cout << "naive_hwc2chw_cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;

    start_time = clock();
    vision::Tensor naive_write_hwc_tensor = naive_chw_tensor.change_layout(DLayout::NHWC);
    std::cout << "naive_chw2hwc_cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;


    // neon
    start_time = clock();
    uint8_t* chw_data = (uint8_t*)malloc(sizeof(uint8_t) * h*w*c);
    uint8_t* hwc_data = (uint8_t*)hwc_tensor.data;
    hwc_2_chw_neon_u8(hwc_data, chw_data, w, h, c);
    vision::Tensor neon_chw_tensor(w,h,c,NCHW,INT8);
    neon_chw_tensor.data = chw_data;
    std::cout << "neon_hwc2chw_cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;


    start_time = clock();
    uint8_t* hwc_data_2 = (uint8_t*)malloc(sizeof(uint8_t) * h*w*c);
    chw_2_hwc_neon_u8((uint8_t*)neon_chw_tensor.data, hwc_data_2, w, h, c);
    vision::Tensor neon_write_hwc_tensor(w,h,c,NCHW,INT8);
    neon_write_hwc_tensor.data = hwc_data_2;
    std::cout << "neon_chw2hwc_cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
    Mat hwc_mat = vision::TensorConverter::convert_to<cv::Mat>(neon_write_hwc_tensor, true);
    imwrite("output/lakers_hwc.jpg", hwc_mat);

    return 0;
}