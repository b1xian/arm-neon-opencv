//
// Created by b1xian on 2020-10-29.
//

#include <arm_neon.h>
#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <cmath>

#include <opencv2/opencv.hpp>

#include "../vision/common/tensor.h"
#include "../vision/common/tensor_converter.h"

#include "neon_normalize/layout_change.cpp"
#include "neon_normalize/dtype_change.cpp"
#include "neon_normalize/mean_std_dev.cpp"
#include "neon_warpaffine/warp_affine.cpp"


using namespace std;
using namespace cv;
using namespace vision;

int main () {

    assert(1 == 2);

    cv::Mat mat_src = imread("res/face1280720.jpg");
    int h = mat_src.rows;
    int w = mat_src.cols;
    int c = mat_src.channels();

    float* m = (float*)malloc(6*sizeof(float));
    m[0] = 0.849158;
    m[1] = 0.012257;
    m[2] = -474.827;
    m[3] = -0.01225;
    m[4] = 0.849158;
    m[5] = -379.18;
    cv::Mat mat_M(2, 3, CV_32FC1, m);

    int flags = INTER_LINEAR;
    int borderMode = BORDER_CONSTANT;
    cv::Scalar sca_border(0, 0, 0, 0);

    int out_h = 240;
    int out_w = 240;
    cv::Mat mat_dst;
    clock_t start_time = clock();
    cv::warpAffine(mat_src, mat_dst, mat_M, cv::Size(out_w, out_h), flags, borderMode, sca_border);
    std::cout << "opencv warp_affine cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s"  << std::endl;
    cv::imwrite("output/warp_affine_opencv.jpg", mat_dst);


    cv::Mat warp_mat_naive = cv::Mat(cv::Size(out_w, out_h), mat_src.type(), cv::Scalar::all(0));
    start_time = clock();

    uint8_t* src = (uint8_t*)mat_src.data;
    uint8_t* dst = (uint8_t*)warp_mat_naive.data;
    warp_affine_naive_hwc_u8(src, w, h, c, dst, out_w, out_h, m);
    std::cout << "naive warp_affine cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s"  << std::endl;
    cv::imwrite("output/warp_affine_naive.jpg", warp_mat_naive);
}


