//
// Created by b1xian on 2020-10-29.
//

#include <arm_neon.h>
#include <iostream>
#include <stdlib.h>

#include <opencv2/opencv.hpp>

#include "vision/tensor.h"
#include "vision/tensor_converter.h"

#include "neon_normalize/layout_change.cpp"
#include "neon_normalize/dtype_change.cpp"
#include "neon_normalize/mean_std_dev.cpp"


using namespace std;
using namespace cv;
using namespace vision;


int main () {


    Mat src_mat = imread("res/lakers.jpeg");
    int h = src_mat.rows;
    int w = src_mat.cols;
    int c = src_mat.channels();

    vector<Mat> bgr_mats;
    cv::split(src_mat, bgr_mats);

    clock_t start_time = clock();
    cv::Mat mat_mean;
    cv::Mat mat_stddev;
    cv::meanStdDev(src_mat, mat_mean, mat_stddev);
    std::cout << "opencv_meanstddev_cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;





    return 0;
}