//
// Created by b1xian on 2020-10-14.
//

#include <iostream>
#include <time.h>

#include <arm_neon.h>
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>

#include "vision/tensor.h"
#include "vision/tensor_converter.h"

#include "paddle-resize/image_resize.h"

using namespace std;

void resize_naive(cv::Mat& matSrc, cv::Mat& matDst) {
    uchar* dataDst = matDst.data;
    int stepDst = matDst.step;
    uchar* dataSrc = matSrc.data;
    int stepSrc = matSrc.step;
    int iWidthSrc = matSrc.cols;
    int iHiehgtSrc = matSrc.rows;
    float scale_x = iWidthSrc / matDst.cols;
    float scale_y = iHiehgtSrc / matDst.rows;

    for (int j = 0; j < matDst.rows; ++j)
    {
        // SrcY=(dstY+0.5) * (srcHeight/dstHeight)-0.5
        float fy = (float)((j + 0.5) * scale_y - 0.5);
        // v 整数部分
        int sy = cvFloor(fy);
        sy = std::min(sy, iHiehgtSrc - 2);
        sy = std::max(0, sy);
        // v 小数部分
        fy -= sy;
        short cbufy[2];
        cbufy[0] = cv::saturate_cast<short>((1.f - fy) * 2048);
        cbufy[1] = 2048 - cbufy[0];

        for (int i = 0; i < matDst.cols; ++i)
        {
            // SrcX=(dstX+0.5)* (srcWidth/dstWidth) -0.5
            float fx = (float)((i + 0.5) * scale_x - 0.5);
            // u 整数部分
            int sx = cvFloor(fx);
            // u 小数部分
            fx -= sx;

            if (sx < 0) {
                fx = 0, sx = 0;
            }
            if (sx >= iWidthSrc - 1) {
                fx = 0, sx = iWidthSrc - 2;
            }

            short cbufx[2];
            cbufx[0] = cv::saturate_cast<short>((1.f - fx) * 2048);
            cbufx[1] = 2048 - cbufx[0];

            for (int k = 0; k < matSrc.channels(); ++k)
            {
                // f(i+u,j+v) = (1-u)(1-v)f(i,j) + (1-u)vf(i,j+1) + u(1-v)f(i+1,j) + uvf(i+1,j+1)
                *(dataDst+ j*stepDst + 3*i + k) = (*(dataSrc + sy*stepSrc + 3*sx + k) * cbufx[0] * cbufy[0] +
                                                   *(dataSrc + (sy+1)*stepSrc + 3*sx + k) * cbufx[0] * cbufy[1] +
                                                   *(dataSrc + sy*stepSrc + 3*(sx+1) + k) * cbufx[1] * cbufy[0] +
                                                   *(dataSrc + (sy+1)*stepSrc + 3*(sx+1) + k) * cbufx[1] * cbufy[1]) >> 22;
            }
        }
    }
}


int main() {
    cv::Mat matSrc = cv::imread("res/lakers.jpeg", 0);
    int h = matSrc.rows;
    int w = matSrc.cols;
    int c = matSrc.channels();
    int dst_h = (int) h / 2.7;
    int dst_w = (int) w / 2.7;
    cv::Mat matDst1(dst_h, dst_w, CV_8UC(c));
    cv::Mat matDst2(dst_h, dst_w, CV_8UC(c));
    cv::Mat matDst3(dst_h, dst_w, CV_8UC(c));

    clock_t start_time = clock();
    resize_naive(matSrc, matDst1);
    std::cout << "naive_cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
    cv::imwrite("output/lakers_linear_our.jpg", matDst1);

    start_time = clock();
    cv::resize(matSrc, matDst2, matDst1.size(), 0, 0, 1);
    std::cout << "opencv_cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
    cv::imwrite("output/lakers_linear_opencv.jpg", matDst2);

    start_time = clock();
    ImageResize* is = new ImageResize();
    is->choose((uint8_t*) matSrc.data,
               (uint8_t*) matDst3.data,
               ImageFormat::GRAY,
               w, h,
               dst_w, dst_h);
    std::cout << "paddle_cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
    cv::imwrite("output/lakers_linear_paddle.jpg", matDst3);

    return 0;
}

