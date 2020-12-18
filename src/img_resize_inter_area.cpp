//
// Created by b1xian on 2020-10-14.
//

#include <iostream>
#include <time.h>
#include <math.h>

#include <arm_neon.h>
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>

#include "../vision/common/tensor.h"
#include "../vision/common/tensor_converter.h"

#include "paddle-resize/image_resize.h"

using namespace std;

void resize_inter_area_naive(uint8_t* src,
                            int w_in,
                            int h_in,
                            int c,
                            uint8_t* dst,
                            int w_out,
                            int h_out) {
    double inv_scale_x = (double)(w_out / w_in);
    double inv_scale_y = (double)(h_out / h_in);
    double scale_x = 1./inv_scale_x, scale_y = 1./inv_scale_y;
    // TODO 四舍五入吗
    int iscale_x = static_cast<int>(scale_x);
    int iscale_y = static_cast<int>(scale_y);
    int k, sx, sy, dx, dy;

    int area = iscale_x*iscale_y;
//    size_t srcstep = src.step / src.elemSize1();
    size_t srcstep = w_in * c;
    int* _ofs = new int[area + w_out*c];
//    AutoBuffer<int> _ofs(area + dsize.width*cn);
    int* ofs = _ofs; // area个值
    int* xofs = ofs + area; // w_out*c个值

    for( sy = 0, k = 0; sy < iscale_y; sy++ )
        for( sx = 0; sx < iscale_x; sx++ )
            ofs[k++] = (int)(sy*srcstep + sx*c);

    for( dx = 0; dx < w_out; dx++ )
    {
        int j = dx * c;
        sx = iscale_x * j;
        for( k = 0; k < c; k++ )
            xofs[j + k] = sx + k;
    }

//    ResizeAreaFastFunc func = areafast_tab[depth];
//    CV_Assert( func != 0 );
//    func( src, dst, ofs, xofs, iscale_x, iscale_y );
}


int main() {
    cv::Mat mat_src = cv::imread("res/lakers25601440.jpeg", 1);
    int h = mat_src.rows;
    int w = mat_src.cols;
    int c = mat_src.channels();


    // 一行元素占用的字节数 2560*3
    std::cout << "mat_src step:" << mat_src.step << std::endl;
    std::cout << "mat_src step1_0:" << mat_src.step1(0) << std::endl;
    // 一个元素占用的字节数 3
    std::cout << "mat_src step1_1:" << mat_src.step1(1) << std::endl;
    // 一个元素占用的字节数 3
    std::cout << "mat_src elemSize:" << mat_src.elemSize() << std::endl;
    // 一个元素一个通道占用的字节数 1
    std::cout << "mat_src elemSize1:" << mat_src.elemSize1() << std::endl;

//    int dst_h = (int) h / 2;
//    int dst_w = (int) w / 2;
    int dst_w = 1920;
    int dst_h = 1080;
    std::cout << "resize w:" << dst_w << std::endl;
    std::cout << "resize h:" << dst_h << std::endl;
    cv::Mat resize_mat_opencv(dst_h, dst_w, CV_8UC(c));
    cv::Mat resize_mat_naive(dst_h, dst_w, CV_8UC(c));

    clock_t start_time = clock();
    cv::resize(mat_src, resize_mat_opencv, resize_mat_opencv.size(), 0, 0, 1);
    std::cout << "opencv_cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
    cv::imwrite("output/inter_area_opencv.jpg", resize_mat_opencv);

    start_time = clock();
    std::cout << "naive_cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
    cv::imwrite("output/inter_area_naive.jpg", resize_mat_naive);

    return 0;
}

