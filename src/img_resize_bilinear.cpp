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

void resize_naive_mat(cv::Mat& matSrc, cv::Mat& matDst) {
    uchar* dataDst = matDst.data;
    int stepDst = matDst.step;
    uchar* dataSrc = matSrc.data;
    int stepSrc = matSrc.step;
    int iWidthSrc = matSrc.cols;
    int iHiehgtSrc = matSrc.rows;
    float scale_x = static_cast<float>(iWidthSrc) / matDst.cols;
    float scale_y = static_cast<float>(iHiehgtSrc) / matDst.rows;

    for (int j = 0; j < matDst.rows; j++)
    {
        // SrcY=(dstY+0.5) * (srcHeight/dstHeight)-0.5
        float fy = (float)((j + 0.5) * scale_y - 0.5);
        // v 整数部分
        int sy = cvFloor(fy);
        fy -= sy;

        if (sy < 0) {
            sy = 0;
            fy = 0.f;
        }
        if (sy >= iHiehgtSrc - 1) {
            sy = iHiehgtSrc - 2;
            fy = 1.f;
        }

        // v 小数部分
        short cbufy[2];
        cbufy[0] = cv::saturate_cast<short>((1.f - fy) * 2048);
//        cbufy[1] = 2048 - cbufy[0];
        cbufy[1] = 2048 * fy;

        for (int i = 0; i < matDst.cols; i++)
        {
            // SrcX=(dstX+0.5)* (srcWidth/dstWidth) -0.5
            float fx = (float)((i + 0.5) * scale_x - 0.5);
            // u 整数部分
            int sx = cvFloor(fx);
            // u 小数部分
            fx -= sx;

            if (sx < 0) {
                sx = 0;
                fx = 0.f;
            }
            // 如果x越界
            if (sx >= iWidthSrc - 1) {
                sx = iWidthSrc - 2;
                fx = 1.f;
            }

            short cbufx[2];
            cbufx[0] = cv::saturate_cast<short>((1.f - fx) * 2048);
//            cbufx[1] = 2048 - cbufx[0];
            cbufx[1] = 2048 * fx;

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


#define SATURATE_CAST_SHORT(X)                                               \
  (int16_t)::std::min(                                                       \
      ::std::max(static_cast<int>(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), \
      SHRT_MAX);

void resize_naive_uint8(const uint8_t* src,
                        int w_in,
                        int h_in,
                        int c,
                        uint8_t* dst,
                        int w_out,
                        int h_out) {
    float scale_x = static_cast<float>(w_in) / w_out;
    float scale_y = static_cast<float>(h_in) / h_out;

    for (int dy = 0; dy < h_out; dy++) {
        float fy = static_cast<float>((dy + 0.5) * scale_y - 0.5);
        int sy = floor(fy);
        fy -= sy;
        if (sy < 0) {
            sy = 0; fy = 0.f;
        }
        if (sy >= h_in - 1) {
            sy = h_in - 2; fy = 1.f;
        }

        short cbufy[2];
        cbufy[0] = SATURATE_CAST_SHORT((1.f - fy) * 2048);
        cbufy[1] = SATURATE_CAST_SHORT(2048 * fy);

        for (int dx = 0; dx < w_out; dx++) {
            float fx = static_cast<float>((dx + 0.5) * scale_x - 0.5);
            int sx = floor(fx);
            fx -= sx;

            if (sx < 0) {
                sx = 0;
                fx = 0.f;
            }
            if (sx >= w_in - 1) {
                sx = w_in - 2;
                fx = 1.f;
            }

            short cbufx[2];
            cbufx[0] = SATURATE_CAST_SHORT((1.f - fx) * 2048);
            cbufx[1] = SATURATE_CAST_SHORT(2048 * fx);

            int lt_ofs = (sy * w_in + sx) * c;
            int rt_ofs = (sy * w_in + sx + 1) * c;
            int lb_ofs = ((sy + 1) * w_in + sx) * c;
            int rb_ofs = ((sy + 1) * w_in + sx + 1) * c;
            int dst_ofs = (dy * w_out + dx) * c;
            for (int k = 0; k < c; k++) {
                *(dst + dst_ofs + k) =
                        (*(src + lt_ofs + k) * cbufx[0] * cbufy[0] +
                         *(src + lb_ofs + k) * cbufx[0] * cbufy[1] +
                         *(src + rt_ofs + k) * cbufx[1] * cbufy[0] +
                         *(src + rb_ofs + k) * cbufx[1] * cbufy[1]) >> 22;
            }
        }
    }
}

//template <typename T>
void resize_naive(const char* src,
                        int w_in,
                        int h_in,
                        int c,
                        char* dst,
                        int w_out,
                        int h_out) {
    float scale_x = static_cast<float>(w_in) / w_out;
    float scale_y = static_cast<float>(h_in) / h_out;

    for (int dy = 0; dy < h_out; dy++) {
        float fy = static_cast<float>((dy + 0.5) * scale_y - 0.5);
        int sy = floor(fy);
        fy -= sy;
        if (sy < 0) {
            sy = 0; fy = 0.f;
        }
        if (sy >= h_in - 1) {
            sy = h_in - 2; fy = 1.f;
        }

        short cbufy[2];
        cbufy[0] = SATURATE_CAST_SHORT((1.f - fy) * 2048);
        cbufy[1] = SATURATE_CAST_SHORT(2048 * fy);

        for (int dx = 0; dx < w_out; dx++) {
            float fx = static_cast<float>((dx + 0.5) * scale_x - 0.5);
            int sx = floor(fx);
            fx -= sx;

            if (sx < 0) {
                sx = 0;
                fx = 0.f;
            }
            if (sx >= w_in - 1) {
                sx = w_in - 2;
                fx = 1.f;
            }

            short cbufx[2];
            cbufx[0] = SATURATE_CAST_SHORT((1.f - fx) * 2048);
            cbufx[1] = SATURATE_CAST_SHORT(2048 * fx);

            int lt_ofs = (sy * w_in + sx) * c;
            int rt_ofs = (sy * w_in + sx + 1) * c;
            int lb_ofs = ((sy + 1) * w_in + sx) * c;
            int rb_ofs = ((sy + 1) * w_in + sx + 1) * c;
            int dst_ofs = (dy * w_out + dx) * c;
            for (int k = 0; k < c; k++) {
                *(dst + dst_ofs + k) =
                        (*(src + lt_ofs + k) * cbufx[0] * cbufy[0] +
                         *(src + lb_ofs + k) * cbufx[0] * cbufy[1] +
                         *(src + rt_ofs + k) * cbufx[1] * cbufy[0] +
                         *(src + rb_ofs + k) * cbufx[1] * cbufy[1]) >> 22;
            }
        }
    }
}

void resize_naive_float(const float* src,
                        int w_in,
                        int h_in,
                        int c,
                        float* dst,
                        int w_out,
                        int h_out) {
    float scale_x = static_cast<float>(w_in) / w_out;
    float scale_y = static_cast<float>(h_in) / h_out;

    for (int dy = 0; dy < h_out; dy++) {
        float fy = static_cast<float>((dy + 0.5) * scale_y - 0.5);
        int sy = floor(fy);
        fy -= sy;
        if (sy < 0) {
            sy = 0; fy = 0.f;
        }
        if (sy >= h_in - 1) {
            sy = h_in - 2; fy = 1.f;
        }

        float cbufy[2];
        cbufy[0] = 1.f - fy;
        cbufy[1] = fy;

        for (int dx = 0; dx < w_out; dx++) {
            float fx = static_cast<float>((dx + 0.5) * scale_x - 0.5);
            int sx = floor(fx);
            fx -= sx;

            if (sx < 0) {
                sx = 0;
                fx = 0.f;
            }
            if (sx >= w_in - 1) {
                sx = w_in - 2;
                fx = 1.f;
            }

            float cbufx[2];
            cbufx[0] = 1.f - fx;
            cbufx[1] = fx;

            int lt_ofs = (sy * w_in + sx) * c;
            int rt_ofs = (sy * w_in + sx + 1) * c;
            int lb_ofs = ((sy + 1) * w_in + sx) * c;
            int rb_ofs = ((sy + 1) * w_in + sx + 1) * c;
            int dst_ofs = (dy * w_out + dx) * c;
            for (int k = 0; k < c; k++) {
                *(dst + dst_ofs + k) =
                        (*(src + lt_ofs + k) * cbufx[0] * cbufy[0] +
                         *(src + lb_ofs + k) * cbufx[0] * cbufy[1] +
                         *(src + rt_ofs + k) * cbufx[1] * cbufy[0] +
                         *(src + rb_ofs + k) * cbufx[1] * cbufy[1]);
            }
        }
    }
}


int main() {
    cv::Mat matSrc = cv::imread("res/lakers25601440.jpeg", 1);
    int h = matSrc.rows;
    int w = matSrc.cols;
    int c = matSrc.channels();
//    int dst_h = (int) h / 2;
//    int dst_w = (int) w / 2;
    int dst_w = 320;
    int dst_h = 180;
    std::cout << "resize w:" << dst_w << std::endl;
    std::cout << "resize h:" << dst_h << std::endl;
    cv::Mat matDst1(dst_h, dst_w, CV_8UC(c));
    cv::Mat matDst2(dst_h, dst_w, CV_8UC(c));
    cv::Mat matDst3(dst_h, dst_w, CV_8UC(c));

    clock_t start_time = clock();
    resize_naive((char*)matSrc.data, w, h, c, (char*)matDst1.data, dst_w, dst_h);
    std::cout << "naive_u8_cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
    cv::imwrite("output/resize_linear_naive_u8.jpg", matDst1);

    start_time = clock();
    cv::Mat matSrc_f32;
    matSrc.convertTo(matSrc_f32, CV_32FC(c));
    cv::Mat matDst1_f32(dst_h, dst_w, CV_32FC(c));
    resize_naive_float((float*)matSrc_f32.data, w, h, c, (float*)matDst1_f32.data, dst_w, dst_h);
//    cv::resize(matSrc_f32, matDst1_f32, matDst1_f32.size(), 0, 0, 1);
    std::cout << "naive_fp32_cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
    cv::imwrite("output/resize_linear_naive_f32.jpg", matDst1_f32);

    start_time = clock();
    cv::resize(matSrc, matDst2, matDst1.size(), 0, 0, 1);
    std::cout << "opencv_cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
    cv::imwrite("output/resize_linear_opencv.jpg", matDst2);

    start_time = clock();
    ImageResize* is = new ImageResize();
    is->choose((uint8_t*) matSrc.data,
               (uint8_t*) matDst3.data,
               ImageFormat::RGB,
               w, h,
               dst_w, dst_h);
    std::cout << "paddle_cost: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
    cv::imwrite("output/resize_linear_neon.jpg", matDst3);

    return 0;
}

