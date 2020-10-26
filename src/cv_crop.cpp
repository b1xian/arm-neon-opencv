//
// Created by v_guojinlong on 2020-10-12.
//

#include <iostream>
#include <time.h>

#include <arm_neon.h>
#include "opencv2/opencv.hpp"

#include "vision/tensor.h"
#include "vision/tensor_converter.h"

using namespace std;

typedef struct VACV_RECT{
    int left;
    int top;
    int width;
    int height;
} VACV_RECT;

void crop_neon_chw(vision::Tensor& src_tensor, vision::Tensor& crop_tensor, VACV_RECT& rect) {
    /*
     * 128位寄存器，
     * fp32:每次操作4个像素
     * fp16:每次操作8个像素
     * int8:每次操作16个像素
     */
    uint8_t* src_data = (uint8_t*)src_tensor.data;
    uint8_t* dst_data = (uint8_t*)malloc(sizeof(uint8_t) * crop_tensor.size());
    // todo 仅对于uint8数据类型
    // todo 考虑不能被整除的情况
    int count = 128 / 8;
    int get_count = rect.width / count;
    int remainder = rect.width % count;
    if (remainder != 0) {
        get_count += 1;
    }
    // 使用neon指令crop每一列需要操作的次数
    cout << "get_count:" << get_count << endl;
    uint8x16_t C;
    for (int i = 0; i < rect.height; i++) {
        for (int k = 0; k < get_count; k++) {
            int channel_offset = src_tensor.w * (rect.top+i) + rect.left + k*count;
            int dst_offset = crop_tensor.w*i + k*count;

            C = vld1q_u8(src_data + src_tensor.stride*1 + channel_offset);
            vst1q_u8(dst_data + crop_tensor.stride*1 + dst_offset, C);
//            for (int c = 0; c < src_tensor.c; c++) {
//                C = vld1q_u8(src_data + src_tensor.stride*c + channel_offset);
//                vst1q_u8(dst_data + crop_tensor.stride*c + dst_offset, C);
//            }
        }
    }
    crop_tensor.data = dst_data;
}

void crop_neon_hwc(vision::Tensor& src_tensor, vision::Tensor& crop_tensor, VACV_RECT& rect) {
    /*
     * 128位寄存器，
     * fp32:每次操作4个像素
     * fp16:每次操作8个像素
     * int8:每次操作16个像素
     */
    uint8_t* src_data = (uint8_t*)src_tensor.data;
    uint8_t* dst_data = (uint8_t*)malloc(sizeof(uint8_t) * crop_tensor.size());
    // todo 仅对于uint8数据类型
    // todo 考虑不能被整除的情况
    int count = 128 / 8;
    int num8x16 = rect.width / count;
    int remainder = rect.width % count;
    if (remainder != 0) {
        num8x16 += 1;
    }
    // 使用neon指令crop每一列需要操作的次数
    cout << "num8x16:" << num8x16 << endl;
//    uint8x16_t C;
    uint8x16x3_t intlv_rgb;
    for (int i=0; i < rect.height; i++) {
        for (int k=0; k < num8x16; k++) {
            int offset = src_tensor.w*(rect.top+i)*3 + rect.left*3;
            intlv_rgb = vld3q_u8(src_data+offset+3*16*k);
            vst3q_u8(dst_data + 16*k, intlv_rgb);
        }
    }
    crop_tensor.data = dst_data;
}


void print_chw_tensor(vision::Tensor& tensor) {
    cout << "chw tensor:" << tensor.c << endl;
    unsigned char* data = (unsigned char*)tensor.data;
    for (int c = 0; c < tensor.c; c++) {
        for (int i = 0; i < tensor.stride; i++) {
            cout << int(data[tensor.stride*c + i]) << " ";
            if ((i+1) % tensor.w == 0) {
                cout << endl;
            }
        }
        cout << "----------------" << endl;
    }
}


int main() {
//    cv::Mat img = cv::imread("./create.jpg", 1);
    cv::Mat img = cv::imread("./lakers.jpeg", 1);
    cout << img.cols <<endl;
    cout << img.rows <<endl;
    cout << img.channels() <<endl;

    cv::Mat crop;
    clock_t start_time = clock();
//    crop = img(cv::Rect(5, 5, 16, 16));
    crop = img(cv::Rect(1095, 540, 210, 256)).clone();
    clock_t end_time = clock();
    std::cout << "cv_cost: " << (double)(end_time - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
    cv::imwrite("./cv_crop.jpg", crop);
//    vision::Tensor cv_tensor = vision::TensorConverter::convert_from<cv::Mat>(crop);
//    cv_tensor.change_layout(vision::DLayout::NCHW);
//    cout << "cv_tensor" << endl;
//    print_chw_tensor(cv_tensor);


    vision::Tensor tensor = vision::TensorConverter::convert_from<cv::Mat>(img);
    tensor.change_layout(vision::DLayout::NCHW);
    VACV_RECT rect;
//    rect.left   = 5;
//    rect.top    = 5;
//    rect.width  = 16;
//    rect.height = 16;
    rect.left   = 1095;
    rect.top    = 540;
    rect.height = 256;
    rect.width  = 210;
    vision::Tensor crop_tensor(rect.width, rect.height, 1, vision::DLayout::NCHW);
    start_time = clock();
//    crop_neon_hwc(tensor, crop_tensor, rect);
    crop_neon_chw(tensor, crop_tensor, rect);
    end_time = clock();
    std::cout << "neon_cost: " << (double)(end_time - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
    cout << "crop_tensor" << endl;
//    print_chw_tensor(crop_tensor);
//    print_chw_tensor(crop_tensor);
//    crop_tensor.change_layout(vision::DLayout::NHWC);
    cv::Mat neon_crop(rect.height, rect.width, CV_8UC1, crop_tensor.data);
//    cv::Mat neon_crop = vision::TensorConverter::convert_to<cv::Mat>(crop_tensor);
    cv::imwrite("./neon_crop.jpg", neon_crop);

    return 0;
}

int main() {
//    cv::Mat img = cv::imread("./create.jpg", 1);
    cv::Mat img = cv::imread("./lakers.jpeg", 1);
    cout << img.cols <<endl;
    cout << img.rows <<endl;
    cout << img.channels() <<endl;

    cv::Mat crop;
    clock_t start_time = clock();
//    crop = img(cv::Rect(5, 5, 16, 16));
    crop = img(cv::Rect(1095, 540, 210, 256)).clone();
    clock_t end_time = clock();
    std::cout << "cv_cost: " << (double)(end_time - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
    cv::imwrite("./cv_crop.jpg", crop);
//    vision::Tensor cv_tensor = vision::TensorConverter::convert_from<cv::Mat>(crop);
//    cv_tensor.change_layout(vision::DLayout::NCHW);
//    cout << "cv_tensor" << endl;
//    print_chw_tensor(cv_tensor);


    vision::Tensor tensor = vision::TensorConverter::convert_from<cv::Mat>(img);
    tensor.change_layout(vision::DLayout::NCHW);
    VACV_RECT rect;
//    rect.left   = 5;
//    rect.top    = 5;
//    rect.width  = 16;
//    rect.height = 16;
    rect.left   = 1095;
    rect.top    = 540;
    rect.height = 256;
    rect.width  = 210;
    vision::Tensor crop_tensor(rect.width, rect.height, 1, vision::DLayout::NCHW);
    start_time = clock();
//    crop_neon_hwc(tensor, crop_tensor, rect);
    crop_neon_chw(tensor, crop_tensor, rect);
    end_time = clock();
    std::cout << "neon_cost: " << (double)(end_time - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
    cout << "crop_tensor" << endl;
//    print_chw_tensor(crop_tensor);
//    print_chw_tensor(crop_tensor);
//    crop_tensor.change_layout(vision::DLayout::NHWC);
    cv::Mat neon_crop(rect.height, rect.width, CV_8UC1, crop_tensor.data);
//    cv::Mat neon_crop = vision::TensorConverter::convert_to<cv::Mat>(crop_tensor);
    cv::imwrite("./neon_crop.jpg", neon_crop);

    return 0;
}