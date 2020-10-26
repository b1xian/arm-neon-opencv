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
    // todo 仅对于uint8数据类型
    uint8_t* src_data = (uint8_t*)src_tensor.data;
    uint8_t* dst_data = (uint8_t*)malloc(sizeof(uint8_t) * crop_tensor.size());
    int count = 128 / 8;
    uint8x16x3_t intlv_rgb;
    uint8x16_t intlv_grey;

    int crop_row_num8x16 = rect.width / count;
    int remain = rect.width % count;
    if (remain != 0) {
        crop_row_num8x16 += 1;
    }
    int src_offset;
    int dst_offset;
    for (int i = 0; i < rect.height; i++) {
        int row_index = rect.top + i;
        for (int k = 0; k < crop_row_num8x16; k++) {
            src_offset = (row_index*src_tensor.w + rect.left+k*count) * src_tensor.c;
            dst_offset = (i*crop_tensor.w + k*count) * src_tensor.c;

            if (src_tensor.c == 3) {
                intlv_rgb = vld3q_u8(src_data + src_offset);
                vst3q_u8(dst_data + dst_offset, intlv_rgb);
            } else if (src_tensor.c == 1) {
                intlv_grey = vld1q_u8(src_data + src_offset);
                vst1q_u8(dst_data + dst_offset, intlv_grey);
            }
        }
    }
    crop_tensor.data = dst_data;
}

void print_hwc_tensor(vision::Tensor& tensor) {
    unsigned char* data = (unsigned char*)tensor.data;
    for (int i = 0; i < tensor.stride; i++) {
        for (int j = 0; j < tensor.c; j++) {
            cout << "pix-" << i << " channel-" << j << " :" << int(data[i + j*tensor.c]) << endl;
        }
    }
}


int main() {
    cv::Mat img = cv::imread("./lakers.jpeg", 0);

    int rect_left   = 800;
    int rect_top    = 700;
    int rect_height = 670;
    int rect_width  = 321;

    cv::Mat crop;
    clock_t start_time = clock();
    crop = img(cv::Rect(rect_left, rect_top, rect_width, rect_height)).clone();
    clock_t end_time = clock();
    std::cout << "cv_cost: " << (double)(end_time - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
    cv::imwrite("./cv_crop.jpg", crop);
    vision::Tensor cv_tensor = vision::TensorConverter::convert_from<cv::Mat>(crop);
//    cout << "cv_tensor:" << endl;
//    print_hwc_tensor(cv_tensor);


    vision::Tensor tensor = vision::TensorConverter::convert_from<cv::Mat>(img);
    VACV_RECT rect;
    rect.left   = rect_left;
    rect.top    = rect_top;
    rect.height = rect_height;
    rect.width  = rect_width;
    vision::Tensor crop_tensor(rect.width, rect.height, tensor.c, vision::DLayout::NHWC);
    start_time = clock();
    crop_neon_hwc(tensor, crop_tensor, rect);
    end_time = clock();
    std::cout << "neon_cost: " << (double)(end_time - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
//    cout << "neon_tensor:" << endl;
//    print_hwc_tensor(crop_tensor);
    cv::Mat neon_crop(rect.height, rect.width, CV_8UC(crop_tensor.c), crop_tensor.data);
    cv::imwrite("./neon_crop.jpg", neon_crop);

    return 0;
}

