//
// Created by v_guojinlong on 2020-10-12.
//

#include <iostream>
#include <time.h>

#include <arm_neon.h>
#include "opencv2/opencv.hpp"

#include "vision/tensor.h"
#include "vision/tensor_converter.h"
using namespace vision;
using namespace std;

typedef struct VACV_RECT{
    int left;
    int top;
    int width;
    int height;
} VACV_RECT;

struct  VRect {
        float left;
        float top;
        float right;
        float bottom;
        VRect(float _left, float _top, float _right, float _bottom)
        : left(_left), top(_top), right(_right), bottom(_bottom) {}
        void set(float left, float top, float right, float bottom);
        float width() const;
        float height() const;
        bool contains(float x, float y);
};
void VRect::set(float l, float t, float r, float b) {
    left = l;
    top = t;
    right = r;
    bottom = b;
}

float VRect::width() const {
    return right - left;
}

float VRect::height() const {
    return bottom - top;
}

bool VRect::contains(float x, float y) {
    return left < right && top < bottom  // check for empty first
           && x >= left && x < right && y >= top && y < bottom;
}
static void crop_neon_hwc(const vision::Tensor& src, vision::Tensor& dst, VRect& rect) {
    dst.layout = NHWC;
    dst.h = int(rect.height());
    dst.w = int(rect.width());
    dst.c = src.c;
    dst.stride = dst.h * dst.w;

    if (src.dtype == INT8) {
        uint8_t* src_data = (uint8_t*)src.data;
        uint8_t* dst_data = (uint8_t*)malloc(sizeof(uint8_t) * dst.size());
        int count = 16; // 128 / 8
        uint8x16x3_t intlv_rgb;
        uint8x16_t intlv_grey;

        int crop_row_num8x16 = int(dst.w / count);
        int row_remain = dst.w % count;
        int src_offset;
        int dst_offset;
        for (int i = 0; i < dst.h; i++) {
            int src_row_index = int(rect.top) + i;
            int src_row_ofs = src_row_index * src.w;
            int dst_row_ofs = i * dst.w;
            for (int k = 0; k < crop_row_num8x16; k++) {
                src_offset = (src_row_ofs + int(rect.left) + k * count) * src.c;
                dst_offset = (dst_row_ofs + k * count) * dst.c;

                if (src.c == 3) {
                    intlv_rgb = vld3q_u8(src_data + src_offset);
                    vst3q_u8(dst_data + dst_offset, intlv_rgb);
                } else if (src.c == 1) {
                    intlv_grey = vld1q_u8(src_data + src_offset);
                    vst1q_u8(dst_data + dst_offset, intlv_grey);
                }
            }
            src_offset += count * src.c;
            dst_offset += count * dst.c;
            if (row_remain > 0) {
                for (int j = 0; j < row_remain; j++) {
                    int remain_ofs = j * dst.c;
                    for (int channel = 0; channel < src.c; channel++) {
                        *(dst_data + dst_offset + remain_ofs + channel) = *(src_data + src_offset + remain_ofs + channel);
                    }
                }
            }
        }

        dst.dtype = INT8;
        dst.data = dst_data;
    } else if (src.dtype == FP32) {
        float32_t* src_data = (float32_t*)src.data;
        float32_t* dst_data = (float32_t*)malloc(sizeof(float32_t) * dst.size());
        int count = 4; // 128 / 32
        float32x4x3_t intlv_rgb;
        float32x4_t intlv_grey;

        int crop_row_num32x4 = int(dst.w / count);
        int row_remain = dst.w % count;
        int src_offset;
        int dst_offset;
        for (int i = 0; i < dst.h; i++) {
            int src_row_index = int(rect.top) + i;
            int src_row_ofs = src_row_index * src.w;
            int dst_row_ofs = i * dst.w;
            for (int k = 0; k < crop_row_num32x4; k++) {
                src_offset = (src_row_ofs + int(rect.left) + k * count) * src.c;
                dst_offset = (dst_row_ofs + k * count) * dst.c;

                if (src.c == 3) {
                    intlv_rgb = vld3q_f32(src_data + src_offset);
                    vst3q_f32(dst_data + dst_offset, intlv_rgb);
                } else if (src.c == 1) {
                    intlv_grey = vld1q_f32(src_data + src_offset);
                    vst1q_f32(dst_data + dst_offset, intlv_grey);
                }
            }
            src_offset += count * src.c;
            dst_offset += count * dst.c;
            if (row_remain > 0) {
                for (int j = 0; j < row_remain; j++) {
                    int remain_ofs = j * dst.c;
                    for (int channel = 0; channel < src.c; channel++) {
                        *(dst_data + dst_offset + remain_ofs + channel) = *(src_data + src_offset + remain_ofs + channel);
                    }
                }
            }
        }

        dst.dtype = FP32;
        dst.data = dst_data;
    }
}

static void crop_neon_chw(const vision::Tensor& src, vision::Tensor& dst, const VRect& rect) {
    dst.layout = NCHW;
    dst.h = int(rect.height());
    dst.w = int(rect.width());
    dst.c = src.c;
    dst.stride = dst.h * dst.w;

    if (src.dtype == INT8) {
        uint8_t* src_data = (uint8_t*)src.data;
        uint8_t* dst_data = (uint8_t*)malloc(sizeof(uint8_t) * dst.size());
        int count = 16; // 128 / 8
        uint8x16_t b_int8x16;
        uint8x16_t g_int8x16;
        uint8x16_t r_int8x16;

        int crop_row_num8x16 = int(dst.w / count);
        int row_remain = dst.w % count;
        int b_src_channel_ofs = 0;
        int g_src_channel_ofs = src.w * src.h;
        int r_src_channel_ofs = src.w * src.h * 2;
        int b_dst_channel_ofs = 0;
        int g_dst_channel_ofs = dst.w * dst.h;
        int r_dst_channel_ofs = dst.w * dst.h* 2;
        for (int i = 0; i < dst.h; i++) {
            int src_row_index = int(rect.top) + i;
            int src_row_ofs = b_src_channel_ofs + src_row_index * src.w + int(rect.left);
            int dst_row_ofs = b_dst_channel_ofs + i * dst.w;
            int k = 0;
            for (k = 0; k < crop_row_num8x16; k++) {
                int crop_ofs = k * count;
                int b_src_offset = b_src_channel_ofs + src_row_ofs + crop_ofs;
                int b_dst_offset = b_dst_channel_ofs + dst_row_ofs + crop_ofs;
                b_int8x16 = vld1q_u8(src_data + b_src_offset);
                vst1q_u8(dst_data + b_dst_offset, b_int8x16);

                int g_src_offset = g_src_channel_ofs + src_row_ofs + crop_ofs;
                int g_dst_offset = g_dst_channel_ofs + dst_row_ofs + crop_ofs;
                g_int8x16 = vld1q_u8(src_data + g_src_offset);
                vst1q_u8(dst_data + g_dst_offset, g_int8x16);

                int r_src_offset = r_src_channel_ofs + src_row_ofs + crop_ofs;
                int r_dst_offset = r_dst_channel_ofs + dst_row_ofs + crop_ofs;
                r_int8x16 = vld1q_u8(src_data + r_src_offset);
                vst1q_u8(dst_data + r_dst_offset, r_int8x16);
            }

            if (row_remain > 0) {
                int crop_ofs = k * count;
                int b_src_offset = b_src_channel_ofs + src_row_ofs + crop_ofs;
                int b_dst_offset = b_dst_channel_ofs + dst_row_ofs + crop_ofs;
                int g_src_offset = g_src_channel_ofs + src_row_ofs + crop_ofs;
                int g_dst_offset = g_dst_channel_ofs + dst_row_ofs + crop_ofs;
                int r_src_offset = r_src_channel_ofs + src_row_ofs + crop_ofs;
                int r_dst_offset = r_dst_channel_ofs + dst_row_ofs + crop_ofs;
                for (int j = 0; j < row_remain; j++) {
                    *(dst_data + b_dst_offset + j) = *(src_data + b_src_offset + j);
                    *(dst_data + g_dst_offset + j) = *(src_data + g_src_offset + j);
                    *(dst_data + r_dst_offset + j) = *(src_data + r_src_offset + j);
                }
            }
        }

        dst.dtype = INT8;
        dst.data = dst_data;
    } else if (src.dtype == FP32) {
        float32_t* src_data = (float32_t*)src.data;
        float32_t* dst_data = (float32_t*)malloc(sizeof(float32_t) * dst.size());
        int count = 16; // 128 / 8
        float32x4_t b_float32x4;
        float32x4_t g_float32x4;
        float32x4_t r_float32x4;

        int crop_row_num32x4 = int(dst.w / count);
        int row_remain = dst.w % count;
        int b_src_channel_ofs = 0;
        int g_src_channel_ofs = src.w * src.h;
        int r_src_channel_ofs = src.w * src.h * 2;
        int b_dst_channel_ofs = 0;
        int g_dst_channel_ofs = dst.w * dst.h;
        int r_dst_channel_ofs = dst.w * dst.h* 2;
        for (int i = 0; i < dst.h; i++) {
            int src_row_index = int(rect.top) + i;
            int src_row_ofs = b_src_channel_ofs + src_row_index * src.w + int(rect.left);
            int dst_row_ofs = b_dst_channel_ofs + i * dst.w;
            int k = 0;
            for (k = 0; k < crop_row_num32x4; k++) {
                int crop_ofs = k * count;
                int b_src_offset = b_src_channel_ofs + src_row_ofs + crop_ofs;
                int b_dst_offset = b_dst_channel_ofs + dst_row_ofs + crop_ofs;
                b_float32x4 = vld1q_f32(src_data + b_src_offset);
                vst1q_f32(dst_data + b_dst_offset, b_float32x4);

                int g_src_offset = g_src_channel_ofs + src_row_ofs + crop_ofs;
                int g_dst_offset = g_dst_channel_ofs + dst_row_ofs + crop_ofs;
                g_float32x4 = vld1q_f32(src_data + g_src_offset);
                vst1q_f32(dst_data + g_dst_offset, g_float32x4);

                int r_src_offset = r_src_channel_ofs + src_row_ofs + crop_ofs;
                int r_dst_offset = r_dst_channel_ofs + dst_row_ofs + crop_ofs;
                r_float32x4 = vld1q_f32(src_data + r_src_offset);
                vst1q_f32(dst_data + r_dst_offset, r_float32x4);
            }

            if (row_remain > 0) {
                int crop_ofs = k * count;
                int b_src_offset = b_src_channel_ofs + src_row_ofs + crop_ofs;
                int b_dst_offset = b_dst_channel_ofs + dst_row_ofs + crop_ofs;
                int g_src_offset = g_src_channel_ofs + src_row_ofs + crop_ofs;
                int g_dst_offset = g_dst_channel_ofs + dst_row_ofs + crop_ofs;
                int r_src_offset = r_src_channel_ofs + src_row_ofs + crop_ofs;
                int r_dst_offset = r_dst_channel_ofs + dst_row_ofs + crop_ofs;
                for (int j = 0; j < row_remain; j++) {
                    *(dst_data + b_dst_offset + j) = *(src_data + b_src_offset + j);
                    *(dst_data + g_dst_offset + j) = *(src_data + g_src_offset + j);
                    *(dst_data + r_dst_offset + j) = *(src_data + r_src_offset + j);
                }
            }
        }

        dst.dtype = FP32;
        dst.data = dst_data;
    }
}


int main() {
    cv::Mat img = cv::imread("res/lakers25601440.jpeg", 1);

    int rect_left   = 1900;
    int rect_top    = 500;
    int rect_height = 170;
    int rect_width  = 150;
//    int rect_left   = 200;
//    int rect_top    = 200;
//    int rect_height = 807;
//    int rect_width  = 807;

    cv::Mat crop;
    clock_t start_time = clock();
    crop = img(cv::Rect(rect_left, rect_top, rect_width, rect_height)).clone();
    clock_t end_time = clock();
    std::cout << "cv_cost: " << (double)(end_time - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
    cv::imwrite("output/crop_opencv.jpg", crop);
    vision::Tensor cv_tensor = vision::TensorConverter::convert_from<cv::Mat>(crop);


    vision::Tensor tensor = vision::TensorConverter::convert_from<cv::Mat>(img);
    VRect rect(rect_left, rect_top, rect_left + rect_width, rect_top + rect_height);
    vision::Tensor crop_tensor(rect_width, rect_height, tensor.c, NHWC);
    start_time = clock();
    crop_neon_hwc(tensor, crop_tensor, rect);
    end_time = clock();
    std::cout << "neon_hwc_cost: " << (double)(end_time - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
    cv::Mat neon_crop(rect_height, rect_width, CV_8UC(crop_tensor.c), crop_tensor.data);
    cv::imwrite("output/crop_neon.jpg", neon_crop);

    vision::Tensor tensor_chw = tensor.change_layout(NCHW);
    vision::Tensor crop_tensor_chw(rect_width, rect_height, tensor.c, NCHW);
    start_time = clock();
    crop_neon_chw(tensor_chw, crop_tensor_chw, rect);
    end_time = clock();
    vision::Tensor crop_tensor_hwc = crop_tensor_chw.change_layout(NHWC);
    std::cout << "neon_chw_cost: " << (double)(end_time - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
    cv::Mat neon_crop_chw(rect_height, rect_width, CV_8UC(crop_tensor_hwc.c), crop_tensor_hwc.data);
    cv::imwrite("output/crop_neon_chw.jpg", neon_crop_chw);




    return 0;
}

