#include "crop.h"

#include <iostream>
#include <math.h>

#ifdef USE_OPENCV
#include "opencv2/opencv.hpp"
#endif

#if defined (USE_NEON) and __ARM_NEON
#include "arm_neon.h"
#endif

#include "../common/tensor_converter.h"

namespace va_cv {

using namespace vision;

void Crop::crop(const vision::Tensor &src, vision::Tensor &dst, const VRect &rect) {
#if defined (USE_NEON) and __ARM_NEON
    crop_neon(src, dst, rect);
#else
    crop_naive(src, dst, rect);
#endif // USE_NEON
}

void Crop::crop_opencv(const vision::Tensor& src, vision::Tensor& dst, const vision::VRect& rect) {
#ifdef USE_OPENCV
    const auto& mat_src = vision::TensorConverter::convert_to<cv::Mat>(src);
    cv::Rect cv_rect(cvRound(rect.left), cvRound(rect.top),
                     cvRound(rect.right - rect.left),
                     cvRound(rect.bottom - rect.top));
    cv::Mat mat_dst;
    mat_dst = mat_src(cv_rect).clone();
    dst = vision::TensorConverter::convert_from<cv::Mat>(mat_dst, true);
#endif
}

    void Crop::crop_naive_chw(const vision::Tensor& src, vision::Tensor& dst,
                              int crop_left, int crop_top, int crop_width, int crop_height) {

        if (src.dtype == INT8) {
            dst.create(crop_width, crop_height, src.c, NCHW, INT8);
            unsigned char* src_data = (unsigned char*)src.data;
            unsigned char* dst_data = (unsigned char*)dst.data;

            for (int k = 0; k < src.c; k++) {
                int src_channel_ofs = src.w * src.h * k;
                int dst_channel_ofs = dst.w * dst.h * k;
                for (int i = 0; i < dst.h; i++) {
                    int src_row_index = crop_top + i;
                    int src_row_ofs = src_channel_ofs + src_row_index * src.w + crop_left;
                    int dst_row_ofs = dst_channel_ofs + i * dst.w;
                    for (int j = 0; j < dst.w; j++) {
                        *(dst_data + dst_row_ofs + j) = *(src_data + src_row_ofs + j);
                    }
                }
            }
        } else if (src.dtype == FP32) {
            dst.create(crop_width, crop_height, src.c, NCHW, FP32);
            float* src_data = (float*)src.data;
            float* dst_data = (float*)dst.data;

            for (int k = 0; k < src.c; k++) {
                int src_channel_ofs = src.w * src.h * k;
                int dst_channel_ofs = dst.w * dst.h * k;
                for (int i = 0; i < dst.h; i++) {
                    int src_row_index = crop_top + i;
                    int src_row_ofs = src_channel_ofs + src_row_index * src.w + crop_left;
                    int dst_row_ofs = dst_channel_ofs + i * dst.w;
                    for (int j = 0; j < dst.w; j++) {
                        *(dst_data + dst_row_ofs + j) = *(src_data + src_row_ofs + j);
                    }
                }
            }
        }
    }

    void Crop::crop_naive_hwc_rgb(const vision::Tensor& src, vision::Tensor& dst,
                                  int crop_left, int crop_top, int crop_width, int crop_height) {
        if (src.dtype == INT8) {
            dst.create(crop_width, crop_height, src.c, NHWC, INT8);
            unsigned char* src_data = (unsigned char*)src.data;
            unsigned char* dst_data = (unsigned char*)dst.data;

            int src_offset = 0;
            int dst_offset = 0;
            for (int i = 0; i < dst.h; i++) {
                int src_row_index = crop_top + i;
                int src_row_ofs = src_row_index * src.w;
                int dst_row_ofs = i * dst.w;
                for (int j = 0; j < dst.w; j++) {
                    src_offset = (src_row_ofs + crop_left + j) * src.c;
                    dst_offset = (dst_row_ofs + j) * dst.c;
                    for (int k = 0; k < src.c; k++) {
                        *(dst_data + dst_offset + k) = *(src_data + src_offset + k);
                    }
                }
            }
        } else if (src.dtype == FP32) {
            dst.create(crop_width, crop_height, src.c, NHWC, FP32);
            float* src_data = (float*)src.data;
            float* dst_data = (float*)dst.data;

            int src_offset = 0;
            int dst_offset = 0;
            for (int i = 0; i < dst.h; i++) {
                int src_row_index = crop_top + i;
                int src_row_ofs = src_row_index * src.w;
                int dst_row_ofs = i * dst.w;
                for (int j = 0; j < dst.w; j++) {
                    src_offset = (src_row_ofs + crop_left + j) * src.c;
                    dst_offset = (dst_row_ofs + j) * dst.c;
                    for (int k = 0; k < src.c; k++) {
                        *(dst_data + dst_offset + k) = *(src_data + src_offset + k);
                    }
                }
            }
        }
    }

    void Crop::crop_naive(const vision::Tensor& src, vision::Tensor& dst, const vision::VRect& rect) {
        int crop_left   = static_cast<int>(rect.left);
        int crop_top    = static_cast<int>(rect.top);
        int crop_width  = static_cast<int>(rect.width());
        int crop_height = static_cast<int>(rect.height());

        if (src.dtype == INT8 || src.dtype == FP32) {
            if (src.layout == NHWC) {
                crop_naive_hwc_rgb(src, dst, crop_left, crop_top, crop_width, crop_height);
            } else {
                crop_naive_chw(src, dst, crop_left, crop_top, crop_width, crop_height);
            }
        } else {
            crop_opencv(src, dst, rect);
        }
    }

    void Crop::crop_sse(const vision::Tensor& src, vision::Tensor& dst, const vision::VRect& rect) {
        // todo:
    }

#if defined (USE_NEON) and __ARM_NEON
    void Crop::crop_neon(const vision::Tensor& src, vision::Tensor& dst, const vision::VRect& rect) {
    int crop_left   = static_cast<int>(rect.left);
    int crop_top    = static_cast<int>(rect.top);
    int crop_width  = static_cast<int>(rect.width());
    int crop_height = static_cast<int>(rect.height());

    if ((src.dtype == INT8 || src.dtype == FP32) && (src.c == 1 || src.c == 3)) {
        if (src.c == 1 || src.layout == NHWC) {
            crop_neon_hwc_rgb_ir(src, dst, crop_left, crop_top, crop_width, crop_height);
        } else {
            crop_neon_chw_rgb(src, dst, crop_left, crop_top, crop_width, crop_height);
        }
    } else {
        crop_opencv(src, dst, rect);
    }
}

void Crop::crop_neon_hwc_rgb_ir(const vision::Tensor& src, vision::Tensor& dst,
                                int crop_left, int crop_top, int crop_width, int crop_height) {
    if (src.dtype == INT8) {
        dst.create(crop_width, crop_height, src.c, NHWC, INT8);
        uint8_t* src_data = (uint8_t*)src.data;
        uint8_t* dst_data = (uint8_t*)dst.data;
        int count = 16; // 128 / 8
        uint8x16x3_t intlv_rgb;
        uint8x16_t intlv_grey;

        int crop_row_num8x16 = int(dst.w / count);
        int row_remain = dst.w % count;

        int src_offset = 0;
        int dst_offset = 0;
        for (int i = 0; i < dst.h; i++) {
            int src_row_index = crop_top + i;
            int src_row_ofs = src_row_index * src.w;
            int dst_row_ofs = i * dst.w;

            src_offset = (src_row_ofs + crop_left) * src.c;
            dst_offset = (dst_row_ofs) * dst.c;
            for (int k = 0; k < crop_row_num8x16; k++) {
                src_offset = (src_row_ofs + crop_left + k * count) * src.c;
                dst_offset = (dst_row_ofs + k * count) * dst.c;

                if (src.c == 3) {
                    intlv_rgb = vld3q_u8(src_data + src_offset);
                    vst3q_u8(dst_data + dst_offset, intlv_rgb);
                } else if (src.c == 1) {
                    intlv_grey = vld1q_u8(src_data + src_offset);
                    vst1q_u8(dst_data + dst_offset, intlv_grey);
                }
            }
            if (crop_row_num8x16 > 0) {
                src_offset += count * src.c;
                dst_offset += count * dst.c;
            }
            if (row_remain > 0) {
                for (int j = 0; j < row_remain; j++) {
                    int remain_ofs = j * dst.c;
                    for (int channel = 0; channel < src.c; channel++) {
                        *(dst_data + dst_offset + remain_ofs + channel) = *(src_data + src_offset + remain_ofs + channel);
                    }
                }
            }
        }
    } else if (src.dtype == FP32) {
        dst.create(crop_width, crop_height, src.c, NHWC, FP32);
        float32_t* src_data = (float32_t*)src.data;
        float32_t* dst_data = (float32_t*)dst.data;
        int count = 4; // 128 / 32
        float32x4x3_t intlv_rgb;
        float32x4_t intlv_grey;

        int crop_row_num32x4 = int(dst.w / count);
        int row_remain = dst.w % count;
        int src_offset = 0;
        int dst_offset = 0;
        for (int i = 0; i < dst.h; i++) {
            int src_row_index = crop_top + i;
            int src_row_ofs = src_row_index * src.w;
            int dst_row_ofs = i * dst.w;

            src_offset = (src_row_ofs + crop_left) * src.c;
            dst_offset = (dst_row_ofs) * dst.c;
            for (int k = 0; k < crop_row_num32x4; k++) {
                src_offset = (src_row_ofs + crop_left + k * count) * src.c;
                dst_offset = (dst_row_ofs + k * count) * dst.c;

                if (src.c == 3) {
                    intlv_rgb = vld3q_f32(src_data + src_offset);
                    vst3q_f32(dst_data + dst_offset, intlv_rgb);
                } else if (src.c == 1) {
                    intlv_grey = vld1q_f32(src_data + src_offset);
                    vst1q_f32(dst_data + dst_offset, intlv_grey);
                }
            }
            if (crop_row_num32x4 > 0) {
                src_offset += count * src.c;
                dst_offset += count * dst.c;
            }
            if (row_remain > 0) {
                for (int j = 0; j < row_remain; j++) {
                    int remain_ofs = j * dst.c;
                    for (int channel = 0; channel < src.c; channel++) {
                        *(dst_data + dst_offset + remain_ofs + channel) = *(src_data + src_offset + remain_ofs + channel);
                    }
                }
            }
        }
    }
}

void Crop::crop_neon_chw_rgb(const vision::Tensor& src, vision::Tensor& dst,
                             int crop_left, int crop_top, int crop_width, int crop_height) {
    if (src.dtype == INT8) {
        dst.create(crop_width, crop_height, src.c, NCHW, INT8);
        uint8_t* src_data = (uint8_t*)src.data;
        uint8_t* dst_data = (uint8_t*)dst.data;
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
            int src_row_index = crop_top + i;
            int src_row_ofs = b_src_channel_ofs + src_row_index * src.w + crop_left;
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
    } else if (src.dtype == FP32) {
        dst.create(crop_width, crop_height, src.c, NCHW, FP32);
        float32_t* src_data = (float32_t*)src.data;
        float32_t* dst_data = (float32_t*)dst.data;
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
            int src_row_index = crop_top + i;
            int src_row_ofs = b_src_channel_ofs + src_row_index * src.w + crop_left;
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

    }
}
#endif // USE_NEON

} // namespace va_cv