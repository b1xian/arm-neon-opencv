//
// Created by b1xian on 2020-10-12.
//
#include <arm_neon.h>
#include <iostream>
#include <stdlib.h>


static void hwc_2_chw_c(uint8_t* r, uint8_t* g, uint8_t* b, uint8_t* rgb, int len_color) {
    /*
     * Take the elements of "rgb" and store the individual colors "r", "g", and "b".
     */
    for (int i=0; i < len_color; i++) {
        r[i] = rgb[3*i];
        g[i] = rgb[3*i+1];
        b[i] = rgb[3*i+2];
    }
}

static void hwc_2_chw_neon(uint8_t *r, uint8_t *g, uint8_t *b, uint8_t *rgb, int len_color) {
    /*
     * Take the elements of "rgb" and store the individual colors "r", "g", and "b"
     */
    int num8x16 = len_color / 16;
    uint8x16x3_t intlv_rgb;
    for (int i=0; i < num8x16; i++) {
        intlv_rgb = vld3q_u8(rgb+3*16*i);
        vst1q_u8(r+16*i, intlv_rgb.val[0]);
        vst1q_u8(g+16*i, intlv_rgb.val[1]);
        vst1q_u8(b+16*i, intlv_rgb.val[2]);
    }
}

static void hwc_2_chw_neon_u8(uint8_t *src_data, uint8_t *dst_data, int w, int h, int c) {
    int stride = w * h;
    if (c == 1){
        memcpy(dst_data, src_data, stride);
        return;
    }

    int num8x16 = int(stride / 16);
    int remain = stride % 16;
    uint8x16x3_t intlv_rgb;
    int i = 0;
    for (i = 0; i < num8x16; i++) {
        intlv_rgb = vld3q_u8(src_data + 3 * 16 * i);
        vst1q_u8(dst_data + 16 * i, intlv_rgb.val[0]);
        vst1q_u8(dst_data + stride + 16 * i, intlv_rgb.val[1]);
        vst1q_u8(dst_data + stride * 2 + 16 * i, intlv_rgb.val[2]);
    }
    if (remain > 0) {
        for (int j = 0; j < remain; j++) {
            *(dst_data + 16 * i + j) = *(src_data + 3 * 16 * i + j * 3);
            *(dst_data + stride + 16 * i + j) = *(src_data+ 3 * 16 * i + j * 3 + 1);
            *(dst_data + stride * 2 + 16 * i + j) = *(src_data + 3 * 16 * i + j * 3 + 2);
        }
    }
}

static void chw_2_hwc_neon_u8(uint8_t *src_data, uint8_t *dst_data, int w, int h, int c) {
    int stride = w * h;
    if (c == 1){
        memcpy(dst_data, src_data, stride);
        return;
    }

    int num8x16 = stride / 16;
    int remain = stride % 16;
    uint8x16x3_t intlv_rgb;
    uint8x16_t intlv_b;
    uint8x16_t intlv_g;
    uint8x16_t intlv_r;
    int i = 0;
    for (i = 0; i < num8x16; i++) {
        intlv_b = vld1q_u8(src_data + 16 * i);
        intlv_g = vld1q_u8(src_data + stride + 16 * i);
        intlv_r = vld1q_u8(src_data + stride * 2 + 16 * i);
        intlv_rgb.val[0] = intlv_b;
        intlv_rgb.val[1] = intlv_g;
        intlv_rgb.val[2] = intlv_r;
        vst3q_u8(dst_data +  3 * 16 * i, intlv_rgb);
    }
    if (remain > 0) {
        for (int j = 0; j < remain; j++) {
            *(dst_data + 3 * 16 * i + j * 3) = *(src_data + 16 * i + j);
            *(dst_data + 3 * 16 * i + j * 3 + 1) = *(src_data + stride + 16 * i + j);
            *(dst_data + 3 * 16 * i + j * 3 + 2) = *(src_data + stride * 2 + 16 * i + j);
        }
    }
}

static void hwc_2_chw_neon_fp32(float *src_data, float *dst_data, int w, int h, int c) {
    int stride = w * h;
    if (c == 1){
        memcpy(dst_data, src_data, stride);
        return;
    }

    int num32x4 = stride / 4;
    int remain = stride % 4;
    float32x4x3_t fp32lv_rgb;
    int i = 0;
    for (i = 0; i < num32x4; i++) {
        fp32lv_rgb = vld3q_f32(src_data + 3 * 4 * i);
        vst1q_f32(dst_data + 4 * i, fp32lv_rgb.val[0]);
        vst1q_f32(dst_data + stride + 4 * i, fp32lv_rgb.val[1]);
        vst1q_f32(dst_data + stride * 2 + 4 * i, fp32lv_rgb.val[2]);
    }
    if (remain > 0) {
        for (int j = 0; j < remain; j++) {
            *(dst_data + 4 * i + j) = *(src_data + 3 * 4 * i + j * 3);
            *(dst_data + stride + 4 * i + j) = *(src_data+ 3 * 4 * i + j * 3 + 1);
            *(dst_data + stride * 2 + 4 * i + j) = *(src_data + 3 * 4 * i + j * 3 + 2);
        }
    }
}

static void chw_2_hwc_neon_fp32(float *src_data, float *dst_data, int w, int h, int c) {
    int stride = w * h;
    if (c == 1){
        memcpy(dst_data, src_data, stride);
        return;
    }

    int num32x4 = stride / 4;
    int remain = stride % 4;
    float32x4x3_t intlv_rgb;
    float32x4_t intlv_b;
    float32x4_t intlv_g;
    float32x4_t intlv_r;
    int i = 0;
    for (i = 0; i < num32x4; i++) {
        intlv_b = vld1q_f32(src_data + 4 * i);
        intlv_g = vld1q_f32(src_data + stride + 4 * i);
        intlv_r = vld1q_f32(src_data + stride * 2 + 4 * i);
        intlv_rgb.val[0] = intlv_b;
        intlv_rgb.val[1] = intlv_g;
        intlv_rgb.val[2] = intlv_r;
        vst3q_f32(dst_data + 3 * 4 * i, intlv_rgb);
    }
    if (remain > 0) {
        for (int j = 0; j < remain; j++) {
            *(dst_data + 3 * 4 * i + j * 3) = *(src_data + 4 * i + j);
            *(dst_data + 3 * 4 * i + j * 3 + 1) = *(src_data + stride + 4 * i + j);
            *(dst_data + 3 * 4 * i + j * 3 + 2) = *(src_data + stride * 2 + 4 * i + j);
        }
    }
}

