// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// ncnn license
// Tencent is pleased to support the open source community by making ncnn
// available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "image_resize.h"

#include <arm_neon.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <algorithm>
#include <iostream>

void ImageResize::choose(const uint8_t* src,
                         uint8_t* dst,
                         ImageFormat srcFormat,
                         int srcw,
                         int srch,
                         int dstw,
                         int dsth) {
  resize(src, dst, srcFormat, srcw, srch, dstw, dsth);
}

void resize_three_channel(
    const uint8_t* src, int w_in, int h_in, uint8_t* dst, int w_out, int h_out);

void bgr_resize(const uint8_t* src,
                uint8_t* dst,
                int w_in,
                int h_in,
                int w_out,
                int h_out) {
  if (w_out == w_in && h_out == h_in) {
    memcpy(dst, src, sizeof(uint8_t) * w_in * h_in * 3);
    return;
  }
  // y
  resize_three_channel(src, w_in * 3, h_in, dst, w_out * 3, h_out);
}
void resize_three_channel(const uint8_t* src,
                          int w_in,
                          int h_in,
                          uint8_t* dst,
                          int w_out,
                          int h_out) {
  const int resize_coef_bits = 11;
  // 放大2048倍
  const int resize_coef_scale = 1 << resize_coef_bits;
  double scale_x = static_cast<double>(w_in) / w_out;
  double scale_y = static_cast<double>(h_in) / h_out;
  // w_out*2:保存每行每个元素的1-u和u; h_out*2，保存每列每个元素的1-v和v
  int* buf = new int[w_out * 2 + h_out * 2];
  // 每列，每个元素的u保存，起始位置
  int* xofs = buf;          // new int[w];
  // 每行，每个元素的v保存 起始位置
  int* yofs = buf + w_out;  // new int[h];
  // ialpha需要w*2,将int32*w作为int16*w*2使用
  int16_t* ialpha =
      reinterpret_cast<int16_t*>(buf + w_out + h_out);  // new int16_t[w * 2];
  // ibeta需要h*2,将int32*w作为int16*w*2使用
  int16_t* ibeta =
      reinterpret_cast<int16_t*>(buf + w_out * 2 + h_out);  // new short[h * 2];
  float fx = 0.f;
  float fy = 0.f;
  int sx = 0.f;
  int sy = 0.f;
#define SATURATE_CAST_SHORT(X)                                               \
  (int16_t)::std::min(                                                       \
      ::std::max(static_cast<int>(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), \
      SHRT_MAX);
  // #pragma omp parallel for
  /*
   * 计算i   :    int32*w
   *    j   :    int32*h
   *    1-u :    int16*w
   *    u   :    int16*w
   *    1-v :    int16*h
   *    v   :    int16*h
   *    保存至buf
   */
  for (int dx = 0; dx < w_out / 3; dx++) {
    // src_x = (dstX + 0.5) * (srcWidth/dstWidth) - 0.5
    fx = static_cast<float>((dx + 0.5) * scale_x - 0.5);
    // sx:src_x的整数部分 i
    sx = floor(fx);
    // fx:src_x的小数部分 u
    fx -= sx;
    if (sx < 0) {
      sx = 0;
      fx = 0.f;
    }
    // 如果x越界
    if (sx >= w_in - 1) {
      sx = w_in - 2;
      fx = 1.f;
    }
    // dx: 0 ~ w_out/3,保存src_x的整数部分 i
    xofs[dx] = sx * 3;
    // 1 - u
    float a0 = (1.f - fx) * resize_coef_scale;
    // u
    float a1 = fx * resize_coef_scale;
    // 保存一行中每个元素的1-u 和 u
    ialpha[dx * 2] = SATURATE_CAST_SHORT(a0);
    ialpha[dx * 2 + 1] = SATURATE_CAST_SHORT(a1);
  }
  // #pragma omp parallel for
  for (int dy = 0; dy < h_out; dy++) {
    fy = static_cast<float>((dy + 0.5) * scale_y - 0.5);
    sy = floor(fy);
    fy -= sy;
    if (sy < 0) {
      sy = 0;
      fy = 0.f;
    }
    if (sy >= h_in - 1) {
      sy = h_in - 2;
      fy = 1.f;
    }
    // dy: 0 ~ h,保存src_y的整数部分 j
    yofs[dy] = sy;
    // 1 - v
    float b0 = (1.f - fy) * resize_coef_scale;
    // v
    float b1 = fy * resize_coef_scale;
    // 保存一列中每个元素的1-v 和 v
    ibeta[dy * 2] = SATURATE_CAST_SHORT(b0);
    ibeta[dy * 2 + 1] = SATURATE_CAST_SHORT(b1);
  }
#undef SATURATE_CAST_SHORT
  // loop body
  int16_t* rowsbuf0 = new int16_t[w_out + 1];
  int16_t* rowsbuf1 = new int16_t[w_out + 1];
  int16_t* rows0 = rowsbuf0;
  int16_t* rows1 = rowsbuf1;
  int prev_sy1 = -1;
  for (int dy = 0; dy < h_out; dy++) {
      // sy : src_y的整数部分，本行元素的sy一致
    int sy = yofs[dy];
    if (sy == prev_sy1) {
      // hresize one row
      int16_t* rows0_old = rows0;
      rows0 = rows1;
      rows1 = rows0_old;
      const uint8_t* S1 = src + w_in * (sy + 1);
      const int16_t* ialphap = ialpha;
      int16_t* rows1p = rows1;
      for (int dx = 0; dx < w_out / 3; dx++) {
        int sx = xofs[dx];
        int16_t a0 = ialphap[0];
        int16_t a1 = ialphap[1];
        const uint8_t* S1p = S1 + sx;
        int tmp = dx * 3;
        rows1p[tmp] = (S1p[0] * a0 + S1p[3] * a1) >> 4;
        rows1p[tmp + 1] = (S1p[1] * a0 + S1p[4] * a1) >> 4;
        rows1p[tmp + 2] = (S1p[2] * a0 + S1p[5] * a1) >> 4;
        ialphap += 2;
      }
    } else {
      // hresize two rows // 取src中的两行
      const uint8_t* S0 = src + w_in * (sy);
      const uint8_t* S1 = src + w_in * (sy + 1);
      // 取每行每个元素的1-u 和 u
      const int16_t* ialphap = ialpha;
      int16_t* rows0p = rows0;
      int16_t* rows1p = rows1;
      // 遍历该行每个元素
      for (int dx = 0; dx < w_out / 3; dx++) {
          // 当前元素的sx
        int sx = xofs[dx]; // sx:src_x的整数部分 i
        int16_t a0 = ialphap[0];    // 当前元素的1-u
        int16_t a1 = ialphap[1];    // 当前元素的u
        // 需要src中的两行
        const uint8_t* S0p = S0 + sx;
        const uint8_t* S1p = S1 + sx;
        int tmp = dx * 3; // 三通道
        // rows0p : new int16_t[w_out + 1]
        // b通道
        // TODO 右移4位
        // S0p[0]:对应原图左上像素值 a0:1-u S0p[3] 对应原图右上像素值， a1:u
        rows0p[tmp] = (S0p[0] * a0 + S0p[3] * a1) >> 4;
        // S0p[0]:对应原图左下像素值 a0:1-u S0p[3] 对应原图右下像素值， a1:u
        rows1p[tmp] = (S1p[0] * a0 + S1p[3] * a1) >> 4;
        // g通道
        rows0p[tmp + 1] = (S0p[1] * a0 + S0p[4] * a1) >> 4;
        rows1p[tmp + 1] = (S1p[1] * a0 + S1p[4] * a1) >> 4;
        // r通道
        rows0p[tmp + 2] = (S0p[2] * a0 + S0p[5] * a1) >> 4;
        rows1p[tmp + 2] = (S1p[2] * a0 + S1p[5] * a1) >> 4;
        ialphap += 2; // 每个元素的1-u和u
      }
    }
    // 标记剩余行数
    prev_sy1 = sy + 1;
    // vresize
    int16_t b0 = ibeta[0];  //dst当前列的1-v
    int16_t b1 = ibeta[1];  //dst当前列的v
    int16_t* rows0p = rows0;
    int16_t* rows1p = rows1;
    // dst当前列的首地址
    uint8_t* dp_ptr = dst + w_out * (dy);
    // 每次操作8个元素,一行需要多少次操作
    int cnt = w_out >> 3;
    // 最终每行剩余多少个元素未操作
    int remain = w_out - (cnt << 3);
    int16x4_t _b0 = vdup_n_s16(b0);
    int16x4_t _b1 = vdup_n_s16(b1);
    int32x4_t _v2 = vdupq_n_s32(2);
    // f(i+u,j+v) = (1-u)(1-v)f(i,j) + (1-u)vf(i,j+1) + u(1-v)f(i+1,j) + uvf(i+1,j+1)
    // 已计算部分 rows0: f(i,j)*(1-u)   + f(i+1,j)*u
    //          rows1: f(i,j+1)*(1-u) + f(i+1,j+1)*u
    for (; cnt > 0; cnt--) {
        // 取前四个像素的rows0和rows1
      int16x4_t _rows0p_sr4 = vld1_s16(rows0p);
      int16x4_t _rows1p_sr4 = vld1_s16(rows1p);
      // dst每个像素乘以y插值
      // 当前完成计算：f(i,j)*(1-u)*(1-v)
      int32x4_t _rows0p_sr4_mb0 = vmull_s16(_rows0p_sr4, _b0);
      // 当前完成计算：f(i+1,j)*u*(1-v)
      int32x4_t _rows1p_sr4_mb1 = vmull_s16(_rows1p_sr4, _b1);

      // 取后四个像素的rows0和rows1
      int16x4_t _rows0p_1_sr4 = vld1_s16(rows0p + 4);
      int16x4_t _rows1p_1_sr4 = vld1_s16(rows1p + 4);
      // dst每个像素乘以y插值
      int32x4_t _rows0p_1_sr4_mb0 = vmull_s16(_rows0p_1_sr4, _b0);
      int32x4_t _rows1p_1_sr4_mb1 = vmull_s16(_rows1p_1_sr4, _b1);

      // TODO 右移16位
      int32x4_t _acc = _v2;
      // 前四个像素，rows0 + rows1
      _acc = vsraq_n_s32(_acc, _rows0p_sr4_mb0, 16);  // _acc >> 16 + _rows0p_sr4_mb0 >> 16
      _acc = vsraq_n_s32(_acc, _rows1p_sr4_mb1, 16);

      int32x4_t _acc_1 = _v2;
      // 后四个像素，rows0 + rows1
      _acc_1 = vsraq_n_s32(_acc_1, _rows0p_1_sr4_mb0, 16);
      _acc_1 = vsraq_n_s32(_acc_1, _rows1p_1_sr4_mb1, 16);

      // 截断为int16 TODO 右移2位，对应之前左移的22位
      int16x4_t _acc16 = vshrn_n_s32(_acc, 2);  // _acc >> 2
      int16x4_t _acc16_1 = vshrn_n_s32(_acc_1, 2);

      // 拼接前后8个像素
      uint8x8_t _dout = vqmovun_s16(vcombine_s16(_acc16, _acc16_1));

      // 设置dst像素值
      vst1_u8(dp_ptr, _dout);
      dp_ptr += 8;
      rows0p += 8;
      rows1p += 8;
    }
    // 余数remain部分计算
    for (; remain; --remain) {
      // D[x] = (rows0[x]*b0 + rows1[x]*b1) >> INTER_RESIZE_COEF_BITS;
      *dp_ptr++ =
          (uint8_t)(((int16_t)((b0 * (int16_t)(*rows0p++)) >> 16) +
                     (int16_t)((b1 * (int16_t)(*rows1p++)) >> 16) + 2) >>
                    2);
    }
    ibeta += 2;
  }
  delete[] buf;
  delete[] rowsbuf0;
  delete[] rowsbuf1;
}
void resize_one_channel(
    const uint8_t* src, int w_in, int h_in, uint8_t* dst, int w_out, int h_out);
void resize_one_channel_uv(
    const uint8_t* src, int w_in, int h_in, uint8_t* dst, int w_out, int h_out);
void nv21_resize(const uint8_t* src,
                 uint8_t* dst,
                 int w_in,
                 int h_in,
                 int w_out,
                 int h_out) {
  if (w_out == w_in && h_out == h_in) {
    memcpy(dst, src, sizeof(uint8_t) * w_in * static_cast<int>(1.5 * h_in));
    return;
  }
  //     return;
  int y_h = h_in;
  int uv_h = h_in / 2;
  const uint8_t* y_ptr = src;
  const uint8_t* uv_ptr = src + y_h * w_in;
  // out
  int dst_y_h = h_out;
  int dst_uv_h = h_out / 2;
  uint8_t* dst_ptr = dst + dst_y_h * w_out;
  // y
  resize_one_channel(y_ptr, w_in, y_h, dst, w_out, dst_y_h);
  // uv
  resize_one_channel_uv(uv_ptr, w_in, uv_h, dst_ptr, w_out, dst_uv_h);
}

void resize_one_channel(const uint8_t* src,
                        int w_in,
                        int h_in,
                        uint8_t* dst,
                        int w_out,
                        int h_out) {
  const int resize_coef_bits = 11;
  const int resize_coef_scale = 1 << resize_coef_bits;

  double scale_x = static_cast<double>(w_in) / w_out;
  double scale_y = static_cast<double>(h_in) / h_out;

  int* buf = new int[w_out * 2 + h_out * 2];

  int* xofs = buf;          // new int[w];
  int* yofs = buf + w_out;  // new int[h];

  int16_t* ialpha =
      reinterpret_cast<int16_t*>(buf + w_out + h_out);  // new short[w * 2];
  int16_t* ibeta =
      reinterpret_cast<int16_t*>(buf + w_out * 2 + h_out);  // new short[h * 2];

  float fx = 0.f;
  float fy = 0.f;
  int sx = 0;
  int sy = 0;

#define SATURATE_CAST_SHORT(X)                                               \
  (int16_t)::std::min(                                                       \
      ::std::max(static_cast<int>(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), \
      SHRT_MAX);
  for (int dx = 0; dx < w_out; dx++) {
    fx = static_cast<float>((dx + 0.5) * scale_x - 0.5);
    sx = floor(fx);
    fx -= sx;

    if (sx < 0) {
      sx = 0;
      fx = 0.f;
    }
    if (sx >= w_in - 1) {
      sx = w_in - 2;
      fx = 1.f;
    }

    xofs[dx] = sx;

    float a0 = (1.f - fx) * resize_coef_scale;
    float a1 = fx * resize_coef_scale;

    ialpha[dx * 2] = SATURATE_CAST_SHORT(a0);
    ialpha[dx * 2 + 1] = SATURATE_CAST_SHORT(a1);
  }
  for (int dy = 0; dy < h_out; dy++) {
    fy = static_cast<float>((dy + 0.5) * scale_y - 0.5);
    sy = floor(fy);
    fy -= sy;

    if (sy < 0) {
      sy = 0;
      fy = 0.f;
    }
    if (sy >= h_in - 1) {
      sy = h_in - 2;
      fy = 1.f;
    }

    yofs[dy] = sy;

    float b0 = (1.f - fy) * resize_coef_scale;
    float b1 = fy * resize_coef_scale;

    ibeta[dy * 2] = SATURATE_CAST_SHORT(b0);
    ibeta[dy * 2 + 1] = SATURATE_CAST_SHORT(b1);
  }
#undef SATURATE_CAST_SHORT
  // loop body
  int16_t* rowsbuf0 = new int16_t[w_out + 1];
  int16_t* rowsbuf1 = new int16_t[w_out + 1];
  int16_t* rows0 = rowsbuf0;
  int16_t* rows1 = rowsbuf1;

  int prev_sy1 = -1;
  for (int dy = 0; dy < h_out; dy++) {
    int sy = yofs[dy];

    if (sy == prev_sy1) {
      // hresize one row
      int16_t* rows0_old = rows0;
      rows0 = rows1;
      rows1 = rows0_old;
      const uint8_t* S1 = src + w_in * (sy + 1);
      const int16_t* ialphap = ialpha;
      int16_t* rows1p = rows1;
      for (int dx = 0; dx < w_out; dx++) {
        int sx = xofs[dx];
        int16_t a0 = ialphap[0];
        int16_t a1 = ialphap[1];

        const uint8_t* S1p = S1 + sx;
        rows1p[dx] = (S1p[0] * a0 + S1p[1] * a1) >> 4;

        ialphap += 2;
      }
    } else {
      // hresize two rows
      const uint8_t* S0 = src + w_in * (sy);
      const uint8_t* S1 = src + w_in * (sy + 1);

      const int16_t* ialphap = ialpha;
      int16_t* rows0p = rows0;
      int16_t* rows1p = rows1;
      for (int dx = 0; dx < w_out; dx++) {
        int sx = xofs[dx];
        int16_t a0 = ialphap[0];
        int16_t a1 = ialphap[1];

        const uint8_t* S0p = S0 + sx;
        const uint8_t* S1p = S1 + sx;
        rows0p[dx] = (S0p[0] * a0 + S0p[1] * a1) >> 4;
        rows1p[dx] = (S1p[0] * a0 + S1p[1] * a1) >> 4;

        ialphap += 2;
      }
    }

    prev_sy1 = sy + 1;

    // vresize
    int16_t b0 = ibeta[0];
    int16_t b1 = ibeta[1];

    int16_t* rows0p = rows0;
    int16_t* rows1p = rows1;
    uint8_t* dp_ptr = dst + w_out * (dy);

    int cnt = w_out >> 3;
    int remain = w_out - (cnt << 3);
    int16x4_t _b0 = vdup_n_s16(b0);
    int16x4_t _b1 = vdup_n_s16(b1);
    int32x4_t _v2 = vdupq_n_s32(2);

    for (cnt = w_out >> 3; cnt > 0; cnt--) {
      int16x4_t _rows0p_sr4 = vld1_s16(rows0p);
      int16x4_t _rows1p_sr4 = vld1_s16(rows1p);
      int16x4_t _rows0p_1_sr4 = vld1_s16(rows0p + 4);
      int16x4_t _rows1p_1_sr4 = vld1_s16(rows1p + 4);

      int32x4_t _rows0p_sr4_mb0 = vmull_s16(_rows0p_sr4, _b0);
      int32x4_t _rows1p_sr4_mb1 = vmull_s16(_rows1p_sr4, _b1);
      int32x4_t _rows0p_1_sr4_mb0 = vmull_s16(_rows0p_1_sr4, _b0);
      int32x4_t _rows1p_1_sr4_mb1 = vmull_s16(_rows1p_1_sr4, _b1);

      int32x4_t _acc = _v2;
      _acc = vsraq_n_s32(_acc, _rows0p_sr4_mb0, 16);
      _acc = vsraq_n_s32(_acc, _rows1p_sr4_mb1, 16);

      int32x4_t _acc_1 = _v2;
      _acc_1 = vsraq_n_s32(_acc_1, _rows0p_1_sr4_mb0, 16);
      _acc_1 = vsraq_n_s32(_acc_1, _rows1p_1_sr4_mb1, 16);

      int16x4_t _acc16 = vshrn_n_s32(_acc, 2);
      int16x4_t _acc16_1 = vshrn_n_s32(_acc_1, 2);

      uint8x8_t _dout = vqmovun_s16(vcombine_s16(_acc16, _acc16_1));

      vst1_u8(dp_ptr, _dout);

      dp_ptr += 8;
      rows0p += 8;
      rows1p += 8;
    }
    for (; remain; --remain) {
      // D[x] = (rows0[x]*b0 + rows1[x]*b1) >> INTER_RESIZE_COEF_BITS;
      *dp_ptr++ =
          (uint8_t)(((int16_t)((b0 * (int16_t)(*rows0p++)) >> 16) +
                     (int16_t)((b1 * (int16_t)(*rows1p++)) >> 16) + 2) >>
                    2);
    }
    ibeta += 2;
  }

  delete[] buf;
  delete[] rowsbuf0;
  delete[] rowsbuf1;
}

void resize_one_channel_uv(const uint8_t* src,
                           int w_in,
                           int h_in,
                           uint8_t* dst,
                           int w_out,
                           int h_out) {
  const int resize_coef_bits = 11;
  const int resize_coef_scale = 1 << resize_coef_bits;

  double scale_x = static_cast<double>(w_in) / w_out;
  double scale_y = static_cast<double>(h_in) / h_out;

  int* buf = new int[w_out * 2 + h_out * 2];

  int* xofs = buf;          // new int[w];
  int* yofs = buf + w_out;  // new int[h];

  int16_t* ialpha =
      reinterpret_cast<int16_t*>(buf + w_out + h_out);  // new int16_t[w * 2];
  int16_t* ibeta = reinterpret_cast<int16_t*>(buf + w_out * 2 +
                                              h_out);  // new int16_t[h * 2];

  float fx = 0.f;
  float fy = 0.f;
  int sx = 0.f;
  int sy = 0.f;

#define SATURATE_CAST_SHORT(X)                                               \
  (int16_t)::std::min(                                                       \
      ::std::max(static_cast<int>(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), \
      SHRT_MAX);
  for (int dx = 0; dx < w_out / 2; dx++) {
    fx = static_cast<float>((dx + 0.5) * scale_x - 0.5);
    sx = floor(fx);
    fx -= sx;

    if (sx < 0) {
      sx = 0;
      fx = 0.f;
    }
    if (sx >= w_in - 1) {
      sx = w_in - 2;
      fx = 1.f;
    }

    xofs[dx] = sx;

    float a0 = (1.f - fx) * resize_coef_scale;
    float a1 = fx * resize_coef_scale;

    ialpha[dx * 2] = SATURATE_CAST_SHORT(a0);
    ialpha[dx * 2 + 1] = SATURATE_CAST_SHORT(a1);
  }
  for (int dy = 0; dy < h_out; dy++) {
    fy = static_cast<float>((dy + 0.5) * scale_y - 0.5);
    sy = floor(fy);
    fy -= sy;

    if (sy < 0) {
      sy = 0;
      fy = 0.f;
    }
    if (sy >= h_in - 1) {
      sy = h_in - 2;
      fy = 1.f;
    }

    yofs[dy] = sy;

    float b0 = (1.f - fy) * resize_coef_scale;
    float b1 = fy * resize_coef_scale;

    ibeta[dy * 2] = SATURATE_CAST_SHORT(b0);
    ibeta[dy * 2 + 1] = SATURATE_CAST_SHORT(b1);
  }

#undef SATURATE_CAST_SHORT
  // loop body
  int16_t* rowsbuf0 = new int16_t[w_out + 1];
  int16_t* rowsbuf1 = new int16_t[w_out + 1];
  int16_t* rows0 = rowsbuf0;
  int16_t* rows1 = rowsbuf1;

  int prev_sy1 = -1;
  for (int dy = 0; dy < h_out; dy++) {
    int sy = yofs[dy];
    if (sy == prev_sy1) {
      // hresize one row
      int16_t* rows0_old = rows0;
      rows0 = rows1;
      rows1 = rows0_old;
      const uint8_t* S1 = src + w_in * (sy + 1);

      const int16_t* ialphap = ialpha;
      int16_t* rows1p = rows1;
      for (int dx = 0; dx < w_out / 2; dx++) {
        int sx = xofs[dx] * 2;
        int16_t a0 = ialphap[0];
        int16_t a1 = ialphap[1];
        const uint8_t* S1p = S1 + sx;
        int tmp = dx * 2;
        rows1p[tmp] = (S1p[0] * a0 + S1p[2] * a1) >> 4;
        rows1p[tmp + 1] = (S1p[1] * a0 + S1p[3] * a1) >> 4;

        ialphap += 2;
      }
    } else {
      // hresize two rows
      const uint8_t* S0 = src + w_in * (sy);
      const uint8_t* S1 = src + w_in * (sy + 1);

      const int16_t* ialphap = ialpha;
      int16_t* rows0p = rows0;
      int16_t* rows1p = rows1;
      for (int dx = 0; dx < w_out / 2; dx++) {
        int sx = xofs[dx] * 2;
        int16_t a0 = ialphap[0];
        int16_t a1 = ialphap[1];

        const uint8_t* S0p = S0 + sx;
        const uint8_t* S1p = S1 + sx;
        int tmp = dx * 2;
        rows0p[tmp] = (S0p[0] * a0 + S0p[2] * a1) >> 4;
        rows1p[tmp] = (S1p[0] * a0 + S1p[2] * a1) >> 4;

        rows0p[tmp + 1] = (S0p[1] * a0 + S0p[3] * a1) >> 4;
        rows1p[tmp + 1] = (S1p[1] * a0 + S1p[3] * a1) >> 4;
        ialphap += 2;
      }
    }
    prev_sy1 = sy + 1;

    // vresize
    int16_t b0 = ibeta[0];
    int16_t b1 = ibeta[1];

    int16_t* rows0p = rows0;
    int16_t* rows1p = rows1;
    uint8_t* dp_ptr = dst + w_out * (dy);

    int cnt = w_out >> 3;
    int remain = w_out - (cnt << 3);
    int16x4_t _b0 = vdup_n_s16(b0);
    int16x4_t _b1 = vdup_n_s16(b1);
    int32x4_t _v2 = vdupq_n_s32(2);
    for (cnt = w_out >> 3; cnt > 0; cnt--) {
      int16x4_t _rows0p_sr4 = vld1_s16(rows0p);
      int16x4_t _rows1p_sr4 = vld1_s16(rows1p);
      int16x4_t _rows0p_1_sr4 = vld1_s16(rows0p + 4);
      int16x4_t _rows1p_1_sr4 = vld1_s16(rows1p + 4);

      int32x4_t _rows0p_sr4_mb0 = vmull_s16(_rows0p_sr4, _b0);
      int32x4_t _rows1p_sr4_mb1 = vmull_s16(_rows1p_sr4, _b1);
      int32x4_t _rows0p_1_sr4_mb0 = vmull_s16(_rows0p_1_sr4, _b0);
      int32x4_t _rows1p_1_sr4_mb1 = vmull_s16(_rows1p_1_sr4, _b1);

      int32x4_t _acc = _v2;
      _acc = vsraq_n_s32(
          _acc, _rows0p_sr4_mb0, 16);  // _acc >> 16 + _rows0p_sr4_mb0 >> 16
      _acc = vsraq_n_s32(_acc, _rows1p_sr4_mb1, 16);

      int32x4_t _acc_1 = _v2;
      _acc_1 = vsraq_n_s32(_acc_1, _rows0p_1_sr4_mb0, 16);
      _acc_1 = vsraq_n_s32(_acc_1, _rows1p_1_sr4_mb1, 16);

      int16x4_t _acc16 = vshrn_n_s32(_acc, 2);  // _acc >> 2
      int16x4_t _acc16_1 = vshrn_n_s32(_acc_1, 2);

      uint8x8_t _dout = vqmovun_s16(vcombine_s16(_acc16, _acc16_1));

      vst1_u8(dp_ptr, _dout);

      dp_ptr += 8;
      rows0p += 8;
      rows1p += 8;
    }
    for (; remain; --remain) {
      // D[x] = (rows0[x]*b0 + rows1[x]*b1) >> INTER_RESIZE_COEF_BITS;
      *dp_ptr++ =
          (uint8_t)(((int16_t)((b0 * (int16_t)(*rows0p++)) >> 16) +
                     (int16_t)((b1 * (int16_t)(*rows1p++)) >> 16) + 2) >>
                    2);
    }
    ibeta += 2;
  }

  delete[] buf;
  delete[] rowsbuf0;
  delete[] rowsbuf1;
}

void compute_xy(int srcw,
                int srch,
                int dstw,
                int dsth,
                int num,
                double scale_x,
                double scale_y,
                int* xofs,
                int* yofs,
                int16_t* ialpha,
                int16_t* ibeta);
// use bilinear method to resize
void resize(const uint8_t* src,
            uint8_t* dst,
            ImageFormat srcFormat,
            int srcw,
            int srch,
            int dstw,
            int dsth) {
  int size = srcw * srch;
  if (srcw == dstw && srch == dsth) {
    if (srcFormat == NV12 || srcFormat == NV21) {
      size = srcw * (static_cast<int>(1.5 * srch));
    } else if (srcFormat == BGR || srcFormat == RGB) {
      size = 3 * srcw * srch;
    } else if (srcFormat == BGRA || srcFormat == RGBA) {
      size = 4 * srcw * srch;
    }
    memcpy(dst, src, sizeof(uint8_t) * size);
    return;
  }

  int w_out = dstw;
  int w_in = srcw;
  int num = 1;
  int orih = dsth;
  if (srcFormat == GRAY) {
    num = 1;
  } else if (srcFormat == NV12 || srcFormat == NV21) {
    nv21_resize(src, dst, srcw, srch, dstw, dsth);
    return;
    num = 1;
    int hout = static_cast<int>(0.5 * dsth);
    dsth += hout;
  } else if (srcFormat == BGR || srcFormat == RGB) {
    bgr_resize(src, dst, srcw, srch, dstw, dsth);
    printf("resize \n");
    return;
    w_in = srcw * 3;
    w_out = dstw * 3;
    num = 3;
  } else if (srcFormat == BGRA || srcFormat == RGBA) {
    w_in = srcw * 4;
    w_out = dstw * 4;
    num = 4;
  }
  double scale_x = static_cast<double>(srcw) / dstw;
  double scale_y = static_cast<double>(srch) / dsth;

  int* buf = new int[dstw * 2 + dsth * 3];
  int* xofs = buf;
  int* yofs = buf + dstw;
  int16_t* ialpha = reinterpret_cast<int16_t*>(buf + dstw + dsth);
  int16_t* ibeta = reinterpret_cast<int16_t*>(buf + 2 * dstw + dsth);

  compute_xy(
      srcw, srch, dstw, orih, num, scale_x, scale_y, xofs, yofs, ialpha, ibeta);

  int* xofs1 = nullptr;
  int* yofs1 = nullptr;
  int16_t* ialpha1 = nullptr;
  if (orih < dsth) {  // uv
    int tmp = dsth - orih;
    xofs1 = new int[dstw];
    yofs1 = new int[tmp];
    ialpha1 = new int16_t[dstw];
    compute_xy(srcw,
               srch / 2,
               dstw / 2,
               tmp,
               2,
               scale_x,
               scale_y,
               xofs1,
               yofs1,
               ialpha1,
               ibeta + orih * 2);
  }
  int cnt = w_out >> 3;
  int remain = w_out % 8;
  int32x4_t _v2 = vdupq_n_s32(2);
  int prev_sy1 = -1;
  int16_t* rowsbuf0 = new int16_t[w_out + 1];
  int16_t* rowsbuf1 = new int16_t[w_out + 1];
#pragma omp parallel for
  for (int dy = 0; dy < dsth; dy++) {
    int sy = yofs[dy];
    if (dy >= orih) {
      xofs = xofs1;
      yofs = yofs1;
      ialpha = ialpha1;
      num = 2;
      sy = yofs1[dy - orih] + srch;
    }

    // hresize two rows
    const uint8_t* S0 = src + w_in * (sy);
    const uint8_t* S1 = src + w_in * (sy + 1);
    const int16_t* ialphap = ialpha;
    int16_t* rows0p = rowsbuf0;
    int16_t* rows1p = rowsbuf1;
    for (int dx = 0; dx < w_out; dx += num) {
      int sx = xofs[dx / num];
      int16_t a0 = ialphap[0];
      int16_t a1 = ialphap[1];
      const uint8_t* S0pl = S0 + sx;
      const uint8_t* S0pr = S0 + sx + num;
      const uint8_t* S1pl = S1 + sx;
      const uint8_t* S1pr = S1 + sx + num;
      for (int i = 0; i < num; i++) {
        *rows0p++ = ((*S0pl++) * a0 + (*S0pr++) * a1) >> 4;
        *rows1p++ = ((*S1pl++) * a0 + (*S1pr++) * a1) >> 4;
      }
      ialphap += 2;
    }

    int16_t b0 = ibeta[0];
    int16_t b1 = ibeta[1];
    uint8_t* dp_ptr = dst + dy * w_out;
    rows0p = rowsbuf0;
    rows1p = rowsbuf1;
    int16x8_t _b0 = vdupq_n_s16(b0);
    int16x8_t _b1 = vdupq_n_s16(b1);
    int re_cnt = cnt;
    if (re_cnt > 0) {
#ifdef __aarch64__
      asm volatile(
          "1: \n"
          "ld1 {v0.8h}, [%[rows0p]], #16 \n"
          "ld1 {v1.8h}, [%[rows1p]], #16 \n"
          "orr v6.16b, %[_v2].16b, %[_v2].16b \n"
          "orr v7.16b, %[_v2].16b, %[_v2].16b \n"
          "smull v2.4s, v0.4h, %[_b0].4h \n"
          "smull2 v4.4s, v0.8h, %[_b0].8h \n"
          "smull v3.4s, v1.4h, %[_b1].4h \n"
          "smull2 v5.4s, v1.8h, %[_b1].8h \n"

          "ssra v6.4s, v2.4s, #16 \n"
          "ssra v7.4s, v4.4s, #16 \n"
          "ssra v6.4s, v3.4s, #16 \n"
          "ssra v7.4s, v5.4s, #16 \n"

          "shrn v0.4h, v6.4s, #2 \n"
          "shrn2 v0.8h, v7.4s, #2 \n"
          "subs %w[cnt], %w[cnt], #1 \n"
          "sqxtun v1.8b, v0.8h \n"
          "st1 {v1.8b}, [%[dp]], #8 \n"
          "bne 1b \n"
          : [rows0p] "+r"(rows0p),
            [rows1p] "+r"(rows1p),
            [cnt] "+r"(re_cnt),
            [dp] "+r"(dp_ptr)
          : [_b0] "w"(_b0), [_b1] "w"(_b1), [_v2] "w"(_v2)
          : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
#else
      asm volatile(
          "mov        r4, #2          \n"
          "vdup.s32   q12, r4         \n"
          "0:                         \n"
          "vld1.s16   {d2-d3}, [%[rows0p]]!\n"
          "vld1.s16   {d6-d7}, [%[rows1p]]!\n"
          "vorr.s32   q10, q12, q12   \n"
          "vorr.s32   q11, q12, q12   \n"

          "vmull.s16  q0, d2, %e[_b0]     \n"
          "vmull.s16  q1, d3, %e[_b0]     \n"
          "vmull.s16  q2, d6, %e[_b1]     \n"
          "vmull.s16  q3, d7, %e[_b1]     \n"

          "vsra.s32   q10, q0, #16    \n"
          "vsra.s32   q11, q1, #16    \n"
          "vsra.s32   q10, q2, #16    \n"
          "vsra.s32   q11, q3, #16    \n"

          "vshrn.s32  d20, q10, #2    \n"
          "vshrn.s32  d21, q11, #2    \n"
          "subs       %[cnt], #1          \n"
          "vqmovun.s16 d20, q10        \n"
          "vst1.8     {d20}, [%[dp]]!    \n"
          "bne        0b              \n"
          : [rows0p] "+r"(rows0p),
            [rows1p] "+r"(rows1p),
            [cnt] "+r"(re_cnt),
            [dp] "+r"(dp_ptr)
          : [_b0] "w"(_b0), [_b1] "w"(_b1)
          : "cc",
            "memory",
            "r4",
            "q0",
            "q1",
            "q2",
            "q3",
            "q8",
            "q9",
            "q10",
            "q11",
            "q12");

#endif  // __aarch64__
    }
    for (int i = 0; i < remain; i++) {
      //             D[x] = (rows0[x]*b0 + rows1[x]*b1) >>
      //             INTER_RESIZE_COEF_BITS;
      *dp_ptr++ =
          (uint8_t)(((int16_t)((b0 * (int16_t)(*rows0p++)) >> 16) +
                     (int16_t)((b1 * (int16_t)(*rows1p++)) >> 16) + 2) >>
                    2);
    }
    ibeta += 2;
  }
  if (orih < dsth) {  // uv
    delete[] xofs1;
    delete[] yofs1;
    delete[] ialpha1;
  }
  delete[] buf;
  delete[] rowsbuf0;
  delete[] rowsbuf1;
}
// compute xofs, yofs, alpha, beta
void compute_xy(int srcw,
                int srch,
                int dstw,
                int dsth,
                int num,
                double scale_x,
                double scale_y,
                int* xofs,
                int* yofs,
                int16_t* ialpha,
                int16_t* ibeta) {
  float fy = 0.f;
  float fx = 0.f;
  int sy = 0;
  int sx = 0;
  const int resize_coef_bits = 11;
  const int resize_coef_scale = 1 << resize_coef_bits;
#define SATURATE_CAST_SHORT(X)                                               \
  (int16_t)::std::min(                                                       \
      ::std::max(static_cast<int>(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), \
      SHRT_MAX);

  for (int dx = 0; dx < dstw; dx++) {
    fx = static_cast<float>((dx + 0.5) * scale_x - 0.5);
    sx = floor(fx);
    fx -= sx;

    if (sx < 0) {
      sx = 0;
      fx = 0.f;
    }
    if (sx >= srcw - 1) {
      sx = srcw - 2;
      fx = 1.f;
    }

    xofs[dx] = sx * num;

    float a0 = (1.f - fx) * resize_coef_scale;
    float a1 = fx * resize_coef_scale;
    ialpha[dx * 2] = SATURATE_CAST_SHORT(a0);
    ialpha[dx * 2 + 1] = SATURATE_CAST_SHORT(a1);
  }
  for (int dy = 0; dy < dsth; dy++) {
    fy = static_cast<float>((dy + 0.5) * scale_y - 0.5);
    sy = floor(fy);
    fy -= sy;
    if (sy < 0) {
      sy = 0;
      fy = 0.f;
    }
    if (sy >= srch - 1) {
      sy = srch - 2;
      fy = 1.f;
    }
    yofs[dy] = sy;
    float b0 = (1.f - fy) * resize_coef_scale;
    float b1 = fy * resize_coef_scale;
    ibeta[dy * 2] = SATURATE_CAST_SHORT(b0);
    ibeta[dy * 2 + 1] = SATURATE_CAST_SHORT(b1);
  }
#undef SATURATE_CAST_SHORT
}
