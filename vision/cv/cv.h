#ifndef VISION_CV_H
#define VISION_CV_H

#include <vector>

#include "../common/tensor.h"
#include "../common/vision_structs.h"

namespace va_cv {

struct VSize {
    int w;
    int h;
    VSize() : w(0), h(0) {}
    VSize(int _w, int _h) : w(_w), h(_h) {}
};

struct VScalar {
    double v0;
    double v1;
    double v2;
    double v3;

    VScalar() : v0(0), v1(0), v2(0), v3(0) {}
};

// 插值模式
enum VInterMode {
    INTER_NEAREST = 0,
    INTER_LINEAR = 1,
    INTER_CUBIC = 2,
    INTER_AREA = 3,
    INTER_LANCZOS4 = 4,
    INTER_MAX = 7,
    WARP_INVERSE_MAP = 16
};

// 边界模式
enum VBorderMode {
    BORDER_REPLICATE = 1,
    BORDER_CONSTANT = 0,
    BORDER_REFLECT = 2,
    BORDER_WRAP = 3,
    BORDER_REFLECT_101 = 4,
    BORDER_REFLECT101 = 4,
    BORDER_TRANSPARENT = 5,
    BORDER_DEFAULT = 4,
    BORDER_ISOLATED = 16
};

// 模板匹配模式
enum VMatchMode {
    TM_SQDIFF = 0,
    TM_SQDIFF_NORMED = 1,
    TM_CCORR = 2,
    TM_CCORR_NORMED = 3,
    TM_CCOEFF = 4,
    TM_CCOEFF_NORMED = 5
};

// 颜色转换模式
enum InputImageFormat {
    COLOR_GRAY2RGB = 8,
    COLOR_GRAY2BGR = COLOR_GRAY2RGB,
    COLOR_YUV2RGB_NV12 = 90,
    COLOR_YUV2BGR_NV12 = 91,
    COLOR_YUV2RGB_NV21 = 92,
    COLOR_YUV2BGR_NV21 = 93,
    COLOR_YUV2RGBA_NV12 = 94,
    COLOR_YUV2BGRA_NV12 = 95,
    COLOR_YUV2RGBA_NV21 = 96,
    COLOR_YUV2BGRA_NV21 = 97,
    COLOR_YUV2BGR_YV12 = 99
};

/**
 * @brief 图像缩放
 * @param src 输入
 * @param dst 输出
 * @param dsize 目标尺寸
 * @param fx
 * @param fy
 * @param interpolation 插值模式
 */
void resize(const vision::Tensor& src, vision::Tensor& dst,
            VSize dsize, double fx = 0, double fy = 0,
            int interpolation = INTER_LINEAR);

/**
 * @brief 颜色转换
 * @param src 输入
 * @param dst 输出
 * @param code 颜色转换模式
 */
void cvt_color(const vision::Tensor& src, vision::Tensor& dst, int code);

/**
 * @brief 图像标准化
 * @param src 输入
 * @param dst 输出
 * @param mean 均值
 * @param stddev 标准差
 */
void normalize(const vision::Tensor& src, vision::Tensor& dst,
               const vision::Tensor& mean = vision::Tensor(),
               const vision::Tensor& stddev = vision::Tensor());

/**
 * @brief 仿射变换
 * @param src 输入
 * @param dst 输出
 * @param M 仿射变换矩阵
 * @param dsize 目标尺寸
 * @param flags 插值模式
 * @param borderMode 边界模式
 * @param borderValue 边界值
 */
void warp_affine(const vision::Tensor& src, vision::Tensor& dst,
                 const vision::Tensor& M, VSize dsize,
                 int flags = INTER_LINEAR,
                 int borderMode = BORDER_CONSTANT,
                 const VScalar& borderValue = VScalar());

/**
 * @brief 仿射变换
 * @param src 输入
 * @param dst 输出
 * @param scale 缩放系数
 * @param rot 旋转角度
 * @param dsize 目标尺寸
 * @param aux_param 修正参数（非通用）
 * @param flags
 * @param borderMode
 * @param borderValue
 */
void warp_affine(const vision::Tensor& src, vision::Tensor& dst,
                 float scale, float rot, VSize dsize,
                 const VScalar& aux_param = VScalar(),
                 int flags = INTER_LINEAR,
                 int borderMode = BORDER_CONSTANT,
                 const VScalar& borderValue = VScalar());

/**
 * @brief 图像缩放，然后标准化
 * @param src 输入
 * @param dst 输出
 * @param dsize 目标尺寸
 * @param fx
 * @param fy
 * @param interpolation 插值模式
 * @param mean 均值
 * @param stddev 标准差
 */
void resize_normalize(const vision::Tensor& src, vision::Tensor& dst,
                      VSize dsize, double fx = 0, double fy = 0,
                      int interpolation = INTER_LINEAR,
                      const vision::Tensor& mean = vision::Tensor(),
                      const vision::Tensor& stddev = vision::Tensor());

/**
 * @brief 图仿射变换，然后标准化
 * @param src 输入
 * @param dst 输出
 * @param M 放射变换矩阵
 * @param dsize 目标尺寸
 * @param flags
 * @param borderMode
 * @param borderValue
 * @param mean 均值
 * @param stddev 标准差
 */
void warp_affine_normalize(const vision::Tensor& src, vision::Tensor& dst,
                           const vision::Tensor& M, VSize dsize,
                           int flags = INTER_LINEAR,
                           int borderMode = BORDER_CONSTANT,
                           const VScalar& borderValue = VScalar(),
                           const vision::Tensor& mean = vision::Tensor(),
                           const vision::Tensor& stddev = vision::Tensor());

/**
 * @brief 仿射变换，然后标准化
 * @param src 输入
 * @param dst 输出
 * @param scale 缩放系数
 * @param rot 旋转角度
 * @param dsize 目标尺寸
 * @param aux_param 修正参数（非通用）
 * @param flags
 * @param borderMode
 * @param borderValue
 * @param mean 均值
 * @param stddev 标准差
 */
void warp_affine_normalize(const vision::Tensor& src, vision::Tensor& dst,
                           float scale, float rot, VSize dsize,
                           const VScalar& aux_param = VScalar(),
                           int flags = INTER_LINEAR,
                           int borderMode = BORDER_CONSTANT,
                           const VScalar& borderValue = VScalar(),
                           const vision::Tensor& mean = vision::Tensor(),
                           const vision::Tensor& stddev = vision::Tensor());

/**
 * @brief 图像剪切
 * @param src 输入
 * @param dst 输出
 * @param rect ROI 区域
 */
void crop(const vision::Tensor& src, vision::Tensor& dst, const vision::VRect& rect);

/**
 * @brief 模板匹配
 * @param src 输入
 * @param target 匹配模板
 * @param result 匹配结果
 * @param method 匹配方法
 */
void match_template(const vision::Tensor& src, const vision::Tensor& target,
                    vision::Tensor& result, int method);

/**
 * @brief 求矩阵中最大值、最小值的位置
 * @param src 输入矩阵
 * @param minVal 最小值，nullptr 表示不需要
 * @param maxVal 最大值，nullptr 表示不需要
 * @param minIdx 最小值的位置，nullptr 表示不需要
 * @param maxIdx 最大值的位置，nullptr 表示不需要
 * @param mask 掩膜
 */
void minMaxIdx(const vision::Tensor& src, double* minVal, double* maxVal,
               int* minIdx = nullptr, int* maxIdx = nullptr, const vision::Tensor& mask=vision::Tensor());

/**
 * @brief 压缩图片
 * @param src 输入图片
 * @param target 输出图片
 * @param format 压缩格式
 */
void imencode(const vision::Tensor& src, std::vector<unsigned char>& buf, const char* format);

} // namespace va_cv

#endif // VISION_CV_H