#ifndef IMAGE_UTIL_H
#define IMAGE_UTIL_H

#include <math.h>

class ImageUtil {
public:

    /**
     * 计算两个图片数据的相似度
     * @param first
     * @param second
     * @return
     */
    template<typename T>
    static float compare_image_data(const T *first, const T *second, int len) {
        float result = 0.0f;
        if (first == nullptr || second == nullptr) {
            return result;
        }

        float norm1 = 0.000001f;
        float norm2 = 0.000001f;
        float dot = 0.0f;
        for (int i = 0; i < len; i++) {
            dot += static_cast<float>(first[i]) * static_cast<float>(second[i]);
            norm1 += static_cast<float>(first[i]) * static_cast<float>(first[i]);
            norm2 += static_cast<float>(second[i]) * static_cast<float>(second[i]);
        }
        result = dot / sqrt(norm1 * norm2);
        return result;
    }

    /**
     * BGR转换为NV21格式
     * @param src 输入BGR图像数据
     * @param dst 输出NV21图像数据
     * @param width 图像宽度
     * @param height 图像高度
     */
    static void bgr2nv21(unsigned char *src, unsigned char *dst, int width, int height);
};

#endif // IMAGE_UTIL_H