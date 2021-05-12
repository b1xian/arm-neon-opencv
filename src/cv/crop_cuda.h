#ifndef VISION_CROP_CUDA_H
#define VISION_CROP_CUDA_H

namespace va_cv {

class CropCuda {

public:
    static void crop_cuda_chw_int8(const unsigned char *src, int src_width, int src_height, int src_channel,
                                    unsigned char *dst,
                                    int crop_left, int crop_top, int crop_width, int crop_height);

    static void crop_cuda_rgb_hwc_int8(const unsigned char *src, int src_width, int src_height,
                                    unsigned char *dst,
                                    int crop_left, int crop_top, int crop_width, int crop_height);
};

}

#endif //VISION_CROP_CUDA_H
