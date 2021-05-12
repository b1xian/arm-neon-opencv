#ifndef VISION_CUDA_DEVICE_H
#define VISION_CUDA_DEVICE_H

namespace va_cv {

class CudaDevice {

public:
    static int get_device_count();

    static int set_device(int device);

};

}

#endif //VISION_CUDA_DEVICE_H
