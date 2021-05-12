#include "cuda_device.h"

#include <cstdlib>
#include <math.h>

#include "../common/macro.h"

namespace va_cv {

int CudaDevice::get_device_count() {
    int dev_count = 0;
    int err = cudaGetDeviceCount( &dev_count );
    return err == cudaSuccess ? dev_count : -1;
}

int CudaDevice::set_device(int device) {
    return cudaSetDevice( device );
}

}