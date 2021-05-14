#include "va_allocator.h"
#include <cstdlib>
#include <iostream>

#if defined (USE_CUDA)
#include "va_cuda_allocator.cuh"
#endif


namespace vision {

void VaAllocator::allocate(void** data, int len) {
    if (len > 0) {
#if defined (USE_CUDA)
        VaCudaAllocator::cuda_host_alloc_mapped(data, len);
#else
        *data = malloc(len);
#endif
    }
}

void VaAllocator::deallocate(void* ptr) {

    if (ptr) {
#if defined (USE_CUDA)
        VaCudaAllocator::cuda_free_host(ptr);
#else
        free(ptr);
#endif
    }
}

int VaAllocator::align_size(int len) {
    return len;
}

int VaAllocator::align_size(int sz, int n) {
    return (sz + n - 1) & -n;
}

} // namespace vision