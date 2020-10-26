#include "va_allocator.h"
#include <cstdlib>

namespace vision {

void* VaAllocator::allocate(int len) {
    if (len <= 0) {
        return nullptr;
    }

    return malloc(len);
}

void VaAllocator::deallocate(void* ptr) {
    if (ptr) {
        free(ptr);
    }
}

int VaAllocator::align_size(int len) {
    return len;
}

int VaAllocator::align_size(int sz, int n) {
    return (sz + n - 1) & -n;
}

} // namespace vision