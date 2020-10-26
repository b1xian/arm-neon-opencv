#ifndef VISION_VA_ALLOCATOR_H
#define VISION_VA_ALLOCATOR_H

namespace vision {

class VaAllocator {
public:
    static void* allocate(int len);
    static void deallocate(void* ptr);
    static int align_size(int len);
    static int align_size(int len, int size);
};

} // namespace vision

#endif //VISION_VA_ALLOCATOR_H
