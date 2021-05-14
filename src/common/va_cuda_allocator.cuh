//
// Created by baiduiov on 2021/5/13.
//

#ifndef VISION_CUDA_ALLOCATE_CUH
#define VISION_CUDA_ALLOCATE_CUH


namespace vision {

    class VaCudaAllocator {
    public:
        static void cuda_malloc(void** data, int len);
        static void cuda_memcpy(void *dst, void *src, int len, int copy_kind);
        static void cuda_free(void* data);


        static void cuda_host_alloc(void** data, int len, int flags);
        static void cuda_free_host(void* data);

        static void cuda_host_alloc_mapped(void** data, int len);
        static void cuda_host_alloc_pinned(void** data, int len);
    };

} // namespace vision


#endif //VISION_CUDA_ALLOCATE_CUH
