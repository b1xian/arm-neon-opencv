//
// Created by baiduiov on 2021/5/13.
//
#include "va_cuda_allocator.cuh"

namespace vision {

    void VaCudaAllocator::cuda_malloc(void** data, int len) {
        cudaMalloc(data, len);
    }

    void VaCudaAllocator::cuda_memcpy(void *dst, void *src, int len, int copy_kind) {
        return cudaMemcpy(dst, src, len, copy_kind);
    }

    void VaCudaAllocator::cuda_free(void* data) {
        cudaFree(data);
    }

    void VaCudaAllocator::cuda_host_alloc(void** data, int len, int flags) {
        cudaHostAlloc(data, len, flags);
    }

    void VaCudaAllocator::cuda_free_host(void* data) {
        cudaFreeHost(data);
    }

    void VaCudaAllocator::cuda_host_alloc_mapped(void** data, int len) {
        cuda_host_alloc(data, len, cudaHostAllocWriteCombined | cudaHostAllocMapped);
    }

    void VaCudaAllocator::cuda_host_alloc_pinned(void** data, int len) {
        cuda_host_alloc(data, len, cudaHostAllocDefault);
    }
}