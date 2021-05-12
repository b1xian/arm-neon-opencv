//#include "crop_cuda.h"
//
//#include <stdio.h>
//#include <cstdlib>
//#include <math.h>
//#include <iostream>
//
//#include "../common/macro.h"
//
//#define PIXEL_PER_THREAD 128
//
//namespace va_cv {
//
//texture<unsigned char, 2>  tex_src;
//__constant__ int rect[5];
//
//
//__global__ void kernel_crop_grey(unsigned char *dst ) {
//    // map from threadIdx/BlockIdx to pixel position(on dst)
//    int tid = threadIdx.x + blockIdx.x * blockDim.x;
//    while (tid < rect[2] * rect[3]) {
//        int dst_x = tid % rect[2];
//        int dst_y = tid / rect[2];
//        dst[tid] = tex2D(tex_src, dst_x + rect[0], dst_y + rect[1]);
//
//        tid += blockDim.x * gridDim.x;
//    }
//}
//
//void CropCuda::crop_cuda_grey_int8(const unsigned char *src, int src_width, int src_height,
//                              unsigned char *dst,
//                              int crop_left, int crop_top, int crop_width, int crop_height) {
//    // crop rect, use const value
//    int *rect_vec = new int[5]{crop_left, crop_top, crop_width, crop_height, src_width};
//    cudaMemcpyToSymbol( rect, rect_vec, sizeof(int) * 5);
//
//
//    int dst_size = crop_width * crop_height;
//    int src_size = src_width * src_height;
//    // 使用cudaHostAlloc代替cudaMalloc
//    unsigned char *dev_src, *dev_dst;
//    cudaHostAlloc( (void**)&dev_dst, dst_size * sizeof(unsigned char), cudaHostAllocDefault ) ;
//    cudaHostAlloc( (void**)&dev_src, src_size * sizeof(unsigned char), cudaHostAllocDefault ) ;
//    cudaMemcpy( dev_src, src, src_size * sizeof(unsigned char), cudaMemcpyHostToDevice );
//
//    // src使用紋理內存
//    cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned char>();
//    int err = cudaBindTexture2D( NULL, tex_src, dev_src, desc, src_width, src_height,
//                       sizeof(unsigned char) * src_width );
//    if (err != cudaSuccess) {
//        printf("bind failed!!! %d\n", err);
//    }
//
//    // 設備函數
//    dim3    blocks((dst_size + PIXEL_PER_THREAD - 1) / PIXEL_PER_THREAD);
//    dim3    threads(PIXEL_PER_THREAD);
//    kernel_crop_grey<<<blocks,threads>>>( dev_dst );
//
//    // 讀取dst內存
//    cudaMemcpy(dst, dev_dst, dst_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
//
//    // 回收內存
//    cudaFreeHost( dev_dst );
//    cudaFreeHost( dev_src );
//    cudaUnbindTexture( tex_src );
//
//    delete[] rect_vec;
//}
//
//}