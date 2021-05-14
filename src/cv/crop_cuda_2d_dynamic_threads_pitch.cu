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
//__global__ void kernel_crop_chw(unsigned char *dst, size_t pitch_dst) {
//    // map from threadIdx/BlockIdx to pixel position(on dst)
//
//    int tid = threadIdx.x + blockIdx.x * blockDim.x;
//    while (tid < pitch_dst * rect[3]) {
//        int dst_x = tid % pitch_dst;
//        int dst_y = tid / pitch_dst;
//        dst[tid] = tex2D(tex_src, dst_x + rect[0], dst_y + rect[1]);
//
//        tid += blockDim.x * gridDim.x;
//    }
//}
//
//void CropCuda::crop_cuda_chw_int8(const unsigned char *src, int src_width, int src_height, int src_channel,
//                              unsigned char *dst,
//                              int crop_left, int crop_top, int crop_width, int crop_height) {
//    // crop rect, use const value
//    int *rect_vec = new int[5]{crop_left, crop_top, crop_width, crop_height, src_width};
//    cudaMemcpyToSymbol( rect, rect_vec, sizeof(int) * 5);
//
//    int dst_size = crop_width * crop_height;
//    int src_size = src_width * src_height;
//    // dst使用cuda malloc
//    unsigned char *dev_src, *dev_dst;
//    size_t pitch_dst = 0;
//    // pitch的訪問效率更高，但拷貝效率不受影響。。。
//    cudaMallocPitch((void**)&dev_dst, &pitch_dst, crop_width * sizeof(unsigned char), crop_height);
//    // src cudaMallocPitch
//    size_t pitch_src = 0;
//    cudaMallocPitch((void**)&dev_src, &pitch_src, src_width * sizeof(unsigned char), src_height);
//    cudaMemcpy2D(dev_src, pitch_src, src, src_width * sizeof(unsigned char),
//                 src_width * sizeof(unsigned char), src_height, cudaMemcpyHostToDevice);
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
//    kernel_crop_chw<<<blocks,threads>>>( dev_dst, pitch_dst );
//
//    // 讀取dst內存
//    cudaMemcpy2D(dst, crop_width * sizeof(unsigned char), dev_dst, pitch_dst,
//                 crop_width * sizeof(unsigned char), crop_height, cudaMemcpyDeviceToHost);
//    // 回收內存
//    cudaFree(dev_dst);
//    cudaFree(dev_src);
//    cudaUnbindTexture( tex_src );
//
//    delete[] rect_vec;
//}
//
//}