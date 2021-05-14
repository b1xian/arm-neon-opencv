//#include "crop_cuda.h"
//
//#include <stdio.h>
//#include <cstdlib>
//#include <math.h>
//#include <iostream>
//
//#include "../common/macro.h"
//
//#define PIXEL_PER_THREAD 256
//
//namespace va_cv {
//
//texture<unsigned char, 2>  srcTexture2D;
//__constant__ int rect[7];
//
//
//extern "C" __global__ void kernel_crop_chw( const unsigned char *src, unsigned char *dst, int* dev_rect ) {
//    // map from threadIdx/BlockIdx to pixel position(on dst)
//    int tid = threadIdx.x + blockIdx.x * blockDim.x;
//    while (tid < dev_rect[2] * dev_rect[3]) {
//        int dst_x = tid % dev_rect[2];
//        int dst_y = tid / dev_rect[2];
//        int src_ofs = dev_rect[4] * dst_y + dst_x;
//        for (int i = 0; i < dev_rect[6]; ++i) {
//            int src_channel_ofs = i * dev_rect[4] * dev_rect[5];
//            int dst_channel_ofs = i * dev_rect[2] * dev_rect[3];
//            dst[dst_channel_ofs + tid] = src[src_channel_ofs + src_ofs];
//        }
//
//        tid += blockDim.x * gridDim.x;
//    }
//}
//
//
//extern "C" __global__ void kernel_crop_rgb_hwc_int8( const unsigned char *src, unsigned char *dst, int *dev_rect ) {
//    // map from threadIdx/BlockIdx to pixel position(on dst)
//    int tid = threadIdx.x + blockIdx.x * blockDim.x;
//    while (tid < dev_rect[2] * dev_rect[3]) {
//        int dst_x = tid % dev_rect[2];
//        int dst_y = tid / dev_rect[2];
//        int src_ofs_r = (dev_rect[4] * dst_y + dst_x) * 3;
//
//        int dst_ofs_r = tid * 3;
//        int dst_ofs_g = dst_ofs_r + 1;
//        int dst_ofs_b = dst_ofs_r + 2;
//        int src_ofs_g = src_ofs_r + 1;
//        int src_ofs_b = src_ofs_r + 2;
//        dst[dst_ofs_r] = src[src_ofs_r];
//        dst[dst_ofs_g] = src[src_ofs_g];
//        dst[dst_ofs_b] = src[src_ofs_b];
//        tid += blockDim.x * gridDim.x;
//    }
//}
//
//void CropCuda::crop_cuda_chw_int8(const unsigned char *src, int src_width, int src_height, int src_channel,
//                              unsigned char *dst,
//                              int crop_left, int crop_top, int crop_width, int crop_height) {
//    // crop rect, use const value
//    int rect_vec[7] = {crop_left, crop_top, crop_width, crop_height, src_width, src_height, src_channel};
//    int *dev_rect;
//    cudaMalloc( (void**)&dev_rect, 7 * sizeof(int) ) ;
//    cudaMemcpy( dev_rect, rect_vec, 7 * sizeof(int), cudaMemcpyHostToDevice );
//
//
//
//    int dst_size = crop_width * crop_height * src_channel;
//    int src_size = src_width * src_height * src_channel;
//    unsigned char *dev_src, *dev_dst;
//    cudaMalloc( (void**)&dev_src, src_size * sizeof(unsigned char) ) ;
//    cudaMalloc( (void**)&dev_dst, dst_size * sizeof(unsigned char) ) ;
//    cudaMemcpy( dev_src, src, src_size * sizeof(unsigned char), cudaMemcpyHostToDevice );
//
//    dim3    blocks((dst_size + PIXEL_PER_THREAD - 1) / PIXEL_PER_THREAD);
//    dim3    threads(PIXEL_PER_THREAD);
//    kernel_crop_chw<<<blocks,threads>>>( dev_src, dev_dst, dev_rect );
//
//    cudaMemcpy(dst, dev_dst, dst_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
//
//    cudaFree(dev_src);
//    cudaFree(dev_dst);
//}
//
//
//void CropCuda::crop_cuda_rgb_hwc_int8(const unsigned char *src, int src_width, int src_height,
//                                   unsigned char *dst,
//                                   int crop_left, int crop_top, int crop_width, int crop_height) {
//
//    cudaEvent_t     start, stop;
//    float elapsed_time;
//    cudaEventCreate( &start );
//    cudaEventCreate( &stop );
//
//    // crop rect, use const value
//    int rect_vec[5] = {crop_left, crop_top, crop_width, crop_height, src_width};
//
////    cudaEventRecord( start, 0 );
//    int dst_size = crop_width * crop_height * 3;
//    int src_size = src_width * src_height * 3;
//    unsigned char *dev_src, *dev_dst;
//    int *dev_rect;
//    cudaMalloc((void**)&dev_rect, sizeof(int) * 5);
//    cudaMalloc( (void**)&dev_src, src_size * sizeof(unsigned char) ) ;
//    cudaMalloc( (void**)&dev_dst, dst_size * sizeof(unsigned char) ) ;
////    cudaEventRecord( stop, 0 );
////    cudaEventSynchronize( stop );
////    cudaEventElapsedTime( &elapsed_time, start, stop );
////    printf("1 cudaMalloc cost %f ms\n", elapsed_time);
//
//
////    cudaEventRecord( start, 0 );
//    cudaMemcpy( dev_rect, rect_vec, sizeof(int) * 5, cudaMemcpyHostToDevice );
//    cudaMemcpy( dev_src, src, src_size * sizeof(unsigned char), cudaMemcpyHostToDevice );
////    cudaEventRecord( stop, 0 );
////    cudaEventSynchronize( stop );
////    cudaEventElapsedTime( &elapsed_time, start, stop );
////    printf("2 cudaMemcpy to device cost %f ms\n", elapsed_time);
//
//    dim3    blocks((dst_size + PIXEL_PER_THREAD - 1) / PIXEL_PER_THREAD);
//    dim3    threads(PIXEL_PER_THREAD);
////    cudaEventRecord( start, 0 );
//    kernel_crop_rgb_hwc_int8<<<blocks,threads>>>( dev_src, dev_dst, dev_rect );
////    cudaEventRecord( stop, 0 );
////    cudaEventSynchronize( stop );
////    cudaEventElapsedTime( &elapsed_time, start, stop );
////    printf("3 kernel_crop_rgb_hwc_int8 cost %f ms\n", elapsed_time);
//
////    cudaEventRecord( start, 0 );
//    cudaMemcpy(dst, dev_dst, dst_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
////    cudaEventRecord( stop, 0 );
////    cudaEventSynchronize( stop );
////    cudaEventElapsedTime( &elapsed_time, start, stop );
////    printf("4 cudaMemcpy to host cost %f ms\n", elapsed_time);
//
////    cudaEventRecord( start, 0 );
//    cudaFree(dev_src);
//    cudaFree(dev_dst);
//    cudaFree(dev_rect);
////    cudaEventRecord( stop, 0 );
////    cudaEventSynchronize( stop );
////    cudaEventElapsedTime( &elapsed_time, start, stop );
////    printf("5 free memory cost %f ms\n", elapsed_time);
//    cudaEventDestroy( start );
//    cudaEventDestroy( stop );
//}
//
//}