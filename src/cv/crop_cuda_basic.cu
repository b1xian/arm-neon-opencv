//#include "crop_cuda.h"
//
//#include <stdio.h>
//#include <cstdlib>
//#include <math.h>
//#include <iostream>
//
//#include "../common/macro.h"
//
//
//namespace va_cv {
//
//texture<unsigned char, 2>  srcTexture2D;
//__constant__ int rect[4];
//
//
//extern "C" __global__ void kernel_crop_grey( const unsigned char *src, unsigned char *dst ) {
//    // map from threadIdx/BlockIdx to pixel position(on dst)
//    int dst_x = threadIdx.x;
//    int dst_y = blockIdx.x;
//
//    if (dst_x <= rect[2] && dst_y <= rect[3]){
//        int dst_ofs = dst_y * rect[2] + dst_x;
//        int src_ofs = 1280 * dst_y + dst_x;
//
//        dst[dst_ofs] = src[src_ofs];
//    }
//}
//
//
//void CropCuda::crop_cuda_grey_int8(const unsigned char *src, int src_width, int src_height,
//                              unsigned char *dst,
//                              int crop_left, int crop_top, int crop_width, int crop_height) {
//    // crop rect, use const value
//    int rect_vec[4] = {crop_left, crop_top, crop_width, crop_height};
//    cudaMemcpyToSymbol( rect, rect_vec, sizeof(int) * 4);
//
//
//    int dst_size = crop_width * crop_height;
//    int src_size = src_width * src_height;
//    unsigned char *dev_src, *dev_dst;
//    cudaMalloc( (void**)&dev_src, src_size * sizeof(unsigned char) ) ;
//    cudaMalloc( (void**)&dev_dst, dst_size * sizeof(unsigned char) ) ;
//    cudaMemcpy( dev_src, src, src_size * sizeof(unsigned char), cudaMemcpyHostToDevice );
//
//    kernel_crop_grey<<<crop_height,crop_width>>>( dev_src, dev_dst );
//
//    cudaMemcpy(dst, dev_dst, dst_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
//
//    cudaFree(dev_src);
//    cudaFree(dev_dst);
//}
//
//}