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
//extern "C" __global__ void kernel_crop_grey(unsigned char *dst ) {
//    // map from threadIdx/BlockIdx to pixel position(on dst)
//    int dst_x = threadIdx.x;
//    int dst_y = blockIdx.x;
//
//    if (dst_x <= rect[2] && dst_y <= rect[3]){
//        int dst_ofs = dst_y * rect[2] + dst_x;
//        int src_x = dst_x;
//        int src_y = dst_y;
//        dst[dst_ofs] = tex2D(srcTexture2D, src_x, src_y);
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
//    // 每個block處理一行數據，一共height個block
////    const int threadsPerBlock = crop_width;
////    const int blocksPerGrid = crop_height;
////    dim3    grids(blocksPerGrid);
////    dim3    threads(threadsPerBlock);
//
//
//    int dst_size = crop_width * crop_height;
//    int src_size = src_width * src_height;
//    unsigned char *dev_src, *dev_dst;
//    cudaMalloc( (void**)&dev_dst, dst_size * sizeof(unsigned char) ) ;
//    cudaMalloc( (void**)&dev_src, src_size * sizeof(unsigned char) ) ;
//    cudaMemcpy( dev_src, src, src_size * sizeof(unsigned char), cudaMemcpyHostToDevice );
//
//    cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned char>();
//    cudaBindTexture2D( NULL, srcTexture2D, dev_src, desc, src_width, src_height,
//                       sizeof(unsigned char) * src_width );
//
//    kernel_crop_grey<<<crop_height,crop_width>>>( dev_dst );
//
//    cudaMemcpy(dst, dev_dst, dst_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
//
//    cudaUnbindTexture( srcTexture2D );
//    cudaFree(dev_dst);
//    cudaFree(dev_src);
//}
//
//}