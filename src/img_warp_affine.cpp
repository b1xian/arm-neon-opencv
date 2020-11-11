//
// Created by b1xian on 2020-10-29.
//

#include <arm_neon.h>
#include <iostream>
#include <stdlib.h>
#include <cmath>

#include <opencv2/opencv.hpp>

#include "vision/tensor.h"
#include "vision/tensor_converter.h"

#include "neon_normalize/layout_change.cpp"
#include "neon_normalize/dtype_change.cpp"
#include "neon_normalize/mean_std_dev.cpp"
#include "neon_warpaffine/warp_affine.cpp"


using namespace std;
using namespace cv;
using namespace vision;


int main() {


    Mat src_mat = imread("res/lakers.jpeg", 0);
    int h = src_mat.rows;
    int w = src_mat.cols;
    int c = src_mat.channels();

    clock_t start_time = clock();
    cv::Mat warp_mat = Mat::zeros(src_mat.rows, src_mat.cols, src_mat.type());

    Point2f srcTri[3];
    Point2f dstTri[3];
    srcTri[0] = Point2f(0, 0);
    srcTri[1] = Point2f(src_mat.cols - 1, 0);
    srcTri[2] = Point2f(0, src_mat.rows - 1);
    dstTri[0] = Point2f(src_mat.cols*0.0, src_mat.rows*0.33);
    dstTri[1] = Point2f(src_mat.cols*0.85, src_mat.rows*0.25);
    dstTri[2] = Point2f(src_mat.cols*0.15, src_mat.rows*0.7);
    cv::Mat m_mat = cv::getAffineTransform(srcTri, dstTri);

    cv::warpAffine(src_mat, warp_mat, m_mat, warp_mat.size());
    cv::imwrite("output/warp_affine_opencv.jpg", warp_mat);
    std::cout << "opencv : " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s"  << std::endl;

    float* m = (float*) m_mat.data;

    const double degree = 45;
    double angle = degree * CV_PI / 180.;
    double alpha = cos(angle);
    double beta = sin(angle);
    int iWidth = w;
    int iHeight = h;
    int iNewWidth = cvRound(iWidth * fabs(alpha) + iHeight * fabs(beta));
    int iNewHeight = cvRound(iHeight * fabs(alpha) + iWidth * fabs(beta));

    m[0] = alpha;
    m[1] = beta;
    m[2] = (1 - alpha) * iWidth / 2. - beta * iHeight / 2.;
    m[3] = -m[1];
    m[4] = m[0];
    m[5] = beta * iWidth / 2. + (1 - alpha) * iHeight / 2.;

    double D = m[0]*m[4] - m[1]*m[3];
    D = D != 0 ? 1./D : 0;
    double A11 = m[4]*D, A22 = m[0]*D;
    m[0] = A11; m[1] *= -D;
    m[3] *= -D; m[4] = A22;
    double b1 = -m[0]*m[2] - m[1]*m[5];
    double b2 = -m[3]*m[2] - m[4]*m[5];
    m[2] = b1; m[5] = b2;
    std::cout << m[0] << "  " << m[1] << "  "  << m[2] << "  "  << std::endl;
    std::cout << m[3] << "  " << m[4] << "  "  << m[5] << "  "  << std::endl;
    start_time = clock();
    cv::Mat warp_mat_neon = cv::Mat(cv::Size(w, h), src_mat.type(), cv::Scalar::all(0));
    warp_affine_naive((uint8_t*) src_mat.data, w, h,
                              (uint8_t*) warp_mat_neon.data, w, h, m);
    std::cout << "neon : " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s"  << std::endl;
    cv::imwrite("output/warp_affine_neon.jpg", warp_mat_neon);


//    start_time = clock();
//    const double degree = 45;
//    double angle = degree * CV_PI / 180.;
//    double alpha = cos(angle);
//    double beta = sin(angle);
//    int iWidth = w;
//    int iHeight = h;
//    int iNewWidth = cvRound(iWidth * fabs(alpha) + iHeight * fabs(beta));
//    int iNewHeight = cvRound(iHeight * fabs(alpha) + iWidth * fabs(beta));
//
//    m[0] = alpha;
//    m[1] = beta;
//    m[2] = (1 - alpha) * iWidth / 2. - beta * iHeight / 2.;
//    m[3] = -m[1];
//    m[4] = m[0];
//    m[5] = beta * iWidth / 2. + (1 - alpha) * iHeight / 2.;
//
//    double D = m[0]*m[4] - m[1]*m[3];
//    D = D != 0 ? 1./D : 0;
//    double A11 = m[4]*D, A22 = m[0]*D;
//    m[0] = A11; m[1] *= -D;
//    m[3] *= -D; m[4] = A22;
//    double b1 = -m[0]*m[2] - m[1]*m[5];
//    double b2 = -m[3]*m[2] - m[4]*m[5];
//    m[2] = b1; m[5] = b2;
//
//    cv::Mat matDst1 = cv::Mat(cv::Size(iNewWidth, iNewHeight), src_mat.type(), cv::Scalar::all(0));
//
//    for (int y=0; y<iNewHeight; ++y)
//    {
//        for (int x=0; x<iNewWidth; ++x)
//        {
//            //int tmpx = cvFloor(m[0] * x + m[1] * y + m[2]);
//            //int tmpy = cvFloor(m[3] * x + m[4] * y + m[5]);
//            float fx = m[0] * x + m[1] * y + m[2];
//            float fy = m[3] * x + m[4] * y + m[5];
//
//            int sy  = cvFloor(fy);
//            fy -= sy;
//            //sy = std::min(sy, iHeight-2);
//            //sy = std::max(0, sy);
//            if (sy < 0 || sy >= iHeight) continue;
//
//            short cbufy[2];
//            cbufy[0] = cv::saturate_cast<short>((1.f - fy) * 2048);
//            cbufy[1] = 2048 - cbufy[0];
//
//            int sx = cvFloor(fx);
//            fx -= sx;
//            //if (sx < 0) {
//            //	fx = 0, sx = 0;
//            //}
//            //if (sx >= iWidth - 1) {
//            //	fx = 0, sx = iWidth - 2;
//            //}
//            if (sx < 0 || sx >= iWidth) continue;
//
//            short cbufx[2];
//            cbufx[0] = cv::saturate_cast<short>((1.f - fx) * 2048);
//            cbufx[1] = 2048 - cbufx[0];
//
//            for (int k=0; k<src_mat.channels(); ++k)
//            {
//                if (sy == iHeight - 1 || sx == iWidth - 1) {
//                    continue;
//                } else {
//                    matDst1.at<cv::Vec3b>(y, x)[k] = (src_mat.at<cv::Vec3b>(sy, sx)[k] * cbufx[0] * cbufy[0] +
//                                                      src_mat.at<cv::Vec3b>(sy+1, sx)[k] * cbufx[0] * cbufy[1] +
//                                                      src_mat.at<cv::Vec3b>(sy, sx+1)[k] * cbufx[1] * cbufy[0] +
//                                                      src_mat.at<cv::Vec3b>(sy+1, sx+1)[k] * cbufx[1] * cbufy[1]) >> 22;
//                }
//            }
//        }
//    }
//    cv::imwrite("output/warp_affine_naive.jpg", matDst1);
//    std::cout << "naive : " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "s"  << std::endl;
    return 0;
}


#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

//全局变量
String src_windowName = "原图像";
String warp_windowName = "仿射变换";
String warp_rotate_windowName = "仿射旋转变换";
String rotate_windowName = "图像旋转";

int main1()
{
    Point2f srcTri[3];
    Point2f dstTri[3];

    Mat rot_mat(2, 3, CV_32FC1);
    Mat warp_mat(2, 3, CV_32FC1);
    Mat srcImage, warp_dstImage, warp_rotate_dstImage, rotate_dstImage;

    //加载图像
    srcImage = imread("res/lakers.jpeg");

    //创建仿射变换目标图像与原图像尺寸类型相同
    warp_dstImage = Mat::zeros(srcImage.rows, srcImage.cols, srcImage.type());

    //设置三个点来计算仿射变换
    srcTri[0] = Point2f(0, 0);
    srcTri[1] = Point2f(srcImage.cols - 1, 0);
    srcTri[2] = Point2f(0, srcImage.rows - 1);

    dstTri[0] = Point2f(srcImage.cols*0.0, srcImage.rows*0.33);
    dstTri[1] = Point2f(srcImage.cols*0.85, srcImage.rows*0.25);
    dstTri[2] = Point2f(srcImage.cols*0.15, srcImage.rows*0.7);

    //计算仿射变换矩阵
    warp_mat = getAffineTransform(srcTri, dstTri);

    //对加载图形进行仿射变换操作
    warpAffine(srcImage, warp_dstImage, warp_mat, warp_dstImage.size());

    //计算图像中点顺时针旋转50度，缩放因子为0.6的旋转矩阵
    Point center = Point(warp_dstImage.cols/2, warp_dstImage.rows/2);
    double angle = -50.0;
    double scale = 0.6;

    //计算旋转矩阵
    rot_mat = getRotationMatrix2D(center, angle, scale);

    //旋转已扭曲图像
    warpAffine(warp_dstImage, warp_rotate_dstImage, rot_mat, warp_dstImage.size());

    //将原图像旋转
    warpAffine(srcImage, rotate_dstImage, rot_mat, srcImage.size());

    //显示变换结果
    namedWindow(src_windowName, WINDOW_AUTOSIZE);
    imshow(src_windowName, srcImage);

    namedWindow(warp_windowName, WINDOW_AUTOSIZE);
    imshow(warp_windowName, warp_dstImage);

    namedWindow(warp_rotate_windowName, WINDOW_AUTOSIZE);
    imshow(warp_rotate_windowName, warp_rotate_dstImage);

    namedWindow(rotate_windowName, WINDOW_AUTOSIZE);
    imshow(rotate_windowName, rotate_dstImage);

    waitKey(0);

    return 0;
}