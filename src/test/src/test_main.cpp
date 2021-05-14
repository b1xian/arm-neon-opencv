#include <iostream>

#include "impl/test_change_dtype.h"
#include "impl/test_change_layout.h"
#include "impl/test_crop.h"
#include "impl/test_cvt_color.h"
#include "impl/test_normalize.h"
#include "impl/test_resize.h"
#include "impl/test_warp_affine.h"
#include "profile/cv_profile.h"

using namespace vacv;
using namespace std;

int main(int argc, char** argv) {

    std::vector<CvProfile::SpeedResult> speed_result;
    std::vector<CvProfile::OutputResult> output_result;
    CvProfile::TestFuncList func_list {
            {TestCrop::test_crop_hwc_5x5, "test_crop_hwc_5x5"},
//            {TestCrop::test_crop_hwc_5x5_FP32, "test_crop_hwc_5x5_FP32"},
            {TestCrop::test_crop_hwc_320x180, "test_crop_hwc_320x180"},
            {TestCrop::test_crop_hwc_640x360, "test_crop_hwc_640x360"},
            {TestCrop::test_crop_hwc_1280x720, "test_crop_hwc_1280x720"},
//            {TestCrop::test_crop_hwc_1920x1080, "test_crop_hwc_1920x1080"},
//            {TestCrop::test_crop_chw_320x180, "test_crop_chw_320x180"},
//            {TestCrop::test_crop_chw_320x180_FP32, "test_crop_chw_320x180_FP32"},
//            {TestCrop::test_crop_chw_640x360, "test_crop_chw_640x360"},
//            {TestCrop::test_crop_chw_5x5, "test_crop_chw_5x5"},
//            {TestCrop::test_crop_chw_5x5_FP32, "test_crop_chw_5x5_FP32"},
//
//            {TestResize::test_resize_bilinear_hwc_u8_320x180, "test_resize_bilinear_hwc_u8_320x180"},
//            {TestResize::test_resize_bilinear_chw_u8_320x180, "test_resize_bilinear_chw_u8_320x180"},
//            {TestResize::test_resize_bilinear_hwc_fp32_320x180, "test_resize_bilinear_hwc_fp32_320x180"},
//            {TestResize::test_resize_bilinear_chw_fp32_320x180, "test_resize_bilinear_chw_fp32_320x180"},
//            {TestResize::test_resize_cubic_hwc_fp32_320x180, "test_resize_cubic_hwc_fp32_320x180"},
//            {TestResize::test_resize_cubic_chw_fp32_320x180, "test_resize_cubic_chw_fp32_320x180"},
//
//            {TestChangeDtype::test_change_dtype_u8_to_fp32_176x144, "test_change_dtype_u8_to_fp32_176x144"},
//            {TestChangeDtype::test_change_dtype_fp32_to_u8_176x144, "test_change_dtype_fp32_to_u8_176x144"},
//
//            {TestChangeLayout::test_change_layout_hwc_to_chw_u8_176x144, "test_change_layout_hwc_to_chw_u8_176x144"},
//            {TestChangeLayout::test_change_layout_hwc_to_chw_fp32_176x144, "test_change_layout_hwc_to_chw_fp32_176x144"},
//
//            {TestNormalize::test_normalize_hwc_176x144, "test_normalize_hwc_176x144"},
//            {TestNormalize::test_normalize_chw_176x144, "test_normalize_chw_176x144"},
//            {TestNormalize::test_normalize_hwc_284x214, "test_normalize_hwc_284x214"},
//
//            {TestWarpAffine::test_warp_affine_hwc_u8, "test_warp_affine_hwc_u8"},
//            {TestWarpAffine::test_warp_affine_hwc_fp32, "test_warp_affine_hwc_fp32"},
//            {TestWarpAffine::test_get_rotation_matrix_hwc_u8, "test_get_rotation_matrix_hwc_u8"},
//            {TestWarpAffine::test_get_rotation_matrix_hwc_fp32, "test_get_rotation_matrix_hwc_fp32"},
//
//            {TestWarpAffine::test_warp_affine_chw_u8, "test_warp_affine_chw_u8"},
//            {TestWarpAffine::test_warp_affine_chw_fp32, "test_warp_affine_chw_fp32"},
//            {TestWarpAffine::test_get_rotation_matrix_chw_u8, "test_get_rotation_matrix_chw_u8"},
//
//            {TestCvtColor::test_nv21_to_bgr_176x144, "test_nv21_to_bgr_176x144"},
//            {TestCvtColor::test_nv21_to_bgr_640x360, "test_nv21_to_bgr_640x360"},
//            {TestCvtColor::test_nv21_to_bgr_1280x720, "test_nv21_to_bgr_1280x720"},
//            {TestCvtColor::test_nv21_to_bgr_1920x1080, "test_nv21_to_bgr_1920x1080"},
//            {TestCvtColor::test_nv21_to_bgr_2560x1440, "test_nv21_to_bgr_2560x1440"},
    };

    CvProfile::profile(func_list, nullptr, nullptr, speed_result, output_result);
    return 0;
}