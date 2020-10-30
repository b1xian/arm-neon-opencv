cmake_minimum_required( VERSION 2.8 )
project(test_neon)
set(demo_name test_neon)

add_definitions("-Wall -g")
set(CMAKE_BUILD_TYPE Release)

set(LIB_DIR thirdparty)


set(OPENCV_VER 2.4.13.4)
set(OPENCV_DIR ${LIB_DIR}/opencv_${OPENCV_VER})
set(OPENCV_INCLUDE_DIR ${OPENCV_DIR}/${TARGET_OS}-${TARGET_ARCH}/include/)
set(OPENCV_LINK_DIR ${OPENCV_DIR}/${TARGET_OS}-${TARGET_ARCH}/lib/)
set(OPENCV_3RDPARTY_LINK_DIR ${OPENCV_DIR}/${TARGET_OS}-${TARGET_ARCH}/3rdparty/)
message(STATUS "[Dependency] opencv lib_dir=" ${OPENCV_LINK_DIR})
set(OPENCV_LIB opencv_calib3d opencv_highgui opencv_imgproc opencv_core libtiff libpng libjpeg libjasper IlmImf)


aux_source_directory(src/vision vision_srcs)
aux_source_directory(src/paddle-resize paddle_srcs)
aux_source_directory(src/neon_normalize normal_srcs)
add_executable(${demo_name} src/img_resize_cubic.cpp ${vision_srcs} ${paddle_srcs} ${normal_srcs})
target_compile_options(${demo_name} PRIVATE -std=c++11 -O3 -frtti -fexceptions -march=armv8.2-a+fp16)
target_link_directories(${demo_name} PUBLIC ${OPENCV_LINK_DIR} ${OPENCV_3RDPARTY_LINK_DIR})
target_link_libraries(${demo_name} PUBLIC log z ${OPENCV_LIB})
target_include_directories(${demo_name} PUBLIC ${OPENCV_INCLUDE_DIR})
