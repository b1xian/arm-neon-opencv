cmake_minimum_required( VERSION 2.8 )
project(vacv)
set(lib_name vacv)

add_definitions(" -g")
set(CMAKE_BUILD_TYPE Release)

if (ANDROID_ARM_NEON)
    add_definitions(-DUSE_NEON)
endif()

#CUDA
if (NVIDIA_CUDA)
    add_definitions(-DUSE_CUDA)
    find_package(CUDA REQUIRED)
    enable_language(CUDA)
    message("USE CUDA...")
endif()

set(LIB_DIR thirdparty)

# opencv
set(OPENCV_VER 2.4.13.4)
set(OPENCV_DIR ${LIB_DIR}/opencv_${OPENCV_VER})
set(OPENCV_INCLUDE_DIR ${OPENCV_DIR}/${TARGET_OS}-${TARGET_ARCH}/include/)
set(OPENCV_LINK_DIR ${OPENCV_DIR}/${TARGET_OS}-${TARGET_ARCH}/lib/)
set(OPENCV_3RDPARTY_LINK_DIR ${OPENCV_DIR}/${TARGET_OS}-${TARGET_ARCH}/3rdparty/)
message(STATUS "[Dependency] opencv lib_dir=" ${OPENCV_LINK_DIR})
set(OPENCV_LIB opencv_calib3d opencv_highgui opencv_imgproc opencv_core libtiff libpng libjpeg libjasper IlmImf)


aux_source_directory(src/common vision_common)
aux_source_directory(src/cv vision_cv)
aux_source_directory(src/util vision_util)

if (NVIDIA_CUDA)
    cuda_add_library(${lib_name} SHARED ${vision_common} ${vision_cv} ${vision_util})
else()
    add_library(${lib_name} SHARED ${vision_common} ${vision_cv} ${vision_util})
endif ()

target_link_directories(${lib_name} PUBLIC ${OPENCV_LINK_DIR} ${OPENCV_3RDPARTY_LINK_DIR})
target_link_libraries(${lib_name} PUBLIC ${OPENCV_LIB})
target_include_directories(${lib_name} PUBLIC ${OPENCV_INCLUDE_DIR})


if (${TARGET_OS} MATCHES android)
    target_compile_options(${lib_name} PRIVATE -std=c++14 -O3 -frtti -fexceptions -march=armv8.2-a+fp16)
    target_link_libraries(${lib_name} PUBLIC log z)
else()
    target_compile_options(${lib_name} PRIVATE -std=c++14 -O3)
endif ()

add_subdirectory(src/test)