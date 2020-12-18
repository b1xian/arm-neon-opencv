#!/usr/bin/env bash

mkdir build
cd build

cmake   -DANDROID_ABI=arm64-v8a \
        -DTARGET_OS=android \
        -DTARGET_ARCH=arm64-v8a \
        -DCMAKE_CXX_FLAGS="-std=c++11 -frtti -fexceptions" \
        -DANDROID_PLATFORM=android-23 \
        -DCMAKE_TOOLCHAIN_FILE=/Users/v_guojinlong/Desktop/developer/android-ndk-r19c/build/cmake/android.toolchain.cmake \
        -DANDROID_ARM_NEON=ON \
        ..
make
make install

cd ..

DEMO_BIN=./build/vacv_test

adb shell "rm -r /data/local/tmp/vacv_test"
adb push $DEMO_BIN /data/local/tmp/
bin_path="/data/local/tmp/vacv_test"
adb shell "chmod +x ${bin_path}/va_cv_ut"
adb shell "cd ${bin_path} \
       && export LD_LIBRARY_PATH=${bin_path}:${LD_LIBRARY_PATH} \
       && ./va_cv_ut"

#bin_path="/data/local/tmp/test_neon"
#
#adb shell "mkdir ${bin_path}"
#adb shell "rm -r ${bin_path}/output"
#adb shell "mkdir ${bin_path}/output"
#
#adb push test_neon ${bin_path}
#adb push ../res ${bin_path}
#
#adb shell "cd ${bin_path} && ./test_neon"
#
#adb pull ${bin_path}/output ../