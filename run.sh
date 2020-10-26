#!/usr/bin/env bash

cd cmake-build-debug
cmake   -DANDROID_ABI=arm64-v8a \
        -DTARGET_OS=android \
        -DTARGET_ARCH=arm64-v8a \
        -DCMAKE_CXX_FLAGS="-std=c++11 -frtti -fexceptions" \
        -DANDROID_PLATFORM=android-23 \
        -DCMAKE_TOOLCHAIN_FILE=/Users/v_guojinlong/Desktop/developer/ndk17/android-ndk-r17/build/cmake/android.toolchain.cmake \
        -DANDROID_ARM_NEON=ON \
        ..
make

bin_path="/data/local/tmp/test_neon"

adb shell "mkdir ${bin_path}"
adb shell "rm -r ${bin_path}/output"
adb shell "mkdir ${bin_path}/output"

adb push test_neon ${bin_path}
adb push ../res ${bin_path}

adb shell "cd ${bin_path} && ./test_neon"

adb pull ${bin_path}/output ../