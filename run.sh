#!/usr/bin/env bash
show_help() {
    echo "Usage: $0 [option...]" >&2
    echo
    echo "   -t, --target            Set platform target 1-android-armv7a, 2-android-armv8a, 3-android-x86_64, 4-android-x86"
    echo "   -h, --help              show help message"
    echo
}

TARGET="arm64-v8a"
CMAKE_TOOLCHAIN_FILE="/Users/v_guojinlong/Desktop/developer/android-ndk-r19c/build/cmake/android.toolchain.cmake"

# parse arguments
while [ $# != 0 ]
do
  case "$1" in
    -t)
        TARGET_INDEX=$2
        shift
        ;;
    --target)
        TARGET_INDEX=$2
        shift
        ;;
    -h)
        show_help
        exit 1
        ;;
    --help)
        show_help
        exit 1
        ;;
    *)
        ;;
  esac
  shift
done

case "$TARGET_INDEX" in
1)
    TARGET="armeabi-v7a"
    CMAKE_TOOLCHAIN_FILE="/Users/v_guojinlong/Desktop/developer/ndk17/android-ndk-r17/build/cmake/android.toolchain.cmake"
    ;;
2)
    TARGET="arm64-v8a"
    ;;
*)
    echo "Not supported target!"
    exit 1
    ;;
esac

mkdir build
cd build
mkdir ${TARGET}
cd ${TARGET}

cmake   -DANDROID_ABI=${TARGET} \
        -DTARGET_OS=android \
        -DTARGET_ARCH=${TARGET} \
        -DCMAKE_CXX_FLAGS="-std=c++11 -frtti -fexceptions" \
        -DANDROID_PLATFORM=android-23 \
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} \
        -DANDROID_ARM_NEON=ON \
        ../../

make
make install

cd ../../

DEMO_BIN=./build/${TARGET}/vacv_test

adb shell "rm -r /data/local/tmp/vacv_test"
adb push ${DEMO_BIN} /data/local/tmp/
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