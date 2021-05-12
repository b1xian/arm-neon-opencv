#!/usr/bin/env bash
show_help() {
    echo "Usage: $0 [option...]" >&2
    echo
    echo "   -t, --target            Set platform target 1-android-armv7a, 2-android-armv8a, 3-linux-x86_64"
    echo "   -h, --help              show help message"
    echo
}

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
    TARGET_OS="android"
    ;;
2)
    TARGET="arm64-v8a"
    TARGET_OS="android"
    ;;
3)
    TARGET="x86_64"
    TARGET_OS="linux"
    ;;
*)
    echo "Not supported target!"
    exit 1
    ;;
esac

if [[ $TARGET_OS == "android" ]]; then
  if [ "$ANDROID_NDK_HOME" = "" ]; then
    echo "ERROR: Please set ANDROID_NDK_HOME environment"
    exit
  fi
  echo "===== ANDROID_NDK_HOME=$ANDROID_NDK_HOME"
  CMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake
fi

mkdir build
cd build
mkdir ${TARGET_OS}-${TARGET}
cd ${TARGET_OS}-${TARGET}

echo "===== cmake target: ${TARGET_OS}-${TARGET}"
if [ "$TARGET_OS" = "android" ]; then
       cmake  -DANDROID_ABI=${TARGET} \
              -DTARGET_OS=${TARGET_OS} \
              -DTARGET_ARCH=${TARGET} \
              -DCMAKE_CXX_FLAGS="-std=c++11 -frtti -fexceptions" \
              -DANDROID_PLATFORM=android-23 \
              -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} \
              -DANDROID_ARM_NEON=ON \
              ../../
#  elif [ "$TARGET_OS" = "osx" ]; then
#      echo "===== cmake target: osx-x86_64"
#       cmake   -DTARGET_OS=osx \
#               -DTARGET_ARCH=x86_64 \
#               -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
#               -DUSE_EXT_MODEL=$USE_EXTERNAL_MODEL \
#               ../..
elif [ "$TARGET_OS" = "linux" ]; then
     cmake   -DTARGET_OS=${TARGET_OS} \
             -DTARGET_ARCH=${TARGET} \
             -DCMAKE_CXX_FLAGS="-std=c++11 -frtti -fexceptions" \
             -DNVIDIA_CUDA=ON          \
             ../..
fi

make
make install

cd ../../

DEMO_BIN=./build/${TARGET_OS}-${TARGET}/vacv_test

if [ "$TARGET_OS" = "android" ]; then
  adb shell "rm -r /data/local/tmp/vacv_test"
  adb push ${DEMO_BIN} /data/local/tmp/
  bin_path="/data/local/tmp/vacv_test"
  adb shell "chmod +x ${bin_path}/va_cv_ut"
  adb shell "cd ${bin_path} \
         && export LD_LIBRARY_PATH=${bin_path}:${LD_LIBRARY_PATH} \
         && ./va_cv_ut"
elif [ "$TARGET_OS" = "linux" ]; then
  cd ${DEMO_BIN}
  chmod +x va_cv_ut
  ./va_cv_ut
fi

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