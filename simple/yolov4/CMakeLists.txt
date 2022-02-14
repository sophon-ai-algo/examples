cmake_minimum_required(VERSION 2.8)
add_compile_options(-std=c++11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(TARGET_X86_PCIE "target platform is x86_pcie" ON)
option(TARGET_ARM "target platform is arm" OFF)
option(TARGET_ARM_PCIE "target platform: arm_pcie" OFF)

# User need to set this before compiling.
set(BMNNSDK2_TOP $ENV{HOME}/bmnnsdk2/bmnnsdk2-latest)

set(bmnnsdk_inc_dirs ${BMNNSDK2_TOP}/include
        ${BMNNSDK2_TOP}/include/bmruntime
        ${BMNNSDK2_TOP}/include/bmlib
        ${BMNNSDK2_TOP}/include/third_party/boost/include
        ${BMNNSDK2_TOP}/include/opencv/opencv4
        ${BMNNSDK2_TOP}/include/ffmpeg
        CACHE INTERNAL "")

if (TARGET_X86_PCIE)
    set(bmnnsdk_link_dirs ${BMNNSDK2_TOP}/lib/bmnn/pcie
            ${BMNNSDK2_TOP}/lib/opencv/x86
            ${BMNNSDK2_TOP}/lib/thirdparty/x86
            ${BMNNSDK2_TOP}/lib/opencv/x86
            ${BMNNSDK2_TOP}/lib/ffmpeg/x86
            ${BMNNSDK2_TOP}/lib/decode/x86
            CACHE INTERNAL "")

    set(bmnnsdk_link_libs bmrt bmlib bmcv
            opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio
            avfilter avformat avcodec avutil swresample swscale bmvideo bmjpuapi bmjpulite
            boost_system boost_filesystem boost_thread boost_chrono
            protobuf gflags glog pthread
            CACHE INTERNAL "")

endif ()

include_directories(${bmnnsdk_inc_dirs})
link_directories(${bmnnsdk_link_dirs})

#cpp_cv_bmcv_bmrt_postprocess
add_executable(yolov3_cpp_cv_cv_bmrt_postprocess cpp_cv_bmcv_bmrt_postprocess/main.cpp
        cpp_cv_bmcv_bmrt_postprocess/yolov3.cpp)
target_link_libraries(yolov3_cpp_cv_cv_bmrt_postprocess ${bmnnsdk_link_libs})




