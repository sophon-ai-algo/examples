cmake_minimum_required(VERSION 3.2)
project(inference_framework)

set(CMAKE_CXX_STANDARD 14)

option(HAVE_TRACKER "Enable tracker" ON)
option(USE_BM_FFMPEG "Use bm ffmpeg" ON)
option(USE_BM_OPENCV "Use bm opencv" ON)
option(USE_QTGUI "Whether has GUI support" OFF)

add_compile_options(-Wno-deprecated-declarations -fPIC)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
# Global Variables
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(TOP ${CMAKE_SOURCE_DIR})

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH x86)
endif()

if (USE_QTGUI)
    if (${TARGET_ARCH} STREQUAL "soc")
        set(QTDIR /opt/qt5.14-18.04-soc)
        set(Qt5Widgets_DIR /opt/qt5.14-18.04-soc/lib/cmake/Qt5Widgets)
    else()
        set(QTDIR /opt/Qt5.12.9/5.12.9/gcc_64)
        set(Qt5Widgets_DIR /opt/Qt5.12.9/5.12.9/gcc_64/lib/cmake/Qt5Widgets)
    endif()
endif()

# User need to set this before compiling.
if (DEFINED ENV{REL_TOP})
    set(BMNNSDK2_TOP $ENV{REL_TOP})
else()
    set(BMNNSDK2_TOP $ENV{HOME}/bmnnsdk2/bmnnsdk2-latest)
endif()

set(bmnnsdk_inc_dirs ${BMNNSDK2_TOP}/include
        ${BMNNSDK2_TOP}/include/bmruntime
        ${BMNNSDK2_TOP}/include/bmlib
        ${BMNNSDK2_TOP}/include/third_party/boost/include
        CACHE INTERNAL "")

set(bm_opencv_inc_dirs  ${BMNNSDK2_TOP}/include/opencv/opencv4)
set(bm_ffmpeg_inc_dirs  ${BMNNSDK2_TOP}/include/ffmpeg ${BMNNSDK2_TOP}/include/bmlib)
set(bm_ffmpeg_link_libs avfilter avformat avcodec avutil avdevice swresample swscale)
set(bm_opencv_link_libs opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio opencv_dnn)

set(bmnnsdk_link_libs bmrt bmcv bmvideo bmjpuapi bmjpulite bmlib boost_system boost_filesystem protobuf gflags glog pthread
        CACHE INTERNAL "")

if (${TARGET_ARCH} STREQUAL "x86")
    set(bm_ffmpeg_link_dirs ${BMNNSDK2_TOP}/lib/ffmpeg/x86 ${BMNNSDK2_TOP}/lib/decode/x86
            ${BMNNSDK2_TOP}/lib/bmnn/pcie
            ${BMNNSDK2_TOP}/lib/thirdparty/x86)
    set(bm_opencv_link_dirs ${BMNNSDK2_TOP}/lib/opencv/x86)
    set(bmnnsdk_link_dirs ${BMNNSDK2_TOP}/lib/bmnn/pcie)

elseif (${TARGET_ARCH} STREQUAL "arm64")
    # arm-pcie
    set(bm_ffmpeg_link_dirs ${BMNNSDK2_TOP}/lib/ffmpeg/arm_pcie ${BMNNSDK2_TOP}/lib/decode/arm_pcie
            ${BMNNSDK2_TOP}/lib/bmnn/arm_pcie
            ${BMNNSDK2_TOP}/lib/thirdparty/soc)
    set(bm_opencv_link_dirs ${BMNNSDK2_TOP}/lib/opencv/arm_pcie)
    set(bmnnsdk_link_dirs ${BMNNSDK2_TOP}/lib/bmnn/arm_pcie)

else() # soc
    set(bm_ffmpeg_link_dirs ${BMNNSDK2_TOP}/lib/ffmpeg/soc ${BMNNSDK2_TOP}/lib/decode/soc
            ${BMNNSDK2_TOP}/lib/bmnn/soc
            ${BMNNSDK2_TOP}/lib/thirdparty/soc)
    set(bm_opencv_link_dirs ${BMNNSDK2_TOP}/lib/opencv/soc)
    set(bmnnsdk_link_dirs ${BMNNSDK2_TOP}/lib/bmnn/soc)
endif ()

if (USE_BM_FFMPEG)
    set(ffmpeg_inc_dirs ${bm_ffmpeg_inc_dirs} CACHE INTERNAL "")
    set(ffmpeg_link_dirs ${bm_ffmpeg_link_dirs} CACHE INTERNAL "")
    set(ffmpeg_link_libs ${bm_ffmpeg_link_libs} CACHE INTERNAL "")
else()
    set(ffmpeg_inc_dirs /usr/local/include CACHE INTERNAL "")
    set(ffmpeg_link_dirs /usr/local/lib CACHE INTERNAL "")
    set(ffmpeg_link_libs avcodec avutil avformat swscale CACHE INTERNAL "")
endif()

if (USE_BM_OPENCV)
    set(opencv_inc_dirs ${bm_opencv_inc_dirs})
    set(opencv_link_dirs ${bm_opencv_link_dirs})
    set(opencv_link_libs ${bm_opencv_link_libs})
else()
    find_package(OpenCV REQUIRED)
    set(opencv_inc_dirs ${OpenCV_INCLUDE_DIRS})
    set(opencv_link_dirs ${OpenCV_LIBRARY_DIRS})
    set(opencv_link_libs ${OpenCV_LIBS})
endif()


message(STATUS "ffmpeg_inc = ${ffmpeg_inc_dirs}")
message(STATUS "ffmpeg_link_dirs = ${ffmpeg_link_dirs}")
message(STATUS "opencv_inc = ${opencv_inc_dirs}")
message(STATUS "opencv_link_dirs = ${opencv_link_dirs}")

set(common_inc_dirs ${TOP} ${TOP}/tracker ${TOP}/bmutility ${TOP}/bmgui ${TOP}/third_party/jsoncpp)
message(STATUS "common_inc=${common_inc_dirs}")

add_subdirectory(bmutility)
if (USE_QTGUI)
    add_subdirectory(bmgui)
    add_subdirectory(bmgui-lite)
endif()
add_subdirectory(third_party/jsoncpp)
add_subdirectory(third_party/live555)

if (USE_BM_FFMPEG)
   add_subdirectory(examples)
   if (HAVE_TRACKER)
       add_subdirectory(tracker)
       add_subdirectory(cvs10)
   endif()
endif()

add_subdirectory(ddr_reduction)

