cmake_minimum_required(VERSION 3.2)

set(CMAKE_CXX_STANDARD 14)

add_compile_options(-Wno-deprecated-declarations)
# User need to set this before compiling.
if (DEFINED ENV{REL_TOP})
    set(BMNNSDK2_TOP $ENV{REL_TOP})
else()
    set(BMNNSDK2_TOP $ENV{HOME}/bmnnsdk2/bmnnsdk2-latest)
endif()

if (NOT DEFINED PRODUCTFORM)
    set(PRODUCTFORM x86)
endif()

set(ffmpeg_inc_dirs  ${BMNNSDK2_TOP}/include/ffmpeg 
                     ${BMNNSDK2_TOP}/include/bmlib)
set(ffmpeg_link_dirs ${BMNNSDK2_TOP}/lib/ffmpeg/${PRODUCTFORM}
                    ${BMNNSDK2_TOP}/lib/decode/${PRODUCTFORM}
                    )
if (${PRODUCTFORM} STREQUAL "x86")
   list(APPEND ffmpeg_link_dirs ${BMNNSDK2_TOP}/lib/bmnn/pcie)
elseif (${PRODUCTFORM} STREQUAL "soc")
   list(APPEND ffmpeg_link_dirs ${BMNNSDK2_TOP}/lib/bmnn/soc)
elseif (${PRODUCTFORM} STREQUAL "arm_pcie")
   list(APPEND ffmpeg_link_dirs ${BMNNSDK2_TOP}/lib/bmnn/arm_pcie)
elseif (${PRODUCTFORM} STREQUAL "mips64")
    list(APPEND ffmpeg_link_dirs ${BMNNSDK2_TOP}/lib/bmnn/mips64)
elseif (${PRODUCTFORM} STREQUAL "sw64")
    list(APPEND ffmpeg_link_dirs ${BMNNSDK2_TOP}/lib/bmnn/sw64)
elseif (${PRODUCTFORM} STREQUAL "loongarch64")
    list(APPEND ffmpeg_link_dirs ${BMNNSDK2_TOP}/lib/bmnn/loongarch64)
endif()

set(ffmpeg_link_libs avfilter avformat 
                     avcodec avutil swresample swscale 
                     bmvideo bmjpuapi bmjpulite
                     bmlib 
                     )

message(STATUS "[jpeg_demos: inc = ${bm_ffmpeg_inc_dirs}")

include_directories(${ffmpeg_inc_dirs})
link_directories(${ffmpeg_link_dirs})

add_executable(simple_ffmpeg_jpeg_enc_from_file simple_ffmpeg_jpeg_enc_from_file.cpp)
target_link_libraries(simple_ffmpeg_jpeg_enc_from_file ${ffmpeg_link_libs} avdevice)

add_executable(simple_ffmpeg_video_enc simple_ffmpeg_video_enc.cpp)
target_link_libraries(simple_ffmpeg_video_enc ${ffmpeg_link_libs} avdevice)

add_executable(simple_ffmpeg_video_jpeg_dec simple_ffmpeg_video_jpeg_dec.cpp)
target_link_libraries(simple_ffmpeg_video_jpeg_dec ${ffmpeg_link_libs} avdevice)
