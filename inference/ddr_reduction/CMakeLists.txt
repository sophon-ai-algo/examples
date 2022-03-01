cmake_minimum_required(VERSION 3.2)
project(ddr_reduction_demo)
add_compile_options(-Wno-deprecated-declarations)

option(USE_BM_OPENCV "use bm_opencv" OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(${common_inc_dirs} ${bmnnsdk_inc_dirs} ${ffmpeg_inc_dirs} ${opencv_inc_dirs})
link_directories(${ffmpeg_link_dirs} ${bmnnsdk_link_dirs} ${opencv_link_dirs})


message(STATUS "opencv_inc_dirs=${opencv_inc_dirs}")
message(STATUS "opencv_link_dirs=${opencv_link_dirs}")
message(STATUS "total libs:${opencv_link_libs} ${ffmpeg_link_libs} ${bmnnsdk_link_libs}")

# app
add_library(ddr_reduction ddr_reduction.cpp)

add_executable(dr_demo ddr_reduction_unittest.cpp)
if (USE_QTGUI)
    target_link_libraries(dr_demo ddr_reduction bmutility ${opencv_link_libs} ${ffmpeg_link_libs} ${bmnnsdk_link_libs} bmgui-lite)
else()
    target_link_libraries(dr_demo ddr_reduction bmutility ${opencv_link_libs} ${ffmpeg_link_libs} ${bmnnsdk_link_libs})
endif()

