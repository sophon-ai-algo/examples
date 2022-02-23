cmake_minimum_required(VERSION 3.2)
project(bmutility)
set(CMAKE_CXX_STANDARD 14)
set(UTILITY_TOP ${CMAKE_CURRENT_SOURCE_DIR})
add_compile_options(-Wno-deprecated-declarations)

include_directories(${common_inc_dirs} ${bmnnsdk_inc_dirs} ${opencv_inc_dirs} ${ffmpeg_inc_dirs})
link_directories(${ffmpeg_link_dirs} ${opencv_link_dirs} ${bmnnsdk_link_dirs})

message(ffmpeg_inc = ${ffmpeg_inc_dirs})
message(ffmpeg_link_libs = ${ffmpeg_link_libs})

add_library(bmutility
        bmutility_pool.cpp
        stream_sei.cpp
        stream_demuxer.cpp
        stream_decode.cpp
	stream_cvdecode.cpp
        bmutility_timer.cpp
        bmutility_string.cpp)

add_executable(stream_unittest stream_unittest.cpp)
target_link_libraries(stream_unittest bmutility ${ffmpeg_link_libs} bmlib pthread)

add_executable(base64_unittest base64_unittest.cpp)
target_link_libraries(base64_unittest bmutility pthread)
