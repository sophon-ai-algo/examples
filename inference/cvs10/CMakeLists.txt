cmake_minimum_required(VERSION 3.2)

set(CMAKE_CXX_STANDARD 14)

add_compile_options(-Wno-deprecated-declarations)
add_definitions(-DUSE_FFMPEG=1)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(USE_QTGUI)
    add_definitions(-DUSE_QTGUI=1)
endif()

set(src_files main.cpp worker.cpp)
add_definitions(-DHAVE_FACEDETECT_MODULE=1)
list(APPEND src_files face_detector.cpp face_extract.cpp resnet50.cpp)

include_directories(${common_inc_dirs} ${bmnnsdk_inc_dirs} ${ffmpeg_inc_dirs} ${opencv_inc_dirs}
        ${TOP}/ddr_reduction)
link_directories(${ffmpeg_link_dirs} ${bmnnsdk_link_dirs} ${opencv_link_dirs})


add_executable(cvs10 ${src_files})
if (USE_QTGUI)
    target_link_libraries(cvs10 bmutility ${opencv_link_libs} ${ffmpeg_link_libs} ${bmnnsdk_link_libs} ddr_reduction bmtracker bmgui jsoncpp)
else()
    target_link_libraries(cvs10 bmutility ${opencv_link_libs} ${ffmpeg_link_libs} ${bmnnsdk_link_libs} ddr_reduction bmtracker jsoncpp pthread)
endif()


