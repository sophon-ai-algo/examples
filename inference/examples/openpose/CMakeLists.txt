cmake_minimum_required(VERSION 3.2)

set(CMAKE_CXX_STANDARD 14)
add_compile_options(-Wno-deprecated-declarations)

set(src_files main.cpp worker.cpp openpose.cpp pose_postprocess.cpp)
add_executable(openpose_demo ${src_files})
if (USE_QTGUI)
    target_link_libraries(openpose_demo bmutility ${opencv_link_libs} ${ffmpeg_link_libs} ${bmnnsdk_link_libs} bmgui jsoncpp)
else()
    target_link_libraries(openpose_demo bmutility ${opencv_link_libs} ${ffmpeg_link_libs} ${bmnnsdk_link_libs} jsoncpp pthread)
endif()
