cmake_minimum_required(VERSION 3.2)

set(CMAKE_CXX_STANDARD 14)
add_compile_options(-Wno-deprecated-declarations)

set(src_files main.cpp worker.cpp face_detector.cpp)
add_executable(facedetect_demo ${src_files})
if (USE_QTGUI)
    target_link_libraries(facedetect_demo bmutility ${opencv_link_libs} ${ffmpeg_link_libs} ${bmnnsdk_link_libs} bmgui jsoncpp)
else()
    target_link_libraries(facedetect_demo bmutility ${opencv_link_libs} ${ffmpeg_link_libs} ${bmnnsdk_link_libs} jsoncpp pthread)
endif()



