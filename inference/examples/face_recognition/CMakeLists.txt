cmake_minimum_required(VERSION 3.2)

set(CMAKE_CXX_STANDARD 14)
add_compile_options(-Wno-deprecated-declarations)

set(src_files main.cpp
        face_detector.cpp
        face_landmark.cpp
        face_extract.cpp
        face_worker.cpp
        )
add_executable(facerecogintion_demo ${src_files})
if (USE_QTGUI)
    target_link_libraries(facerecogintion_demo bmutility ${opencv_link_libs} ${ffmpeg_link_libs} ${bmnnsdk_link_libs} bmgui jsoncpp)
else()
    target_link_libraries(facerecogintion_demo bmutility ${opencv_link_libs} ${ffmpeg_link_libs} ${bmnnsdk_link_libs} jsoncpp pthread)
endif()