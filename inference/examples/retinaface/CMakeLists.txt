
add_library(retinaface SHARED retinaface.cpp)

target_link_libraries(retinaface PUBLIC
    ${ffmpeg_link_libs} ${opencv_link_libs} ${bmnnsdk_link_libs} glog bmutility)

target_include_directories(retinaface PUBLIC
    ${common_inc_dirs} ${bmnnsdk_inc_dirs}
    ${opencv_inc_dirs} ${ffmpeg_inc_dirs})

find_package(exiv2 REQUIRED)
target_link_libraries(retinaface PUBLIC exiv2)
target_compile_definitions(retinaface PUBLIC USE_EXIV2=1)

add_executable(retinaface_detect detect.cpp)
target_link_libraries(retinaface_detect retinaface)

add_executable(retinaface_widerface_val widerface_val.cpp)
target_link_libraries(retinaface_widerface_val retinaface stdc++fs)

add_executable(retinaface_test_throughput retinaface_test_throughput.cpp)
target_link_libraries(retinaface_test_throughput retinaface)

add_executable(retinaface_demo main.cpp worker.cpp)
if (USE_QTGUI)
    target_link_libraries(retinaface_demo bmutility ${opencv_link_libs} ${ffmpeg_link_libs} ${bmnnsdk_link_libs} retinaface bmgui jsoncpp)
else()
    target_link_libraries(retinaface_demo bmutility ${opencv_link_libs} ${ffmpeg_link_libs} ${bmnnsdk_link_libs} retinaface jsoncpp pthread)
endif()
