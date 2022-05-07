find_package(exiv2 REQUIRED)

add_library(ruilai_lib ruilai_api.cpp worker.cpp resnet50.cpp retinaface.cpp mobilenetv2.cpp wsdan.cpp)
add_executable(ruilai_comp main.cpp)
#add_executable(ruilai_comp test.cpp)

target_include_directories(ruilai_lib PUBLIC
    ${common_inc_dirs} ${bmnnsdk_inc_dirs}
    ${opencv_inc_dirs} ${ffmpeg_inc_dirs})
message(----- ${exiv2_LIBRARY})
target_compile_definitions(ruilai_lib PUBLIC USE_EXIV2=1)

if (USE_QTGUI)
    target_link_libraries(ruilai_lib bmutility ${opencv_link_libs} ${ffmpeg_link_libs} ${bmnnsdk_link_libs} glog bmutility bmgui jsoncpp exiv2)
else()
    target_link_libraries(ruilai_lib bmutility ${opencv_link_libs} ${ffmpeg_link_libs} ${bmnnsdk_link_libs} glog bmutility jsoncpp pthread exiv2)
endif()

target_link_libraries(ruilai_comp ruilai_lib)
