cmake_minimum_required(VERSION 3.2)
project(videoui)
set(CMAKE_CXX_STANDARD  14)
add_compile_options(-std=c++14)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_PREFIX_PATH $ENV{QTDIR})

set(inc_dirs ${common_inc_dirs} ${bmnnsdk_inc_dirs} ${ffmpeg_inc_dirs} ${opencv_inc_dirs} ${TOP}/third_party/libyuv/include)
set(link_dirs ${ffmpeg_link_dirs} ${opencv_link_dirs})
set(link_libs ${opencv_link_libs} ${ffmpeg_link_libs} Qt5::Widgets pthread)

message(STATUS "[bmgui]inc_dirs=${inc_dirs}")
message(STATUS "[bmgui]link_dirs=${link_dirs}")
message(STATUS "[bmgui]link_libs=${link_libs}")

include_directories(${inc_dirs})
link_directories(${link_dirs})

find_package(Qt5Widgets)

add_library(bmgui SHARED bmgui.cpp mainwindow.cpp
        container_widget.cpp
        video_pixmap_widget.cpp
        video_widget.cpp)

target_link_libraries(bmgui
        ${link_libs}
        )

add_executable(videoui_demo bmgui_unittest.cpp)
target_link_libraries(videoui_demo bmgui)
