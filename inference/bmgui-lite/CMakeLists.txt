cmake_minimum_required(VERSION 3.2)
project(bmgui-lite)
set(CMAKE_CXX_STANDARD 14)
set(UTILITY_TOP ${CMAKE_CURRENT_SOURCE_DIR})
add_compile_options(-Wno-deprecated-declarations)

set(CMAKE_AUTOGEN_VERBOSE ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
#set(QTDIR /opt/Qt5.12.9/5.12.9/gcc_64)
set(Qt5Widgets_DIR ${QTDIR}/lib/cmake/Qt5Widgets)
set(Qt5Core_DIR ${QTDIR}/lib/cmake/Qt5Core)
set(Qt5Concurrent_DIR ${QTDIR}/lib/cmake/Qt5Concurrent)
set(Qt5Test_DIR ${QTDIR}/lib/cmake/Qt5Test)

find_package(Qt5Widgets REQUIRED)
find_package(Qt5Core REQUIRED)
find_package(Qt5Concurrent REQUIRED)
find_package(Qt5Test REQUIRED)



include_directories(${Qt5Widgets_INCLUDE_DIRS} ${Qt5Core_INCLUDE_DIRS}
        ${Qt5Concurrent_INCLUDE_DIRS}
        ${Qt5Test_INCLUDE_DIRS} ${TOP}/bmgui-lite ${bmnnsdk_inc_dirs} ${ffmpeg_inc_dirs} ${opencv_inc_dirs})

link_directories(${ffmpeg_link_dirs} ${bmnnsdk_link_dirs} ${opencv_link_dirs})

add_library(bmgui-lite SHARED window_QT.cpp)
target_link_libraries(bmgui-lite Qt5::Widgets Qt5::Test ${opencv_link_libs} ${ffmpeg_link_libs} ${bmnnsdk_link_libs})
