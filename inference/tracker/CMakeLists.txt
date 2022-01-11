cmake_minimum_required(VERSION 3.2)
project(bmtracker)

set(TOP ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${common_inc_dirs} ${bmnnsdk_inc_dirs} ${bm_ffmpeg_inc_dirs} ${bm_opencv_inc_dirs} /usr/include/eigen3)

add_library(bmtracker bm_tracker.cpp
        ${TOP}/KalmanFilter/kalmanfilter.cpp
        ${TOP}/KalmanFilter/linear_assignment.cpp
        ${TOP}/KalmanFilter/nn_matching.cpp
        ${TOP}/KalmanFilter/track.cpp
        ${TOP}/KalmanFilter/tracker.cpp
        ${TOP}/MunkresAssignment/hungarianoper.cpp
        ${TOP}/MunkresAssignment/munkres/munkres.cpp
        ${TOP}/DeepAppearanceDescriptor/model.cpp
        )
