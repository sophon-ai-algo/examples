cmake_minimum_required(VERSION 3.2)
project(jsoncpp)
add_library(jsoncpp json_reader.cpp
        json_value.cpp
        json_writer.cpp)
