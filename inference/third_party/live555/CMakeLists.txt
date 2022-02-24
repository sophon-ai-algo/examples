cmake_minimum_required(VERSION 3.2)

aux_source_directory(BasicUsageEnvironment BasicUsageEnvironment)
aux_source_directory(groupsock groupsock)
aux_source_directory(liveMedia liveMedia)
aux_source_directory(UsageEnvironment UsageEnvironment)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/BasicUsageEnvironment/include
        ${CMAKE_CURRENT_SOURCE_DIR}/UsageEnvironment/include
        ${CMAKE_CURRENT_SOURCE_DIR}/groupsock/include
        ${CMAKE_CURRENT_SOURCE_DIR}/liveMedia/include)

if (${TARGET_ARCH} STREQUAL "x86")
    add_compile_options(-m64 -fPIC)
    add_definitions(-DSOCKLEN_T=socklen_t -D_LARGEFILE_SOURCE=1 -D_FILE_OFFSET_BITS=64)
elseif (${TARGET_ARCH} STREQUAL "soc")
    add_definitions(-DSOCKLEN_T=socklen_t -DNO_SSTREAM=1 -D_LARGEFILE_SOURCE=1 -D_FILE_OFFSET_BITS=64)
endif()

#add_custom_target(live555_LIB ALL)
#add_custom_command(
#        TARGET live555_LIB
#        COMMAND cd ${TOP}/third_party/live555 && ./genMakefiles linux-64bit
#        COMMAND cd ${TOP}/third_party/live555 && $(MAKE)
#        COMMAND echo live 555 compile successfully
#)
#add_custom_target(
#        TARGET video_stitch_demo
#        DEPENDS LIVE555_LIB
#)

add_library(live555 ${BasicUsageEnvironment} ${groupsock} ${liveMedia} ${UsageEnvironment})