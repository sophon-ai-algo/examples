# set system name and processor type
set(CMAKE_SYSTEM_NAME        Linux)
set(CMAKE_SYSTEM_PROCESSOR   aarch64)

# set cross compiler
set(CROSS_COMPILE aarch64-linux-gnu-)
set(CMAKE_C_COMPILER         ${CROSS_COMPILE}gcc)
set(CMAKE_CXX_COMPILER       ${CROSS_COMPILE}g++)

# search for programs in the build host dir
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# search lib and header in the target dir
message(${BMNNSDK2_TOP}/3rdparty)
SET(CMAKE_FIND_ROOT_PATH $ENV{REL_TOP}/3rdparty)
SET(CMAKE_CXX_FLAGS "-Wl,--allow-shlib-undefined" CACHE STRING "" FORCE)
set(CMAKE_FIND_ROOT_PATH_LIBRARY      ONLY)
set(CMAKE_FIND_ROOT_PATH_INCLUDE      ONLY)
#  set(CMAKE_C_COMPILER $ENV{BMWS}/bm_prebuilt_toolchains/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc)
#  set(CMAKE_CXX_COMPILER $ENV{BMWS}/bm_prebuilt_toolchains/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++)
