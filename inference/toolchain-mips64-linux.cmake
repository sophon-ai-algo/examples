# set system name and processor type
set(CMAKE_SYSTEM_NAME        Linux)
set(CMAKE_SYSTEM_PROCESSOR   mips64)

# set cross compiler
set(CROSS_COMPILE mips-linux-gnu-)
set(CMAKE_C_COMPILER         ${CROSS_COMPILE}gcc)
set(CMAKE_CXX_COMPILER       ${CROSS_COMPILE}g++)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wa,-mips64r2 -mabi=64 -march=gs464e" CACHE "c flags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wa,-mips64r2 -mabi=64 -march=gs464e" CACHE "ci++ flags")


# search for programs in the build host dir
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# search lib and header in the target dir
set(CMAKE_FIND_ROOT_PATH_LIBRARY      ONLY)
set(CMAKE_FIND_ROOT_PATH_INCLUDE      ONLY)
