include(FindPackageHandleStandardArgs)

set(required_vars glog_LIBRARY glog_INCLUDE_DIR)
set(include_dirs)
if (DEFINED BM_PREBUILT_TOOLCHAINS)
    set(path ${BM_PREBUILT_TOOLCHAINS}/${CMAKE_SYSTEM_PROCESSOR})
    find_library(glog_LIBRARY libglog.a PATHS ${path}/glog/lib)
    find_path(gflags_INCLUDE_DIR gflags/gflags.h PATHS ${path}/gflags/include NO_DEFAULT_PATH)
    list(APPEND required_vars gflags_INCLUDE_DIR)
    find_path(glog_INCLUDE_DIR glog/logging.h PATHS ${path}/glog/include)
    list(APPEND include_dirs ${gflags_INCLUDE_DIR} ${glog_INCLUDE_DIR})
else()
    find_library(glog_LIBRARY glog)
    find_path(glog_INCLUDE_DIR logging.h PATH_SUFFIXES glog)
    list(APPEND include_dirs ${glog_INCLUDE_DIR})
endif()

find_package_handle_standard_args(
    glog
    FOUND_VAR glog_FOUND
    REQUIRED_VARS ${required_vars})

if (glog_FOUND)
    if (NOT TARGET glog::glog)
        add_library(glog::glog SHARED IMPORTED)
        set_target_properties(
            glog::glog PROPERTIES
            IMPORTED_LOCATION "${glog_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${include_dirs}")

        if (DEFINED BM_PREBUILT_TOOLCHAINS)
            set_target_properties(
                glog::glog PROPERTIES
                SKIP_BUILD_RPATH TRUE)
        endif()
    endif()
endif()
