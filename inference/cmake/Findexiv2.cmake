include(FindPackageHandleStandardArgs)

find_library(exiv2_LIBRARY exiv2)
find_path(exiv2_INCLUDE_DIR exiv2/exiv2.hpp)

find_package_handle_standard_args(
    exiv2
    FOUND_VAR exiv2_FOUND
    REQUIRED_VARS exiv2_INCLUDE_DIR exiv2_LIBRARY)

if (exiv2_FOUND)
    if (NOT TARGET exiv2::exiv2)
        add_library(exiv2::exiv2 SHARED IMPORTED)
        set_target_properties(
            exiv2::exiv2 PROPERTIES
            IMPORTED_LOCATION "${exiv2_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${exiv2_INCLUDE_DIR}")
    endif()
endif()
