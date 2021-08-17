cmake_minimum_required(VERSION 3.6)

project(grandin)

set(CMAKE_CXX_STANDARD 14)

find_package(PkgConfig REQUIRED)

pkg_check_modules(GSTLIBS REQUIRED
        gobject-2.0
        glib-2.0
        gstreamer-1.0)

pkg_check_modules(GLIB REQUIRED glib-2.0)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
message(STATUS "!!!!!!!! ${GSTREAMER_LIBRARY_DIRS}  ${GSTREAMER_INCLUDE_DIRS}  ${GSTLIBS_LIBRARIES}")

include_directories(${GLIB_INCLUDE_DIRS} ${GSTREAMER_INCLUDE_DIRS})
link_directories(${GLIB_LIBRARY_DIRS} ${GSTREAMER_LIBRARY_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR} vendor/gstreamer)

find_package(gflags REQUIRED)
find_package(glog REQUIRED)
find_package(GTest REQUIRED)

add_subdirectory(vendor)
add_subdirectory(gstpp)
add_subdirectory(server)

