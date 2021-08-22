cmake_minimum_required(VERSION 3.6)


option(BUILD_UI ON)

set(BUILD_UI ON)
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -O0 -g -ggdb3")
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O2")
set(CUDA_VER "11.1")
set(NVDS_VER "5.1")

project(grandin)

find_package(PkgConfig REQUIRED)

pkg_check_modules(GLIB REQUIRED glib-2.0 gobject-2.0)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-video-1.0 )
message(STATUS "!!!!!!!! ${GSTREAMER_LIBRARY_DIRS}  ${GSTREAMER_INCLUDE_DIRS}  ${GLIB_LIBRARIES} ${GTK3_LINK_LIBRARIES}")

include_directories(${GLIB_INCLUDE_DIRS} ${GSTREAMER_INCLUDE_DIRS})
link_directories(${GLIB_LIBRARY_DIRS} ${GSTREAMER_LIBRARY_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR} vendor/gstreamer)

if (BUILD_UI)
pkg_check_modules(GTK REQUIRED gtk+-3.0)
include_directories(${GTK_INCLUDE_DIRS} ${GTK_INCLUDE_DIRS})
link_directories(${GTK_LIBRARY_DIRS} ${GTK_LIBRARY_DIRS})
endif ()

find_package(gflags REQUIRED)
find_package(glog REQUIRED)
find_package(GTest REQUIRED)


add_subdirectory(vendor)
add_subdirectory(gstpp)
add_subdirectory(deepstream-samples)
#add_subdirectory(server)

