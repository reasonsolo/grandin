cmake_minimum_required(VERSION 3.6)

project(grandin)

#find_package(gstreamer REQUIRED)

include_directories(vendor/gstreamer)

add_subdirectory(vendor)
add_subdirectory(gstpp)
add_subdirectory(server)

