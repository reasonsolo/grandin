cmake_minimum_required(VERSION 3.6)

project(grandin)


find_package(gstreamer required)

add_subdirectory(gstpp)
add_subdirectory(server)