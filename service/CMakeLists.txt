find_library(LIBRT rt)
find_package(OpenSSL REQUIRED)

find_package(workflow REQUIRED CONFIG HINTS "${CMAKE_CURRENT_SOURCE_DIR}/../vendor/workflow")
include_directories(${OPENSSL_INCLUDE_DIR} ${WORKFLOW_INCLUDE_DIR})
link_directories(${WORKFLOW_LIB_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../vendor/workflow/_include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../vendor/workflow/workflow/_lib)

add_executable(grandin
 grandin_main.cc
 auth_manager.cc
 video_manager.cc
 app_manager.cc
 uri_params.cc
)
target_link_libraries(grandin fmt gflags glog workflow pthread ssl crypto rt)
