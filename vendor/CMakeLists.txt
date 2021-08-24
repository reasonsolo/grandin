add_subdirectory(fmt)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/fmt/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/)

add_subdirectory(workflow)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/workflow/_include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/workflow/_lib)