add_library(gstpp
  element.cc
  pipeline.cc
  message.cc
)
target_link_libraries(gstpp PUBLIC
  ${GSTLIBS_LIBRARIES}
  glog
)

add_subdirectory(samples)