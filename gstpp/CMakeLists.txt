add_library(gstpp
  element.cc
)
target_link_libraries(gstpp 
${GSTLIBS_LIBRARIES}
  glog
)