add_library(gstpp
  element.cc
  pipeline.cc
  message.cc
  pad.cc
  bus.cc
  recorder.cc
)
target_link_libraries(gstpp PUBLIC
  ${GLIB_LIBRARIES}
  ${GSTREAMER_LIBRARIES}
  fmt
  glog
)

if (BUILD_UI)
add_library(gstpp_ui gtk_player.cc)
target_link_libraries(gstpp_ui PUBLIC gstpp ${GTK_LIBRARIES})
endif()

add_subdirectory(samples)