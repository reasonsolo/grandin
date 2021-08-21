add_executable(basic-tutorial-12 basic-tutorial-12.cc)
target_link_libraries(basic-tutorial-12 gstpp)

if (BUILD_UI)
add_executable(ui_sample ui_sample.cc)
target_link_libraries(ui_sample gstpp_ui)
endif ()