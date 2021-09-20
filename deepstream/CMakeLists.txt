add_library(testapp testapp.cc)
target_link_libraries(testapp PUBLIC gstpp gstpp_ui
gflags glog ${NVDS_LIBS} -Wl,-rpath,${NVDS_LIB_DIR})

add_executable(testapp_main testapp_main.cc)
target_link_libraries(testapp_main testapp)