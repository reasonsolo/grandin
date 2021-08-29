add_executable(utils_test utils_test.cc)
target_link_libraries(utils_test fmt glog gtest gtest_main pthread ssl crypto)