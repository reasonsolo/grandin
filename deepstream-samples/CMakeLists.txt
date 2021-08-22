set(NVDS_LIB_DIR /opt/nvidia/deepstream/deepstream-${NVDS_VER}/lib/)
set(NVDS_INC_DIR /opt/nvidia/deepstream/deepstream-${NVDS_VER}/sources/includes)
set(CUDA_LIB_DIR /usr/local/cuda-${CUDA_VER}/lib64)
set(CUDA_INC_DIR /usr/local/cuda-${CUDA_VER}/include)
include_directories(${NVDS_INC_DIR} ${CUDA_INC_DIR})
link_directories(${NVDS_LIB_DIR} ${CUDA_LIB_DIR})

set(NVDS_LIBS cudart nvdsgst_meta nvds_meta cuda)
add_executable(deepstream-test1 deepstream-test1.cc)
target_link_libraries(deepstream-test1 gstpp gstpp_ui
gflags glog
${NVDS_LIBS} -Wl,-rpath,${NVDS_LIB_DIR})