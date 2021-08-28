// author: zlz

#include <functional>
#include <thread>
#include <chrono>
#include "deepstream/testapp.h"

#include <cuda_runtime_api.h>
#include <gstnvdsmeta.h>
#include <gflags/gflags.h>

int main(int argc, char** argv) {
  int32_t current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
 
  cudaGetDeviceProperties(&prop, current_device);
  gtk_init(&argc, &argv);
  gst_init(&argc, &argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  grd::deepstream::TestApp app;
  app.Init();
  app.Start();

  using namespace std::chrono_literals;
  bool stop = false;
  while (!stop) {
    app.bus()->PostApplicationMessage([](auto bus, auto msg) { LOG(INFO) << "APPLICATION MESSAGE!!!!"; });
    std::this_thread::sleep_for(1s);
  }

  app.Stop();

  return 0;
}