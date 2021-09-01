// author: zlz

#include <functional>
#include <thread>
#include <chrono>
#include "deepstream/testapp.h"

#include <cuda_runtime_api.h>
#include <gstnvdsmeta.h>
#include <gflags/gflags.h>

DEFINE_string(
    src2,
    "/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p_h264.mp4",
    "input stream");

DEFINE_string(
    src3,
    "https://www.freedesktop.org/software/gstreamer-sdk/data/media/sintel_trailer-480p.webm",
    "input stream uri");
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

  LOG(INFO) << "event1";
  using namespace std::chrono_literals;
  app.AddSource("uribin", FLAGS_src3, 
      [](bool success) { LOG(INFO) << "src2 added" << success; },
      [](bool success) { LOG(INFO) << "src2 removed " << success; });

  std::this_thread::sleep_for(1s);
  bool stop = false;
  while (!stop) {
    LOG(INFO) << "event2";
    app.bus()->PostApplicationMessage([](auto bus, auto msg) { LOG(INFO) << "APPLICATION MESSAGE!!!!"; });
    std::this_thread::sleep_for(1s);
  }

  app.Stop();

  return 0;
}