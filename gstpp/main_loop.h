// author: zlz
#pragma once

#include <common/macros.h>
#include <glog/logging.h>
#include <gst/gst.h>

namespace grd {
namespace gstpp {

class GstppMainLoop {
 public:
  GstppMainLoop() { loop_ = g_main_loop_new(nullptr, false); }
  ~GstppMainLoop() { g_main_loop_unref(loop_); }

  void Run() { 
    LOG(INFO) << "loop running...";
    g_main_loop_run(loop_);
  }
  void Quit() {
    g_main_loop_quit(loop_);
    LOG(INFO) << "loop exits";
  }

 private:
  GMainLoop* loop_ = nullptr;
};

}  // namespace gstpp
}  // namespace grd