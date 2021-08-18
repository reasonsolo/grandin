// author: zlz
#include <gst/gst.h>

namespace grd {
namespace gstpp {

class GstppMainLoop {
 public:
  GstppMainLoop() { loop_ = g_main_loop_new(nullptr, false); }
  ~GstppMainLoop() { g_main_loop_unref(loop_); }

  void Run() { g_main_loop_run(loop_); }
  void Quit() { g_main_loop_quit(loop_); }

 private:
  GMainLoop* loop_ = nullptr;
};

}  // namespace gstpp
}  // namespace grd