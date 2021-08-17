// author: zlz
#include <gst/gst.h>

namespace grd {
namespace gstpp {

class GstppMainLoop {
 public:
  GstppMainLoop() { loop_ = g_main_loop_new(nullptr, false); }
  ~GstppMainLoop() { gst_object_unref(loop_); }

  void Run() { g_main_loop_run(loop_); }
  // void Quit();

 private:
  GMainLoop* loop_ = nullptr;
};

}  // namespace gstpp
}  // namespace grd