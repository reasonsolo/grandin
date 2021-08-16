// author: zlz
#include <gst/gst.h>

namespace grd {
namespace gstpp {

class GstLoop {
  public:
  GstLoop();

  void Run();
  void Quit();

  private:

  GMainLoop* loop_ = nullptr;
};

}
}