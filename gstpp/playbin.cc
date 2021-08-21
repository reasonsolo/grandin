// author: zlz
#include "gstpp/playbin.h"

namespace grd {
namespace gstpp {

GstppPlayBin::GstppPlayBin(const std::string&)
    : GstppElement(name, "playbin"),
      bus_(new GstppBus(gst_element_get_bus(element()))) {}

GstppPlayBin::~GstppPlayBin() {
    delete bus_;
}

}  // namespace gstpp
}  // namespace grd