#include "gstpp/element.h"

#include <glog/logging.h>

namespace grd {
namespace gstpp {

GstppElement::GstppElement(const std::string& ele_type,
                           const std::string& ele_name):
                           type_(ele_type), name_(ele_name) {
  element_ = gst_element_factory_make(type_.c_str(), name_.c_str());
  CHECK (element_) << "cannot make gst element " << ele_type << "," << ele_name;
}

GstppElement::~GstppElement() {
  if (!added_to_pipeline_ && element_) {
    gst_object_unref(GST_OBJECT(element_));
  }
}

void GstppElement::LinkTo(GstppElement& downstream) {
  CHECK(gst_element_link(element(), downstream.element_))
  << "cannot link " << *this << "-->" << downstream;
}

bool GstppElement::SetState(ElementState state) {
  if (gst_element_set_state(element(), static_cast<GstState>(state)) != GST_STATE_CHANGE_FAILURE) {
    state_ = state;
    return true;
  }
  return false;
}

}
}