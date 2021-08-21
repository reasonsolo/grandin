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

GstppElement::GstppElement(const std::string& ele_type,
                           const std::string& ele_name,
                           GstElement* elem_ptr):
                           type_(ele_type), name_(ele_name), element_(elem_ptr) {
  // CHECK (element_) << "invalid gst element " << ele_type << "," << ele_name;
}

GstppElement::~GstppElement() {
  if (bus_) {
    delete bus_;
  }
  if (!pipeline_ && element_) {
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

std::ostream& operator<<(std::ostream& os, GstppElement& elem) {
  os << "(" << elem.type() << ":" << elem.name() << ")";
  return os;
}

/* static */
GstppElement* Create(const std::string& type, const std::string& name) {
  auto elem = gst_element_factory_make(type.c_str(), name.c_str());
  if (elem) {
    return new GstppElement(type, name, elem);
  }
  return nullptr;
}

void GstppElement::SetProperty(const std::string& key, const std::string& value) {
  g_object_set(gpointer(element()), key.c_str(), value.c_str(), nullptr);
}

void GstppElement::SetFlag(gint flag) {
    gint query_flags;
    g_object_get(element(), "flags", &query_flags, nullptr);
    gint flags = query_flags | flag;
    g_object_set(element(), "flags", flags, nullptr);
}

void GstppElement::UnSetFlag(gint flag) {
    gint query_flags;
    g_object_get(element(), "flags", &query_flags, nullptr);
    gint flags = query_flags & ~flag;
    g_object_set(element(), "flags", flags, nullptr);
}

void GstppElement::InitBus(GstBus* bus) {
  CHECK(!bus_);
  bus_ = new GstppBus(bus);
}
}
}