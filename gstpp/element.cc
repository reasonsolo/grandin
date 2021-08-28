#include "gstpp/element.h"
#include "gstpp/pad.h"

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
  if (elem_ptr) {
    gst_object_ref(element_);
  }
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
  auto ret = gst_element_set_state(element(), static_cast<GstState>(state));
  switch (ret) {
    case GST_STATE_CHANGE_SUCCESS:
    case GST_STATE_CHANGE_NO_PREROLL:
      return true;
    case GST_STATE_CHANGE_ASYNC:
      ret = gst_element_get_state(element(), NULL, NULL, GST_CLOCK_TIME_NONE);
      return ret == GST_STATE_CHANGE_SUCCESS || ret == GST_STATE_CHANGE_NO_PREROLL;
    case GST_STATE_CHANGE_FAILURE:
      return false;
    default:
      LOG(INFO) << "unexpected state change result " << ret;
      return false;
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

template <>
void GstppElement::SetProperty<std::string>(const std::string& key, std::string value) {
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

GstppPad* GstppElement::GetRequestPad(const std::string& name) {
  auto pad = gst_element_get_request_pad(element(), name.c_str());
  if (pad) {
    return new GstppPad(pad, name);
  }
  LOG(ERROR) << "cannot get pad " << name << " from " << *this;
  return nullptr;
}

GstppPad* GstppElement::GetStaticPad(const std::string& name) {
  auto pad = gst_element_get_static_pad(element(), name.c_str());
  if (pad) {
    return new GstppPad(pad, name);
  }
  LOG(ERROR) << "cannot get pad " << name << " from " << *this;
  return nullptr;
}

void GstppElement::SetNewPadCallback(ElementPadCallback callback) {
  new_pad_cb_ = std::move(callback);
  g_signal_connect(G_OBJECT(element_), "pad-added", G_CALLBACK(&GstppElement::OnPadAdded), this);
}
void GstppElement::SetNewChildCallback(ElementChildCallback callback) {
  new_child_cb_ = std::move(callback);
  g_signal_connect(G_OBJECT(element_), "child-added", G_CALLBACK(&GstppElement::OnChildAdded), this);
}

/* static */
void GstppElement::OnPadAdded(GstElement* elem, GstPad* pad, gpointer data) {
  auto self = static_cast<GstppElement*>(data);
  GstppPad gstpp_pad(pad);
  if (self->new_pad_cb_) { 
    self->new_pad_cb_(&gstpp_pad);
  }
}

/* static */
void GstppElement::OnChildAdded(GstChildProxy* child_proxy, GObject* obj, gchar* name, gpointer data) {
  auto self = static_cast<GstppElement*>(data);
  if (self->new_child_cb_) { 
    self->new_child_cb_(child_proxy, obj, name);
  }

}

/* static */
GstppElement* GstppElement::CreateSourceFromUri(const std::string& uid, const std::string& uri) { 
  std::string name = fmt::format("srcbin:{}", uid.size() > 8u ? uid.substr(uid.size() - 8) : uid);
  LOG(INFO) << "create source element " << name << " from url " << uri;
  auto bin = new GstppElement("uridecodebin", name.c_str());
  bin->SetProperty("uri", uri);
  return bin;
}
}
}