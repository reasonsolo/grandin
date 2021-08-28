// author: zlz
#include "gstpp/pipeline.h"
#include "gstpp/bus.h"
#include <glog/logging.h>

namespace grd {
namespace gstpp {

GstppPipeline::GstppPipeline(const std::string& name)
    : GstppElement("pipeline", name, nullptr),
      p_(gst_pipeline_new(name.c_str())) {
  InitBus(gst_element_get_bus(p_));
  CHECK(p_) << "cannot create pipeline " << name_;
}

GstppPipeline::GstppPipeline(const std::string& name, GstElement* p)
    : GstppPipeline("pipeline", name, p) {
}

GstppPipeline::GstppPipeline(const std::string& type, const std::string& name,
                             GstElement* p)
    : GstppElement(type, name, nullptr), p_(p) {
  CHECK(p_) << "cannot create pipeline " << name_;
  bus_ = new GstppBus(gst_element_get_bus(p));
  LOG(INFO) << (void*) gst_element_get_bus(p);
  LOG(INFO) << "get bus " << (void*)(bus_->bus());
  gst_object_ref(p_);
}

GstppPipeline::~GstppPipeline() {
  if (p_) {
    gst_object_unref(p_);
  }
}

void GstppPipeline::AddElement(GstppElement& element) {
  CHECK(gst_bin_add(GST_BIN(p_), element.element()))
      << "cannot add " << element << " to " << *this;
  element.AddToPipeline(this);
}

void GstppPipeline::RemoveElement(GstppElement& element) {
  gst_bin_remove(GST_BIN(p_), element.element());
  element.RemoveFromPipeline(this);
}

/* static */
GstppPipeline* GstppPipeline::LaunchFrom(const std::string& name,
                                         const std::string& cmd) {
  GError* err = nullptr;
  auto p = gst_parse_launch(cmd.c_str(), &err);
  if (p) {
    auto pipeline = new GstppPipeline(name, p);
    return pipeline;
  } else {
    LOG(ERROR) << "cannot launch ppl " << name << " from command" << cmd
               << ", error: " << err->message;
    g_error_free(err);
  }
  return nullptr;
}

std::ostream& operator<<(std::ostream& os, GstppPipeline& elem) {
  os << "(" << elem.type() << ":" << elem.name() << ")";
  return os;
}

}  // namespace gstpp
}  // namespace grd