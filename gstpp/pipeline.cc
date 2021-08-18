// author: zlz
#include "gstpp/pipeline.h"
#include "gstpp/bus.h"
#include <glog/logging.h>

namespace grd {
namespace gstpp {

GstppPipeline::GstppPipeline(const std::string& name)
    : GstppElement("pipeline", name, nullptr),
      p_(gst_pipeline_new(name.c_str())) {
  CHECK(p_) << "cannot create pipeline " << name_;
  bus_ = new GstppBus(gst_element_get_bus(p_));
}

GstppPipeline::GstppPipeline(const std::string& name, GstElement* p)
    : GstppElement("pipeline", name, nullptr), p_(p) {
  CHECK(p_) << "cannot create pipeline " << name_;
  bus_ = new GstppBus(gst_element_get_bus(p));
  LOG(INFO) << (void*) gst_element_get_bus(p);
  LOG(INFO) << "get bus " << (void*)(bus_->bus());
}

GstppPipeline::~GstppPipeline() {
  if (bus_) {
    delete bus_;
  }
  if (p_) {
    gst_object_unref(p_);
  }
}

GstppPipeline& GstppPipeline::Add(GstppElement* element) {
  CHECK(elements_.count(element) == 0)
      << "element " << *element << " already in " << *this;
  CHECK(gst_bin_add(GST_BIN(p_), element->element()))
      << "cannot add " << *element << " to " << *this;
  elements_.insert(element);
  element->AddToPipeline(this);
  return *this;
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

}  // namespace gstpp
}  // namespace grd