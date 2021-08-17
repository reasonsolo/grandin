// author: zlz
#include "gstpp/pipeline.h"
#include "gstpp/bus.h"

namespace grd {
namespace gstpp {

GstppPipeline::GstppPipeline(const std::string& name)
    : GstppElement::type_("pipeline"),
      GstppElement::name_(name),
      pipeline_(gst_pipeline_new(name.c_str())) {
  CHECK(pipeline_) << "cannot create pipeline " << name_;
  bus_ = new GstppBus(gst_element_get_bus(pipeline_));
}

GstppPipeline::~GstppPipeline() {
  if (bus_) {
    delete bus_;
  }
  if (pipeline_) {
    gst_object_unref(pipeline_);
  }
}

GstppPipeLine& GstppPipeline::Add(GstppElement* element) {
  CHECK(elements_.count(element) == 0)
  << "element " << *element << " already in " << *this;
  CHECK(gst_bin_add(GST_BIN(pipeline_)), element->element)
  << "cannot add " << *element << " to " << *this;
  elements_.insert(element);
  element->AddToPipeline(this);
  return *this;
}


}  // namespace gstpp
}