// author: zlz
#pragma once
#include <string>
#include <vector>
#include "gstpp/element.h"

#include <gst/gst.h>

namespace grd {
namespace gstpp {

class GstppBus;

class GstppPipeline: public GstppElement {
  public:
  GstppPipeline(const std::string& name);
  virtual ~GstppPipeline();

  const std::string& name() const { return name_; }
  GstppBus* bus() { return bus_; }

  GstppPipeline& Add(GstppElement* element);

private:
  GstElement* pipeline_ = nullptr;

  GstppBus* bus_;

  std::set<GstppElement*> elements_;

  friend std::ostream& operator<<(std::ostream&, GstppPipeline&);
};

std::ostream& operator<<(std::ostream& os, GstppPipeline& elem) {
  os << "(" << elem.type() << ":" << elem.name() << ")";
  return os;
}
}
}