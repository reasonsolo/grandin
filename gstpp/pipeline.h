// author: zlz
#pragma once
#include <string>
#include <set>
#include "gstpp/element.h"

#include <gst/gst.h>

namespace grd {
namespace gstpp {

class GstppBus;

class GstppPipeline: public GstppElement {
  public:
  GstppPipeline(const std::string& name);
  GstppPipeline(const std::string& name, GstElement* p);
  virtual ~GstppPipeline();

  const std::string& name() const { return name_; }
  GstppBus* bus() { return bus_; }
  GstElement* element() override { return p_; }

  GstppPipeline& Add(GstppElement* element);

  static GstppPipeline* LaunchFrom(const std::string& name, const std::string& cmd);

private:
  GstElement* p_ = nullptr;
  GstppBus* bus_ = nullptr;

  std::set<GstppElement*> elements_;

  friend std::ostream& operator<<(std::ostream&, GstppPipeline&);
};

std::ostream& operator<<(std::ostream& os, GstppPipeline& elem) {
  os << "(" << elem.type() << ":" << elem.name() << ")";
  return os;
}
}
}