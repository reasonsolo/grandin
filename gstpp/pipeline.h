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
  GstppPipeline(const std::string& type, const std::string& name, GstElement* p);
  GstppPipeline(const std::string& name, GstElement* p);
  virtual ~GstppPipeline();

  GstElement* element() const override { return p_; }

  void AddElement(GstppElement& element);

  template <typename Elem>
  void Add(Elem&& element) {
    AddElement(element)
  }

  template <typename Elem, typename... Rest>
  void Add(Elem&& elem, Rest&&... rest) {
    Add(elem);
    Add(std::forward<Rest>(rest)...);
  }


  static GstppPipeline* LaunchFrom(const std::string& name, const std::string& cmd);

private:
  GstElement* p_ = nullptr;

  std::set<GstppElement*> elements_;

  friend std::ostream& operator<<(std::ostream&, GstppPipeline&);
};

std::ostream& operator<<(std::ostream& os, GstppPipeline& elem);
}
}