// author: zlz
#pragma once

#include <string>
#include <ostream>

#include "common/macros.h"
#include "gstpp/bus.h"

#include "gst/gst.h"

namespace grd {
namespace gstpp {

enum class ElementState {
  RESET = GST_STATE_NULL,
  READY = GST_STATE_READY,
  PLAYING = GST_STATE_PLAYING,
  PAUSED = GST_STATE_PAUSED,
};

  class GstppPipeline;
class GstppElement {
  NOCOPY(GstppElement);
 public:
  GstppElement() = default;
  GstppElement(const std::string& ele_type, const std::string& ele_name);
  virtual ~GstppElement();

  const std::string& name() const { return name_; }
  const std::string& type() const { return type_; }

  ElementState state() const { return state_; }

  void LinkTo(GstppElement& downstream);

  bool Reset() { return SetState(ElementState::RESET); }
  bool Ready() { return SetState(ElementState::READY); }
  bool Play()  { return SetState(ElementState::PLAYING); }
  bool Pause() { return SetState(ElementState::PAUSED); }

  // upstream-->downstream-->downstream1 
  GstppElement& operator--() { return *this; }
  GstppElement& operator>(GstppElement& rhs) {
    this->LinkTo(rhs);
    return rhs;
  }

 protected:
  
  virtual GstElement* element() { return element_; }

  bool SetState(ElementState state);

  std::string type_;
  std::string name_;

  GstElement* element_ = nullptr;
  ElementState state_ = ElementState::RESET;
  GstppPipeline* pipeline_ = nullptr;

  friend std::ostream& operator<<(std::ostream&, GstppElement&);
  friend class GstppPipeline;

};

std::ostream& operator<<(std::ostream& os, GstppElement& elem) {
  os << "(" << elem.type() << ":" << elem.name() << ")";
  return os;
}

}  // namespace gstpp
}  // namespace grd