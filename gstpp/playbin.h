// author: zlz
#pragma once

#include "gstpp/element.h"

#include <gst/gst.h>

namespace grd {
namespace gstpp {

enum class PlayFlags {
  VIDEO = (1 << 0), /* We want video output */
  AUDIO = (1 << 1), /* We want audio output */
  TEXT = (1 << 2)   /* We want subtitle output */
} ;

class GstppPlayBin: public GstppElement {
 public:
  GstppPlayBin(const std::string& name)
      : GstppElement("playbin", name) {
    GstppElement::InitBus(gst_element_get_bus(element()));
  }

  virtual ~GstppPlayBin() { delete bus_; }

  void ShowVideo(bool enable) {
    if (enable) {
      SetFlag((gint)PlayFlags::VIDEO);
    } else {
      UnSetFlag((gint)PlayFlags::VIDEO);
    }
  }
  void ShowAudio(bool enable) {
    if (enable) {
      SetFlag((gint)PlayFlags::AUDIO);
    } else {
      UnSetFlag((gint)PlayFlags::AUDIO);
    }
  }
  void ShowText(bool enable) {
    if (enable) {
      SetFlag((gint)PlayFlags::TEXT);
    } else {
      UnSetFlag((gint)PlayFlags::TEXT);
    }
  }

  
};

}
}  // namespace grd