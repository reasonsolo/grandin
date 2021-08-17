// author: zlz
#pragma once

#include <gst/gst.h>

namespace grd {
namespace gstpp {

class GstppBus {
  public:
    GstppBus(GstBus* bus);

  private:
    ~GstppBus();

    GstBus* bus_ = nullptr;
};


}

}