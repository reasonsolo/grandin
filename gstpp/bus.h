// author: zlz
#pragma once

#include <string>
#include <map>
#include <functional>

#include "gstpp/message.h"

#include <gst/gst.h>
#include <glog/logging.h>

namespace grd {
namespace gstpp {

class GstppBus;
using GstppBusCallback = std::function<void(GstppBus*, GstppMessage*)>;

class GstppBus {
  public:
    GstppBus(GstBus* bus): bus_(bus) {}
    ~GstppBus() {
      if (bus_) {
        g_object_unref(bus_);
      }
    };

    void WatchAndSignalConnect() {
      CHECK(bus_) << "bus is invalid";
      gst_bus_add_signal_watch(bus_);
      g_signal_connect(bus_, "message", G_CALLBACK(&GstppBus::MessageCallback), this);
    }

    void AddMessageCallback(GstppBusCallback cb) {
      CHECK(cb);
      msg_cbs_.push_back(std::move(cb));
    }

    static void MessageCallback(GstBus* bus, GstMessage* msg, GstppBus* self) {
      GstppMessage gstpp_msg(msg);
      for (auto&& cb : self->msg_cbs_) {
        cb(self, &gstpp_msg);
      }
    }

    GstBus* bus() const { return bus_; }

  private:

    GstBus* bus_ = nullptr;
    std::vector<GstppBusCallback> msg_cbs_;
};


}

}