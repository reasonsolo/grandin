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
using GstppMsgCallback = std::function<void(GstppBus*, GstppMessage*)>;

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
      g_signal_connect(bus_, "message", G_CALLBACK(&GstppBus::message_cb), this);
    }

    void SetMessageCallback(GstppMsgCallback cb) {
      msg_cb_ = cb;
    }

    static void message_cb(GstBus* bus, GstMessage* msg, GstppBus* self) {
      if (self->msg_cb_) {
        GstppMessage gstpp_msg(msg);
        self->msg_cb_(self, &gstpp_msg);
      } else {
        LOG(WARNING) << "bus message callback is not set";
      }
    }

    GstBus* bus() const { return bus_; }

  private:

    GstBus* bus_ = nullptr;
    GstppMsgCallback msg_cb_;
};


}

}