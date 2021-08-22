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
using GstppBusMessageCallback = std::function<void(GstppBus*, GstppMessage*)>;
using GstppBusSyncHandlerCallback = std::function<GstBusSyncReply(GstppBus*, GstppMessage*)>;

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

    void EnableSyncHandler() {
      CHECK(bus_) << "bus is invalid";
      gst_bus_set_sync_handler(bus_, (GstBusSyncHandler)&GstppBus::SyncHandlerCallback, this, nullptr);
    }

    void AddMessageCallback(GstppBusMessageCallback cb) {
      CHECK(cb);
      msg_cbs_.push_back(std::move(cb));
    }

    void SetTypedMessageCallback(MessageType type, GstppBusMessageCallback cb) {
      CHECK(cb);
      typed_msg_cb_map_[type] = std::move(cb);
    }

    void AddSyncHandlerCallback(GstppBusSyncHandlerCallback cb) {
      CHECK(cb);
      sync_handler_cbs_.push_back(std::move(cb));
    }

    static void MessageCallback(GstBus* bus, GstMessage* msg, GstppBus* self) {
      GstppMessage gstpp_msg(msg);
      for (auto&& cb : self->msg_cbs_) {
        cb(self, &gstpp_msg);
      }
      auto it = self->typed_msg_cb_map_.find(gstpp_msg.type());
      if (it != self->typed_msg_cb_map_.end() && it->second) {
        it->second(self, &gstpp_msg);
      }
    }

    static void SyncHandlerCallback(GstBus* bus, GstMessage* msg, GstppBus* self) {
      GstppMessage gstpp_msg(msg);
      for (auto&& cb : self->sync_handler_cbs_) {
        cb(self, &gstpp_msg);
      }
    }

    GstBus* bus() const { return bus_; }

  private:

    GstBus* bus_ = nullptr;
    std::vector<GstppBusMessageCallback> msg_cbs_;
    std::vector<GstppBusSyncHandlerCallback> sync_handler_cbs_;
    std::map<MessageType, GstppBusMessageCallback> typed_msg_cb_map_;
};


}

}