// author: zlz
#pragma once

#include <string>
#include <map>
#include <unordered_map>
#include <functional>

#include "gstpp/message.h"

#include <gst/gst.h>
#include <glog/logging.h>

namespace grd {
namespace gstpp {

class GstppBus;
using GstppBusSignalCallback = std::function<void(const std::string& signal_name)>;
using GstppBusMessageCallback = std::function<void(GstppBus*, GstppMessage*)>;
using GstppBusSyncHandlerCallback = std::function<GstBusSyncReply(GstppBus*, GstppMessage*)>;

class GstppBus {
  public:
    GstppBus(GstBus* bus): bus_(bus) {
      gst_object_ref(bus);
    }
    ~GstppBus() {
      if (bus_watch_id_) { 
        g_source_remove(bus_watch_id_);
      }
      if (bus_) {
        g_object_unref(bus_);
      }
    };

    void AddWatch();

    void WatchAndSignalConnect();

    void EnableSyncHandler();

    void AddMessageCallback(GstppBusMessageCallback cb);

    void SetTypedMessageCallback(MessageType type, GstppBusMessageCallback cb);

    void AddSyncHandlerCallback(GstppBusSyncHandlerCallback cb);

    void PostApplicationMessage(GstppBusMessageCallback cb);

    static bool MessageCallback(GstBus* bus, GstMessage* msg, GstppBus* self);

    static void SyncHandlerCallback(GstBus* bus, GstMessage* msg, GstppBus* self);

    GstBus* bus() const { return bus_; }

  private:
   bool is_watching_ = false;
   GstBus* bus_ = nullptr;
   std::vector<GstppBusMessageCallback> msg_cbs_;
   std::vector<GstppBusSyncHandlerCallback> sync_handler_cbs_;
   std::map<MessageType, GstppBusMessageCallback> typed_msg_cb_map_;

  guint bus_watch_id_ = 0;
};


}

}