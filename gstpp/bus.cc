// author: zlz
#include "gstpp/bus.h"

namespace grd {
namespace gstpp {

void GstppBus::AddWatch() {
  CHECK(bus_watch_id_ == 0);
  CHECK(!is_watching_) << "bus is already watching";
  bus_watch_id_ =
      gst_bus_add_watch(bus_, (GstBusFunc)(&GstppBus::MessageCallback), this);
      is_watching_ = true;
  LOG(INFO) << "add watch " << bus_watch_id_;
}

void GstppBus::WatchAndSignalConnect() {
  CHECK(bus_) << "bus is invalid";
  CHECK(!is_watching_) << "bus is already watching";
  gst_bus_add_signal_watch(bus_);
  g_signal_connect(bus_, "message", G_CALLBACK(&GstppBus::MessageCallback),
                   this);
  is_watching_ = true;
}

void GstppBus::EnableSyncHandler() {
  CHECK(bus_) << "bus is invalid";
  gst_bus_set_sync_handler(
      bus_, (GstBusSyncHandler)&GstppBus::SyncHandlerCallback, this, nullptr);
}

void GstppBus::AddMessageCallback(GstppBusMessageCallback cb) {
  CHECK(cb);
  msg_cbs_.push_back(std::move(cb));
}

void GstppBus::SetTypedMessageCallback(MessageType type,
                                       GstppBusMessageCallback cb) {
  CHECK(cb);
  typed_msg_cb_map_[type] = std::move(cb);
}

void GstppBus::AddSyncHandlerCallback(GstppBusSyncHandlerCallback cb) {
  CHECK(cb);
  sync_handler_cbs_.push_back(std::move(cb));
}

void GstppBus::PostApplicationMessage(GstppBusMessageCallback cb) {
  GstppMessage* msg =
      new GstppMessage(nullptr, std::move(cb));  // GST_OBJECT_CAST(bus_), cb);
  gst_bus_post(bus_, msg->msg());
}

/* static */ bool GstppBus::MessageCallback(GstBus* bus, GstMessage* msg,
                                            GstppBus* self) {
  GstppMessage* gstpp_msg;
  if (msg->type != GST_MESSAGE_APPLICATION) {
    gstpp_msg = new GstppMessage(msg);
  } else {
    auto structure = gst_message_get_structure(msg);
    guint64 ptr = 0;
    gst_structure_get_uint64(structure, "ptr", &ptr);
    gstpp_msg = reinterpret_cast<GstppMessage*>(ptr);
  }
  CHECK(gstpp_msg);
  LOG(INFO) << "get message "
            << gst_message_type_get_name(
                   static_cast<GstMessageType>(gstpp_msg->type()));
  for (auto&& cb : self->msg_cbs_) {
    cb(self, gstpp_msg);
  }
  auto it = self->typed_msg_cb_map_.find(gstpp_msg->type());
  if (it != self->typed_msg_cb_map_.end() && it->second) {
    it->second(self, gstpp_msg);
  }
  delete gstpp_msg;
  return true;
}

/* static*/ void GstppBus::SyncHandlerCallback(GstBus* bus, GstMessage* msg,
                                               GstppBus* self) {
  GstppMessage gstpp_msg(msg);
  for (auto&& cb : self->sync_handler_cbs_) {
    cb(self, &gstpp_msg);
  }
}

}  // namespace gstpp
}  // namespace grd