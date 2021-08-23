// author: zlz
#include "gstpp/message.h"
#include <gst/gst.h>

namespace grd {
namespace gstpp {

GstppMessage::GstppMessage(GstMessage* msg)
    : msg_(msg), type_(static_cast<MessageType>(GST_MESSAGE_TYPE(msg))) {
  gst_message_ref(msg_);
}

GstppMessage::~GstppMessage() {
  gst_message_unref(msg_);
}

GstppMessage::GstppMessage(GstObject* obj, std::any data)
    : msg_(gst_message_new_application(
          obj, gst_structure_new("gstppmessage", "ptr", G_TYPE_UINT64,
                                 reinterpret_cast<guint64>(this), nullptr))),
      type_(MessageType::APPLICATION),
      data_(data) {
  gst_message_ref(msg_);
}

std::string GstppMessage::AsError() {
  GError* error;
  gchar* debug;
  gst_message_parse_error(msg_, &error, &debug);
  std::string error_msg(error->message);
  g_error_free(error);
  g_free(debug);
  return error_msg;
}

int32_t GstppMessage::AsBufferingPercent() {
  gint percent = 0;
  gst_message_parse_buffering(msg_, &percent);
  return percent;
}

}  // namespace gstpp
}  // namespace grd