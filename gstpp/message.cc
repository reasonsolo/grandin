// author: zlz
#include "gstpp/message.h"
#include <gst/gst.h>

namespace grd {
namespace gstpp {

GstppMessage::GstppMessage(GstMessage* msg)
    : msg_(msg), type_(static_cast<GstppMessageType>(GST_MESSAGE_TYPE(msg))) {
    
}

GstppMessage::~GstppMessage() {
  // pass
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