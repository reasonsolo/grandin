// author: zlz
#pragma once

#include <cstdint>
#include <any>
#include <string>
#include <functional>
#include <gst/gst.h>
#include "gst/video/videooverlay.h"

namespace grd {
namespace gstpp {

#define __GST_MSG_TYPE(X) X = GST_MESSAGE_##X
enum class MessageType {
  __GST_MSG_TYPE(ERROR),
  __GST_MSG_TYPE(EOS),
  __GST_MSG_TYPE(BUFFERING),
  __GST_MSG_TYPE(CLOCK_LOST),
  __GST_MSG_TYPE(STATE_CHANGED),
  __GST_MSG_TYPE(ELEMENT),
  __GST_MSG_TYPE(APPLICATION),
  NONE,
};
#undef __GST_MSG_TYPE

class GstppMessage {
 public:
  GstppMessage(GstMessage* msg);
  GstppMessage(GstObject* obj, std::any data);
  ~GstppMessage();

  MessageType type() const { return type_; }
  GstMessage* msg() const { return msg_; }
  std::any& data() { return data_; }

  bool IsVideoOverlayPrepareWindowHandlerMessage() const {
    return gst_is_video_overlay_prepare_window_handle_message(msg_);
  }
  std::string AsError();
  int32_t AsBufferingPercent();


 private:
  GstMessage* msg_ = nullptr;
  MessageType type_ = MessageType::NONE;
  std::any data_;
};
}  // namespace gstpp
}  // namespace grd