// author: zlz
#pragma once

#include <cstdint>
#include <string>
#include <gst/gst.h>

namespace grd {
namespace gstpp {

#define __GST_MSG_TYPE(X) X = GST_MESSAGE_##X
enum class GstppMessageType {
  __GST_MSG_TYPE(ERROR),
  __GST_MSG_TYPE(EOS),
  __GST_MSG_TYPE(BUFFERING),
  __GST_MSG_TYPE(CLOCK_LOST),
  NONE,
};
#undef __GST_MSG_TYPE

class GstppMessage {
 public:
  GstppMessage(GstMessage* msg);
  ~GstppMessage();

  GstppMessageType type() const { return type_; }

  std::string AsError();
  int32_t AsBufferingPercent();

 private:
  GstMessage* msg_ = nullptr;
  GstppMessageType type_ = GstppMessageType::NONE;
};
}  // namespace gstpp
}  // namespace grd