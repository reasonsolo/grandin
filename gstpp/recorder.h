// author: zlz
#pragma once

#include "gst/gst.h"
#include "gstpp/element.h"
#include "gstpp/app.h"
#include "common/macros.h"

namespace grd {
namespace gstpp {

class GstppRecorder {
  NOCOPY(GstppRecorder);

 public:
  GstppRecorder(GstppApp* app, const std::string& file_path);
  ~GstppRecorder();

  void Start(int32_t max_ms);

 private:
  void Stop();

  GstppApp* app_ = nullptr;
  GstppElement* tee_ = nullptr;
  GstppElement* queue_ = nullptr;
  GstppElement* encoder_ = nullptr;
  GstppElement* muxer_ = nullptr;
  GstppElement* filesink_ = nullptr;

  std::string file_path_;
  int64_t start_record_ms_ = 0;
  int64_t max_record_ms_ = 0;
  bool is_recording_ = true;

};

}  // namespace gstpp
}  // namespace grd