// author: zlz
#pragma once

#include <cuda_runtime_api.h>
#include <gflags/gflags.h>
#include <glib.h>
#include <gst/gst.h>
#include <gstnvdsmeta.h>
#include <stdio.h>

#include <chrono>
#include <functional>
#include <thread>
#include <deque>

#include "gstpp/app.h"
#include "gstpp/element.h"
#include "gstpp/gtk_player.h"
#include "gstpp/main_loop.h"
#include "gstpp/pad.h"

namespace grd {
namespace deepstream {

using ::grd::gstpp::GstppApp;
using ::grd::gstpp::GstppElement;
using ::grd::gstpp::SrcStartCallback;
using ::grd::gstpp::SrcStopCallback;

class TestApp : public grd::gstpp::GstppApp {
 public:
  TestApp();
  virtual ~TestApp() {
    Stop();
    for (auto&& src : dynamic_source_list_) {
      delete src.srcbin;
    }
  }

  bool Init() override {
    InitPipeline();
    InitBus();
    return true;
  }

  void AddSource(const std::string& name, const std::string& uri,
                 SrcStartCallback start_cb, SrcStopCallback stop_cb) override {
    LOG(INFO) << "add source " << uri;
    gstpp::GstppApp::RunInLoop([=](gstpp::GstppApp* app) {
      this->AddSourceFromUri(name, uri, start_cb, stop_cb);
    });
  }
  void RemoveSource(const std::string& name, SrcStopCallback stop_cb) override {
    gstpp::GstppApp::RunInLoop([=](gstpp::GstppApp* app) {
      bool ret = this->RemoveSourceByName(name);
      if (stop_cb) stop_cb(ret);
    });
  }

 private:
  bool AddSourceFromUri(const std::string& name, const std::string& uri,
                        SrcStartCallback start_cb, SrcStopCallback stop_cb);
  bool RemoveSourceByName(const std::string& name);
  bool RemoveSourceByIdx(const int32_t idx);

  void InitBus();
  void InitPipeline();

  int32_t frame_number_ = 0;
  GstppElement source_;
  GstppElement h264parser_;
  GstppElement decoder_;
  GstppElement streammux_;
  GstppElement pgie_;
  GstppElement nvvidconv_;
  GstppElement nvosd_;
  GstppElement sink_;

  struct DynamicSource {
    GstppElement* srcbin;
    int32_t sink_slot;
    int64_t source_count;
    SrcStartCallback start_cb;
    SrcStopCallback stop_cb;
    int64_t create_timestamp;
  };

  std::deque<int32_t> available_sink_slots_;
  std::atomic<int64_t> source_count_{0};
  std::vector<DynamicSource> dynamic_source_list_;
  std::unordered_map<std::string, int32_t> dynamic_source_name_idx_map_;
};

}  // namespace deepstream
}  // namespace grd
