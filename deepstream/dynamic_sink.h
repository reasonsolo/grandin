// author: zlz
#pragma once

#include <atomic>
#include <string>
#include <vector>

#include "gstpp/app.h"
#include "gstpp/element.h"
#include "gstpp/pipeline.h"

namespace grd {
namespace deepstream {

using ::grd::gstpp::GstppApp;
using ::grd::gstpp::GstppPipeline;
using ::grd::gstpp::GstppElement;
using ::grd::gstpp::GstppPad;

class SinkBranch {
 public:
  SinkBranch(GstppPipeline* pipeline, GstppElement* upper);
  ~SinkBranch();

  bool Start(const std::string& save_path);
  bool Stop();

  bool is_sinking() const { return is_sinking_; }

 private:

  int32_t sink_id_ = 0;
  std::string src_pad_name_;
  std::atomic<bool> is_sinking_{false};

  GstppPipeline* pipeline_ = nullptr;
  GstppElement* upper_ = nullptr;

  GstppElement queue_ ;
  GstppElement encoder_;
  GstppElement muxer_;
  GstppElement filesink_;

  static std::atomic<int32_t> sink_id_counter_;
};

class DynamicSink {
 public:
 DynamicSink(GstppApp* app, size_t max_branch_num);
 ~DynamicSink();

 GstppElement* tee() { return tee_; }
 GstppElement* demux() { return &nvstreamdemux_; }

 // both methods should be invoked in gloop
 bool SaveBranch(int32_t branch, const std::string& user, const std::string& uid, int64_t max_ms);
 bool StopSaveBranch(int32_t branch);

private:
  GstppApp* app_ = nullptr;
  GstppElement* tee_ = nullptr;
  GstppElement nvstreamdemux_;
  GstppPad* tee_pad_;
  GstppPad* demux_pad_;

  std::vector<SinkBranch*> sink_branches_;
};

}
}