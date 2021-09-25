// author: zlz

#include "deepstream/dynamic_sink.h"

#include "gstpp/pad.h"
#include "common/utils.h"

#include <fmt/format.h>
#include <gflags/gflags.h>

DEFINE_string(base_save_dir, "/tmp/saved_files", "save bas dir");

namespace grd {
namespace deepstream {

std::atomic<int32_t> SinkBranch::sink_id_counter_{0};

SinkBranch::SinkBranch(GstppPipeline* p, GstppElement* upper)
    : sink_id_(sink_id_counter_++),
      src_pad_name_(fmt::format("src_{}", sink_id_)),
      pipeline_(p),
      upper_(upper),
      queue_("queue", fmt::format("queue_{}", sink_id_)),
      encoder_("x264enc", fmt::format("x264enc_{}", sink_id_)),
      muxer_("mp4mux", fmt::format("mp4mux_{}", sink_id_)),
      filesink_("filesink", fmt::format("filesink_{}", sink_id_)) {

    pipeline_->Add(queue_, encoder_, muxer_, filesink_);
    queue_ --> encoder_ --> muxer_ --> filesink_;
}

SinkBranch::~SinkBranch() {
  Stop();
  pipeline_->RemoveElement(queue_);
  pipeline_->RemoveElement(encoder_);
  pipeline_->RemoveElement(muxer_);
  pipeline_->RemoveElement(filesink_);
}

bool SinkBranch::Start(const std::string& save_path) {
  CHECK(!is_sinking_) << sink_id_ << " is sinking ";
  LOG(INFO) << "save to " << save_path;
  filesink_.SetProperty("location", save_path);
  auto src_pad = upper_->GetStaticPad(src_pad_name_);
  auto sink_pad = queue_.GetStaticPad("sink");

  // FIXME(zlz) : this won't work
  (*src_pad) -->-- (*sink_pad);
  queue_.SyncStateWithParent();
  encoder_.SyncStateWithParent();
  muxer_.SyncStateWithParent();
  filesink_.SyncStateWithParent();
  is_sinking_ = true;

  return true;
}

bool SinkBranch::Stop() {
  if (!is_sinking_) {
    LOG(ERROR) << "sink branch " << sink_id_ << " is not sinking";
    return false;
  }
  if (!queue_.Reset()) {
      LOG(ERROR) << "cannot reset state of sink branch " << sink_id_;
      return false;
  }
  auto src_pad = upper_->GetStaticPad(src_pad_name_);
  auto sink_pad = queue_.GetRequestPad("sink");

  (*src_pad) --/-- (*sink_pad);
  is_sinking_ = false;
  return true;
}

DynamicSink::DynamicSink(GstppApp* app, size_t max_branch_num)
    : app_(app),
      tee_(app->output_tee()),
      nvstreamdemux_("nvstreamdemux", "nvstreamdemux") {
  CHECK(tee_) << "cannot get output tee for app";
  GstPadTemplate* pad_templ = gst_element_class_get_pad_template(GST_ELEMENT_GET_CLASS(tee_->element()), "src_%u");
  tee_pad_ = new GstppPad(gst_element_request_pad(tee_->element(), pad_templ, nullptr, nullptr));
  demux_pad_ = nvstreamdemux_.GetStaticPad("sink");
  (*tee_pad_) --> (*demux_pad_);
  app_->pipeline()->AddElement(nvstreamdemux_);

  gst_object_unref(pad_templ);

  for (size_t i = 0; i < max_branch_num; i++) {
    auto pad = nvstreamdemux_.GetRequestPad(fmt::format("src_{}", i));
    delete pad;
    sink_branches_.push_back(new SinkBranch(app->pipeline(), &nvstreamdemux_));
  }
}

DynamicSink::~DynamicSink() {
  (*tee_pad_) --/-- (*demux_pad_);
  app_->pipeline()->RemoveElement(nvstreamdemux_);
  delete tee_pad_;
  delete demux_pad_;

  for (auto branch : sink_branches_) {
    delete branch;
  }
}

bool DynamicSink::SaveBranch(int32_t branch, const std::string& sub_dir,
                             const std::string& uid, int64_t max_ms) {
  CHECK(branch < sink_branches_.size())
      << "cannot sink branch " << branch << " max " << sink_branches_.size();
  std::string full_dir_path = fmt::format("{}/{}", FLAGS_base_save_dir, sub_dir);
  if (!SysUtils::MkDirP(full_dir_path)) {
    LOG(ERROR) << "cannot mkdir " << full_dir_path;
    return false;
  }
  std::string full_save_path = fmt::format("{}/{}.mp4", full_dir_path, uid);
  auto sink_branch = sink_branches_[branch];
  bool ret = sink_branch->Start(full_save_path);
  if (ret) {
    LOG(INFO) << "start sink branch " << branch << " uid " << uid << " to "
              << full_save_path;
    app_->RunInMs([sink_branch](GstppApp* app) { sink_branch->Stop(); },
                  max_ms);
  } else {
    LOG(INFO) << "cannot sink branch " << branch << " uid " << uid;
  }
  return ret;
}

bool DynamicSink::StopSaveBranch(int32_t branch) {
  CHECK(branch < sink_branches_.size())
      << "cannot stop branch " << branch << " max " << sink_branches_.size();
  return sink_branches_[branch]->Stop();
}

}  // namespace deepstream
}  // namespace grd