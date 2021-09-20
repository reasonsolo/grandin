// author: zlz
#include "service/video_manager.h"
#include "service/app_manager.h"
#include "gstpp/app.h"
#include "common/monitor.h"
#include "service/uri_params.h"
#include <gflags/gflags.h>

DEFINE_int32(max_src_ms, 3 * 1000, "max real-world microseconds that a source can stay in pipleline");

namespace grd {
namespace service {

void VideoManager::Init() {}

void VideoManager::ProcessNewVideoReq(WFHttpTask* t, HttpRequestInfo* req_info) {
  MONITOR_COUNTER("video.new_qps", 1);
  auto&& qmap = req_info->query_map;
  std::string app_name = qmap.count(kAppParam) > 0 ? qmap[kAppParam] : kDefaultApp;
  auto app = AppManager::GetInstance().GetApp(app_name);
  if (app == nullptr) {
    json::json resp;
    resp["code"] = static_cast<int>(RespCode::APP_ERROR);
    resp["msg"] = "invalid app";
    return HttpUtils::RespondJson(t, resp);
  } else {
    VideoInput* input = CreateVideoInput(qmap);
    LOG(INFO) << "video input created";
    {
      Lock lock(mtx_);
      video_map_[input->uid] = input;
    }
    auto respond_later = HttpUtils::RespondLater(t, input->uid);
    app->AddSource(
        input->uid, input->uri,
        /* start callback =  */
        [input, t, respond_later, this](bool success) {
          if (success) {
            MONITOR_COUNTER(std::format("video.{}.new_success", input->app_name), 1);
            input->status = VideoStatus::INPROGRESS;
          } else {
            MONITOR_COUNTER(std::format("video.{}.new_error", input->app_name), 1);
            input->status = VideoStatus::ERROR;
            input->error_msg = "error on start";
          }
          LOG(INFO) << "add source " << input->uri << " " << success;
          json::json resp;
          json::json data;
          data["id"] = input->uid;
          resp["code"] = static_cast<int>(input->status);
          resp["msg"] = input->error_msg;
          resp["data"] = data;
          HttpUtils::RespondJson(t, resp);
          WFTaskFactory::count_by_name(input->uid);
        },
        /* stop callback = */
        [input, t, respond_later, this](bool success) {
          if (success) {
            MONITOR_COUNTER(
                std::format("video.{}.finish_success", input->app_name), 1);
            input->status = VideoStatus::STOPPED;
          } else {
            MONITOR_COUNTER(
                std::format("video.{}.finish_erro", input->app_name), 1);
            input->status = VideoStatus::ERROR;
            input->error_msg = "error on finish";
          }
        });
    app->RemoveSourceWithTimeout(
        input->uid,
        [input](bool success) {
          LOG(INFO) << input->uid << " was removed from pipeline, ret " << success;
          input->status = success ? VideoStatus::STOPPED : VideoStatus::ERROR;
        },
        FLAGS_max_src_ms);
    //json::json resp;
    //resp["code"] = VideoStatus::INPROGRESS;
    //resp["id"] = input->uid;
    //HttpUtils::TimedRespond(t, 1000 * 5, resp, input->uid);
  }
}

void VideoManager::ProcessQueryVideoReq(WFHttpTask* t, HttpRequestInfo* req_info) {
  auto& qmap = req_info->query_map;
  std::string& uid = qmap["id"];
  json::json json_resp;
  json::json data;
  json_resp["code"] = 0;
  MONITOR_COUNTER("video.query_qps", 1);
  {
    Lock lock(mtx_);
    auto it = video_map_.find(uid);
    if (it == video_map_.end() || it->second == nullptr) {
      data["status"] = static_cast<int>(VideoStatus::NOTEXIST);
    } else {
      data["status"] = static_cast<int>(it->second->status);
    }
  }
  json_resp["data"] = data;
  HttpUtils::RespondJson(t, json_resp);
}

void VideoManager::ProcessDelVideoReq(WFHttpTask* t, HttpRequestInfo* req_info) {
  MONITOR_COUNTER("video.del_qps", 1);
  json::json json_resp;
  json_resp["code"] = 0;
  HttpUtils::RespondJson(t, json_resp);
}

/* static */
VideoInput* VideoManager::CreateVideoInput(QueryMap& qmap) {
  auto video_input = new VideoInput;
  video_input->uid = UniqueIdUtils::GenUniqueId();
  video_input->uri = qmap[kUri];
  video_input->user = qmap[kUser];
  return video_input;
}

}  // namespace service
}  // namespace grd