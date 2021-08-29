// author: zlz
#include "service/video_manager.h"
#include "service/app_manager.h"
#include "gstpp/app.h"
#include "common/monitor.h"
#include "service/uri_params.h"

namespace grd {
namespace service {

void VideoManager::Init() {}

void VideoManager::ProcessNewVideoReq(WFHttpTask* t, QueryMap* qmap) {
  MONITOR_COUNTER("video.new_qps", 1);
  std::string app_name = qmap->count(kAppParam) > 0 ? (*qmap)[kAppParam] : kDefaultApp;
  auto app = AppManager::GetInstance().GetApp(app_name);
  if (app == nullptr) {
    json::json resp;
    resp["code"] = static_cast<int>(RespCode::APP_ERROR);
    resp["msg"] = "invalid app";
    return HttpUtils::RespondJson(t, resp);
  } else {
    VideoInput* input = CreateVideoInput(qmap);
    {
      Lock lock(mtx_);
      video_map_[input->uid] = input;
    }
    auto respond_later = HttpUtils::RespondLater(t);
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
          json::json resp;
          json::json data;
          data["id"] = input->uid;
          resp["code"] = static_cast<int>(input->status);
          resp["msg"] = input->error_msg;
          resp["data"] = data;
          HttpUtils::RespondJson(t, resp);
          respond_later->count();
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
  }
  json::json json_resp;
  json_resp["code"] = 0;
  HttpUtils::RespondJson(t, json_resp);
}

void VideoManager::ProcessQueryVideoReq(WFHttpTask* t, QueryMap* qmap) {
  std::string uid = (*qmap)["id"];
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

void VideoManager::ProcessDelVideoReq(WFHttpTask* t, QueryMap* qmap) {
  MONITOR_COUNTER("video.del_qps", 1);
  json::json json_resp;
  json_resp["code"] = 0;
  HttpUtils::RespondJson(t, json_resp);
}

/* static */
VideoInput* VideoManager::CreateVideoInput(QueryMap* qmap) {
    auto video_input = new VideoInput;
    video_input->uid = UniqueIdUtils::GenUniqueId();
    video_input->uri = (*qmap)[kUri];
    video_input->user = (*qmap)[kUser];
    return video_input;
}

}  // namespace service
}  // namespace grd