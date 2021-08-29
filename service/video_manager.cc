// author: zlz
#include "service/video_manager.h"

namespace grd {
namespace service {

void VideoManager::Init() {}

void VideoManager::ProcessNewVideoReq(WFHttpTask* t, QueryMap* qmap) {
  json::json json_resp;
  json_resp["code"] = 0;
  HttpUtils::RespondJson(t, &json_resp);
}

void VideoManager::ProcessQueryVideoReq(WFHttpTask* t, QueryMap* qmap) {
  json::json json_resp;
  json_resp["code"] = 0;
  HttpUtils::RespondJson(t, &json_resp);
}

void VideoManager::ProcessDelVideoReq(WFHttpTask* t, QueryMap* qmap) {
  json::json json_resp;
  json_resp["code"] = 0;
  HttpUtils::RespondJson(t, &json_resp);
}

}  // namespace service
}  // namespace grd