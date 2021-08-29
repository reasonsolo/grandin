// author: zlz
#pragma once

#include <string>
#include <tuple>
#include <unordered_map>

#include "common/macros.h"
#include "common/utils.h"
#include "workflow/HttpMessage.h"
#include "workflow/WFTaskFactory.h"

namespace grd {
namespace service {

enum class RespCode {
  OK = 0,
  APP_ERROR = 101,
};

enum class VideoStatus {
  PENDING = 0,
  INPROGRESS = 1,
  STOPPED = 2,
  ERROR = 3,
  NOTEXIST = 4,
};

struct VideoInput {
  std::string uri;
  std::string app_name;
  std::string user;
  std::string uid;
  VideoStatus status = VideoStatus::PENDING;
  std::string error_msg;

  void Play() { status = VideoStatus::INPROGRESS; }
  void Stop() { status = VideoStatus::STOPPED; }
  void Error(const std::string& msg) {
    status = VideoStatus::ERROR;
    error_msg = msg;
  }

};

class VideoManager {
  SINGLETON(VideoManager);

 public:
  ~VideoManager() = default;

  void Init();
  void ProcessNewVideoReq(WFHttpTask* t, QueryMap* query_map);
  void ProcessQueryVideoReq(WFHttpTask* t, QueryMap* query_map);
  void ProcessDelVideoReq(WFHttpTask* t, QueryMap* query_map);

  std::tuple<VideoStatus, std::string> AddVideoInput(std::string uri,
                                                     std::string app_name);
  VideoStatus QueryVideoInput(const std::string& uid);

 private:

  static VideoInput* CreateVideoInput(QueryMap* qmap);
  void RespondError(WFHttpTask* t, const RespCode code, const std::string& msg);

  Mutex mtx_;
  std::unordered_map<std::string, VideoInput*> video_map_;
};
}  // namespace service
}  // namespace grd
