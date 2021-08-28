// author: zlz
#ifndef GRANDING_SERVICE_VIDEO_MANAGER_H_
#define GRANDING_SERVICE_VIDEO_MANAGER_H_

#include <tuple>
#include <string>
#include <unordered_map>

namespace grd {
namespace service {

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
 public:
  ~VideoManager() = default;

  std::tuple<VideoStatus, std::string> AddVideoInput(std::string uri, std::string app_name);
  VideoStatus QueryVideoInput(const std::string& uid);

  static VideoManager& GetInstance() ;

 private:
  VideoManager() = default;
  std::unordered_map<std::string, VideoInput> video_map_;
};
}
}  // namespace grd

#endif