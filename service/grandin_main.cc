// author: zlz

#include <gflags/gflags.h>

#include "common/utils.h"
#include "service/http_router.h"
#include "service/auth_manager.h"
#include "service/app_manager.h"
#include "service/video_manager.h"

#include "workflow/HttpMessage.h"
#include "workflow/WFFacilities.h"
#include "workflow/WFHttpServer.h"


DEFINE_int32(port, 8001, "port");

using grd::HttpRequestInfo;
using grd::service::HttpRouter;
using grd::service::AuthManager;
using grd::service::AppManager;
using grd::service::VideoManager;

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, false);

  // init modules
  AuthManager::GetInstance().Init();
  AppManager::GetInstance().Init();
  VideoManager::GetInstance().Init();

  // init routings
  HttpRouter router;
  router.Route("/video/new", [](WFHttpTask* t, HttpRequestInfo* req_info) {
      return VideoManager::GetInstance().ProcessNewVideoReq(t, req_info);
  });
  router.Route("/video/query", [](WFHttpTask* t, HttpRequestInfo* req_info) {
      return VideoManager::GetInstance().ProcessQueryVideoReq(t, req_info);
  });
  router.Route("/video/del", [](WFHttpTask* t, HttpRequestInfo* req_info) {
      return VideoManager::GetInstance().ProcessDelVideoReq(t, req_info);
  });

  // start serving
  WFHttpServer server([&router](WFHttpTask* t) {
      router.Process(t);
  });
  server.start(FLAGS_port);
  server.wait_finish();
  return 0;
}
