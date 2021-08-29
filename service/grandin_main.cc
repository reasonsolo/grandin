// author: zlz

#include <gflags/gflags.h>

#include "service/http_router.h"
#include "service/auth_manager.h"
#include "service/app_manager.h"
#include "service/video_manager.h"

#include "workflow/HttpMessage.h"
#include "workflow/WFFacilities.h"
#include "workflow/WFHttpServer.h"


DEFINE_int32(port, 8001, "port");

using grd::QueryMap;
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
  router.Route("/video/new", [](WFHttpTask* t, QueryMap* query_map) {
      return VideoManager::GetInstance().ProcessNewVideoReq(t, query_map);
  });
  router.Route("/video/query", [](WFHttpTask* t, QueryMap* query_map) {
      return VideoManager::GetInstance().ProcessQueryVideoReq(t, query_map);
  });
  router.Route("/video/del", [](WFHttpTask* t, QueryMap* query_map) {
      return VideoManager::GetInstance().ProcessDelVideoReq(t, query_map);
  });

  // start serving
  WFHttpServer server([&router](WFHttpTask* t) {
      router.Process(t);
  });
  server.start(FLAGS_port);
  server.wait_finish();
  return 0;
}
