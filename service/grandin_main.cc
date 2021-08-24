// author: zlz

#include <gflags/gflags.h>

#include "service/http_router.h"
#include "service/auth_manager.h"

#include "workflow/HttpMessage.h"
#include "workflow/WFFacilities.h"
#include "workflow/WFHttpServer.h"


DEFINE_int32(port, 8001, "port");

using grd::service::HttpRouter;
using grd::service::AuthManager;

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, false);

  // init modules
  AuthManager::GetInstance().Init();

  // init routings
  HttpRouter router;
  router.Route("/auth", [](WFHttpTask* t) {
    AuthManager::GetInstance().ProcessAuthRequest(t);
  });
  router.Route("/video/new", [](WFHttpTask* t) {
      // TODO
  });
  router.Route("/video/query", [](WFHttpTask* t) {
      // TODO
  });
  router.Route("/video/del", [](WFHttpTask* t) {
      // TODO
  });

  // start serving
  WFHttpServer server([&router](WFHttpTask* t) {
      router.Process(t);
  });
  server.start(FLAGS_port);
  server.wait_finish();
  return 0;
}