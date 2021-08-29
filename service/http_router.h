// author: zlz

#pragma once 

#include <string>
#include <functional>
#include <unordered_map>

#include <glog/logging.h>

#include "common/utils.h"
#include "service/auth_manager.h"

#include "workflow/HttpMessage.h"
#include "workflow/WFHttpServer.h"
#include "workflow/WFTaskFactory.h"

namespace grd {
namespace service {

using HttpHandler = std::function<void(WFHttpTask*, QueryMap*)>;

// TODO: do real routing
class HttpRouter {
 public:
  HttpRouter() = default;
  ~HttpRouter() = default;

  void Route(const std::string& pattern, HttpHandler handler) {
    CHECK(handler) << "invalid handler for " << pattern;
    router_map_[pattern] = std::move(handler);
  }

  void Process(WFHttpTask* task) {
    protocol::HttpRequest* req = task->get_req();
    protocol::HttpResponse* resp= task->get_resp();

    std::string uri = req->get_request_uri();
    LOG(INFO) << "Route uri: " << uri;
    auto it = router_map_.find(uri);
    if (it == router_map_.end()) {
      Respond404(resp);
    } else {
      auto query_map = HttpUtils::GetQueryMap(task->get_req());
      if (AuthManager::GetInstance().Authorize(&query_map)) {
        LOG(INFO) << "user " << query_map["user"] << " authorized";
        it->second(task, &query_map);
      } else {
        LOG(WARNING) << "user " << query_map["user"] << " failed authorization";
        Respond403(resp);
      }
    }
  }

 private:
  void HttpRespond(const char* code, const char* body,
                   protocol::HttpResponse* resp) {
    resp->set_status_code(code);
    resp->append_output_body("<html>");
    resp->append_output_body(body);
    resp->append_output_body("</html>");
  }

  void Respond404(protocol::HttpResponse* resp) {
    HttpRespond("404", "404 Not found.", resp);
  }

  void Respond403(protocol::HttpResponse* resp) {
    HttpRespond("403", "403 Not authorized.", resp);
  }

  std::unordered_map<std::string, HttpHandler> router_map_;
};

}  // namespace service
}  // namespace grd