// author: zlz

#pragma once 

#include <string>
#include <functional>

#include <glog/logging.h>

#include "workflow/HttpMessage.h"
#include "workflow/WFHttpServer.h"
#include "workflow/WFTaskFactory.h"

namespace grd {
namespace service {

using HttpHandler = std::function<void(WFHttpTask*)>;

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
      resp->set_status_code("404");
      resp->append_output_body("<html>404 Not Found.</html>");
    } else {
      it->second(task);
    }
  }

 private:
  std::unordered_map<std::string, HttpHandler> router_map_;
};

}
}