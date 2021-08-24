// author: zlz

#pragma once

#include <string>
#include <set>

#include "workflow/HttpMessage.h"
#include "workflow/WFHttpServer.h"

namespace grd {
namespace service {

class AuthManager {
 public:
  AuthManager() = default;
  ~AuthManager() = default;

  void Init() {}

  void ProcessAuthRequest(WFHttpTask* task) {
    protocol::HttpResponse* resp = task->get_resp();
    resp->set_status_code("200");
    resp->append_output_body("<html>auth request</html>");
  }

  bool Authorize(const std::string& nonce, const std::string& timestamp, const std::string& sha1);

  static AuthManager& GetInstance() { 
      static AuthManager instance;
      return instance;
  }
 private:
  std::set<std::string> known_appids_;

};
}
}