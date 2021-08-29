// author: zlz

#pragma once

#include <string>
#include <unordered_map>

#include "workflow/HttpMessage.h"
#include "workflow/WFHttpServer.h"

#include "common/utils.h"
#include "common/macros.h"

namespace grd {
namespace service {

class AuthManager {
  SINGLETON(AuthManager);
 public:
  ~AuthManager() = default;

  void Init();

  bool Authorize(QueryMap* qmap);

  bool Authorize(const std::string& nonce, const std::string& timestamp,
                 const std::string& user, const std::string& sha1);

 private:
  void InitAppIds();

  std::unordered_map<std::string, std::string> user_appid_map_;

};
}
}