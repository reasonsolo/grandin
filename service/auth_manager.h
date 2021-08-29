// author: zlz

#pragma once

#include <string>
#include <unordered_map>

#include "workflow/HttpMessage.h"
#include "workflow/WFHttpServer.h"

#include "common/utils.h"

namespace grd {
namespace service {

class AuthManager {
 public:
  AuthManager() = default;
  ~AuthManager() = default;

  void Init();

  bool Authorize(QueryMap* qmap);

  bool Authorize(const std::string& nonce, const std::string& timestamp,
                 const std::string& user, const std::string& sha1);

  static AuthManager& GetInstance() { 
      static AuthManager instance;
      return instance;
  }
 private:

  void InitAppIds();

  std::unordered_map<std::string, std::string> user_appid_map_;

};
}
}