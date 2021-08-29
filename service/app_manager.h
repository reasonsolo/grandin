// author: zlz
#ifndef GRANDIN_SERVICE_APP_MANAGER_H_
#define GRANDIN_SERVICE_APP_MANAGER_H_

#include <map>
#include <string>

#include "common/macros.h"

namespace grd {
namespace gstpp {
class GstppApp;
}

namespace service {
class AppManager {
  SINGLETON(AppManager);
 public:
  ~AppManager() = default;

  void Init() {}

  gstpp::GstppApp* GetApp(const std::string& name) { return nullptr; }

 private:
  std::map<std::string, gstpp::GstppApp*> app_map_;
};
}  // namespace service
}  // namespace grd

#endif 