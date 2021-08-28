// author: zlz
#ifndef GRANDIN_SERVICE_APP_MANAGER_H_
#define GRANDIN_SERVICE_APP_MANAGER_H_ 

#include <map>
#include <string>

namespace grd {
namespace gstpp {
class GstppApp;
}

namespace service {
class AppManager {
 public:
  ~AppManager();

  void Init();

  static gstpp::GstppApp* GetApp();
  static AppManager& GetInstance();

 private:
  AppManager();

  std::map<std::string, gstpp::GstppApp*> app_map_;
};
}  // namespace service
}  // namespace grd

#endif 