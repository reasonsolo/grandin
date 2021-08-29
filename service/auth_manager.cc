// author: zlz
#include "service/auth_manager.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <fstream>

#include "common/utils.h"
#include "vendor/fmt/include/fmt/format.h"

DEFINE_string(appid_file, "appid.txt", "appid file path");

namespace grd {
namespace service {

static const std::string kAppId = "appid";
static const std::string kNonce = "nonce";
static const std::string kUser = "user";
static const std::string kTimestamp = "timestamp";
static const std::string kSha1 = "sha1";

void AuthManager::Init() {
    InitAppIds();
}

void AuthManager::InitAppIds() {
    std::ifstream ifs(FLAGS_appid_file, std::ios_base::in);
    std::string line;
    int32_t line_count = 0;
    while (std::getline(ifs, line)) {
      line_count++;
      auto segs = StringUtils::Split(line);
      CHECK(segs.size() == 2 && !segs[0].empty() && !segs[1].empty())
          << "invalid line " << line_count;
      user_appid_map_[segs[0]]  = segs[1];
    }
    // TODO(zlz): remove this
    user_appid_map_[std::string("zlz")] = "123";
    LOG(INFO) << line_count << " user-appid pairs are loaded";
}

bool AuthManager::Authorize(const std::string& nonce,
                            const std::string& timestamp,
                            const std::string& user, const std::string& sha1) {
    auto it =user_appid_map_.find(user);
    if (it == user_appid_map_.end()) {
        LOG(WARNING) << "invalid user " << user;
        return false;
    }

    std::string calc_sha1 = CryptoUtils::CalcSha1(
        fmt::format("{}={}&{}={}&{}={}", kAppId, it->second, kNonce, nonce,
                    kTimestamp, timestamp));
    return calc_sha1 == sha1;
}

bool AuthManager::Authorize(QueryMap* qmap) {
  QueryMap& query_map = *qmap;
  if (!Authorize(query_map[kNonce], query_map[kTimestamp], query_map[kUser], query_map[kSha1])) {
    return false;
  } else {
    return true;
  }
}

}  // namespace service
}  // namespace grd