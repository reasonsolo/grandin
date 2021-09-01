// author: zlz
#pragma once

#include <glog/logging.h>
#include <net/if.h>
#include <netinet/in.h>
#include <openssl/sha.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <iomanip>
#include <mutex>

#include "workflow/WFHttpServer.h"
#include "workflow/WFTaskFactory.h"
#include "vendor/fmt/include/fmt/format.h"
#include "vendor/json/single_include/nlohmann/json.hpp"

#define gettid() syscall(SYS_gettid)

namespace grd {

class INetUtils {
 public:
  static std::string GetMacAddress() {
    struct ifreq ifr;
    struct ifconf ifc;
    char buf[1024];
    int success = 0;

    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (sock == -1) {
      return "";
    };

    ifc.ifc_len = sizeof(buf);
    ifc.ifc_buf = buf;
    if (ioctl(sock, SIOCGIFCONF, &ifc) == -1) {
      close(sock);
      return "";
    }

    struct ifreq* it = ifc.ifc_req;
    const struct ifreq* const end = it + (ifc.ifc_len / sizeof(struct ifreq));

    for (; it != end; ++it) {
      strcpy(ifr.ifr_name, it->ifr_name);
      if (ioctl(sock, SIOCGIFFLAGS, &ifr) == 0) {
        if (!(ifr.ifr_flags & IFF_LOOPBACK)) {  // don't count loopback
          if (ioctl(sock, SIOCGIFHWADDR, &ifr) == 0) {
            success = 1;
            break;
          }
        }
      }
    }

    std::string mac_address(6, 0);
    if (success) memcpy(&(mac_address[0]), ifr.ifr_hwaddr.sa_data, 6);
    close(sock);
    return mac_address;
  }
};

class TimeUtils {
 public:
  static int64_t NowUs() {
    struct timeval now;
    gettimeofday(&now, nullptr);
    return now.tv_sec * 1000 * 1000 + now.tv_usec;
  }
};

class UniqueIdUtils {
 public:
  // return unique 32 bytes string
  static std::string GenUniqueId() {
    static std::string mac = INetUtils::GetMacAddress();
    static int64_t mac_hash = std::hash<std::string>{}(mac);
    static std::atomic<int64_t> seq_num{0};
    static thread_local std::mt19937 gen;
    static int32_t tid = static_cast<int32_t>(gettid());

    int64_t seq_id = seq_num++;
    int64_t now_ms = TimeUtils::NowUs() / 1000;
    int32_t random_id = gen();

    return fmt::format(
        "{:08x}{:04x}{:04x}{:04x}{:08x}", static_cast<uint32_t>(now_ms),
        static_cast<uint16_t>(mac_hash), static_cast<uint16_t>(seq_id),
        static_cast<uint16_t>(tid), static_cast<uint32_t>(random_id));
  }
};

class StringUtils {
public:
 static std::vector<std::string> Split(const std::string& input, char delim=' ') {
   std::vector<std::string> result;

   if (input.empty()) return result;

   size_t prev_pos = 0;
   size_t pos;
   std::string token;
   while ((pos = input.find(delim, prev_pos)) != std::string::npos) {
     token.assign(input.data() + prev_pos, pos - prev_pos);

     result.push_back(token);
     prev_pos = pos + 1;
   }

   token.assign(input.data() + prev_pos, input.length() - prev_pos);
   result.push_back(token);
   return result;
 }
};

class CryptoUtils {
 public:
  static std::string CalcSha1(const std::string& input) {
  std::string buf(SHA_DIGEST_LENGTH, 0);
  SHA1(reinterpret_cast<const unsigned char*>(input.data()), input.size(), reinterpret_cast<unsigned char*>(&buf[0]));
  std::string ret;
  ret.reserve(SHA_DIGEST_LENGTH * 2);
  for (size_t i = 0; i < buf.size(); i++) {
    ret.append(fmt::format("{:02x}", (unsigned char)(buf[i])));
  }
  return ret;
  }
};

using QueryMap = std::unordered_map<std::string, std::string>;
struct HttpRequestInfo {
  std::string path;
  QueryMap query_map;
};

namespace json = nlohmann;
class HttpUtils {
 public:
  static char FromHex(char ch) {
    return std::isdigit(ch) ? ch - '0' : std::tolower(ch) - 'a' + 10;
}

static std::string UrlDecode(const std::string& text) {
  char h;
  std::ostringstream escaped;
  escaped.fill('0');

  for (auto i = text.begin(), n = text.end(); i != n; ++i) {
    std::string::value_type c = (*i);
    if (c == '%') {
      if (i[1] && i[2]) {
        h = FromHex(i[1]) << 4 | FromHex(i[2]);
        escaped << h;
        i += 2;
      }
    } else if (c == '+') {
      escaped << ' ';
    } else {
      escaped << c;
    }
  }

  return escaped.str();
}

  static bool ParseUri(const std::string& uri, std::string* path, QueryMap* qmap) {
    auto path_begin = uri.find('/');
    auto param_begin = uri.find('?');
    *path = uri.substr(path_begin, param_begin);
    qmap->clear();
    if (param_begin != std::string::npos) {
      auto param_segs = StringUtils::Split(uri.substr(param_begin + 1), '&');
      for (auto&& seg : param_segs) {
        auto kv = StringUtils::Split(seg, '=');
        if (kv.size() == 2) {
          (*qmap)[kv[0]] = UrlDecode(kv[1]);
        } else if (kv.size() == 1) {
          (*qmap)[kv[0]] = "";
        }
      }
    }
    return true;
  }

  static void RespondJson(WFHttpTask* t, const json::json& resp_json) {
    auto resp = t->get_resp();
    resp->add_header_pair("content-type", "application/javascript");
    resp->set_status_code("200");
    std::string body = resp_json.dump(4);
    resp->append_output_body(static_cast<const void*>(body.data()), body.size());
  }

  static WFCounterTask* RespondLater(WFHttpTask* t) {
    auto counter_task = WFTaskFactory::create_counter_task(1, nullptr);
    *series_of(t)  << counter_task;
    return counter_task;
  }
};

using Mutex = std::mutex;
using Lock = std::unique_lock<Mutex>;


}  // namespace grd
