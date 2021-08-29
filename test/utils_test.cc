// author: zlz
#include "common/utils.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <thread>
#include <chrono>
#include <set>

using namespace grd;
using namespace std::chrono_literals;

TEST(UtilsTest, UniqueIdTest) {
  int iter = 16;
  std::set<std::string> uids;
  for (int i = 0; i < iter; i++) {
      std::this_thread::sleep_for(10ms);
    auto s = UniqueIdUtils::GenUniqueId();
    LOG(INFO) << s;
    uids.insert(s);
  }
  EXPECT_EQ(iter, uids.size());
}

TEST(UtilsTest, CryptoTest) {
  EXPECT_EQ(std::string("7e240de74fb1ed08fa08d38063f6a6a91462a815"), CryptoUtils::CalcSha1("aaa"));
  EXPECT_EQ(std::string("40bd001563085fc35165329ea1ff5c5ecbdbbeef"), CryptoUtils::CalcSha1("123"));
}

TEST(UtilsTest, HttpTest) {
  {
    std::string path;
    QueryMap query_map;
    EXPECT_TRUE(HttpUtils::ParseUri("/test?asd=1", &path, &query_map));
    EXPECT_EQ(std::string("/test"), path);
    EXPECT_EQ(std::string("1"), query_map["asd"]);
  }
  {
    std::string path;
    QueryMap query_map;
    EXPECT_TRUE(HttpUtils::ParseUri("/test/123?asd=1&qqq=2", &path, &query_map));
    EXPECT_EQ(std::string("/test/123"), path);
    EXPECT_EQ(std::string("1"), query_map["asd"]);
    EXPECT_EQ(std::string("2"), query_map["qqq"]);
  }
  {
    std::string path;
    QueryMap query_map;
    EXPECT_TRUE(HttpUtils::ParseUri("/test/123?", &path, &query_map));
    EXPECT_EQ(std::string("/test/123"), path);
    EXPECT_TRUE(query_map.empty());
  }
  {
    std::string path;
    QueryMap query_map;
    EXPECT_TRUE(HttpUtils::ParseUri("/", &path, &query_map));
    EXPECT_EQ(std::string("/"), path);
    EXPECT_TRUE(query_map.empty());
  }
}