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
