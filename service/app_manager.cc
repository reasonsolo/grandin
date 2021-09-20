// author: zlz

#include "service/app_manager.h"
#include "service/auth_manager.h"

#include "deepstream/testapp.h"

namespace grd {
namespace service {

void AppManager::Init() {
    auto testapp = new ::grd::deepstream::TestApp;
    CHECK(testapp->Init());
    testapp->Start();
    app_map_["test"] = testapp;
}

}
}