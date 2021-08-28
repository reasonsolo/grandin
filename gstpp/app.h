// author: zlz
#include <gst/gst.h>
#include <glib.h>

#include <functional>

#include "gstpp/element.h"
#include "gstpp/main_loop.h"
#include "gstpp/pipeline.h"

namespace grd {
namespace gstpp {

class GstppApp;
using GstppAppFunctor = std::function<void(GstppApp*)>;

class GstppApp {
 public:
  GstppApp(const std::string& name)
      : name_(name),
        pipeline_(new GstppPipeline(name_ + "-pipeline")),
        bus_(pipeline_->bus()) {}
  virtual ~GstppApp() {
    pipeline_->Reset();
    delete pipeline_;
  }

  virtual bool Init() = 0;

  void Start() {
    loop_thread_ = std::thread([this] { this->Run(); });
  }

  void Stop() {
    if (loop_thread_.joinable()) {
      RunInLoop([](auto* app) { app->Quit(); });
      loop_thread_.join();
    } else {
      Quit();
    }
  }

  virtual void Run() {
    pipeline_->Play();
    main_loop_.Run();
  }
  virtual void Quit() {
    pipeline_->Reset();
    main_loop_.Quit();
  };

  void RunInLoop(GstppAppFunctor functor) {
    if (!functor) return;
    bus()->PostApplicationMessage([this, functor = std::move(functor)](
                                      auto bus, auto msg) { functor(this); });
  }

  void RunInMs(GstppAppFunctor functor, int32_t ms) {
    auto task = new TimerTask;
    task->functor = std::move(functor);
    task->app = this;
    g_timeout_add(ms, &GstppApp::TimerCallback, task);
  }

  GstppBus* bus() { return bus_; }
  GstppPipeline* pipeline() { return pipeline_; }

 protected:

  static gboolean TimerCallback(gpointer data) {
    auto task = reinterpret_cast<TimerTask*>(data);
    if (task && task->functor) {
      task->functor(task->app);
      delete task;
    }
    return true;
  }

  struct TimerTask {
    GstppApp* app;
    GstppAppFunctor functor;
  };

  std::string name_;
  GstppMainLoop main_loop_;
  GstppPipeline* pipeline_ = nullptr;
  GstppBus* bus_ = nullptr;

  std::thread loop_thread_;
};
}  // namespace gstpp
}  // namespace grd