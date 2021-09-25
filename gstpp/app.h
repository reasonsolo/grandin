// author: zlz
#pragma once
#include <gst/gst.h>
#include <glib.h>

#include <functional>
#include <string>
#include <thread>

#include "gstpp/element.h"
#include "gstpp/main_loop.h"
#include "gstpp/pipeline.h"

namespace grd {
namespace gstpp {

class GstppApp;
using GstppAppFunctor = std::function<void(GstppApp*)>;
using SrcStartCallback = std::function<void(bool)>;
using SrcStopCallback = std::function<void(bool)>;

class GstppApp {
 public:
  GstppApp(const std::string& name)
      : name_(name),
        pipeline_(new GstppPipeline(name_ + "-pipeline")),
        bus_(pipeline_->bus()) {}
  virtual ~GstppApp() {
    LOG(INFO) << "app exit "  << name_;
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

  virtual void AddSource(const std::string& name, const std::string& uri,
                         SrcStartCallback start_cb, SrcStopCallback stop_cb) {
    LOG(FATAL) << "not implemented";
  }
  virtual void RemoveSource(const std::string& name, SrcStopCallback stop_cb) {
    LOG(FATAL) << "not implemented";
  }
  virtual void RemoveSourceWithTimeout(const std::string& name, SrcStopCallback stop_cb, int32_t timeout_ms) {
    RunInMs([stop_cb=std::move(stop_cb), name=name, this](GstppApp* app) {
      this->RemoveSource(name, stop_cb);
    }, timeout_ms);
  }

  GstppBus* bus() { return bus_; }
  GstppPipeline* pipeline() { return pipeline_; }
  virtual GstppElement* output_tee() { return nullptr; }

 protected:

  static gboolean TimerCallback(gpointer data) {
    auto task = reinterpret_cast<TimerTask*>(data);
    LOG(INFO) << "run timer callback " << (void*)task;
    if (task && task->functor) {
      task->functor(task->app);
      delete task;
    }
    return false; // return false to prevent repeative timer
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