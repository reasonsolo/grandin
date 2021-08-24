// author: zlz
#include <functional>

#include "gstpp/element.h"
#include "gstpp/pipeline.h"
#include "gstpp/main_loop.h"

namespace grd {
namespace gstpp {

class GstppApp {
 public:
  GstppApp(const std::string& name)
      : name_(name),
        pipeline_(new GstppPipeline(name_ + "{}-pipeline")),
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
        RunInLoop([](auto* app) {
            app->Quit();
        });
        loop_thread_.join();
      }
  }

  virtual void Run() { main_loop_.Run(); }
  virtual void Quit() { main_loop_.Quit(); };

  void RunInLoop(std::function<void(GstppApp*)> functor) {
      if (!functor) return;
      bus()->PostApplicationMessage([this, functor=std::move(functor)](auto bus, auto msg) {
          functor(this);
      });
  }

  GstppBus* bus() { return bus_; }
  GstppPipeline* pipeline() { return pipeline_; }

 protected:
  std::string name_;
  GstppMainLoop main_loop_;
  GstppPipeline* pipeline_ = nullptr;
  GstppBus* bus_ = nullptr;

  std::thread loop_thread_;
};
}
}