// author: zlz
// from https://gstreamer.freedesktop.org/documentation/tutorials/basic/streaming.html?gi-language=c
#include "gstpp/element.h"
#include "gstpp/pipeline.h"
#include "gstpp/bus.h"
#include "gstpp/main_loop.h"

using namespace grd::gstpp;
int main() {
  gst_init(nullptr, nullptr);
  auto pipeline = GstppPipeline::LaunchFrom(
      "playbin "
      "uri=https://www.freedesktop.org/software/gstreamer-sdk/data/media/sintel_trailer-480p.webm",
      "pipeline");
  CHECK(pipeline);
  auto bus = pipeline->bus();
  auto main_loop = new GstppMainLoop();

  //pipeline->Play();

  bus->AddMessageCallback([&pipeline, &main_loop](GstppBus* bus, GstppMessage* msg) {
    LOG(INFO) << "get message " << GST_MESSAGE_TYPE_NAME(msg->msg());
    switch (msg->type()) {
      case MessageType::ERROR: {
        LOG(ERROR) << "err msg: " << msg->AsError();
        break;
      }
      case MessageType::EOS: {
        LOG(INFO) << "EOS, stop loop";
        pipeline->Reset();
        main_loop->Quit();
        break;
      }
      case MessageType::BUFFERING: {
        int32_t percent = msg->AsBufferingPercent();
        LOG(INFO) << "buffering " << percent << "%";
        if (percent < 100) {
          pipeline->Pause();
        } else {
          pipeline->Play();
        }
        break;
      }
      case MessageType::CLOCK_LOST: {
        pipeline->Pause();
        pipeline->Play();
        break;
      }
      default: {
          break;
      }
    }
  });
  bus->WatchAndSignalConnect();
  pipeline->Play();
  main_loop->Run();

  pipeline->Reset();

  delete main_loop;
  delete pipeline;

  return 0;
}