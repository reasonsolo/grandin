// author: zlz

#include <functional>
#include <thread>
#include <chrono>

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>

#include "gstpp/app.h"
#include "gstpp/element.h"
#include "gstpp/main_loop.h"
#include "gstpp/gtk_player.h"
#include "gstpp/pad.h"

#include <cuda_runtime_api.h>
#include <gstnvdsmeta.h>
#include <gflags/gflags.h>

using namespace grd::gstpp;

DEFINE_string(
    src,
    "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4",
    "input stream");
DEFINE_string(pgie,
              "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/"
              "deepstream-test1/dstest1_pgie_config.txt",
              "pgie config file");
DEFINE_int32(width, 1920, "width");
DEFINE_int32(height, 1080, "height");

const int32_t MAX_DISPLAY_LEN = 64;
const int32_t PGIE_CLASS_ID_VEHICLE = 0;
const int32_t PGIE_CLASS_ID_PERSON = 2;
const gchar PGIE_CLASSES_STR[4][32] = {"Vehicle", "TwoWheeler", "Person", "Roadsign"};
char FONT_NAME[] = "Serif";

GstPadProbeReturn OsdSinkPadBufferProbe(GstppPad* pad, GstPadProbeInfo* info, int32_t* frame_number);

class DsTest1App: public GstppApp {
 public:
  DsTest1App()
      : GstppApp("DsTest1"),
        source_("filesrc", "file-source"),
        h264parser_("h264parse", "h264-parser"),
        decoder_("nvv4l2decoder", "nvv4l2-decoder"),
        streammux_("nvstreammux", "stream-muxer"),
        pgie_("nvinfer", "primary-nvinference-engine"),
        nvvidconv_("nvvideoconvert", "nvvideo-converter"),
        nvosd_("nvdsosd", "nv-onscreendisplay"),
        sink_("nveglglessink", "nvvideo-renderer") {}
  virtual ~DsTest1App() {}

  bool Init() override {
    InitBus();
    InitPipeline();
    return true;
 }

private:

  void InitBus();
  void InitPipeline();

  int32_t frame_number_ = 0;
  GstppElement source_;
  GstppElement h264parser_;
  GstppElement decoder_;
  GstppElement streammux_;
  GstppElement pgie_;
  GstppElement nvvidconv_;
  GstppElement nvosd_;
  GstppElement sink_;

};

int main(int argc, char** argv) {
  int32_t current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;

  cudaGetDeviceProperties(&prop, current_device);
  gtk_init(&argc, &argv);
  gst_init(&argc, &argv);

  DsTest1App app;
  app.Init();
  app.Start();

  using namespace std::chrono_literals;
  bool stop = false;
  while (!stop) {
    app.bus()->PostApplicationMessage([](auto bus, auto msg) { LOG(INFO) << "APPLICATION MESSAGE!!!!"; });
    std::this_thread::sleep_for(1s);
  }

  app.Stop();

  return 0;
}

void DsTest1App::InitBus() {
  bus_->SetTypedMessageCallback(MessageType::EOS,
                                 [this](GstppBus* bus, GstppMessage* msg) {
                                   LOG(INFO) << "EOS, quit loop";
                                   this->main_loop_.Quit();
                                 });
  bus_->SetTypedMessageCallback(
      MessageType::ERROR, [this](GstppBus* bus, GstppMessage* msg) {
        LOG(ERROR) << "got error: " << msg->AsError();
        this->main_loop_.Quit();
      });
  bus_->SetTypedMessageCallback(
      MessageType::APPLICATION, [](GstppBus* bus, GstppMessage* msg) {
        LOG(INFO) << "application message " << (void*)msg;
        try {
          std::any_cast<GstppBusMessageCallback>(msg->data())(bus, msg);
        } catch (const std::bad_any_cast& e) {
          LOG(ERROR) << "cannot process application msg " << e.what()
                     << " type " << msg->data().type().name();
        }
      });
  bus_->WatchAndSignalConnect();
}

void DsTest1App::InitPipeline() {
  source_.SetProperty("location", FLAGS_src);
  streammux_.SetProperty("batch-size", 1);
  streammux_.SetProperty("batched-push-timeout", 40000);
  streammux_.SetProperty("width", FLAGS_width);
  streammux_.SetProperty("height", FLAGS_height);

  pgie_.SetProperty("config-file-path", FLAGS_pgie);

  GstppPad osd_sink_pad(&nvosd_, "sink");

  osd_sink_pad.AddProbeCallback(
      PadProbeType::BUFFER, [this](GstppPad* pad, GstPadProbeInfo* info) {
        return OsdSinkPadBufferProbe(pad, info, &(this->frame_number_));
      });

  auto sink_pad = streammux_.GetRequestPad("sink_0");
  auto src_pad = decoder_.GetStaticPad("src");

  (*src_pad)--> (*sink_pad);
  delete src_pad;
  delete sink_pad;

  pipeline_->Add(source_, h264parser_, decoder_, streammux_, pgie_, nvvidconv_, nvosd_, sink_);

  source_ --> h264parser_ --> decoder_;
  streammux_ --> pgie_ --> nvvidconv_ --> nvosd_ --> sink_;

}

GstPadProbeReturn OsdSinkPadBufferProbe(GstppPad* pad, GstPadProbeInfo* info, int32_t* frame_number) {
  GstBuffer* buf = (GstBuffer*)info->data;
  guint num_rects = 0;
  NvDsObjectMeta* obj_meta = NULL;
  guint vehicle_count = 0;
  guint person_count = 0;
  NvDsMetaList* l_frame = NULL;
  NvDsMetaList* l_obj = NULL;
  NvDsDisplayMeta* display_meta = NULL;

  NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buf);

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
       l_frame = l_frame->next) {
    NvDsFrameMeta* frame_meta = (NvDsFrameMeta*)(l_frame->data);
    int offset = 0;
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
         l_obj = l_obj->next) {
      obj_meta = (NvDsObjectMeta*)(l_obj->data);
      if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE) {
        vehicle_count++;
        num_rects++;
      }
      if (obj_meta->class_id == PGIE_CLASS_ID_PERSON) {
        person_count++;
        num_rects++;
      }
    }
    display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
    NvOSD_TextParams* txt_params = &display_meta->text_params[0];
    display_meta->num_labels = 1;
    txt_params->display_text = reinterpret_cast<char*>(g_malloc0(MAX_DISPLAY_LEN));
    offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Person = %d ",
                      person_count);
    offset = snprintf(txt_params->display_text + offset, MAX_DISPLAY_LEN,
                      "Vehicle = %d ", vehicle_count);

    /* Now set the offsets where the string should appear */
    txt_params->x_offset = 10;
    txt_params->y_offset = 12;

    /* Font , font-color and font-size */
    txt_params->font_params.font_name = FONT_NAME;
    txt_params->font_params.font_size = 10;
    txt_params->font_params.font_color.red = 1.0;
    txt_params->font_params.font_color.green = 1.0;
    txt_params->font_params.font_color.blue = 1.0;
    txt_params->font_params.font_color.alpha = 1.0;

    /* Text background color */
    txt_params->set_bg_clr = 1;
    txt_params->text_bg_clr.red = 0.0;
    txt_params->text_bg_clr.green = 0.0;
    txt_params->text_bg_clr.blue = 0.0;
    txt_params->text_bg_clr.alpha = 1.0;

    nvds_add_display_meta_to_frame(frame_meta, display_meta);
  }

  LOG(INFO) << "Frame Number = " << *frame_number
            << " Number of objects = " << num_rects
            << " Vehicle Count = " << vehicle_count
            << " Person Count = " << person_count;
  (*frame_number)++;

  return GST_PAD_PROBE_OK;
}