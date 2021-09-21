// author: zlz
#include "deepstream/testapp.h"
#include "common/utils.h"

DEFINE_string(
    src,
    "/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264",
    "input stream");
DEFINE_string(pgie,
              "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/"
              "deepstream-test1/dstest1_pgie_config.txt",
              "pgie config file");
DEFINE_int32(width, 1920, "width");
DEFINE_int32(height, 1080, "height");
DEFINE_int32(testapp_max_input, 32, "max input size");
DEFINE_int32(testapp_max_video_len_sec, 30, "max video length in seconds");
DEFINE_int32(testapp_gpu_id, 0, "gpu id");

const int32_t MAX_DISPLAY_LEN = 64;
const int32_t PGIE_CLASS_ID_VEHICLE = 0;
const int32_t PGIE_CLASS_ID_PERSON = 2;
const gchar PGIE_CLASSES_STR[4][32] = {"Vehicle", "TwoWheeler", "Person", "Roadsign"};
char FONT_NAME[] = "Serif";


namespace grd {
namespace deepstream {

using namespace ::grd::gstpp;

GstPadProbeReturn OsdSinkPadBufferProbe(GstppPad* pad, GstPadProbeInfo* info, int32_t* frame_number);

TestApp::TestApp()
    : GstppApp("DsTest1"),
      source_("filesrc", "file-source"),
      h264parser_("h264parse", "h264-parser"),
      decoder_("nvv4l2decoder", "nvv4l2-decoder"),
      streammux_("nvstreammux", "stream-muxer"),
      pgie_("nvinfer", "primary-nvinference-engine"),
      nvvidconv_("nvvideoconvert", "nvvideo-converter"),
      nvosd_("nvdsosd", "nv-onscreendisplay"),
      sink_("nveglglessink", "nvvideo-renderer") {
  dynamic_source_list_.resize(FLAGS_testapp_max_input);
  for (int32_t i = 0; i < FLAGS_testapp_max_input; i++) {
    available_sink_slots_.push_back(i);
  }
}

void TestApp::InitBus() {
  bus_->SetTypedMessageCallback(MessageType::EOS,
                                 [this](GstppBus* bus, GstppMessage* msg) {
                                   LOG(INFO) << "get EOS, keep looping";
                                   //this->main_loop_.Quit();
                                 });
  bus_->SetTypedMessageCallback(
      MessageType::ERROR, [this](GstppBus* bus, GstppMessage* msg) {
        LOG(ERROR) << "got error: " << msg->AsError();
        // this->main_loop_.Quit();
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
  bus_->SetTypedMessageCallback(
      MessageType::ELEMENT, [](GstppBus* bus, GstppMessage* msg) {
        if (msg->HasName("GstRTSPSrcTimeout")) {
          LOG(INFO) << "gstpp rtsp src timeout";
        }
      });
  bus_->AddWatch();
}

void TestApp::InitPipeline() {
  LOG(INFO) << "init pipeline";

  source_.SetProperty("location", FLAGS_src);
  streammux_.SetProperty("batch-size", 1);
  streammux_.SetProperty("batched-push-timeout", 40000);
  streammux_.SetProperty("width", FLAGS_width);
  streammux_.SetProperty("height", FLAGS_height);

  pgie_.SetProperty("config-file-path", FLAGS_pgie);

  auto sink_pad = streammux_.GetRequestPad("sink_0");
  auto src_pad = decoder_.GetStaticPad("src");

  (*src_pad)--> (*sink_pad);
  //delete src_pad;
  //delete sink_pad;

  pipeline_->Add(source_, h264parser_, decoder_, streammux_, pgie_, nvvidconv_, nvosd_, sink_);

  source_ --> h264parser_ --> decoder_;
  streammux_ --> pgie_ --> nvvidconv_ --> nvosd_ --> sink_;

  GstppPad osd_sink_pad(&nvosd_, "sink");
  osd_sink_pad.AddProbeCallback(
      PadProbeType::BUFFER, [this](GstppPad* pad, GstPadProbeInfo* info) {
        return OsdSinkPadBufferProbe(pad, info, &(this->frame_number_));
      });

}

bool TestApp::AddSourceFromUri(const std::string& name, const std::string& uri,
                               SrcStartCallback start_cb,
                               SrcStopCallback stop_cb) {
  LOG(INFO) << "add source in loop " << uri;
  int32_t slot_idx = -1;
  {
    Lock lock(mtx_);
    if (available_sink_slots_.empty()) {
      LOG(INFO) << "no available input slot for " << name;
      if (start_cb) start_cb(false);
      return false;
    }
    slot_idx = available_sink_slots_.back();
    available_sink_slots_.pop_back();
  }
  GstppElement* srcbin = nullptr;
  if (!uri.empty() && uri[0] == '/') {
    srcbin = GstppElement::CreateSourceFromPath(name, uri);
  } else if (uri.substr(0, 4) == "rtsp") {
  } else {
    srcbin = GstppElement::CreateSourceFromUri(name, uri);
  }
  if (srcbin == nullptr) {
    LOG(INFO) << "cannot create srcbin for " << name;
    {
      Lock lock(mtx_);
      available_sink_slots_.push_back(slot_idx);
    }
    if (start_cb) start_cb(false);
    return false;
  }
  dynamic_source_name_idx_map_[name] = slot_idx;
  dynamic_source_list_[slot_idx] =
      DynamicSource{srcbin, slot_idx, source_count_++, std::move(start_cb),
                    std::move(stop_cb), TimeUtils::NowUs()};

  srcbin->SetNewPadCallback([start_cb, name, slot_idx, srcbin, this](GstppPad* pad) {
    GstCaps* caps = gst_pad_query_caps(pad->pad(), NULL);
    const GstStructure* structure = gst_caps_get_structure(caps, 0);
    const gchar* name = gst_structure_get_name(structure);

    LOG(INFO) << "srbin " << srcbin->name() << " add new pad " << name;
    if (strncmp(name, "video", 5) == 0) {
      std::string pad_name = fmt::format("sink_{}", slot_idx);
      LOG(INFO) << "get pad name " << pad_name;
      auto&& sink_pad = this->streammux_.GetRequestPad(pad_name);
      if (pad->LinkTo(*sink_pad)) {
        LOG(INFO) << "link srcbin " << srcbin->name() << " to pipeline";
        auto&& start_cb = this->dynamic_source_list_[slot_idx].start_cb;
        CHECK(srcbin == this->dynamic_source_list_[slot_idx].srcbin)
        << "srcbin " << (void*)srcbin << " listbin " << slot_idx << ": " << this->dynamic_source_list_[slot_idx].srcbin;
        if (start_cb) {
          start_cb(true);
        }
      } else {
        LOG(INFO) << "cannot link srcbin " << srcbin->name() << " to pipeline";
        this->RemoveSourceByName(name);
      }
      delete sink_pad;
      return;
    } else {
      LOG(ERROR) << "unknown cap " << name;
      if (start_cb) start_cb(false);
    }
  });

  srcbin->SetNewChildCallback(
      [slot_idx, srcbin, this](GstChildProxy* proxy, GObject* obj, gchar* name) {
        LOG(INFO) << "srcbin child added " << name;
        if (g_strrstr(name, "decodebin") == name) {
          LOG(INFO) << "decodebin child callback";
        }
        if (g_strrstr(name, "nvv4l2decoder") == name) {
          LOG(INFO) << "nvv4ldecoder child callback";
          g_object_set(obj, "gpu-id", FLAGS_testapp_gpu_id, nullptr);
        }
      });

  pipeline_->Add(*srcbin);

  if (!srcbin->Play()) {
    LOG(ERROR) << "cannot play dynamic srbin " << name;
    Lock lock(mtx_);
    available_sink_slots_.push_back(slot_idx);
    return false;
  }
  return true;
}

bool TestApp::RemoveSourceByName(const std::string& name) {
  if (dynamic_source_name_idx_map_.count(name) == 0) {
    LOG(ERROR) << "source not found " << name;
    return false;
  }
  if (RemoveSourceByIdx(dynamic_source_name_idx_map_[name])) {
    dynamic_source_name_idx_map_.erase(name);
    return true;
  }
  return false;
}

bool TestApp::RemoveSourceByIdx(const int32_t slot_idx) {
  CHECK(size_t(slot_idx) < dynamic_source_list_.size()) << "cannot remote idx " << slot_idx;
  auto&& dynamic_src = dynamic_source_list_[slot_idx];
  auto&& srcbin = dynamic_src.srcbin;
  if (!srcbin) {
    LOG(INFO) << "cannot remove idx " << slot_idx << " is empty";
    return false;
  }
  if (!srcbin->Reset()) {
    LOG(INFO) << "cannot reset srcbin " << srcbin->name() << " idx " << slot_idx;
    return false;
  }
  std::string pad_name = fmt::format("sink_{}", slot_idx);
  auto&& sinkpad = streammux_.GetStaticPad(pad_name);
  auto gst_pad = sinkpad->pad();
  gst_pad_send_event(gst_pad, gst_event_new_flush_stop(false));
  gst_element_release_request_pad(streammux_.element(), gst_pad);
  delete sinkpad;
  pipeline_->RemoveElement(*srcbin);
  { 
    Lock lock(mtx_);
    available_sink_slots_.push_back(slot_idx);
  }
  if (dynamic_source_list_[slot_idx].srcbin) {
    LOG(INFO) << "delete dynamic src " << slot_idx;
    delete dynamic_source_list_[slot_idx].srcbin;
    dynamic_source_list_[slot_idx].srcbin = nullptr;
    LOG(INFO) << "after delete dynamic src " << slot_idx;
  }
  return true;
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
}
}  // namespace grd
