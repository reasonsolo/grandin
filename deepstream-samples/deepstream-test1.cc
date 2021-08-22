// author: zlz

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>

#include "gstpp/element.h"
#include "gstpp/main_loop.h";
#include "gstpp/gtk_player.h"
#include "gstpp/pad.h"

#include <cuda_runtime_api.h>
#include <gstnvdsmeta.h>
#include <gflags/gflags.h>

using namespace grd::gstpp;

DEFINE_string(src, "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4", "input stream");
DEFINE_string(pgie, "dstest1_pgie_config.txt", "pgie config file");
DEFINE_int32(width, 640, "width");
DEFINE_int32(height, 480, "height");

gchar pgie_classes_str[4][32] = {"Vehicle", "TwoWheeler", "Person", "Roadsign"};

GstPadProbeReturn OsdSinkPadBufferProbe(GstppPad* pad, GstPadProbeInfo* info, int32_t* frame_number);

int main(int argc, char** argv) {
  int32_t current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;

  cudaGetDeviceProperties(&prop, current_device);
  gtk_init(&argc, &argv);
  gst_init(&argc, &argv);

  GstppMainLoop loop;

  GstppPipeline pipeline("pipeline");

  GstppElement source("filesrc", "file-source");
  GstppElement h264parser("h264parse", "h264-parser");
  GstppElement decoder("nvv4l2decoder", "nvv4l2-decoder");
  GstppElement streammux("nvstreammux", "stream-muxer");

  GstppElement pgie("nvinfer", "primary-nvinference-engine");
  GstppElement nvvidconv("nvvideoconvert", "nvvideo-converter");
  GstppElement nvosd("nvdsosd", "nv-onscreendisplay");

  GstppElement sink("nveglglessink", "nvvideo-renderer");
  
  source.SetProperty("location", FLAGS_pgie);
  streammux.SetProperty("batch-size", 1);
  streammux.SetProperty("batched-push-timeout", 40000);
  streammux.SetProperty("width", FLAGS_width);
  streammux.SetProperty("height", FLAGS_height);

  pgie.SetProperty("config-file-path", FLAGS_pgie);

  auto* bus = pipeline.bus();
  bus->SetTypedMessageCallback(MessageType::EOS, [&loop](GstppBus* bus, GstppMessage* msg) {
      LOG(INFO) << "EOS, quit loop";
      loop.Quit();
  });
  bus->SetTypedMessageCallback(MessageType::ERROR, [&loop](GstppBus* bus, GstppMessage* msg) {
      LOG(ERROR) << "got error: " << msg->AsError();
      loop.Quit();
  });

  GstppPad osd_sink_pad(nvosd, "sink");

  int32_t frame_numer = 0;
  osd_sink_pad.AddProbeCallback(PadProbeType::BUFFER, [&frame_nubmer](GstppPad* pad, GstPadProbInfo* info) {
    OsdSinkPadBufferProbe(pad, info, &frame_number);
  });

  auto sink_pad = streammux.GetRequestPad("sink_0");
  auto src_pad = decoder.GetStaticPad("src");

  *src_pad -->-- *sink_pad;

  delete sink_pad;
  delete src_pad;

  pipeline.Add(source, h264parser, decoder, streammux, pgie, nvvidconv, nvosd, sink);

  source --> h264parser --> decoder;
  streammux --> pgie --> nvvidconv --> nvosd --> sink;

  pipeline.Play();
  loop.Run();
  pipeline.Reset();

  return 0;
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
    txt_params->display_text = g_malloc0(MAX_DISPLAY_LEN);
    offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Person = %d ",
                      person_count);
    offset = snprintf(txt_params->display_text + offset, MAX_DISPLAY_LEN,
                      "Vehicle = %d ", vehicle_count);

    /* Now set the offsets where the string should appear */
    txt_params->x_offset = 10;
    txt_params->y_offset = 12;

    /* Font , font-color and font-size */
    txt_params->font_params.font_name = "Serif";
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
            << " Person Count = " person_count;
  (*frame_number)++;
}