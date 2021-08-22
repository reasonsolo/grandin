// author: zlz
// plz refer to
// https://gstreamer.freedesktop.org/documentation/tutorials/basic/toolkit-integration.html?gi-language=c
#pragma once

#include <string>

#include "gstpp/pipeline.h"
#include "gstpp/element.h"

#include <gst/gst.h>
#include <gtk/gtk.h>

namespace grd {
namespace gstpp {

class GstppGtkPlayer {
 public:
  GstppGtkPlayer(const std::string& name);
  ~GstppGtkPlayer();

  void Init(GstppElement* pipeline);
  void Destroy();

  void SetMessageCb(const MessageType msg_type, GstppBusMessageCallback cb) {
      msg_cb_map_[msg_type] = cb;
  }

  void Show(int32_t refresh_sec = 1) { widget_set_->Show(refresh_sec); }

 private:
  static void RealizeCb(GtkWidget* widget, GstppGtkPlayer* self);
  static void PlayCb(GtkButton* button, GstppGtkPlayer* self);
  static void PauseCb(GtkButton* button, GstppGtkPlayer* self);
  static void StopCb(GtkButton* button, GstppGtkPlayer* self);
  static void DeleteCb(GtkWidget* widget, GdkEvent* event, GstppGtkPlayer* self);
  static void SliderCb(GtkRange *range, GstppGtkPlayer* self);
  static gboolean DrawCb(GtkWidget* widget, cairo_t* cr, GstppGtkPlayer* self);
  static gboolean RefreshCb(GstppGtkPlayer* self);
  static void TagsCb(GstElement* pipeline, gint stream, GstppGtkPlayer* self);
  static void MessageCb(GstBus* bus, GstMessage* msg, GstppGtkPlayer* self);

  bool RefreshUi();
  void InstallDefaultCallbacks();
  void EnableVideoOverlay();

  struct WidgetSet {
    GstppGtkPlayer* player = nullptr;
    GtkWidget* main_window = nullptr;
    GtkWidget* video_window = nullptr;
    GtkWidget* main_box = nullptr;
    GtkWidget* main_hbox = nullptr;
    GtkWidget* video_box = nullptr;
    GtkWidget* controls = nullptr;
    GtkWidget* play_button = nullptr;
    GtkWidget* pause_button = nullptr;
    GtkWidget* stop_button = nullptr;
    GtkWidget* slider = nullptr;
    GtkWidget* streams_list = 0;

    gulong slider_update_signal_id = 0;
    guintptr video_window_handle = 0;

    WidgetSet(GstppGtkPlayer* player);
    ~WidgetSet();

    void Show(int32_t refresh_sec = 1);
  };

  WidgetSet* widget_set_ = nullptr;
  GstppElement* element_ = nullptr;
  gint64 duration_ = GST_CLOCK_TIME_NONE;
  ElementState state_ = ElementState::RESET;

};

}  // namespace gstpp
}  // namespace grd