// author: zlz
#include "gstpp/gtk_player.h"
#include "gstpp/message.h"

#include <gdk/gdk.h>
#include <gst/gst.h>
#include <gst/video/videooverlay.h>
#include <gtk/gtk.h>

#if defined(GDK_WINDOWING_X11)
#include <gdk/gdkx.h>
#elif defined(GDK_WINDOWING_WIN32)
#include <gdk/gdkwin32.h>
#elif defined(GDK_WINDOWING_QUARTZ)
#include <gdk/gdkquartz.h>
#endif

namespace grd {
namespace gstpp {

GstppGtkPlayer::GstppGtkPlayer(const std::string& name) {
  InstallDefaultCallbacks();
}

GstppGtkPlayer::~GstppGtkPlayer() { Destroy(); }

void GstppGtkPlayer::Init(GstppElement* element) { 
    CHECK(!element_ && !widget_set_);
    widget_set_ = new WidgetSet(this);
    element_ = element;
    element_->bus()->AddMessageCallback([this](GstppBus* bus, GstppMessage* msg) {
        auto it = this->msg_cb_map_.find(msg->type());
        if (it != this->msg_cb_map_.end() && it->second) {
            it->second(bus, msg);
        }
    });
 }  

void GstppGtkPlayer::Destroy() {
    delete widget_set_;
}

GstppGtkPlayer::WidgetSet::WidgetSet(GstppGtkPlayer* _player)
    : player(_player) {
  main_window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
  g_signal_connect(G_OBJECT(main_window), "delete-event",
                   G_CALLBACK(&GstppGtkPlayer::DeleteCb), player);

  video_window = gtk_drawing_area_new();
  gtk_widget_set_double_buffered(video_window, false);
  g_signal_connect(video_window, "realize",
                   G_CALLBACK(&GstppGtkPlayer::RealizeCb), player);
  g_signal_connect(video_window, "draw", G_CALLBACK(&GstppGtkPlayer::DrawCb),
                   player);

  play_button = gtk_button_new_from_icon_name("media-playback-start",
                                              GTK_ICON_SIZE_SMALL_TOOLBAR);
  g_signal_connect(G_OBJECT(play_button), "clicked",
                   G_CALLBACK(&GstppGtkPlayer::PlayCb), player);

  pause_button = gtk_button_new_from_icon_name("media-playback-pause",
                                               GTK_ICON_SIZE_SMALL_TOOLBAR);
  g_signal_connect(G_OBJECT(pause_button), "clicked",
                   G_CALLBACK(&GstppGtkPlayer::PauseCb), player);

  stop_button = gtk_button_new_from_icon_name("media-playback-stop",
                                              GTK_ICON_SIZE_SMALL_TOOLBAR);
  g_signal_connect(G_OBJECT(stop_button), "clicked",
                   G_CALLBACK(&GstppGtkPlayer::PauseCb), player);

  slider = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, 0, 100, 1);
  gtk_scale_set_draw_value(GTK_SCALE(slider), 0);
  slider_update_signal_id =
      g_signal_connect(G_OBJECT(slider), "value-changed",
                       G_CALLBACK(&GstppGtkPlayer::SliderCb), player);

  streams_list = gtk_text_view_new();
  gtk_text_view_set_editable(GTK_TEXT_VIEW(streams_list), false);

  controls = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
  gtk_box_pack_start(GTK_BOX(controls), play_button, false, false, 2);
  gtk_box_pack_start(GTK_BOX(controls), pause_button, false, false, 2);
  gtk_box_pack_start(GTK_BOX(controls), stop_button, false, false, 2);
  gtk_box_pack_start(GTK_BOX(controls), slider, true, true, 2);

  main_hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
  gtk_box_pack_start(GTK_BOX(main_hbox), video_window, true, true, 0);
  gtk_box_pack_start(GTK_BOX(main_hbox), streams_list, true, true, 0);

  main_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
  gtk_box_pack_start(GTK_BOX(main_box), main_hbox, true, true, 0);
  gtk_box_pack_start(GTK_BOX(main_box), controls, false, false, 0);
  gtk_container_add(GTK_CONTAINER(main_window), main_box);
  gtk_window_set_default_size(GTK_WINDOW(main_window), 640, 480);
}

GstppGtkPlayer::WidgetSet::~WidgetSet() {
  //gtk_widget_destroy(main_window);
}

void GstppGtkPlayer::WidgetSet::Show(int32_t refresh_sec) { 
    gtk_widget_show_all(main_window);
    g_timeout_add_seconds(refresh_sec, (GSourceFunc)(&GstppGtkPlayer::RefreshCb), this);
}

/* static */
void GstppGtkPlayer::PlayCb(GtkButton* button, GstppGtkPlayer* self) {
  self->element_->Play();
}

/* static */
void GstppGtkPlayer::PauseCb(GtkButton* button, GstppGtkPlayer* self) {
  self->element_->Pause();
}

/* static */
void GstppGtkPlayer::StopCb(GtkButton* button, GstppGtkPlayer* self) {
  self->element_->Ready();
}

/* static */
void GstppGtkPlayer::DeleteCb(GtkWidget* widget, GdkEvent* event, GstppGtkPlayer* self) {
    StopCb(nullptr, self);
    gtk_main_quit();
}

/* static */
gboolean GstppGtkPlayer::DrawCb(GtkWidget* widget, cairo_t* cr,
                                GstppGtkPlayer* self) {
  if (self->state_ < ElementState::PAUSED) {
    GtkAllocation allocation;

    /* Cairo is a 2D graphics library which we use here to clean the video window.
     * It is used by GStreamer for other reasons, so it will always be available to us. */
    gtk_widget_get_allocation (widget, &allocation);
    cairo_set_source_rgb (cr, 0, 0, 0);
    cairo_rectangle (cr, 0, 0, allocation.width, allocation.height);
    cairo_fill (cr);
  }

  return false;
}

/* static */
void GstppGtkPlayer::SliderCb(GtkRange* range, GstppGtkPlayer* self) {
    gdouble value = gtk_range_get_value(GTK_RANGE(self->widget_set_->slider));
    gst_element_seek_simple(
        self->element_->element(),
        GST_FORMAT_TIME, static_cast<GstSeekFlags>(GST_SEEK_FLAG_FLUSH | GST_SEEK_FLAG_KEY_UNIT),
        (gint64)(value * GST_SECOND));
}

/* static */
void GstppGtkPlayer::RealizeCb(GtkWidget* widget, GstppGtkPlayer* self) {
  GdkWindow* window = gtk_widget_get_window(widget);
  guintptr window_handle;

  if (!gdk_window_ensure_native(window))
    LOG(ERROR) << "Couldn't create native window needed for GstVideoOverlay!";

    /* Retrieve window handler from GDK */
#if defined(GDK_WINDOWING_WIN32)
  window_handle = (guintptr)GDK_WINDOW_HWND(window);
#elif defined(GDK_WINDOWING_QUARTZ)
  window_handle = gdk_quartz_window_get_nsview(window);
#elif defined(GDK_WINDOWING_X11)
  window_handle = GDK_WINDOW_XID(window);
#endif
  /* Pass it to playbin, which implements VideoOverlay and will forward it to
   * the video sink */
  gst_video_overlay_set_window_handle(
      GST_VIDEO_OVERLAY(self->element_->element()), window_handle);
}

/* static */
gboolean GstppGtkPlayer::RefreshCb(GstppGtkPlayer* self) {
  return self->RefreshUi();
}

bool GstppGtkPlayer::RefreshUi() {
  gint64 current = -1;

  /* We do not want to update anything unless we are in the PAUSED or PLAYING states */
  if (state_ < ElementState::PAUSED) {
    return true;
  }

  /* If we didn't know it yet, query the stream duration */
  if (!GST_CLOCK_TIME_IS_VALID(duration_)) {
    if (!gst_element_query_duration(element_->element(), GST_FORMAT_TIME,
                                    &duration_)) {
      LOG(ERROR) << "Could not query current duration.\n";
    } else {
      /* Set the range of the slider to the clip duration, in SECONDS */
      gtk_range_set_range(GTK_RANGE(widget_set_->slider), 0,
                          (gdouble)duration_ / GST_SECOND);
    }
  }

  if (gst_element_query_position(element_->element(), GST_FORMAT_TIME, &current)) {
    /* Block the "value-changed" signal, so the slider_cb function is not called
     * (which would trigger a seek the user has not requested) */
    g_signal_handler_block(widget_set_->slider, widget_set_->slider_update_signal_id);
    /* Set the position of the slider to the current element positoin, in SECONDS */
    gtk_range_set_value(GTK_RANGE(widget_set_->slider), (gdouble)current / GST_SECOND);
    /* Re-enable the signal */
    g_signal_handler_unblock(widget_set_->slider, widget_set_->slider_update_signal_id);
  }
  return true;
}

/* static */
void GstppGtkPlayer::TagsCb(GstElement* element, gint stream, GstppGtkPlayer* self) {
  gst_element_post_message(
      element,
      gst_message_new_application(GST_OBJECT(element),
                                  gst_structure_new_empty("tags-changed")));
}

/* static */
void GstppGtkPlayer::InstallDefaultCallbacks() {
  msg_cb_map_[MessageType::ERROR] = [this](GstppBus* bus, GstppMessage* msg) {
    LOG(ERROR) << "get error:" << msg->AsError();
  };
  msg_cb_map_[MessageType::EOS] = [this](GstppBus* bus, GstppMessage* msg) {
    LOG(INFO) << "got eos reset";
    this->element_->Ready();
    this->state_ = ElementState::READY;
  };
  msg_cb_map_[MessageType::STATE_CHANGED] = [this] (GstppBus* bus, GstppMessage* msg) {
        GstState old_state, new_state, pending_state;
        gst_message_parse_state_changed(msg->msg(), &old_state, &new_state,
                                        &pending_state);
        if (GST_MESSAGE_SRC(msg) == GST_OBJECT(this->element_->element())) {
          this->state_ = static_cast<ElementState>(new_state);
          LOG(INFO) << "State change from "
                    << gst_element_state_get_name(old_state) << " to "
                    << gst_element_state_get_name(new_state);
          if (old_state == GST_STATE_READY && new_state == GST_STATE_PAUSED) {
            /* For extra responsiveness, we refresh the GUI as soon as we reach
             * the PAUSED state */
            this->RefreshUi();
          }
        }
      };
}

}  // namespace gstpp
}  // namespace grd