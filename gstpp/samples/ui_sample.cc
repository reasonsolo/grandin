// author: zlz
#include "gstpp/gtk_player.h"
#include "gstpp/playbin.h"

using namespace grd::gstpp;

int main(int argc, char** argv) {
  gtk_init(&argc, &argv);
  gst_init(&argc, &argv);

  GstppPlayBin playbin("playbin");

  playbin.SetProperty(
      "uri", argc > 1 ? argv[1]
                      : "https://www.freedesktop.org/software/gstreamer-sdk/"
                        "data/media/sintel_trailer-480p.webm");
  CHECK(playbin.Play()) << "cannot change pipeline state to play";

  GstppGtkPlayer player(argv[0]);
  player.Init(&playbin);
  player.Show();

  gtk_main();

  playbin.Reset();

  return 0;
}