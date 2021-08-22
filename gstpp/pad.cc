// author: zlz
#include "gstpp/pad.h"

namespace grd {
namespace gstpp {

/* static */
std::vector<GstppPad> GstppPad::GetAllPads(GstppElement* element) {
  std::vector<GstppPad> pads;
  GstIterator* it = gst_element_iterate_pads(element->element());
  bool done = false;
  while (!done) {
    GValue value = {0};
    switch (gst_iterator_next(it, &value)) {
      case GST_ITERATOR_OK: {
        GstPad* pad = g_value_get_object(&value);
        pads.emplace_back(pad);
        g_value_reset(&value);
        break;
      }
      case GST_ITERATOR_RESYNC: {
        gst_iterator_resync(it);
        break;
      }
      case GST_ITERATOR_ERROR:
      case GST_ITERATOR_DONE: {
        done = true;
        break;
      }
    }
  }

  gst_iterator_free(it);
  return pads;
}

/* static */
GstPadProbeReturn GstppPad::ProbeCallback(GstPad* gst_pad, GstPadProbeInfo* info, gpointer entry_ptr) {
    CbEntry* entry = reinterpret_cast<CbEntry*>(entry_ptr);
    CHECK(entry && entry->pad && entry->cb);
    return entry->cb(entry->pad, info);
}

void GstppPad::LinkTo(GstppPad& downstream) {
    CHECK(gst_pad_link(pad_, downstream->pad_) == GST_PAD_LINK_OK)
    << "cannot link " << *this << " --> " << *downstream;
}

std::ostream& operator<<(std::ostream& os, GstppPad& pad) {
  os << fmt::format("pad-{}", pad.name_);
  return os
}
}
}