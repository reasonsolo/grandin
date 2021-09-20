// author: zlz
#include "gstpp/pad.h"

namespace grd {
namespace gstpp {

GstppPad::GstppPad(const std::string& name, PadDirection dir)
    : pad_(gst_pad_new(name.c_str(), static_cast<GstPadDirection>(dir))),
      name_(name) {
  CHECK(pad_) << "cannot create pad " << name;
}

GstppPad::GstppPad(GstppElement* element, const std::string& name)
    : pad_(gst_element_get_static_pad(element->element(), name.c_str())),
      name_(fmt::format("{}@{}", element->name(), name)) {
  CHECK(pad_) << "cannot get element " << *element << " pad " << name;
}

GstppPad::GstppPad(GstPad* pad, const std::string& name): pad_(pad), name_(name) {
  gst_object_ref(pad_);
}

GstppPad::GstppPad(GstPad* pad): pad_(pad) {
  gst_object_ref(pad_);
  gchar* name = gst_pad_get_name(pad);
  name_ = std::string(name, strlen(name));
  g_free(name);
}

GstppPad::~GstppPad() {
  if (pad_) gst_object_unref(pad_);
}

void GstppPad::AddProbeCallback(PadProbeType type, GstppPadProbeCallback cb) {
  CHECK(cb);
  probe_cb_entries_.emplace_back(CbEntry{type, this, std::move(cb)});
  gst_pad_add_probe(pad_, static_cast<GstPadProbeType>(type),
                    &GstppPad::ProbeCallback,
                    (gpointer)(&(probe_cb_entries_.back())), nullptr);
}

/* static */
std::vector<GstppPad*> GstppPad::GetAllPads(GstppElement* element) {
  std::vector<GstppPad*> pads;
  GstIterator* it = gst_element_iterate_pads(element->element());
  bool done = false;
  while (!done) {
    GValue value = {0};
    switch (gst_iterator_next(it, &value)) {
      case GST_ITERATOR_OK: {
        GstPad* pad = reinterpret_cast<GstPad*>(g_value_get_object(&value));
        gchar* gname = gst_pad_get_name(pad);
        std::string name(gname, strlen(gname));
        g_free(gname);
        pads.push_back(new GstppPad(pad, name));
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
  if (entry && entry->pad && entry->cb) {
    return entry->cb(entry->pad, info);
  }
  return GstPadProbeReturn::GST_PAD_PROBE_DROP;
}

bool GstppPad::LinkTo(GstppPad& downstream) {
  if (gst_pad_link(pad_, downstream.pad_) == GST_PAD_LINK_OK) {
    LOG(INFO) << "link " << *this << " to " << downstream;
    return true;
  } 
   LOG(ERROR)  << "cannot link " << *this << " --> " << downstream;
   return false;
}

std::ostream& operator<<(std::ostream& os, GstppPad& pad) {
  os << fmt::format("pad-{}", pad.name_);
  return os;
}
}
}