// author: zlz
#pragma once
#include <string>
#include <vector>
#include <ostream>

#include "gstpp/element.h"

#include <gst/gst.h>
#include <fmt/format.h>
#include <glog/logging.h>

namespace grd {
namespace gstpp {

enum class PadDirection {
    UNKNOWN = GST_PAD_UNKNOWN,
    SRC = GST_PAD_SRC,
    SINK = GST_PAD_SINK,
};

#define GST_PAD_PROBE(X) X = GST_PAD_PROBE_TYPE_ ## X
enum class PadProbeType {
    GST_PAD_PROBE(INVALID),
    GST_PAD_PROBE(IDLE),
    GST_PAD_PROBE(BLOCK),
    GST_PAD_PROBE(BUFFER),
    GST_PAD_PROBE(BUFFER_LIST),
    GST_PAD_PROBE(EVENT_DOWNSTREAM),
    GST_PAD_PROBE(EVENT_UPSTREAM),
    GST_PAD_PROBE(EVENT_BOTH),
    GST_PAD_PROBE(EVENT_FLUSH),
    GST_PAD_PROBE(QUERY_DOWNSTREAM),
    GST_PAD_PROBE(QUERY_UPSTREAM),
    GST_PAD_PROBE(QUERY_BOTH),
    GST_PAD_PROBE(PUSH),
    GST_PAD_PROBE(PULL),
    GST_PAD_PROBE(BLOCKING),
    GST_PAD_PROBE(DATA_DOWNSTREAM),
    GST_PAD_PROBE(DATA_UPSTREAM),
    GST_PAD_PROBE(DATA_BOTH),
    GST_PAD_PROBE(BLOCK_DOWNSTREAM),
    GST_PAD_PROBE(BLOCK_UPSTREAM),
    GST_PAD_PROBE(ALL_BOTH),
    GST_PAD_PROBE(SCHEDULING),
};
#undef GST_PAD_PROBE

using GstppPadProbeCallback = std::function<GstPadProbeReturn(GstppPad*, GstPadProbeInfo*)>;

class GstppPad {
  NOCOPY(GstppPad);
  class CbEntry;
 public:
  GstppPad(const std::string& name, PadDirection dir);
  GstppPad(GstppElement* element, const std::string& name);
  GstppPad(GstPad* pad, const std::string& name);
  GstppPad(GstPad* pad);
  ~GstppPad();

  GstPad* pad() const { return pad_; }
  bool LinkTo(GstppPad& downstream);
  void AddProbeCallback(PadProbeType type, GstppPadProbeCallback cb);

  static std::vector<GstppPad*> GetAllPads(GstppElement* element);
  static GstPadProbeReturn ProbeCallback(GstPad*, GstPadProbeInfo* info, gpointer entry_ptr);

  // upstream-->downstream-->downstream1 
  GstppPad& operator--(int) { return *this; }
  GstppPad& operator--() { return *this; }
  GstppPad& operator>(GstppPad& rhs) {
    this->LinkTo(rhs);
    return rhs;
  }

  friend std::ostream& operator<<(std::ostream& os, GstppPad& pad);
  friend class GstppElement;

 private:
  GstPad* pad_ = nullptr;
  std::string name_;

  struct CbEntry {
    PadProbeType type;
    GstppPad* pad; 
    GstppPadProbeCallback cb;
  };
  std::vector<CbEntry> probe_cb_entries_;
};

std::ostream& operator<<(std::ostream& os, GstppPad& pad);
}
}