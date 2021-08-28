// author: zlz
#pragma once

#include <string>
#include <ostream>
#include <type_traits>

#include "common/macros.h"
#include "gstpp/bus.h"

#include "gst/gst.h"

namespace grd {
namespace gstpp {

#define DEF_STATE_CHANGE(X) X = GST_STATE_CHANGE_ ## X
enum class ChangeStateResult { 
  DEF_STATE_CHANGE(SUCCESS),
  DEF_STATE_CHANGE(FAILURE),
  DEF_STATE_CHANGE(ASYNC),
  DEF_STATE_CHANGE(NO_PREROLL),
};

enum class ElementState {
  RESET = GST_STATE_NULL,
  READY = GST_STATE_READY,
  PLAYING = GST_STATE_PLAYING,
  PAUSED = GST_STATE_PAUSED,
};

class GstppPad;
class GstppElement;

using ElementPadCallback = std::function<void(GstppPad*)>;
using ElementChildCallback = std::function<void(GstChildProxy*, GObject* object, gchar* name)>;

class GstppElement {
  NOCOPY(GstppElement);
 public:
  GstppElement() = default;
  GstppElement(const std::string& ele_type, const std::string& ele_name);
  GstppElement(const std::string& ele_type, const std::string& ele_name, GstElement* elem_ptr);
  virtual ~GstppElement();

  const std::string& name() const { return name_; }
  const std::string& type() const { return type_; }
  virtual GstElement* element() const { return element_; }
  ElementState state() const { return state_; }
  virtual GstppBus* bus() const { return bus_; }

  void LinkTo(GstppElement& downstream);

  bool Reset() { return SetState(ElementState::RESET); }
  bool Ready() { return SetState(ElementState::READY); }
  bool Play()  { return SetState(ElementState::PLAYING); }
  bool Pause() { return SetState(ElementState::PAUSED); }

  ElementState GetState();

  // upstream-->downstream-->downstream1 
  GstppElement& operator--(int) { return *this; }
  GstppElement& operator--() { return *this; }
  GstppElement& operator > (GstppElement& rhs) {
    this->LinkTo(rhs);
    return rhs;
  }

  void SetFlag(gint flag);
  void UnSetFlag(gint flag);

  template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
  void SetProperty(const std::string& key, T value) {
    CHECK(element());
    g_object_set(element(), key.c_str(), value, nullptr);
    T get_val;
    g_object_get(element(), key.c_str(), &get_val, nullptr);
    LOG(INFO) << "set property " << *this << " " << key << ":" << value << " get " << get_val;
    CHECK(get_val == value) << "set " << value << " get " << get_val;
  }

  void SetProperty(const std::string& key, const std::string& value) {
    CHECK(element());
    g_object_set(element(), key.c_str(), value.c_str(), nullptr);
    gchar* get_val;
    g_object_get(element(), key.c_str(), &get_val, nullptr);
    LOG(INFO) << "set property " << *this << " " << key << ":" << value << " get " << get_val;
    CHECK(strcmp(value.c_str(), get_val) == 0) << "set " << value << " get " << get_val;
    g_free(get_val);
  }

  GstppPad* GetStaticPad(const std::string& name);
  GstppPad* GetRequestPad(const std::string& name);

  void SetNewPadCallback(ElementPadCallback callback);
  void SetNewChildCallback(ElementChildCallback callback);

  static GstppElement* Create(const std::string& type, const std::string& name);
  static GstppElement* CreateSourceFromUri(const std::string& name, const std::string& uri);

 protected:

  static void OnPadAdded(GstElement* elem, GstPad* pad, gpointer data);
  static void OnChildAdded(GstChildProxy* child_proxy, GObject* obj, gchar* name, gpointer data);
  
  void AddToPipeline(GstppElement* p) { pipeline_ = p; }
  void RemoveFromPipeline(GstppElement* p) {
    CHECK(p == pipeline_);
    pipeline_ = nullptr;
  }

  void InitElement(GstElement* element) { element_ = element; };
  void InitBus(GstBus* bus);
  bool SetState(ElementState state);

  std::string type_;
  std::string name_;

  GstElement* element_ = nullptr;
  ElementState state_ = ElementState::RESET;
  GstppElement* pipeline_ = nullptr;
  GstppBus* bus_ = nullptr;

  ElementPadCallback new_pad_cb_;
  ElementChildCallback new_child_cb_;

  friend std::ostream& operator<<(std::ostream&, GstppElement&);
  friend class GstppPipeline;

};

std::ostream& operator<<(std::ostream& os, GstppElement& elem);

}  // namespace gstpp
}  // namespace grd