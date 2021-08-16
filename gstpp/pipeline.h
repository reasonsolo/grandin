// author: zlz

#include "gstpp/element"

namespace grd {
namespace gstpp {

class GstPipeLine {
  public:
  PipeLine();
  ~PipeLine();

  bool Init();
  void Destroy();

  bool AddSource(const GstSource* source);
  bool AddSink(const GstSink* sink);

  private:

  GstElement* pipeline_ = nullptr;
  std::set<GstElement*> elements_;
};

}
}