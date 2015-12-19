#include "lambert.h"

Lambert::Lambert(Xray* xray, optix::float3 albedo)
  : Material(xray->getContext()), _albedo(albedo) {
  _material["albedo"]->setFloat(albedo);
  freeze();
}

Lambert* Lambert::make(Xray* xray, const Node& n) {
  return new Lambert(xray, n.getFloat3("albedo"));
}

std::string Lambert::getPtxFile() const {
  return "ptx/lambert.cu.ptx";
}

bool Lambert::doesReflect() const {
  return true;
}

bool Lambert::shouldDirectIlluminate() const {
  return true;
}
