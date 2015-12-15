#include "lambert.h"

Lambert::Lambert(Xray* xray, optix::float3 albedo)
  : Material(xray->getContext()), _albedo(albedo) {
  _material["albedo"]->setFloat(albedo);
  freeze();
}

Lambert* Lambert::make(Xray* xray, const Node& n) {
  return new Lambert(xray, n.getFloat3("albedo"));
}

std::string Lambert::getClosestHitPtxFile() const {
  return "PTX_files/lambert.cu.ptx";
}

std::string Lambert::getClosestHitProgram() const {
  return "radiance";
}
