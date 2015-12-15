#include "constant.h"

Constant::Constant(Xray* xray, optix::float3 color) : Material(xray->getContext()) {
  _material["color"]->set3fv(&color.x);
  freeze();
}

Constant* Constant::make(Xray* xray, const Node& n) {
  return new Constant(xray, n.getFloat3("color"));
}

std::string Constant::getClosestHitPtxFile() const {
  return "PTX_files/constant.cu.ptx";
}

std::string Constant::getClosestHitProgram() const {
  return "radiance";
}
