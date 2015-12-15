#include "phong.h"
#include "CUDA_files/shared.cuh"

Phong::Phong(Xray* xray, float e, optix::float3 c)
  : Material(xray->getContext()), _exp(e), _color(c) {
  _material["scaleBRDF"]->setFloat(c * (e + 2.0f) / XRAY_TWO_PI);
  _material["scaleProb"]->setFloat((e + 1.0f) / XRAY_TWO_PI);
  _material["exponent"]->setFloat(e);
  _material["invExponent"]->setFloat(1.0f / e);
  _material["color"]->setFloat(c);
  freeze();
}

Phong* Phong::make(Xray* xray, const Node& n) {
  return new Phong(xray, n.getFloat("exponent"), n.getFloat3("color"));
}

std::string Phong::getClosestHitPtxFile() const {
  return "PTX_files/phong.cu.ptx";
}

std::string Phong::getClosestHitProgram() const {
  return "radiance";
}
