#include "dielectric.h"

Dielectric::Dielectric(Xray xray, float ior, optix::float3 c)
  : Material(xray.getContext()), _ior(ior), _color(c) {
  _material["r0"]->setFloat(schickR0(ior));
  _material["etaEntering"]->setFloat(IOR_VACUUM / ior);
  _material["etaExiting"]->setFloat(ior / IOR_VACUUM);
  _material["color"]->setFloat(c);
  freeze();
}

Dielectric* Dielectric::make(Xray xray, const Node& n) {
  return new Dielectric(xray, n.getFloat("ior"), n.getFloat3("color"));
}

std::string Dielectric::getClosestHitPtxFile() const {
  return "PTX_files/dielectric.cu.ptx";
}

std::string Dielectric::getClosestHitProgram() const {
  return "radiance";
}

float Dielectric::schickR0(float ior) {
  // Pre-compute values for Fresnel calculations.
  
  float r0_temp = (IOR_VACUUM - ior) / (IOR_VACUUM + ior);
  return r0_temp * r0_temp;
}
