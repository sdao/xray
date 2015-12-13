#include "light.h"

AreaLight::AreaLight(Xray, optix::float3 color)
  : _color(color) {
    _l = Light::make(color);
}

AreaLight* AreaLight::make(Xray xray, const Node& n) {
  return new AreaLight(xray, n.getFloat3("color"));
}

AreaLight::~AreaLight() {}

const Light* AreaLight::getLight() const {
  return &_l;
}
