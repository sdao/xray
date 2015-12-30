#include "light.h"

AreaLight::AreaLight(optix::float3 color) : _color(color) {}

AreaLight* AreaLight::make(const Node& n) {
  return new AreaLight(n.getFloat3("color"));
}

AreaLight::~AreaLight() {}

Light* AreaLight::getLight() const {
  return Light::make(_color);
}
