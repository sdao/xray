#include "light.h"

AreaLight::AreaLight(Xray, optix::float3 color)
  : _color(color) {}

AreaLight* AreaLight::make(Xray xray, const Node& n) {
  return new AreaLight(xray, n.getFloat3("color"));
}

AreaLight::~AreaLight() {}

size_t AreaLight::sizeofDeviceLight() {
  return sizeof(Light);
}

const Light* AreaLight::getLight(RTobject object) const {
  return Light::make(_color, object);
}
