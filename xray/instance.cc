#include "instance.h"
#include "constant.h"
#include "CUDA_files/light.cuh"

Instance::Instance(Xray xray, const Geom* g, const Material* m, const AreaLight* l) {
    _instance = xray.getContext()->createGeometryInstance();
    _instance->setGeometry(g->getGeometry());
    _instance->setMaterialCount(1);

    if (l) {
      _light = l->getLight(_instance->get());
    } else {
      _light = Light::make();
    }
    _instance["light"]->setUserData(AreaLight::sizeofDeviceLight(), _light);

    if (m) {
      _instance->setMaterial(0, m->getMaterial());
    } else {
      _instance->setMaterial(0, Constant(xray, optix::make_float3(0)).getMaterial());
    }
}

Instance::~Instance() {}

optix::GeometryInstance Instance::getGeometryInstance() const {
  return _instance;
}

bool Instance::getLight(const Light** light) const {
  *light = _light;
  return _light->object != nullptr;
}
