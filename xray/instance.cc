#include "instance.h"
#include "constant.h"

Light Instance::_nullLight = Light::make();

Instance::Instance(Xray xray, const Geom* g, const Material* m, const AreaLight* l)
  : _instance(xray.getContext()->createGeometryInstance()) {
    _instance->setGeometry(g->getGeometry());
    _instance->setMaterialCount(1);
    if (m) {
      _instance->setMaterial(0, m->getMaterial());
    } else {
      _instance->setMaterial(0, Constant(xray, optix::make_float3(0)).getMaterial());
    }
    if (l) {
      _instance["light"]->setUserData(sizeof(Light), l->getLight()); 
    } else {
      _instance["light"]->setUserData(sizeof(Light), &Instance::_nullLight);
    }
}

Instance::~Instance() {}

optix::GeometryInstance Instance::getGeometryInstance() const {
  return _instance;
}
