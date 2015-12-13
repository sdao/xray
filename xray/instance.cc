#include "instance.h"
#include "constant.h"

Instance::Instance(Xray xray, const Geom* g, const Material* m)
  : _instance(xray.getContext()->createGeometryInstance()) {
    _instance->setGeometry(g->getGeometry());
    if (m) {
      _instance->setMaterialCount(1);
      _instance->setMaterial(0, m->getMaterial());
    } else {
      _instance->setMaterialCount(1);
      _instance->setMaterial(0, Constant(xray, optix::make_float3(12)).getMaterial());
    }
}

Instance::~Instance() {}

optix::GeometryInstance Instance::getGeometryInstance() const {
  return _instance;
}
