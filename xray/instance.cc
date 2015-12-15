#include "instance.h"
#include "constant.h"
#include "CUDA_files/light.cuh"

Instance::Instance(Xray* xray, const Geom* g, const Material* m, const AreaLight* l, int id)
  : _id(id) {
    _instance = xray->getContext()->createGeometryInstance();
    _instance["id"]->setInt(id);
    _instance->setGeometry(g->getGeometry());
    _instance->setMaterialCount(1);

    if (m) {
      _instance->setMaterial(0, m->getMaterial());
    } else {
      _instance->setMaterial(0, Constant(xray, optix::make_float3(0)).getMaterial());
    }

    if (l) {
      optix::float3 boundsOrigin;
      float boundsRadius;
      g->getBoundingSphere(&boundsOrigin, &boundsRadius);

      _light = l->getLight();
      _light->id = id;
      _light->boundsOrigin = boundsOrigin;
      _light->boundsRadius = boundsRadius;
    } else {
      _light = Light::make();
    }

    _instance["light"]->setUserData(sizeof(Light), _light);
}

Instance::~Instance() {}

Instance* Instance::make(Xray* xray, const Geom* g, const Material* m, const AreaLight* l) {
  return new Instance(xray, g, m, l, xray->getNextID());
}

optix::GeometryInstance Instance::getGeometryInstance() const {
  return _instance;
}

Light* Instance::getLightInstance() const {
  return _light;
}
