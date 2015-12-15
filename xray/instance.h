#pragma once
#define NOMINMAX
#include <optix.h>
#include <optix_world.h>
#include "geom.h"
#include "material.h"
#include "light.h"
#include "xray.h"

class Instance {
  int _id;
  optix::GeometryInstance _instance;
  Light* _light;

  Instance(Xray* xray, const Geom* g, const Material* m, const AreaLight* l, int id);

public:
  ~Instance();
  static Instance* make(Xray* xray, const Geom* g, const Material* m, const AreaLight* l);
  optix::GeometryInstance getGeometryInstance() const;
  Light* getLightInstance() const;
};
