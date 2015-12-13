#pragma once
#define NOMINMAX
#include <optix.h>
#include <optix_world.h>
#include "geom.h"
#include "material.h"
#include "light.h"
#include "xray.h"

class Instance {
  static Light _nullLight;
  
  optix::GeometryInstance _instance;

public:
  Instance(Xray xray, const Geom* g, const Material* m, const AreaLight* l);
  ~Instance();
  optix::GeometryInstance getGeometryInstance() const;
};
