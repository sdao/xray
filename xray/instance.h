#pragma once
#define NOMINMAX
#include <optix.h>
#include <optix_world.h>
#include "geom.h"
#include "material.h"
#include "light.h"
#include "xray.h"

class Instance {
  optix::GeometryInstance _instance;
  const Light* _light;

public:
  Instance(Xray xray, const Geom* g, const Material* m, const AreaLight* l);
  ~Instance();
  optix::GeometryInstance getGeometryInstance() const;
  bool getLight(const Light** light) const;
};
