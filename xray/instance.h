#pragma once
#define NOMINMAX
#include <optix.h>
#include <optix_world.h>
#include "geom.h"
#include "material.h"
#include "xray.h"

class Instance {
  optix::GeometryInstance _instance;

public:
  Instance(Xray xray, const Geom* g, const Material* m);
  ~Instance();
  optix::GeometryInstance getGeometryInstance() const;
};
