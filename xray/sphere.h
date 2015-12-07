#pragma once
#include "geom.h"

class Sphere : public Geom {
  optix::float3 _origin;
  float _radius;
  bool _inverted;

public:
  Sphere(optix::Context ctx, optix::float3 origin, float radius, bool inverted = false);
  virtual optix::Geometry makeOptixGeometry() const override;
};

