#pragma once
#include "geom.h"

class Disc : public Geom {
  optix::float3 _origin;
  optix::float3 _normal;
  float _radiusOuter;
  float _radiusInner;

public:
  Disc(optix::Context ctx, optix::float3 origin, optix::float3 normal, float radiusOuter, float radiusInner);
  virtual optix::Geometry makeOptixGeometry() const override;
};

