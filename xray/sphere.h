#pragma once
#include "geom.h"
#include "node.h"
#include "xray.h"

class Sphere : public Geom {
  optix::float3 _origin;
  float _radius;
  bool _inverted;

protected:
  virtual unsigned getPrimitiveCount() const override;
  virtual std::string getPtxFile() const override;
  virtual std::string getIsectProgram() const override;
  virtual std::string getBoundsProgram() const override;

public:
  Sphere(Xray xray, optix::float3 origin, float radius, bool inverted = false);
  static Sphere* make(Xray xray, const Node& n);
};

