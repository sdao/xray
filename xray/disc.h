#pragma once
#include "geom.h"
#include "node.h"
#include "xray.h"

class Disc : public Geom {
  optix::float3 _origin;
  optix::float3 _normal;
  float _radiusOuter;
  float _radiusInner;
  
protected:
  virtual unsigned getPrimitiveCount() const override;
  virtual std::string getPtxFile() const override;
  virtual std::string getIsectProgram() const override;
  virtual std::string getBoundsProgram() const override;

public:
  Disc(Xray xray, optix::float3 origin, optix::float3 normal, float radiusOuter, float radiusInner);
  static Disc* make(Xray xray, const Node& n);
};

