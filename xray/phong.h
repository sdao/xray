#pragma once
#define NOMINMAX
#include <optix_world.h>
#include "material.h"
#include "node.h"
#include "xray.h"

class Phong : public Material {
  float _exp;
  optix::float3 _color;

protected:
  virtual std::string getClosestHitPtxFile() const override;
  virtual std::string getClosestHitProgram() const override;

public:
  Phong(Xray* xray, float e, optix::float3 c);
  static Phong* make(Xray* xray, const Node& n);
};

