#pragma once
#include "material.h"
#include "node.h"
#include "xray.h"

class Constant : public Material {
protected:
  virtual std::string getClosestHitPtxFile() const override;
  virtual std::string getClosestHitProgram() const override;

public:
  Constant(Xray* xray, optix::float3 color);
  static Constant* make(Xray* xray, const Node& n);
};

