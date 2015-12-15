#pragma once
#include "material.h"
#include "node.h"
#include "xray.h"

class NormalTest : public Material {
protected:
  virtual std::string getClosestHitPtxFile() const override;
  virtual std::string getClosestHitProgram() const override;

public:
  NormalTest(Xray* xray);
  static NormalTest* make(Xray* xray, const Node& n);
};

