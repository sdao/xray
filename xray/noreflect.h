#pragma once
#include "material.h"
#include "node.h"
#include "xray.h"

class NoReflect : public Material {
protected:
  virtual std::string getPtxFile() const override;
  virtual bool doesReflect() const override;
  virtual bool shouldDirectIlluminate() const override;

public:
  NoReflect(Xray* xray);
  static NoReflect* make(Xray* xray);
};

