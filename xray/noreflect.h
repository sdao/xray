#pragma once
#include "material.h"
#include "node.h"
#include "xray.h"

/** A null material that reflects nothing and absorbs all incoming light. */
class NoReflect : public Material {
protected:
  virtual std::string getPtxFile() const override;
  virtual bool doesReflect() const override;
  virtual bool shouldDirectIlluminate() const override;

public:
  NoReflect(Xray* xray);
  static NoReflect* make(Xray* xray);
};

