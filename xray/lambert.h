#pragma once
#define NOMINMAX
#include <optix_world.h>
#include "material.h"
#include "node.h"
#include "xray.h"

/** A Lambert (perfectly diffuse) material. */
class Lambert : public Material {
  optix::float3 _albedo;

protected:
  virtual std::string getPtxFile() const override;
  virtual bool doesReflect() const override;
  virtual bool shouldDirectIlluminate() const override;

public:
  Lambert(Xray* xray, optix::float3 albedo);
  static Lambert* make(Xray* xray, const Node& n);
};

