#pragma once
#define NOMINMAX
#include <optix_world.h>
#include "material.h"
#include "node.h"
#include "xray.h"

#define IOR_VACUUM 1.0f /**< The IOR for glass. */
#define IOR_GLASS 1.5f /**< The IOR for glass. */
#define IOR_DIAMOND 2.4f /**< The IOR for diamond. */

class Dielectric : public Material {
  float _ior;
  optix::float3 _color;

  static float schickR0(float ior);

protected:
  virtual std::string getClosestHitPtxFile() const override;
  virtual std::string getClosestHitProgram() const override;

public:
  Dielectric(Xray* xray, float ior = IOR_GLASS, optix::float3 c = optix::make_float3(1));
  static Dielectric* make(Xray* xray, const Node& n);
};

