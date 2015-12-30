#pragma once
#define NOMINMAX
#include <optix_world.h>
#include "material.h"
#include "node.h"
#include "xray.h"

#define IOR_VACUUM 1.0f /**< The IOR for glass. */
#define IOR_GLASS 1.5f /**< The IOR for glass. */
#define IOR_DIAMOND 2.4f /**< The IOR for diamond. */

/** A dielectric (glossy) material. */
class Dielectric : public Material {
  float _ior;
  optix::float3 _color;

  /** Computer the Schick R(0) approximation for the given IOR. */
  static float schickR0(float ior);

protected:
  virtual std::string getPtxFile() const override;
  virtual bool doesReflect() const override;
  virtual bool shouldDirectIlluminate() const override;

public:
  /**
   * Constructs a dielectric material.
   *
   * @param xray the global Xray state to attach
   * @param ior  the index of reflection on the inside of the material
   * @param c    the color of the material
   */
  Dielectric(
    Xray* xray,
    float ior = IOR_GLASS,
    optix::float3 c = optix::make_float3(1)
  );

  /** Makes a dielectric material from the given node. */
  static Dielectric* make(Xray* xray, const Node& n);
};

