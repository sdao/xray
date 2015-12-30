#pragma once
#define NOMINMAX
#include <optix_world.h>
#include "material.h"
#include "node.h"
#include "xray.h"

/** A Phong (simulated glossy reflection) material. */
class Phong : public Material {
  float _exp;           /**< The Phong exponent. */
  optix::float3 _color; /**< The reflection color (similar to albedo). */

protected:
  virtual std::string getPtxFile() const override;
  virtual bool doesReflect() const override;
  virtual bool shouldDirectIlluminate() const override;

public:
  /**
   * Constructs a Phong material.
   *
   * @param xray the global Xray state to attach
   * @param e    the Phong exponent
   * @param c    the reflection color
   */
  Phong(Xray* xray, float e, optix::float3 c);

  /** Makes a Phong material from the given node. */
  static Phong* make(Xray* xray, const Node& n);
};

