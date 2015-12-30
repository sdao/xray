#pragma once
#define NOMINMAX
#include "node.h"
#include "cuda/light.cuh"

/**
 * A diffuse area light. This host representation can correspond to multiple
 * light instances on the device.
 */
class AreaLight {
  optix::float3 _color;

public:
  /** Constructs a diffuse area light with the given color. */
  AreaLight(optix::float3 color);
  ~AreaLight();
  
  /** Makes a diffuse area light from the given node. */
  static AreaLight* make(const Node& n);
  /** Makes a light instance for the current area light. Caller owns pointer. */
  Light* getLight() const;
};

