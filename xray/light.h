#pragma once
#define NOMINMAX
#include "node.h"
#include "cuda/light.cuh"

class AreaLight {
  optix::float3 _color;

public:
  AreaLight(optix::float3 color);
  ~AreaLight();
  
  static AreaLight* make(const Node& n);
  Light* getLight() const;
};

