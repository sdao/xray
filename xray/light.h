#pragma once
#define NOMINMAX
#include "CUDA_files/light.cuh"
#include "xray.h"
#include "node.h"

class AreaLight {
  optix::float3 _color;
  Light _l;

public:
  AreaLight(Xray xray, optix::float3 color);
  ~AreaLight();
  
  static AreaLight* make(Xray xray, const Node& n);
  const Light* getLight() const;
};

