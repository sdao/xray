#pragma once
#define NOMINMAX
#include "xray.h"
#include "node.h"
#include "CUDA_files/light.cuh"

class AreaLight {
  optix::float3 _color;

public:
  AreaLight(Xray* xray, optix::float3 color);
  ~AreaLight();
  
  static AreaLight* make(Xray* xray, const Node& n);
  Light* getLight() const;
};

