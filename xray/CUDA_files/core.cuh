#pragma once
#include <optix.h>

enum RayTypes {
  RAY_TYPE_NORMAL = 0,
  RAY_TYPE_SHADOW = 1
};

struct ShadowRayData {
  bool hit;

  __device__ static ShadowRayData make() {
    ShadowRayData data;
    data.hit = false;
    return data;
  }
};

struct NormalRayData {
  optix::float3 radiance;
  int depth;

  __device__ static NormalRayData make() {
    NormalRayData data;
    data.radiance = optix::make_float3(0);
    data.depth = 0;
    return data;
  }
};