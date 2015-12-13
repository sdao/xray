#pragma once
#include <optix.h>
#include <curand_kernel.h>
#include "shared.cuh"

struct ShadowRayData {
  bool hit;

  __device__ static ShadowRayData make() {
    ShadowRayData data;
    data.hit = false;
    return data;
  }
};

struct NormalRayData {
  optix::float3 origin;
  optix::float3 direction;

  optix::float3 radiance;
  optix::float3 beta;

  curandState* rng;

  __device__ static NormalRayData make(optix::float3 or, optix::float3 dir, curandState* rng) {
    NormalRayData data;
    data.origin = or;
    data.direction = dir;
    data.beta = optix::make_float3(1);
    data.radiance = optix::make_float3(0);
    data.rng = rng;
    return data;
  }
};