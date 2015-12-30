#pragma once
#include <optix.h>
#include <curand_kernel.h>
#include "shared.cuh"

enum NormalRayDataFlags {
  RAY_DID_DIRECT_ILLUMINATE = 0x1,
  RAY_SKIP_MATERIAL_COMPUTATION = 0x2,
  RAY_DEAD = 0x4
};

struct NormalRayData {
  optix::float3 origin;
  optix::float3 direction;
  optix::float3 radiance;

  union {
    optix::float3 beta;
    optix::float3 hitNormal;
  };

  union {
    curandState* rng;
    int lastHitId;
  };

  int flags;

  __device__ static NormalRayData make(
    optix::float3 or,
    optix::float3 dir,
    curandState* rng
  ) {
    NormalRayData data;
    data.origin = or;
    data.direction = dir;
    data.radiance = optix::make_float3(0);
    data.beta = optix::make_float3(1);
    data.rng = rng;
    data.flags = 0;
    return data;
  }

  __device__ static NormalRayData makeShadow(
    optix::float3 or,
    optix::float3 dir
  ) {
    NormalRayData data;
    data.origin = or;
    data.direction = dir;
    data.radiance = optix::make_float3(0);
    data.hitNormal = optix::make_float3(0);
    data.lastHitId = -1;
    data.flags = RAY_SKIP_MATERIAL_COMPUTATION;
    return data;
  }
};