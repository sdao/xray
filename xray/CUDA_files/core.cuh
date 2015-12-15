#pragma once
#include <optix.h>
#include <curand_kernel.h>
#include "shared.cuh"

struct ShadowRayData {
  int targetID;
  bool hit;

  __device__ static ShadowRayData make(int id) {
    ShadowRayData data;
    data.targetID = id;
    data.hit = false;
    return data;
  }
};

enum NormalRayDataFlags {
  RAY_DID_DIRECT_ILLUMINATE = 0x1,
  RAY_SKIP_MATERIAL_COMPUTATION = 0x2
};

struct NormalRayData {
  optix::float3 origin;
  optix::float3 direction;
  optix::float3 hitNormal;

  optix::float3 radiance;
  optix::float3 beta;

  curandState* rng;
  int flags;
  int lastHitId;

  __device__ static NormalRayData make(optix::float3 or, optix::float3 dir, curandState* rng) {
    NormalRayData data;
    data.origin = or;
    data.direction = dir;
    data.beta = optix::make_float3(1);
    data.radiance = optix::make_float3(0);
    data.rng = rng;
    data.flags = 0;
    data.lastHitId = -1;
    return data;
  }
};