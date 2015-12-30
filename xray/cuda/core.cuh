#pragma once
#include <optix.h>
#include <curand_kernel.h>
#include "shared.cuh"

enum NormalRayDataFlags {
  RAY_DID_DIRECT_ILLUMINATE = 0x1,
  RAY_DEAD = 0x2
};

/** Payload for all tracing rays (both NEE and non-NEE). */ 
struct NormalRayData {
  optix::float3 origin;    /**< The origin of the ray. */
  optix::float3 direction; /**< The direction of the ray. */
  optix::float3 radiance;  /**< The collected radiance thus far. */
  optix::float3 beta;      /**< The energy remaining after bouncing. */
  curandState* rng;        /**< A pointer to the ray's thread RNG. */
  int flags;               /**< Bitwise flags containing extra ray info. */

  /** Makes a payload with the given origin, direction, and RNG pointer. */
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
};

/** Payload for a single-hit shadow ray. */
struct ShadowRayData {
  optix::float3 hitNormal; /**< The normal at the hit intersection. */
  int lastHitId;           /**< The ID of the closest-hit object. */
  
  /** Makes an empty payload indicating no hit. */
  __device__ static ShadowRayData make() {
    ShadowRayData data;
    data.hitNormal = optix::make_float3(0);
    data.lastHitId = -1;
    return data;
  }
};