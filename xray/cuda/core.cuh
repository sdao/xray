#pragma once
#include <optix.h>
#include <curand_kernel.h>
#include "shared.cuh"

enum RayDataFlags {
  RAY_DID_DIRECT_ILLUMINATE = 0x1,
  RAY_DEAD = 0x2,
  RAY_SHADOW = 0x4
};

/** Payload for all tracing rays (both NEE and non-NEE). */ 
struct RayData {
  optix::float3 origin;      /**< The origin of the ray. */
  optix::float3 direction;   /**< The direction of the ray. */
  optix::float3 radiance;    /**< The collected radiance thus far. */
  union {
    optix::float3 beta;      /**< The energy remaining after bouncing. (N) */
    optix::float3 hitNormal; /**< The normal at the hit intersection. (S) */
  };
  union {
    curandState* rng;        /**< A pointer to the ray's thread RNG. (N) */
    int lastHitId;           /**< The ID of the closest-hit object. (S) */
  };
  int flags;                 /**< Bitwise flags containing extra ray info. */

  /** Makes a trace payload with given origin, direction, and RNG pointer. */
  __device__ static RayData makeTrace(
    optix::float3 or,
    optix::float3 dir,
    curandState* rng
  ) {
    RayData data;
    data.origin = or;
    data.direction = dir;
    data.radiance = optix::make_float3(0);
    data.beta = optix::make_float3(1);
    data.rng = rng;
    data.flags = 0;
    return data;
  }

  /** Makes an empty shadow payload indicating no hit. */
  __device__ static RayData makeShadow() {
    RayData data;
    data.hitNormal = optix::make_float3(0);
    data.lastHitId = -1;
    data.flags = RAY_SHADOW;
    return data;
  }
};