#pragma once
#include <optix.h>
#include <optix_world.h>
#include "shared.cuh"

struct Light {
  optix::float3 color;
  bool valid;

  __host__ __device__ static Light make(optix::float3 c) {
    Light l;
    l.color = c;
    l.valid = true;
    return l;
  }

  __host__ __device__ static Light make() {
    Light l;
    l.valid = false;
    return l;
  }

  __device__ optix::float3 emit(const optix::float3& dir, const optix::float3& n) {
    // Only emit on the normal-facing side of objects, e.g. on the outside of a
    // sphere or on the normal side of a disc.
    if (dot(dir, n) > XRAY_EXTREMELY_SMALL) {
      return optix::make_float3(0);
    }

    return color;
  }
};
