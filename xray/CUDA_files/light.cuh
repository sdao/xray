#pragma once
#include <optix.h>
#include <optix_world.h>
#include "shared.cuh"

struct Light {
  optix::float3 color;
  RTobject object;

  __host__ static Light* make(optix::float3 c, RTobject obj) {
    Light* l = new Light();
    l->color = c;
    l->object = obj;
    return l;
  }

  __host__ static Light* make() {
    Light* l = new Light();
    l->object = nullptr;
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
