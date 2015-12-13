#pragma once
#include <optix.h>
#include <optix_world.h>

/** A very small nonzero value. */
#define XRAY_VERY_SMALL 0.0001f
/** An extremely small nonzero value. */
#define XRAY_EXTREMELY_SMALL 0.000000000001f
/** Pi/4 as a float. **/
#define XRAY_PI_4 0.7853981633974483f
/** 1/Pi as a float. */
#define XRAY_INV_PI 0.3183098861837907f
/** 1/Pi as a float. */
#define XRAY_TWO_PI 6.2831853071795865f

enum RayTypes {
  RAY_TYPE_NORMAL = 0,
  RAY_TYPE_SHADOW = 1
};

namespace shared {
  using namespace optix;

  __host__ __device__ __inline__ Matrix4x4 rotationThenTranslation(float angle, float3 axis, float3 offset) {
    return Matrix4x4::translate(offset) * Matrix4x4::rotate(angle, axis);
  }

}
