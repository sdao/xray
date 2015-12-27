#pragma once
#include <optix.h>
#include <optix_world.h>

/** A very small nonzero value. */
#define XRAY_VERY_SMALL 0.0001f
/** An extremely small nonzero value. */
#define XRAY_EPSILON 0.000000000001f
/** Pi/4 as a float. **/
#define XRAY_PI_4 0.7853981633974483f
/** 1/Pi as a float. */
#define XRAY_INV_PI 0.3183098861837907f
/** 2 * Pi as a single-precision float. */
#define XRAY_TWO_PI 6.2831853071795865f
/** 4 * Pi as a single-precision float. */
#define XRAY_FOUR_PI 12.5663706143591730f
/** The number of steradians in a sphere (4 * Pi). */
#define XRAY_STERADIANS_PER_SPHERE XRAY_FOUR_PI

enum RayTypes {
  RAY_TYPE_NORMAL = 0,
  RAY_TYPE_NEXT_EVENT_ESTIMATION = 1,
  RAY_TYPE_COUNT = 2
};

enum MaterialFlags {
  MATERIAL_REFLECT = 0x1,
  MATERIAL_DIRECT_ILLUMINATE = 0x2
};

namespace shared {
  using namespace optix;

  __host__ __device__ __inline__ Matrix4x4 rotationThenTranslation(float angle, float3 axis, float3 offset) {
    return Matrix4x4::translate(offset) * Matrix4x4::rotate(angle, axis);
  }

  /**
   * Generates an orthonormal coordinate basis. The first vector must be given,
   * and the other two orthogonal vectors will be generated from it.
   * Taken from page 63 of Pharr & Humphreys' Physically-Based Rendering.
   *
   * @param v1 [in]  the first unit (normalized) vector of the basis
   * @param v2 [out] the second unit vector, generated from the first
   * @param v3 [out] the third unit vector, generated from the first
   */
  __host__ __device__ __inline__
  void coordSystem(const float3& v1, float3* v2, float3* v3) {
    if (fabsf(v1.x) > fabsf(v1.y)) {
      float invLen = 1.0f / sqrtf(v1.x * v1.x + v1.z * v1.z);
      *v2 = make_float3(-v1.z * invLen, 0.0f, v1.x * invLen);
    } else {
      float invLen = 1.0f / sqrtf(v1.y * v1.y + v1.z * v1.z);
      *v2 = make_float3(0.0f, v1.z * invLen, -v1.y * invLen);
    }
    *v3 = cross(v1, *v2);
  }

}
