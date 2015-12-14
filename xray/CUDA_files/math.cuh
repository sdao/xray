#pragma once
#define _USE_MATH_DEFINES
#include <optix.h>
#include <curand_kernel.h>
#include <cmath>
#include "shared.cuh"

namespace math {
  using namespace optix;

  __device__ __inline__ bool isPositive(float x) {
    return x > XRAY_EXTREMELY_SMALL;
  }

  /**
   * Determines whether a number is zero, within a small epsilon.
   */
  __device__ __inline__ bool isNearlyZero(float x) {
    return fabsf(x) < XRAY_EXTREMELY_SMALL;
  }

  /**
   * Determines whether a vec's magnitude is zero, within a small epsilon.
   */
  __device__ __inline__ bool isNearlyZero(const float3& v) {
    return isNearlyZero(v.x * v.x + v.y * v.y + v.z * v.z);
  }

  /**
   * Linearly interpolates between x and y. Where a <= 0, x is returned, and
   * where a >= 1, y is returned. No extrapolation will occur.
   */
  __device__ __inline__ float clampedLerp(float x, float y, float a) {
    return lerp(x, y, clamp(a, 0.0f, 1.0f));
  }

  __device__ __inline__ float3 at(const optix::Ray& r, float t) {
    return r.origin + t * r.direction;
  }

  __device__ __inline__ float3 pointXform(float3 v, Matrix4x4 xform) {
    float4 world = xform * make_float4(v.x, v.y, v.z, 1);
    return make_float3(world.x / world.w, world.y / world.w, world.z / world.w);
  }

  __device__ __inline__ float3 vectorXform(float3 v, Matrix4x4 xform) {
    float4 world = xform * make_float4(v.x, v.y, v.z, 0);
    return make_float3(world.x, world.y, world.z);
  }

  __device__ __inline__ float saturate(float x) {
    return x < 0 ? 0 : (x > 1 ? 1 : x);
  }

  __device__ __inline__ uchar4 colorToBgra(const float3& c) {
    return optix::make_uchar4(
      static_cast<unsigned char>(saturate(c.z) * 255.99f), /* B */
      static_cast<unsigned char>(saturate(c.y) * 255.99f), /* G */
      static_cast<unsigned char>(saturate(c.x) * 255.99f), /* R */
      255u /* A */
    );                                                 
  }

  /**
   * Converts a world-space vector to a local coordinate system.
   * The resulting coordinates are (x, y, z), where x is the weight of the
   * tangent, y is the weight of the binormal, and z is the weight of the
   * normal.
   */
  __device__ __inline__ float3 worldToLocal(
    const float3& world,
    const float3& tangent,
    const float3& binormal,
    const float3& normal
  ) {
    return make_float3(
      dot(world, tangent),
      dot(world, binormal),
      dot(world, normal)
    );
  }

  /**
   * Converts a local-space vector back to world-space. The local-space vector
   * should be (x, y, z), where x is the weight of the tangent, y is the weight
   * of the binormal, and z is the weight of the normal.
   */
  __device__ __inline__ float3 localToWorld(
    const float3& local,
    const float3& tangent,
    const float3& binormal,
    const float3& normal
  ) {
    return make_float3(
      tangent.x * local.x + binormal.x * local.y
        + normal.x * local.z,
      tangent.y * local.x + binormal.y * local.y
        + normal.y * local.z,
      tangent.z * local.x + binormal.z * local.y
        + normal.z * local.z
    );
  }

  /**
   * Returns Cos[Theta] of a vector where Theta is the polar angle of the vector
   * in spherical coordinates.
   */
  __device__ __inline__ float cosTheta(const float3& v) { return v.z; }

  /**
   * Returns Abs[Cos[Theta]] of a vector where Theta is the polar angle of the
   * vector in spherical coordinates.
   */
  __device__ __inline__ float absCosTheta(const float3& v) { return fabsf(v.z); }

  __device__ __inline__ float nextFloat(curandState* rngState, float min, float max) {
    return min + (max - min) * curand_uniform(rngState);
  }

  __device__ __inline__ float2 nextFloat2(curandState* rngState, float min, float max) {
    return make_float2(nextFloat(rngState, min, max), nextFloat(rngState, min, max));
  }

  /**
   * Samples a unit disk, ensuring that the samples are uniformally distributed
   * throughout the area of the disk.
   *
   * Taken from Pharr & Humphreys' p. 667.
   *
   * @param rng      the per-thread RNG in use
   * @param dx [out] the x-coordinate of the sample
   * @param dy [out] the y-coordinate of the sample
   */
  __device__ __inline__ void areaSampleDisk(curandState* rngState, float* dx, float* dy) {
    float2 sample = square_to_disk(nextFloat2(rngState, 0.0f, 1.0f));
    *dx = sample.x;
    *dy = sample.y;
  }

  /**
   * Returns the probability that the given direction was sampled from a unit
   * hemisphere using a cosine-weighted distribution. (It does not matter
   * whether the hemisphere is on the positive or negative Z-axis.)
   */
  __device__ __inline__ float cosineSampleHemispherePDF(const float3& direction) {
    return absCosTheta(direction) * XRAY_INV_PI;
  }

  /**
   * Samples a unit hemisphere with a cosine-weighted distribution.
   * Directions with a higher cosine value (more parallel to the normal) are
   * more likely to be chosen than those with a lower cosine value (more
   * perpendicular to the normal).
   *
   * Taken from Pharr & Humphreys p. 669.
   *
   * @param rng     the per-thread RNG in use
   * @param flipped whether to sample from the hemisphere on the negative
   *                Z-axis instead; false will sample from the positive
   *                hemisphere and true will sample from the negative hemisphere
   * @returns       a cosine-weighted random vector in the hemisphere;
   *                the pointer must not be null
   */
  __device__ __inline__ float3 cosineSampleHemisphere(curandState* rngState, bool flipped) {
    float3 ret;
    areaSampleDisk(rngState, &ret.x, &ret.y);
    ret.z = sqrtf(max(0.0f, 1.0f - ret.x * ret.x - ret.y * ret.y));
    if (flipped) {
      ret.z *= -1.0f;
    }
    return ret;
  }

  /**
   * Determines if two vectors in the same local coordinate space are in the
   * same hemisphere.
   */
  __device__ __inline__ bool localSameHemisphere(const float3& u, const float3& v) {
    return u.z * v.z >= 0.0f;
  }

  /**
   * Computes the 1-dimensional Mitchell filter with B = 1/3 and C = 1/3 for a
   * scaled offset from the pixel center. The values are not normalized.
   *
   * Pharr and Humphreys suggest on p. 398 of PBR that values of B and C should
   * be chosen such that B + 2C = 1.
   * GPU Gems <http://http.developer.nvidia.com/GPUGems/gpugems_ch24.html>
   * suggests the above values of B = 1/3 and C = 1/3.
   *
   * @param x the scaled x-offset from the pixel center, -1 <= x <= 1
   */
  __device__ __inline__ float mitchellFilter(float x) {
    const float B = 1.0f / 3.0f;
    const float C = 1.0f / 3.0f;

    x = fabsf(2.0f * x); // Convert to the range [0, 2].

    if (x > 1.0f && x < 2.0f) {
      return ((-B - 6.0f * C) * (x * x * x)
        + (6.0f * B + 30.0f * C) * (x * x)
        + (-12.0f * B - 48.0f * C) * x
        + (8.0f * B + 24.0f * C)) * (1.0f / 6.0f);
    } else if (x < 1.0f) {
      return ((12.0f - 9.0f * B - 6.0f * C) * (x * x * x)
        + (-18.0f + 12.0f * B + 6.0f * C) * (x * x)
        + (6.0f - 2.0f * B)) * (1.0f / 6.0f);
    } else {
      return 0.0f;
    }
  }

  /**
   * Evaluates a 2-dimensional Mitchell filter at a specified offset from the
   * pixel center by separating and computing the 1-dimensional Mitchell
   * filter for the x- and y- offsets.
   *
   * @param x     the x-offset from the pixel center, -width <= x <= width
   * @param y     the y-offset from the pixel center, -width <= x <= width
   * @param width the maximum x- or y- offset sampled from the pixel center
   * @returns the value of the filter
   */
  __device__ __inline__ float mitchellFilter(float x, float y, float width = 2.0f) {
    return mitchellFilter(x / width) * mitchellFilter(y / width);
  }

  __device__ __inline__ bool isNaN(float x) {
    return x != x;
  }

  __device__ __inline__ bool isNaN(const float3& v) {
    return isNaN(v.x) | isNaN(v.y) | isNaN(v.z);
  }

}