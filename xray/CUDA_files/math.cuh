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
   * Linearly interpolates between x and y. Where a = 0, x is returned, and
   * where a = 1, y is returned. If a < 0 or a > 1, this function will
   * extrapolate.
   */
  __device__ __inline__ float lerp(float x, float y, float a) {
    return x + a * (y - x);
  }

  /** Clamps a value x between a and b. */
  __device__ __inline__ float clamp(float x, float a = 0.0f, float b = 1.0f) {
    return x < a ? a : (x > b ? b : x);
  }

  /** Clamps a value x between a and b. */
  template< class T >
  __device__ __inline__ T clampAny(T x, T a, T b) {
    return x < a ? a : (x > b ? b : x);
  }

  /**
   * Linearly interpolates between x and y. Where a <= 0, x is returned, and
   * where a >= 1, y is returned. No extrapolation will occur.
   */
  __device__ __inline__ float clampedLerp(float x, float y, float a) {
    return lerp(x, y, clamp(a));
  }

  __device__ __inline__ float3 at(const optix::Ray& r, float t) {
    return r.origin + t * r.direction;
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
  __device__ __inline__
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

  __device__ __inline__ uchar4 colorToRgba(const float3& c) {
    return optix::make_uchar4(
      static_cast<unsigned char>(saturate(c.x) * 255.99f), /* R */
      static_cast<unsigned char>(saturate(c.y) * 255.99f), /* G */
      static_cast<unsigned char>(saturate(c.z) * 255.99f), /* B */
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
    float sx = nextFloat(rngState, -1.0f, 1.0f);
    float sy = nextFloat(rngState, -1.0f, 1.0f);

    // Handle degeneracy at the origin.
    if (sx == 0.0f && sy == 0.0f) {
      *dx = 0.0f;
      *dy = 0.0f;
      return;
    }

    float r;
    float theta;
    if (sx >= -sy) {
      if (sx > sy) {
        // Region 1.
        r = sx;
        if (sy > 0.0f) {
          theta = sy / r;
        } else {
          theta = 8.0f + sy / r;
        }
      } else {
        // Region 2.
        r = sy;
        theta = 2.0f - sx / r;
      }
    } else {
      if (sx <= sy) {
        // Region 3.
        r = -sx;
        theta = 4.0f - sy / r;
      } else {
        // Region 4.
        r = -sy;
        theta = 6.0f + sx / r;
      }
    }
    theta *= XRAY_PI_4;
    *dx = r * cosf(theta);
    *dy = r * sinf(theta);
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

    if (x > 1.0f) {
      return ((-B - 6.0f * C) * (x * x * x)
        + (6.0f * B + 30.0f * C) * (x * x)
        + (-12.0f * B - 48.0f * C) * x
        + (8.0f * B + 24.0f * C)) * (1.0f / 6.0f);
    } else {
      return ((12.0f - 9.0f * B - 6.0f * C) * (x * x * x)
        + (-18.0f + 12.0f * B + 6.0f * C) * (x * x)
        + (6.0f - 2.0f * B)) * (1.0f / 6.0f);
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

}