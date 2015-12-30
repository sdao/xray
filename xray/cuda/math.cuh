#pragma once
#define _USE_MATH_DEFINES
#include <optix.h>
#include <curand_kernel.h>
#include <cmath>
#include "shared.cuh"

namespace math {
  using namespace optix;

  /**
   * Determines whether a number is positive, above a small epsilon.
   */
  __device__ __inline__ bool isPositive(float x) {
    return x > XRAY_EPSILON;
  }

  /**
   * Determines whether a number is zero, within a small epsilon.
   */
  __device__ __inline__ bool isNearlyZero(float x) {
    return fabsf(x) < XRAY_EPSILON;
  }

  /**
   * Determines whether a vec's magnitude is zero, within a small epsilon.
   */
  __device__ __inline__ bool isNearlyZero(const float3& v) {
    return isNearlyZero(dot(v, v));
  }

  /**
   * Determines whether a vec's magnitude is exactly zero, with no epsilon
   * check.
   */
  __device__ __inline__ bool isVectorExactlyZero(const float3& v) {
    return v.x == 0.0f & v.y == 0.0f & v.z == 0.0f;
  }

  /**
   * Linearly interpolates between x and y. Where a <= 0, x is returned, and
   * where a >= 1, y is returned. No extrapolation will occur.
   */
  __device__ __inline__ float clampedLerp(float x, float y, float a) {
    return lerp(x, y, clamp(a, 0.0f, 1.0f));
  }

  /**
   * Computes a point along a ray at the given parameter.
   */
  __device__ __inline__ float3 at(const optix::Ray& r, float t) {
    return r.origin + t * r.direction;
  }

  /**
   * Transforms a point stored in a 3-comp vec using the specified transform.
   */
  __device__ __inline__ float3 pointXform(float3 v, Matrix4x4 xform) {
    float4 world = xform * make_float4(v.x, v.y, v.z, 1);
    return make_float3(world.x / world.w, world.y / world.w, world.z / world.w);
  }

  /**
   * Transforms a vector stored in a 3-comp vec using the specified transform.
   */
  __device__ __inline__ float3 vectorXform(float3 v, Matrix4x4 xform) {
    float4 world = xform * make_float4(v.x, v.y, v.z, 0);
    return make_float3(world.x, world.y, world.z);
  }

  /**
   * Clamps the given value to the range [0, 1].
   */
  __device__ __inline__ float saturate(float x) {
    return clamp(x, 0.0f, 1.0f);
  }

  /**
   * Converts a floating-point RGB color vector to its BGRA integer
   * representation.
   */
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
  __device__ __inline__
  float absCosTheta(const float3& v) { return fabsf(v.z); }

  /**
   * Returns a random float in the range (min, max].
   */
  __device__ __inline__
  float nextFloat(curandState* rngState, float min, float max) {
    return min + (max - min) * curand_uniform(rngState);
  }

  /**
   * Returns a random 2-vector with components in the range (min, max].
   */
  __device__ __inline__
  float2 nextFloat2(curandState* rngState, float min, float max) {
    return make_float2(
      nextFloat(rngState, min, max),
      nextFloat(rngState, min, max)
    );
  }

  /**
   * Returns a random integer in the range [min, max).
   */
  __device__ __inline__
  int nextInt(curandState* rngState, int min, int max) {
    return int(floorf(
      math::nextFloat(rngState, float(min), float(max) - XRAY_VERY_SMALL)
    ));
  }

  /**
   * Returns a random Gaussian-distributed (normal) float using the standard
   * normal distribution (mean=0, stdev=1).
   */
  __device__ __inline__ float nextGaussian(curandState* rngState) {
    return curand_normal(rngState);
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
  __device__ __inline__
  void areaSampleDisk(curandState* rngState, float* dx, float* dy) {
    float2 sample = square_to_disk(nextFloat2(rngState, 0.0f, 1.0f));
    *dx = sample.x;
    *dy = sample.y;
  }

  /**
   * Returns the probability that the given direction was sampled from a unit
   * hemisphere using a cosine-weighted distribution. (It does not matter
   * whether the hemisphere is on the positive or negative Z-axis.)
   */
  __device__ __inline__
  float cosineSampleHemispherePDF(const float3& direction) {
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
  __device__ __inline__
  float3 cosineSampleHemisphere(curandState* rngState, bool flipped) {
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
  __device__ __inline__
  bool localSameHemisphere(const float3& u, const float3& v) {
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

    if (x > 1.0f & x < 2.0f) {
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
  __device__ __inline__
  float mitchellFilter(float x, float y, float width = 2.0f) {
    return mitchellFilter(x / width) * mitchellFilter(y / width);
  }

  /**
   * Determines whether the value is NaN (Not-a-Number).
   */
  __device__ __inline__ bool isNaN(float x) {
    return x != x;
  }

  /**
   * Determines whether any of the components of the vector is NaN.
   */
  __device__ __inline__ bool isNaN(const float3& v) {
    return isNaN(v.x) | isNaN(v.y) | isNaN(v.z);
  }

  /**
   * Calculates the power heuristic for multiple importance sampling of
   * two separate functions.
   *
   * See Pharr & Humphreys p. 693 for more information.
   *
   * @param nf   number of samples taken with a Pf distribution
   * @param fPdf probability according to the Pf distribution
   * @param ng   number of samples taken with a Pg distribution
   * @param gPdf probability according to the Pg distribution
   * @returns    the weight according to the power heuristic
   */
  __device__ __inline__
  float powerHeuristic(int nf, float fPdf, int ng, float gPdf) {
    float f = nf * fPdf;
    float g = ng * gPdf;

    return (f * f) / (f * f + g * g);
  }

  /**
   * Determines whether a point is contained within the sphere.
   *
   * @param or the origin of the sphere
   * @param r  the radius of the sphere
   * @param v  the point to check
   * @returns  whether v is in the sphere defined by or and r
   */
  __device__ __inline__
  bool sphereContains(const float3& or, float r, const float3& v) {
    return dot(v - or, v - or) <= (r * r);
  }

  /**
   * Returns the probability that any solid angle was sampled uniformly
   * from a unit sphere.
   */
  __device__ __inline__ float uniformSampleSpherePDF() {
    return 1.0f / XRAY_STERADIANS_PER_SPHERE;
  }

  /**
   * Uniformly samples from a unit sphere, with respect to the sphere's
   * surface area.
   *
   * @param rng the per-thread RNG in use
   * @returns   the uniformly-distributed random vector in the sphere
   */
  __device__ __inline__ float3 uniformSampleSphere(curandState* rng) {
    // See MathWorld <http://mathworld.wolfram.com/SpherePointPicking.html>.
    float x = nextGaussian(rng);
    float y = nextGaussian(rng);
    float z = nextGaussian(rng);
    float a = 1.0f / sqrtf(x * x + y * y + z * z);

    return make_float3(a * x, a * y, a * z);
  }

  /**
   * Returns the probability that any solid angle already inside the given
   * cone was sampled uniformly from the cone. The cone is defined by the
   * half-angle of the subtended (apex) angle.
   *
   * @param halfAngle the half-angle of the cone
   * @returns         the probability that the angle was sampled
   */
  __device__ __inline__ float uniformSampleConePDF(float halfAngle) {
    const float solidAngle = XRAY_TWO_PI * (1.0f - cosf(halfAngle));
    return 1.0f / solidAngle;
  }

  /**
   * Returns the proabability that the given solid angle was sampled
   * uniformly from the given cone. The cone is defined by the half-angle of
   * the subtended (apex) angle. The probability is uniform if the direction
   * is actually in the cone, and zero if it is outside the cone.
   *
   * @param halfAngle the half-angle of the cone
   * @param direction the direction of the sampled vector
   * @returns         the probability that the angle was sampled
   */
  __device__ __inline__
  float uniformSampleConePDF(float halfAngle, const float3& direction) {
    const float cosHalfAngle = cosf(halfAngle);
    const float solidAngle = XRAY_TWO_PI * (1.0f - cosHalfAngle);
    if (cosTheta(direction) > cosHalfAngle) {
      // Within the sampling cone.
      return 1.0f / solidAngle;
    } else {
      // Outside the sampling cone.
      return 0.0f;
    }
  }

  /**
   * Generates a random ray in a cone around the positive z-axis, uniformly
   * with respect to solid angle.
   *
   * Handy Mathematica code for checking that this works:
   * \code
   * R[a_] := (h = Cos[Pi/2];
   *   z = RandomReal[{h, 1}];
   *   t = RandomReal[{0, 2*Pi}];
   *   r = Sqrt[1 - z^2];
   *   x = r*Cos[t];
   *   y = r*Sin[t];
   *   {x, y, z})
   *
   * ListPointPlot3D[Map[R, Range[1000]], BoxRatios -> Automatic]
   * \endcode
   *
   * @param rng       the per-thread RNG in use
   * @param halfAngle the half-angle of the cone's opening; must be between 0
   *                  and Pi/2 and in radians
   * @returns         a uniformally-random vector within halfAngle radians of
   *                  the positive z-axis
   */
  __device__ __inline__
  float3 uniformSampleCone(curandState* rng, float halfAngle) {
    float h = cosf(halfAngle);
    float z = nextFloat(rng, h, 1.0f);
    float t = nextFloat(rng, 0.0f, XRAY_TWO_PI);
    float r = sqrtf(1.0f - (z * z));
    float x = r * cosf(t);
    float y = r * sinf(t);

    return make_float3(x, y, z);
  }

}