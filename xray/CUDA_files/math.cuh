#include <optix.h>
#include <limits>


namespace math {
  using namespace optix;

  __host__ __device__ __inline__ bool isPositive(float x) {
    return x > std::numeric_limits<float>::epsilon();
  }

  __host__ __device__ __inline__ float3 at(const optix::Ray& r, float t) {
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