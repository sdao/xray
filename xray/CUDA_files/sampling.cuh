#pragma once
#include <optix.h>
#include <curand_kernel.h>
#include "math.cuh"

namespace sampling {
  
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

}