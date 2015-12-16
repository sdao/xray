#include <optix.h>
#include <optix_cuda.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include "math.cuh"
#include "core.cuh"

enum Sample {
  SAMPLE_RADIANCE = 0,
  SAMPLE_POSITION = 1
};

using namespace optix;

rtDeclareVariable(Matrix4x4, xform, , );
rtDeclareVariable(float3, focalPlaneOrigin, , ); // The lower-left corner of the focal rectangle in camera space.
rtDeclareVariable(float, focalPlaneRight, , ); // The vector pointing from the upper-left corner to the upper-right corner of the focal rectangle in camera space.
rtDeclareVariable(float, focalPlaneUp, , );
rtDeclareVariable(float, lensRadius, , );
rtDeclareVariable(unsigned int, frameNumber, , );
rtDeclareVariable(float3, backgroundColor, , );

rtDeclareVariable(rtObject, sceneRoot, , );
rtBuffer<float3, 3> rawBuffer;
rtBuffer<float4, 2> accumBuffer;
rtBuffer<uchar4, 2> imageBuffer;

rtDeclareVariable(NormalRayData, normalRayData, rtPayload, );

rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, launchDim, rtLaunchDim, );

/**
  * The number of bounces at which a ray is subject to Russian Roulette.
  */
#define RUSSIAN_ROULETTE_DEPTH 5

/**
  * Limits any given sample to the given amount of radiance. This helps to
  * reduce "fireflies" in the output. The lower this value, the more bias will
  * be introduced into the image. For unbiased rendering, set this to
  * std::numeric_limits<float>::max().
  */
#define BIASED_RADIANCE_CLAMPING 50.0

/** Filter radius. */
#define FILTER_WIDTH 2.0f

RT_PROGRAM void camera() {
  curandState rngState;
  curand_init((frameNumber * launchDim.x * launchDim.y) + (launchIndex.x * launchDim.y) + launchIndex.y, 0, 0, &rngState);
  
  float offsetX = math::nextFloat(&rngState, -FILTER_WIDTH, FILTER_WIDTH);
  float offsetY = math::nextFloat(&rngState, -FILTER_WIDTH, FILTER_WIDTH);

  float posX = float(launchIndex.x) + offsetX;
  float posY = float(launchIndex.y) + offsetY;

  float fracX = posX / float(launchDim.x - 1);
  float fracY = posY / float(launchDim.y - 1);

  float3 offset = make_float3(focalPlaneRight * fracX, focalPlaneUp * fracY, 0);
  float3 lookAt = focalPlaneOrigin + offset;

  float3 eye = make_float3(0, 0, 0);
  math::areaSampleDisk(&rngState, &eye.x, &eye.y);
  eye = lensRadius * eye;

  float3 eyeWorld = math::pointXform(eye, xform);
  float3 lookAtWorld = math::pointXform(lookAt, xform);
  float3 dir = normalize(lookAtWorld - eyeWorld);

  NormalRayData data = NormalRayData::make(eyeWorld, dir, &rngState);
  for (int depth = 0; ; ++depth) {
    optix::Ray ray = make_Ray(data.origin, data.direction, RAY_TYPE_NORMAL, XRAY_VERY_SMALL, RT_DEFAULT_MAX);
    rtTrace(sceneRoot, ray, data);

    // End path if the beta is black, since no point in continuing.
    if (data.flags & RAY_DEAD | math::isNearlyZero(data.beta)) {
      break;
    }

    // Do Russian Roulette if this path is "old".
    if (depth >= RUSSIAN_ROULETTE_DEPTH) {
      float rv = math::nextFloat(&rngState, 0.0f, 1.0f);
      float probLive = luminanceCIE(data.beta);

      if (rv < probLive) {
        // The ray lives (more energy = more likely to live).
        // Increase its energy to balance out probabilities.
        data.beta = data.beta / probLive;
      } else {
        // The ray dies.
        break;
      }
    }
  }

  data.radiance = make_float3(
    math::clamp(data.radiance.x, 0.0f, BIASED_RADIANCE_CLAMPING),
    math::clamp(data.radiance.y, 0.0f, BIASED_RADIANCE_CLAMPING),
    math::clamp(data.radiance.z, 0.0f, BIASED_RADIANCE_CLAMPING)
  );

  rawBuffer[make_uint3(SAMPLE_RADIANCE, launchIndex.x, launchIndex.y)] = data.radiance;
  rawBuffer[make_uint3(SAMPLE_POSITION, launchIndex.x, launchIndex.y)] = make_float3(posX, posY, 0);
}

RT_PROGRAM void commit() {
  float posX = launchIndex.x;
  float posY = launchIndex.y;

  int minX = clamp(int(ceilf(posX - FILTER_WIDTH - XRAY_VERY_SMALL)), 0, int(launchDim.x - 1));
  int maxX = clamp(int(floorf(posX + FILTER_WIDTH + XRAY_VERY_SMALL)), 0, int(launchDim.x - 1));
  int minY = clamp(int(ceilf(posY - FILTER_WIDTH - XRAY_VERY_SMALL)), 0, int(launchDim.y - 1));
  int maxY = clamp(int(floorf(posY + FILTER_WIDTH + XRAY_VERY_SMALL)), 0, int(launchDim.y - 1));
  
  float4 currentAccum = accumBuffer[launchIndex];

  for (int yy = minY; yy <= maxY; ++yy) {
    for (int xx = minX; xx <= maxX; ++xx) {
      float3 newRadiance = rawBuffer[make_uint3(SAMPLE_RADIANCE, xx, yy)];
      float3 newPosition = rawBuffer[make_uint3(SAMPLE_POSITION, xx, yy)];

      float weight = math::mitchellFilter(
        posX - newPosition.x,
        posY - newPosition.y,
        FILTER_WIDTH
      );

      currentAccum.x += newRadiance.x * weight;
      currentAccum.y += newRadiance.y * weight;
      currentAccum.z += newRadiance.z * weight;
      currentAccum.w += weight;
    }
  }

  accumBuffer[launchIndex] = currentAccum;

  float3 color = make_float3(currentAccum.x, currentAccum.y, currentAccum.z) / currentAccum.w;
  imageBuffer[launchIndex] = math::colorToBgra(color);
}

RT_PROGRAM void init() {
  accumBuffer[launchIndex] = make_float4(0);
}

RT_PROGRAM void miss() {
  normalRayData.radiance += backgroundColor * normalRayData.beta;
  normalRayData.beta = make_float3(0);
}
