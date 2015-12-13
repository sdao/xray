#include <optix.h>
#include <optix_cuda.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <curand_kernel.h>
#include "math.cuh"
#include "core.cuh"
#include "sampling.cuh"

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

rtDeclareVariable(rtObject, sceneRoot, , );
rtBuffer<float3, 3> rawBuffer;
rtBuffer<float4, 2> accumBuffer;
rtBuffer<uchar4, 2> imageBuffer;

rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, launchDim, rtLaunchDim, );

RT_PROGRAM void camera() {
  curandState rngState;
  curand_init((frameNumber * launchDim.x * launchDim.y) + (launchIndex.x * launchDim.y) + launchIndex.y, 0, 0, &rngState);

  float fracX = float(launchIndex.x) / float(launchDim.x - 1);
  float fracY = float(launchIndex.y) / float(launchDim.y - 1);

  float3 offset = make_float3(focalPlaneRight * fracX, focalPlaneUp * fracY, 0);
  float3 lookAt = focalPlaneOrigin + offset;

  float3 eye = make_float3(0, 0, 0);
  sampling::areaSampleDisk(&rngState, &eye.x, &eye.y);
  eye = lensRadius * eye;

  float3 eyeWorld = math::pointXform(eye, xform);
  float3 lookAtWorld = math::pointXform(lookAt, xform);
  float3 dir = normalize(lookAtWorld - eyeWorld);
  
  optix::Ray ray = make_Ray(eyeWorld, dir, RAY_TYPE_NORMAL, XRAY_VERY_SMALL, RT_DEFAULT_MAX);

  NormalRayData data = NormalRayData::make();
  rtTrace(sceneRoot, ray, data);

  rawBuffer[make_uint3(SAMPLE_RADIANCE, launchIndex.x, launchIndex.y)] = data.radiance;
  rawBuffer[make_uint3(SAMPLE_POSITION, launchIndex.x, launchIndex.y)] = make_float3(launchIndex.x, launchIndex.y, 0);
}

RT_PROGRAM void commit() {
  float3 newRadiance = rawBuffer[make_uint3(SAMPLE_RADIANCE, launchIndex.x, launchIndex.y)];
  float3 newPosition = rawBuffer[make_uint3(SAMPLE_POSITION, launchIndex.x, launchIndex.y)];

  float4 currentAccum = accumBuffer[launchIndex];
  currentAccum = make_float4(
    currentAccum.x + newRadiance.x,
    currentAccum.y + newRadiance.y,
    currentAccum.z + newRadiance.z,
    currentAccum.w + 1.0f
  );
  accumBuffer[launchIndex] = currentAccum;
  float3 color = make_float3(currentAccum.x, currentAccum.y, currentAccum.z) / currentAccum.w;

  imageBuffer[launchIndex] = math::colorToRgba(color);
}

RT_PROGRAM void init() {
  accumBuffer[launchIndex] = make_float4(0);
}
