#pragma once
#include <optix.h>
#include <optix_cuda.h>
#include <optix_world.h>
#include "core.cuh"
#include "light.cuh"
#include "math.cuh"

using namespace optix;

rtDeclareVariable(Light, light, , ); 
rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(NormalRayData, normalRayData, rtPayload, );
rtDeclareVariable(float, isectDist, rtIntersectionDistance, );
rtDeclareVariable(float3, isectNormal, attribute isectNormal, );
rtDeclareVariable(int, isectHitId, attribute isectHitId, );
rtDeclareVariable(int, numLights, , ); 
rtBuffer<Light, 1> lightsBuffer;

__device__ void scatter(NormalRayData& rayData, float3 normal, float3 pos);
__device__ bool shouldDirectIlluminate() { return false; }

__device__ optix::float3 uniformSampleOneLight(
  const NormalRayData& data,
  const optix::float3& isectNormalObj,
  const optix::float3& isectPos
) {
  if (numLights == 0) {
    return optix::make_float3(0);
  }

  int lightIdx = int(floorf(math::nextFloat(data.rng, 0.0f, float(numLights) - XRAY_VERY_SMALL)));
  Light light = lightsBuffer[min(lightIdx, numLights - 1)];

  // P[this light] = 1 / numLights, so 1 / P[this light] = numLights.
  return float(numLights) * light.directIlluminate(
    data, isectNormalObj, isectPos
  );
}

RT_PROGRAM void radiance() {
  normalRayData.lastHitId = isectHitId;
  normalRayData.hitNormal = isectNormal;

  if (normalRayData.flags & RAY_SKIP_MATERIAL_COMPUTATION) {
    return;
  }

  if (math::isNaN(isectNormal)) {
    normalRayData.beta = make_float3(0);
    return;
  }

  float3 isectNormalObj = rtTransformNormal(RT_OBJECT_TO_WORLD, isectNormal); 
  float3 isectPos = normalRayData.origin + normalRayData.direction * isectDist;

  // Regular illumination on light at current step.
  if (light.id != -1 && !(normalRayData.flags & RAY_DID_DIRECT_ILLUMINATE)) {
    normalRayData.radiance += normalRayData.beta * light.emit(normalRayData.direction, isectNormalObj);
  }
  
  // Next event estimation with light at next step.
  if (shouldDirectIlluminate()) {
    normalRayData.radiance += normalRayData.beta * uniformSampleOneLight(normalRayData, isectNormalObj, isectPos);
    normalRayData.flags |= RAY_DID_DIRECT_ILLUMINATE;
  } else {
    normalRayData.flags &= ~RAY_DID_DIRECT_ILLUMINATE;
  }

  // Material evaluation at current step.
  scatter(normalRayData, isectNormalObj, isectPos);
}
