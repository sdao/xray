#include <optix.h>
#include <optix_cuda.h>
#include <optix_world.h>
#include "core.cuh"

/* Data for closest-hit programs. */
rtDeclareVariable(ShadowRayData, shadowRayData, rtPayload, );
rtDeclareVariable(float3, isectNormal, attribute isectNormal, );
rtDeclareVariable(int, isectHitId, attribute isectHitId, );

RT_PROGRAM void radiance() {
  shadowRayData.hitNormal = isectNormal;
  shadowRayData.lastHitId = isectHitId;
}