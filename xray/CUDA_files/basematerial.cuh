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

__device__ void scatter(NormalRayData& rayData, float3 normal, float3 pos);

RT_PROGRAM void radiance() {
  if (math::isNaN(isectNormal)) {
    normalRayData.beta = make_float3(0);
    return;
  }

  float3 isectNormalObj = rtTransformNormal(RT_OBJECT_TO_WORLD, isectNormal); 
  float3 isectPos = normalRayData.origin + normalRayData.direction * isectDist;

  if (light.object != nullptr) {
    normalRayData.radiance += normalRayData.beta * light.emit(normalRayData.direction, isectNormalObj);
  }
  
  scatter(normalRayData, isectNormalObj, isectPos);
}