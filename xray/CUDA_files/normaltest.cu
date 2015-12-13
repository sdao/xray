#pragma once

#include <optix.h>
#include <optix_cuda.h>
#include <optixu/optixu_math_namespace.h>
#include "core.cuh";

using namespace optix;

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(NormalRayData, normalRayData, rtPayload, );
rtDeclareVariable(float3, isectNormal, attribute isectNormal, ); 

RT_PROGRAM void normalTestRadiance()
{
  float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, isectNormal));
  n.x = fabsf(n.x);
  n.y = fabsf(n.y);
  n.z = fabsf(n.z);
	normalRayData.radiance += n * normalRayData.beta;
  normalRayData.beta = make_float3(0);
}