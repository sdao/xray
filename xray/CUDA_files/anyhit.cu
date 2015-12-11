#pragma once
#include <optix.h>
#include <optix_cuda.h>
#include <optixu/optixu_math_namespace.h>
#include "core.cuh"

rtDeclareVariable(ShadowRayData, shadowRayData, rtPayload, );

RT_PROGRAM void anyHit() {
	shadowRayData.hit = true;
  rtTerminateRay();
}