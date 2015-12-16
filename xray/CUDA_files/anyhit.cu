#pragma once
#include <optix.h>
#include <optix_cuda.h>
#include <optixu/optixu_math_namespace.h>
#include "core.cuh"

rtDeclareVariable(NormalRayData, normalRayData, rtPayload, );

RT_PROGRAM void anyHit() {
	normalRayData.flags &= ~RAY_DEAD;
  rtTerminateRay();
}