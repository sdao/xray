#pragma once

#include <optix.h>
#include <optix_cuda.h>
#include <optixu/optixu_math_namespace.h>
#include "core.cuh";
#include "basematerial.cuh"

using namespace optix;

__device__ void scatter(NormalRayData& rayData, float3 normal, float3 pos) {
  normal.x = fabsf(normal.x);
  normal.y = fabsf(normal.y);
  normal.z = fabsf(normal.z);
	rayData.radiance += normal * rayData.beta;
  rayData.beta = make_float3(0);
}