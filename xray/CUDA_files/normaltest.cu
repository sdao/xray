#pragma once

#include <optix.h>
#include <optix_cuda.h>
#include <optixu/optixu_math_namespace.h>
#include "core.cuh";
#include "basematerial.cuh"

using namespace optix;

__device__ float3 evalBSDFLocal(const float3& incoming, const float3& outgoing) {
  return make_float3(0);
}

__device__ float evalPDFLocal(const float3& incoming, const float3& outgoing) {
  return 0.0f;
}

__device__ void sampleLocal(
  curandState* rng,
  const float3& incoming,
  float3* outgoingOut,
  float3* bsdfOut,
  float* pdfOut
) {
  *outgoingOut = make_float3(0);
  *bsdfOut = make_float3(0);
  *pdfOut = 0.0f;
}

__device__ void scatter(NormalRayData& rayData, float3 normal, float3 pos) {
  normal.x = fabsf(normal.x);
  normal.y = fabsf(normal.y);
  normal.z = fabsf(normal.z);
	rayData.radiance += normal * rayData.beta;
  rayData.beta = make_float3(0);
}

__device__ __inline__ bool shouldDirectIlluminate() {
  return false;
}