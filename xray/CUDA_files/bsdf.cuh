#include <optix.h>
#include <optix_cuda.h>
#include <optix_world.h>
#include "core.cuh"
#include "math.cuh"
#include "shared.cuh"
#include "basematerial.cuh"

using namespace optix;

rtDeclareVariable(float3, albedo, , ); 

__device__ float3 evalBSDFLocal(const float3& incoming, const float3& outgoing);

__device__ float evalPDFLocal(const float3& incoming, const float3& outgoing);

__device__ void scatter(NormalRayData& rayData, float3 normal, float3 pos) {
  float3 outgoingWorld;
  float3 bsdf;
  float pdf;
  sampleWorld(rayData.rng, normal, -rayData.direction, &outgoingWorld, &bsdf, &pdf);

  float3 scale;
  if (pdf > 0.0f) {
    scale = bsdf * fabsf(dot(normal, outgoingWorld)) / pdf;
  } else {
    scale = make_float3(0, 0, 0);
  }

  rayData.origin = pos + outgoingWorld * XRAY_VERY_SMALL;
  rayData.direction = outgoingWorld;
  rayData.beta *= scale;
}