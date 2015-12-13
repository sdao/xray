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

__device__ void sampleLocal(
  curandState* rng,
  const float3& incoming,
  float3* outgoingOut,
  float3* bsdfOut,
  float* pdfOut
);

__device__ void sampleWorld(
  curandState* rng,
  const float3& isectNormalObj,
  const float3& incoming,
  float3* outgoingOut,
  float3* bsdfOut,
  float* pdfOut
) {
  float3 tangent;
  float3 binormal;
  math::coordSystem(isectNormalObj, &tangent, &binormal);

  // BSDF computation expects incoming ray to be in local-space.
  float3 incomingLocal = math::worldToLocal(
    incoming,
    tangent,
    binormal,
    isectNormalObj
  );

  // Sample BSDF for direction, color, and probability.
  float3 outgoingLocal;
  float3 tempBsdf;
  float tempPdf;
  sampleLocal(rng, incomingLocal, &outgoingLocal, &tempBsdf, &tempPdf);

  // Rendering expects outgoing ray to be in world-space.
  float3 outgoingWorld = math::localToWorld(
    outgoingLocal,
    tangent,
    binormal,
    isectNormalObj
  );

  *outgoingOut = outgoingWorld;
  *bsdfOut = tempBsdf;
  *pdfOut = tempPdf;
}

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