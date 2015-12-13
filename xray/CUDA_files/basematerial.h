#include <optix.h>
#include <optix_cuda.h>
#include <optix_world.h>
#include "core.cuh"
#include "math.cuh"
#include "shared.cuh"

using namespace optix;

rtDeclareVariable(float3, albedo, , ); 

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(NormalRayData, normalRayData, rtPayload, );
rtDeclareVariable(float, isectDist, rtIntersectionDistance, );
rtDeclareVariable(float3, isectNormal, attribute isectNormal, ); 

__device__ float3 evalBSDFLocal(const float3& incoming, const float3& outgoing) {
  if (!math::localSameHemisphere(incoming, outgoing)) {
    return make_float3(0);
  }

  return albedo * XRAY_INV_PI;
}

__device__ void sampleLocal(
  const float3& incoming,
  float3* outgoingOut,
  float3* bsdfOut,
  float* pdfOut
) {
  float3 outgoing = math::cosineSampleHemisphere(normalRayData.rng, incoming.z < 0.0f);

  *outgoingOut = outgoing;
  *bsdfOut = evalBSDFLocal(incoming, outgoing);
  *pdfOut = math::cosineSampleHemispherePDF(outgoing);
}

__device__ void sampleWorld(
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
  sampleLocal(incomingLocal, &outgoingLocal, &tempBsdf, &tempPdf);

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

RT_PROGRAM void radiance() {
  float3 isectNormalObj = rtTransformNormal(RT_OBJECT_TO_WORLD, isectNormal); 
  float3 isectPos = normalRayData.origin + normalRayData.direction * isectDist;

  float3 outgoingWorld;
  float3 bsdf;
  float pdf;
  sampleWorld(isectNormalObj, -normalRayData.direction, &outgoingWorld, &bsdf, &pdf);

  float3 scale;
  if (pdf > 0.0f) {
    scale = bsdf * fabsf(dot(isectNormalObj, outgoingWorld)) / pdf;
  } else {
    scale = make_float3(0, 0, 0);
  }

  normalRayData.origin = isectPos + outgoingWorld * XRAY_VERY_SMALL;
  normalRayData.direction = outgoingWorld;
  normalRayData.beta *= scale;
}