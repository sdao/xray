#pragma once
#include <optix.h>
#include <optix_cuda.h>
#include <optix_world.h>
#include "core.cuh"
#include "light.cuh"
#include "math.cuh"

using namespace optix;

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(NormalRayData, normalRayData, rtPayload, );
rtDeclareVariable(float, isectDist, rtIntersectionDistance, );
rtDeclareVariable(float3, isectNormal, attribute isectNormal, );
rtDeclareVariable(int, isectHitId, attribute isectHitId, );
rtDeclareVariable(Light, light, , ); 
rtDeclareVariable(int, numLights, , ); 
rtDeclareVariable(int, materialFlags, , );
rtBuffer<Light, 1> lightsBuffer;

typedef rtCallableProgramX<float3(const float3& /* incoming */, const float3& /* outgoing */)> evalBSDFLocalFunc;
rtDeclareVariable(evalBSDFLocalFunc, evalBSDFLocal, , ); 

typedef rtCallableProgramX<float(const float3& /* incoming */, const float3& /* outgoing */)> evalPDFLocalFunc;
rtDeclareVariable(evalPDFLocalFunc, evalPDFLocal, , ); 

typedef rtCallableProgramX<void(curandState* /* rng */, const float3& /* incoming */, float3* /* outgoingOut */, float3* /* bsdfOut */, float* /* pdfOut */)> sampleLocalFunc;
rtDeclareVariable(sampleLocalFunc, sampleLocal, , ); 

__device__ __inline__ void sampleWorld(
  curandState* rng,
  const float3& isectNormalObj,
  const float3& incoming,
  float3* outgoingOut,
  float3* bsdfOut,
  float* pdfOut
) {
  float3 tangent;
  float3 binormal;
  shared::coordSystem(isectNormalObj, &tangent, &binormal);

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

__device__ __inline__ void evalWorld(
  const optix::float3& isectNormalObj,
  const optix::float3& incomingWorld,
  const optix::float3& outgoingWorld,
  optix::float3* bsdfOut,
  float* pdfOut
) {
  float3 tangent;
  float3 binormal;
  shared::coordSystem(isectNormalObj, &tangent, &binormal);

  // BSDF and PDF computation expects rays to be in local-space.
  float3 incomingLocal = math::worldToLocal(
    incomingWorld,
    tangent,
    binormal,
    isectNormalObj
  );

  float3 outgoingLocal = math::worldToLocal(
    outgoingWorld,
    tangent,
    binormal,
    isectNormalObj
  );

  
  *bsdfOut = evalBSDFLocal(incomingLocal, outgoingLocal);
  *pdfOut = evalPDFLocal(incomingLocal, outgoingLocal);
}

__device__ __inline__ optix::float3 uniformSampleOneLight(
  const NormalRayData& data,
  const optix::float3& isectNormalObj,
  const optix::float3& isectPos
) {
  if (numLights == 0) {
    return optix::make_float3(0);
  }

  int lightIdx = int(floorf(math::nextFloat(data.rng, 0.0f, float(numLights) - XRAY_VERY_SMALL)));
  Light light = lightsBuffer[min(lightIdx, numLights - 1)];

  // P[this light] = 1 / numLights, so 1 / P[this light] = numLights.
  return float(numLights) * light.directIlluminate(
    data, isectNormalObj, isectPos
  );
}

__device__ __inline__ void scatter(NormalRayData& rayData, float3 normal, float3 pos) {
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

RT_PROGRAM void radiance() {
  if (normalRayData.flags & RAY_SKIP_MATERIAL_COMPUTATION) {
    normalRayData.hitNormal = isectNormal;
    normalRayData.lastHitId = isectHitId;
    return;
  }

  if (math::isNaN(isectNormal)) {
    normalRayData.flags |= RAY_DEAD;
    return;
  }

  float3 isectNormalObj = rtTransformNormal(RT_OBJECT_TO_WORLD, isectNormal); 
  float3 isectPos = normalRayData.origin + normalRayData.direction * isectDist;

  // Regular illumination on light at current step.
  int lume = !(normalRayData.flags & RAY_DID_DIRECT_ILLUMINATE) & (light.id != -1);
  normalRayData.radiance += lume * normalRayData.beta * light.emit(normalRayData.direction, isectNormalObj);
  
  // Next event estimation with light at next step.
  if (materialFlags & MATERIAL_DIRECT_ILLUMINATE) {
    normalRayData.radiance += normalRayData.beta * uniformSampleOneLight(normalRayData, isectNormalObj, isectPos);
    normalRayData.flags |= RAY_DID_DIRECT_ILLUMINATE;
  } else {
    normalRayData.flags &= ~RAY_DID_DIRECT_ILLUMINATE;
  }

  // Material evaluation at current step.
  if (materialFlags & MATERIAL_REFLECT) {
    scatter(normalRayData, isectNormalObj, isectPos);
  } else {
    normalRayData.flags |= RAY_DEAD;
  }
}
