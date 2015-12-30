#pragma once
#include <optix.h>
#include <optix_cuda.h>
#include <optix_world.h>
#include "core.cuh"
#include "light.cuh"
#include "math.cuh"

using namespace optix;

rtDeclareVariable(NormalRayData, normalRayData, rtPayload, );
rtDeclareVariable(float, isectDist, rtIntersectionDistance, );
rtDeclareVariable(float3, isectNormal, attribute isectNormal, );
rtDeclareVariable(int, isectHitId, attribute isectHitId, );
rtDeclareVariable(Light, light, , ); 
rtDeclareVariable(unsigned int, numLights, , ); 
rtDeclareVariable(int, materialFlags, , );
rtBuffer<Light, 1> lightsBuffer;

/**
 * Evaluates the BSDF of an incoming and outgoing ray direction in the local
 * space oriented with the Z-axis along the surface normal of an intersection.
 * Materials must implement this callable function.
 *
 * @param incoming the incoming ray direction, in local space
 * @param outgoing the outgoing ray direction, in local space
 * @returns the evaluated BSDF as a 3-comp RGB vector
 */
typedef rtCallableProgramX<
  float3(const float3& /* incoming */, const float3& /* outgoing */)
> evalBSDFLocalFunc;
rtDeclareVariable(evalBSDFLocalFunc, evalBSDFLocal, , ); 

/**
 * Evaluates the PDF of an outgoing ray direction being sampled given an
 * incoming ray direction in the local space oriented with the Z-axis along the
 * surface normal of an intersection.
 * Materials must implement this callable function.
 *
 * @param incoming the incoming ray direction, in local space
 * @param outgoing the outgoing ray direction, in local space
 * @returns the evaluated scalar PDF
 */
typedef rtCallableProgramX<
  float(const float3& /* incoming */, const float3& /* outgoing */)
> evalPDFLocalFunc;
rtDeclareVariable(evalPDFLocalFunc, evalPDFLocal, , ); 

/**
 * Samples an outgoing ray direction, evaluating the BSDF and PDF of the sample.
 * Samples are computed in the local space oriented with the Z-axis along the
 * surface normal of an intersection.
 * Materials must implement this callable function.
 *
 * @param rng                  the per-thread RNG in use
 * @param isectNormalObj       the normal of the surface at the intersection,
 *                             in the intersection's local space
 * @param incoming             the direction of the incoming ray, in the
 *                             intersection's local space
 * @param outgoingOut    [out] the outgoing ray direction sampled
 * @param bsdfOut        [out] the BSDF of the sample w/r/t the incoming ray
 * @param pdfOut         [out] the PDF of the sample w/r/t the incoming ray
 */
typedef rtCallableProgramX<
  void(
    curandState* /* rng */,
    const float3& /* incoming */,
    float3* /* outgoingOut */,
    float3* /* bsdfOut */,
    float* /* pdfOut */
  )
> sampleLocalFunc;
rtDeclareVariable(sampleLocalFunc, sampleLocal, , ); 

/**
 * Wrapper around sampleLocal.
 * Samples the BSDF and PDF of the material in world-space.
 *
 * @param rng                  the per-thread RNG in use
 * @param isectNormalObj       the normal of the surface at the intersection,
 *                             in object-world space
 * @param incoming             the direction of the incoming ray, in
 *                             object-world space
 * @param outgoingOut    [out] the outgoing ray direction sampled
 * @param bsdfOut        [out] the BSDF of the sample w/r/t the incoming ray
 * @param pdfOut         [out] the PDF of the sample w/r/t the incoming ray
 */
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

/**
 * Wrapper around evalBSDFLocal and evalPDFLocal.
 * Evaluates the BSDF and PDF of the material in world-space.
 *
 * @param isectNormalObj      the normal of the surface at the intersection,
 *                            in object-world space
 * @param incomingWorld       the direction of the incoming ray, in
 *                            object-world space
 * @param outgoingWorld       the outgoing ray direction sampled
 * @param bsdfOut       [out] the BSD w/r/t the incoming and outgoing rays
 * @param pdfOut        [out] the PDF w/r/t the incoming and outgoing rays
 */
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

/**
 * Uniformly picks a random light in the scene and computes its direct
 * contribution to the given intersection point.
 *
 * @param data           the ray data for the ray that hit the intersection
 * @param isectNormalObj the surface normal at the intersection, in object-world
 *                       space
 * @param isectPos       the position of the intersection, in scene-world space
 */
__device__ __inline__ optix::float3 uniformSampleOneLight(
  const NormalRayData& data,
  const optix::float3& isectNormalObj,
  const optix::float3& isectPos
) {
  if (numLights == 0) {
    return optix::make_float3(0);
  }

  int lightIdx = math::nextInt(data.rng, 0, numLights);
  Light light = lightsBuffer[min(lightIdx, numLights - 1)];

  // P[this light] = 1 / numLights, so 1 / P[this light] = numLights.
  return float(numLights) * light.directIlluminate(
    data, isectNormalObj, isectPos
  );
}

/**
  * Calculates the transmittance and scatters another ray from an
  * intersection.
  *
  * @param rayData        [in,out] the incoming ray that hit the intersection;
  *                                it will be modified to reflect the outgoing
  *                                ray that should be cast by changing the
  *                                origin, direction, beta value, and flags
  * @param isectNormalObj          the surface normal at the intersection, in
  *                                object-world space
  * @param pos                     the position of the intersection, in
  *                                scene-world space
  */
__device__ __inline__ void scatter(
  NormalRayData& rayData,
  float3 isectNormalObj,
  float3 pos
) {
  float3 outgoingWorld;
  float3 bsdf;
  float pdf;
  sampleWorld(rayData.rng, isectNormalObj, -rayData.direction, &outgoingWorld, &bsdf, &pdf);

  float3 scale;
  bool dead;
  if (pdf > 0.0f) {
    scale = bsdf * fabsf(dot(isectNormalObj, outgoingWorld)) / pdf;
    dead = false;
  } else {
    scale = make_float3(0, 0, 0);
    dead = true;
  }

  rayData.origin = pos + outgoingWorld * XRAY_VERY_SMALL;
  rayData.direction = outgoingWorld;
  rayData.beta *= scale;
  rayData.flags |= dead ? RAY_DEAD : 0;
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
#if ENABLE_DIRECT_ILLUMINATION
  int lume = !(normalRayData.flags & RAY_DID_DIRECT_ILLUMINATE)
    & (light.id != -1);

  normalRayData.radiance += lume
    * normalRayData.beta
    * light.emit(normalRayData.direction, isectNormalObj);
#else
  int lume = (light.id != -1);

  normalRayData.radiance += lume
    * normalRayData.beta
    * light.emit(normalRayData.direction, isectNormalObj);
#endif
  
  // Next event estimation with light at next step.
#if ENABLE_DIRECT_ILLUMINATION
  if (materialFlags & MATERIAL_DIRECT_ILLUMINATE) {
    normalRayData.radiance += normalRayData.beta
      * uniformSampleOneLight(normalRayData, isectNormalObj, isectPos);
    normalRayData.flags |= RAY_DID_DIRECT_ILLUMINATE;
  } else {
    normalRayData.flags &= ~RAY_DID_DIRECT_ILLUMINATE;
  }
#endif

  // Material evaluation at current step.
  if (materialFlags & MATERIAL_REFLECT) {
    scatter(normalRayData, isectNormalObj, isectPos);
  } else {
    normalRayData.flags |= RAY_DEAD;
  }
}
