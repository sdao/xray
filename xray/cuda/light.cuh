#pragma once
#include <optix.h>
#include <optix_world.h>
#include "shared.cuh"

#ifdef __CUDACC__
#include <optix_cuda.h>
#include "math.cuh"
#endif

#ifdef __CUDACC__
rtDeclareVariable(rtObject, sceneRoot, , );

__device__ void evalWorld(
  const optix::float3& isectNormalObj,
  const optix::float3& incomingWorld,
  const optix::float3& outgoingWorld,
  optix::float3* bsdfOut,
  float* bsdfPdfOut
);

__device__ void sampleWorld(
  curandState* rng,
  const float3& isectNormalObj,
  const float3& incoming,
  float3* outgoingOut,
  float3* bsdfOut,
  float* pdfOut
);
#endif

/**
 * A light instance as represented on the device.
 * Light instances should be created on the host and sent to the device.
 * They cannot be constructed on the device, and in turn, their emission
 * and sampling functions cannot be called on the host.
 */
struct Light {
  optix::float3 color;        /**< The color emitted by the light. */
  optix::float3 boundsOrigin; /**< The origin of the light's bounding sphere. */
  float boundsRadius;         /**< The radius of the light's bounding sphere. */
  int id;                     /**< Each light instance needs a unique ID. */

  __host__ static Light* make(optix::float3 c) {
    Light* l = new Light();
    l->color = c;
    l->id = -1;
    l->boundsOrigin = optix::make_float3(0);
    l->boundsRadius = 0.0f;
    return l;
  }

  __host__ static Light* make() {
    Light* l = new Light();
    l->color = optix::make_float3(0);
    l->id = -1;
    l->boundsOrigin = optix::make_float3(0);
    l->boundsRadius = 0.0f;
    return l;
  }
  
#ifdef __CUDACC__
  __device__ optix::float3 emit(
    const optix::float3& dir,
    const optix::float3& n
  ) const {
    // Only emit on the normal-facing side of objects, e.g. on the outside of a
    // sphere or on the normal side of a disc.
    int outside = dot(dir, n) < -XRAY_EPSILON;
    return outside * color;
  }

  __device__ optix::float3 directIlluminateByLightPDF(
    const NormalRayData& data,
    const optix::float3& isectNormalObj,
    const optix::float3& isectPos
  ) const {
    // Sample random from light PDF.
    optix::float3 outgoingWorld;
    optix::float3 lightColor;
    float lightPdf;
    sampleLight(
      data.rng,
      isectPos,
      &outgoingWorld,
      &lightColor,
      &lightPdf
    );

    if (lightPdf > 0.0f & !math::isVectorExactlyZero(lightColor)) {
      // Evaluate material BSDF and PDF as well.
      optix::float3 bsdf;
      float bsdfPdf;
      evalWorld(
        isectNormalObj,
        -data.direction,
        outgoingWorld,
        &bsdf,
        &bsdfPdf
      );

      if (bsdfPdf > 0.0f & !math::isVectorExactlyZero(bsdf)) {
        float lightWeight = math::powerHeuristic(1, lightPdf, 1, bsdfPdf);
        return (bsdf * lightColor)
          * fabsf(dot(isectNormalObj, outgoingWorld))
          * lightWeight / lightPdf;
      }
    }

    return optix::make_float3(0);
  }

  __device__ optix::float3 directIlluminateByMatPDF(
    const NormalRayData& data,
    const optix::float3& isectNormalObj,
    const optix::float3& isectPos
  ) const {
    // Sample random from BSDF PDF.
    optix::float3 outgoingWorld;
    optix::float3 bsdf;
    float bsdfPdf;
    sampleWorld(
      data.rng,
      isectNormalObj,
      -data.direction,
      &outgoingWorld,
      &bsdf,
      &bsdfPdf
    );

    if (bsdfPdf > 0.0f & !math::isVectorExactlyZero(bsdf)) {
      // Evaluate light PDF as well.
      optix::float3 lightColor;
      float lightPdf;
      evalLight(
        isectPos,
        outgoingWorld,
        &lightColor,
        &lightPdf
      );

      if (lightPdf > 0.0f & !math::isVectorExactlyZero(lightColor)) {
        float bsdfWeight = math::powerHeuristic(1, bsdfPdf, 1, lightPdf);
        return bsdf * lightColor
          * fabsf(dot(isectNormalObj, outgoingWorld))
          * bsdfWeight / bsdfPdf;
      }
    }

    return optix::make_float3(0);
  }

  __device__ void evalLight(
    const optix::float3& point,
    const optix::float3& dirToLight,
    optix::float3* colorOut,
    float* pdfOut
  ) const {
    float pdf;

    if (math::sphereContains(boundsOrigin, boundsRadius, point)) {
      // We're inside the bounding sphere, so sample sphere uniformly.
      pdf = math::uniformSampleSpherePDF();
    } else {
      // We're outside the bounding sphere, so sample by solid angle.
      optix::float3 dirToLightOrigin = boundsOrigin - point;
      float theta = asinf(boundsRadius / length(dirToLightOrigin));

      optix::float3 normal = normalize(dirToLightOrigin);
      optix::float3 tangent;
      optix::float3 binormal;
      shared::coordSystem(normal, &tangent, &binormal);

      optix::float3 dirToLightLocal =
        math::worldToLocal(dirToLight, tangent, binormal, normal);
      pdf = math::uniformSampleConePDF(theta, dirToLightLocal);
    }

    optix::Ray pointToLight = optix::make_Ray(
      point + dirToLight * XRAY_VERY_SMALL,
      dirToLight,
      RAY_TYPE_SHADOW,
      XRAY_VERY_SMALL,
      RT_DEFAULT_MAX
    );
    ShadowRayData checkData = ShadowRayData::make();
    rtTrace(sceneRoot, pointToLight, checkData);
    int idMatch = checkData.lastHitId == id;
    optix::float3 emittedColor =
      idMatch * emit(dirToLight, checkData.hitNormal);

    *colorOut = emittedColor;
    *pdfOut = pdf;
  }

  __device__ void sampleLight(
    curandState* rng,
    const optix::float3& point,
    optix::float3* dirToLightOut,
    optix::float3* colorOut,
    float* pdfOut
  ) const {
    optix::float3 dirToLight;
    float pdf;

    if (math::sphereContains(boundsOrigin, boundsRadius, point)) {
      // We're inside the bounding sphere, so sample sphere uniformly.
      dirToLight = math::uniformSampleSphere(rng);
      pdf = math::uniformSampleSpherePDF();
    } else {
      // We're outside the bounding sphere, so sample by solid angle.
      optix::float3 dirToLightOrigin = boundsOrigin - point;
      float theta = asinf(boundsRadius / length(dirToLightOrigin));

      optix::float3 normal = normalize(dirToLightOrigin);
      optix::float3 tangent;
      optix::float3 binormal;
      shared::coordSystem(normal, &tangent, &binormal);

      dirToLight = math::localToWorld(
        math::uniformSampleCone(rng, theta),
        tangent,
        binormal,
        normal
      );
      pdf = math::uniformSampleConePDF(theta);
    }
     
    optix::Ray pointToLight = optix::make_Ray(
      point + dirToLight * XRAY_VERY_SMALL,
      dirToLight,
      RAY_TYPE_SHADOW,
      XRAY_VERY_SMALL,
      RT_DEFAULT_MAX
    );
    ShadowRayData checkData = ShadowRayData::make();
    rtTrace(sceneRoot, pointToLight, checkData);
    int idMatch = checkData.lastHitId == id;
    optix::float3 emittedColor =
      idMatch * emit(dirToLight, checkData.hitNormal);

    *dirToLightOut = dirToLight;
    *colorOut = emittedColor;
    *pdfOut = pdf;
  }

  __device__ optix::float3 directIlluminate(
    const NormalRayData& data,
    const optix::float3& isectNormalObj,
    const optix::float3& isectPos
  ) const {
    optix::float3 Ld = optix::make_float3(0);

    Ld +=
      directIlluminateByLightPDF(data, isectNormalObj, isectPos);
    Ld +=
      directIlluminateByMatPDF(data, isectNormalObj, isectPos);

    return Ld;
  }
#endif
};
