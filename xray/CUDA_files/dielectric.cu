#include "bsdf.cuh"

rtDeclareVariable(float, r0, , ); /**< The cached R(0) value for Schlick's approximation. */
rtDeclareVariable(float, etaEntering, , ); /**< The refraction ratio nVac / nMaterial. */
rtDeclareVariable(float, etaExiting, , ); /**< Inverse of etaEntering (nMaterial / nVac). */
rtDeclareVariable(float3, color, , );

__device__ float3 evalBSDFLocal(const float3& incoming, const float3& outgoing) {
  // Probabilistically, we are never going to get the exact matching
  // incoming and outgoing vectors.
  return make_float3(0);
}

__device__ void sampleLocal(
  curandState* rng,
  const float3& incoming,
  float3* outgoingOut,
  float3* bsdfOut,
  float* pdfOut
) {
    // Entering = are normal and incoming direction in opposite directions?
  // Recall that the incoming direction is in the normal's local space.
  bool entering = incoming.z > 0.0f;

  float3 alignedNormal; // Normal flipped based on ray direction.
  float eta; // Ratio of indices of refraction.
  if (entering) {
    // Note: geometry will return surface normal pointing outwards.
    // If we are entering, this is the right normal.
    // If we are exiting, since geometry is single-shelled, we will need
    // to flip the normal.
    alignedNormal = make_float3(0, 0, 1);
    eta = etaEntering;
  } else {
    alignedNormal = make_float3(0, 0, -1);
    eta = etaExiting;
  }

  // Calculate reflection vector.
  float3 reflectVector = reflect(
    -incoming,
    alignedNormal
  );

  // Calculate refraction vector.
  float3 refractVector;
  bool didReflect = refract(
    refractVector,
    -incoming,
    -alignedNormal,
    eta
  );

  if (!didReflect) {
    // Total internal reflection. Must reflect.
    *outgoingOut = reflectVector;
    *bsdfOut = color / math::absCosTheta(reflectVector);
    *pdfOut = 1.0f;
    return;
  }

  // Calculates Fresnel reflectance factor using Schlick's approximation.
  // See <http://graphics.stanford.edu/
  // courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf>.
  float cosTemp;
  if (eta < 1.0f) {
    // Equivalent to nIncident < nTransmit.
    // Equivalent to condition: entering == true
    // (e.g. nI = 1 (air), nT = 1.5 (glass))
    // Theta = angle of incidence.
    cosTemp = 1.0f - dot(incoming, alignedNormal);
  } else {
    // Equivalent to condition: entering == false
    // Theta = angle of refraction.
    cosTemp = 1.0f - dot(refractVector, -alignedNormal);
  }

  float cosTemp5 = cosTemp * cosTemp * cosTemp * cosTemp * cosTemp;
  float refl = r0 + (1.0f - r0) * cosTemp5;
  float refr = 1.0f - refl;

  // Importance sampling probabilities.
  // Pr[cast reflect ray] = [0.25, 0.75] based on reflectance.
  float probRefl = math::lerp(0.25f, 0.75f, refl);
  float probRefr = 1.0f - probRefl;

  // Probabilistically choose to refract or reflect.
  if (math::nextFloat(rng, 0.0f, 1.0f) < probRefl) {
    // Higher reflectance = higher probability of reflecting.
    *outgoingOut = reflectVector;
    *bsdfOut = color * refl / math::absCosTheta(reflectVector);
    *pdfOut = probRefl;
  } else {
    *outgoingOut = refractVector;
    *bsdfOut = color * refr / math::absCosTheta(refractVector);
    *pdfOut = probRefr;
  }
}
