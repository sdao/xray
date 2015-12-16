#include <optix.h>
#include <optix_cuda.h>
#include <optix_world.h>
#include "core.cuh"
#include "math.cuh"

rtDeclareVariable(float3, albedo, , );

RT_CALLABLE_PROGRAM float3 evalBSDFLocal(const float3& incoming, const float3& outgoing) {
  int sameHemis = math::localSameHemisphere(incoming, outgoing);
  return sameHemis * albedo * XRAY_INV_PI;
}

RT_CALLABLE_PROGRAM float evalPDFLocal(const float3& incoming, const float3& outgoing) {
  int sameHemis = math::localSameHemisphere(incoming, outgoing);
  return sameHemis * math::cosineSampleHemispherePDF(outgoing);
}

RT_CALLABLE_PROGRAM void sampleLocal(
  curandState* rng,
  const float3& incoming,
  float3* outgoingOut,
  float3* bsdfOut,
  float* pdfOut
) {
  float3 outgoing = math::cosineSampleHemisphere(rng, incoming.z < 0.0f);

  *outgoingOut = outgoing;
  *bsdfOut = evalBSDFLocal(incoming, outgoing);
  *pdfOut = math::cosineSampleHemispherePDF(outgoing);
}
