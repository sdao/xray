#include "bsdf.cuh"

__device__ float3 evalBSDFLocal(const float3& incoming, const float3& outgoing) {
  if (!math::localSameHemisphere(incoming, outgoing)) {
    return make_float3(0);
  }

  return albedo * XRAY_INV_PI;
}

__device__ void sampleLocal(
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
