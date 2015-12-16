#include <optix.h>
#include <optix_cuda.h>
#include <optix_world.h>
#include "core.cuh"

RT_CALLABLE_PROGRAM float3 evalBSDFLocal(const float3& incoming, const float3& outgoing) {
  return make_float3(0);
}

RT_CALLABLE_PROGRAM float evalPDFLocal(const float3& incoming, const float3& outgoing) {
  return 0.0f;
}

RT_CALLABLE_PROGRAM void sampleLocal(
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
