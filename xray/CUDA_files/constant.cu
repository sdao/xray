#include <optix.h>
#include <optix_cuda.h>
#include <optix_world.h>
#include "core.cuh"

rtDeclareVariable(float3, backgroundColor, , );
rtDeclareVariable(NormalRayData, normalRayData, rtPayload, );

RT_PROGRAM void miss()
{
  normalRayData.radiance += backgroundColor * normalRayData.beta;
  normalRayData.beta = make_float3(0);
}

RT_PROGRAM void constantRadiance()
{
  normalRayData.radiance += backgroundColor * normalRayData.beta;
  normalRayData.beta = make_float3(0);
}