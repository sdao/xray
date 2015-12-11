#include <optix.h>
#include <optix_cuda.h>
#include <optix_world.h>
#include "core.cuh"

rtDeclareVariable(float3, backgroundColor, , );
rtDeclareVariable(NormalRayData, normalRayData, rtPayload, );

RT_PROGRAM void miss()
{
  normalRayData.radiance = backgroundColor;
}

RT_PROGRAM void constantRadiance()
{
  normalRayData.radiance = backgroundColor;
}