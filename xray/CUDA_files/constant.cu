#include <optix.h>
#include <optix_cuda.h>
#include <optix_world.h>
#include "core.cuh"
#include "basematerial.cuh"

rtDeclareVariable(float3, color, , );

__device__ void scatter(NormalRayData& rayData, float3 normal, float3 pos) {
  rayData.radiance += color * rayData.beta;
  rayData.beta = make_float3(0);
}