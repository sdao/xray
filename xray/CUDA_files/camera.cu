#include <optix.h>
#include <optix_cuda.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include "math.cuh"
#include "core.cuh"

using namespace optix;

rtDeclareVariable(Matrix4x4, xform, , );
rtDeclareVariable(float3, focalPlaneOrigin, , ); // The lower-left corner of the focal rectangle in camera space.
rtDeclareVariable(float, focalPlaneRight, , ); // The vector pointing from the upper-left corner to the upper-right corner of the focal rectangle in camera space.
rtDeclareVariable(float, focalPlaneUp, , );

rtDeclareVariable(rtObject, sceneRoot, , );
rtBuffer<float3, 2> rawBuffer;
rtBuffer<uchar4, 2> imageBuffer;

rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, launchDim, rtLaunchDim, );

RT_PROGRAM void camera() {
  float fracX = float(launchIndex.x) / float(launchDim.x - 1);
  float fracY = float(launchDim.y - launchIndex.y - 1) / float(launchDim.y - 1);

  float3 offset = make_float3(focalPlaneRight * fracX, focalPlaneUp * fracY, 0);
  float3 lookAt = focalPlaneOrigin + offset;

  float3 eyeWorld = math::pointXform(make_float3(0, 0, 0), xform);
  float3 lookAtWorld = math::pointXform(lookAt, xform);
  float3 dir = normalize(lookAtWorld - eyeWorld);
  
  optix::Ray ray = make_Ray(eyeWorld, dir, RAY_TYPE_NORMAL, XRAY_VERY_SMALL, RT_DEFAULT_MAX);

  NormalRayData data = NormalRayData::make();
  rtTrace(sceneRoot, ray, data);

  imageBuffer[launchIndex] = math::colorToBgra(data.radiance);
}