#include <optix_world.h>
#include <optix_cuda.h>

using namespace optix;

struct RayData {
  float3 radiance;
  int depth;

  __host__ __device__ RayData() : radiance(make_float3(0, 0, 0)), depth(0) {}
};

rtDeclareVariable(float3, camOrigin, , );
rtDeclareVariable(float3, camX, , );
rtDeclareVariable(float3, camY, , );
rtDeclareVariable(float3, camZ, , );
rtDeclareVariable(float3, focalPlaneOrigin, , );
rtDeclareVariable(float, focalPlaneRight, , );
rtDeclareVariable(float, focalPlaneUp, , );

rtDeclareVariable(rtObject, sceneRoot, , );
rtBuffer<float4, 2> outputBuffer;

rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, launchDim, rtLaunchDim, );

float3 __device__ __inline__ cameraToWorld(float3 v) {
  return camOrigin + v.x * camX + v.y * camY + v.z * camZ;
}

RT_PROGRAM void camera() {
  float fracX = float(launchIndex.x) / float(launchDim.x - 1);
  float fracY = float(launchIndex.y) / float(launchDim.y - 1);

  float3 offset = make_float3(focalPlaneRight * fracX, focalPlaneUp * fracY, 0);
  float3 lookAt = focalPlaneOrigin + offset;

  float3 eyeWorld = cameraToWorld(make_float3(0, 0, 0));
  float3 lookAtWorld = cameraToWorld(lookAt);
  float3 dir = normalize(lookAtWorld - eyeWorld);

  optix::Ray ray = optix::make_Ray(eyeWorld, dir, 0u, 1.0e-4f, RT_DEFAULT_MAX);

  RayData data;
  rtTrace(sceneRoot, ray, data);

  outputBuffer[launchIndex] = make_float4(data.radiance, 1.0f);
}