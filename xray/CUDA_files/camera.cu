#include <optix.h>
#include <optix_cuda.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include "math.cuh"

using namespace optix;

struct PerRayData_radiance
{
  float3 result;
  float  importance;
  int    depth;
};

rtDeclareVariable(Matrix4x4, xform, , );
rtDeclareVariable(float3, focalPlaneOrigin, , ); // The upper-left corner of the focal rectangle in camera space.
rtDeclareVariable(float, focalPlaneRight, , ); // The vector pointing from the upper-left corner to the upper-right corner of the focal rectangle in camera space.
rtDeclareVariable(float, focalPlaneUp, , );

//rtDeclareVariable(rtObject, sceneRoot, , );
rtDeclareVariable(rtObject, top_object, , );
//rtBuffer<float4, 2> outputBuffer;
rtBuffer<uchar4, 2> output_buffer;

rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, launchDim, rtLaunchDim, );

#ifdef __CUDACC__
static __device__ __inline__ optix::uchar4 make_color(const optix::float3& c)
{
    return optix::make_uchar4( static_cast<unsigned char>(__saturatef(c.z)*255.99f),  /* B */
                               static_cast<unsigned char>(__saturatef(c.y)*255.99f),  /* G */
                               static_cast<unsigned char>(__saturatef(c.x)*255.99f),  /* R */
                               255u);                                                 /* A */
}
#endif

RT_PROGRAM void camera() {
  float fracX = float(launchIndex.x) / float(launchDim.x - 1);
  float fracY = float(launchIndex.y) / float(launchDim.y - 1);

  float3 offset = make_float3(focalPlaneRight * fracX, focalPlaneUp * fracY, 0);
  float3 lookAt = focalPlaneOrigin + offset;

  float3 eyeWorld = math::pointXform(make_float3(0, 0, 0), xform);
  float3 lookAtWorld = math::pointXform(lookAt, xform);
  float3 dir = normalize(lookAtWorld - eyeWorld);
  
  optix::Ray ray = make_Ray(eyeWorld, dir, 0u, XRAY_VERY_SMALL, RT_DEFAULT_MAX);

  PerRayData_radiance data;
  rtTrace(top_object, ray, data);

  output_buffer[launchIndex] = make_color(data.result);
}