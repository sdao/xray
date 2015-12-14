#include <optix.h>
#include <optix_cuda.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

#include "math.cuh"

using namespace optix;

rtDeclareVariable(float3, origin, , );
rtDeclareVariable(float3, normal, , );
rtDeclareVariable(float, radiusOuter, , );
rtDeclareVariable(float, radiusInner, , );

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float3, isectNormal, attribute isectNormal, );

RT_PROGRAM void discIntersect(int) {
  // See Wikipedia:
  // <http://en.wikipedia.org/wiki/Line%E2%80%93disc_intersection>

  float denom = dot(ray.direction, normal);
  if (denom != 0.0f) {
    float d = dot(origin - ray.origin, normal) / denom;

    if (math::isPositive(d)) {
      // In the plane, but are we in the disc?
      float3 isectPoint = math::at(ray, d);
      float isectToOriginDist = length(isectPoint - origin);
      if (isectToOriginDist <= radiusOuter
          && isectToOriginDist >= radiusInner) {
        // In the disc.
        if (rtPotentialIntersection(d)) {
          isectNormal = normal;
          rtReportIntersection(0);
        }
      }
    }
  }

  // Either no isect was found or it was behind us.
}

RT_PROGRAM void discBounds(int, float result[6]) {
  float3 tangent;
  float3 binormal;
  shared::coordSystem(normal, &tangent, &binormal);

  float3 tr = tangent * radiusOuter;
  float3 br = binormal * radiusOuter;

	Aabb* aabb = (Aabb*) result;
  aabb->set(make_float3(0), make_float3(0));
  aabb->include(origin + tr + br);
  aabb->include(origin - tr - br);
  aabb->include(origin + tr - br);
  aabb->include(origin - tr + br);
}