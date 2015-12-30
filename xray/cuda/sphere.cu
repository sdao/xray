#include <optix.h>
#include <optix_cuda.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

#include "math.cuh"

using namespace optix;

/* For sphere programs. */
rtDeclareVariable(float3, origin, , );
rtDeclareVariable(float, radius, , );
rtDeclareVariable(int, invertMode, , );

/* For intersection/bounds programs. */
rtDeclareVariable(int, id, , );

/* OptiX data for intersection programs. */
rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float3, isectNormal, attribute isectNormal, );
rtDeclareVariable(int, isectHitId, attribute isectHitId, );

RT_PROGRAM void sphereIntersect(int) {
  bool inverted = invertMode != 0;

  float3 diff = ray.origin - origin;
  float3 l = ray.direction;

  // See Wikipedia:
  // <http://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection>
  float a = dot(l, l);
  float b = dot(l, diff);
  float c = dot(diff, diff) - (radius * radius);

  float discriminant = (b * b) - (a * c);

  if (discriminant > 0.0f) {
    discriminant = sqrtf(discriminant);
    // Quadratic has at most 2 results.
    float resPos = (-b + discriminant);
    float resNeg = (-b - discriminant);

    // Neg before pos because we want to return closest isect first.
    if (math::isPositive(resNeg)) {
      if (rtPotentialIntersection(resNeg)) {
        float3 pt = math::at(ray, resNeg);
        float3 normal = normalize(inverted ? origin - pt : pt - origin);

        isectNormal = normal;
        isectHitId = id;
        rtReportIntersection(0);
      }
    } else if (math::isPositive(resPos)) {
      if (rtPotentialIntersection(resPos)) {
        float3 pt = math::at(ray, resPos);
        float3 normal = normalize(inverted ? origin - pt : pt - origin);
      
        isectNormal = normal;
        isectHitId = id;
        rtReportIntersection(0);
      }
    }
  }

  // Either no isect was found or it was behind us.
}

RT_PROGRAM void sphereBounds(int, float result[6]) {
  float3 boundsDiag = make_float3(radius);
  Aabb* aabb = (Aabb*) result;
  aabb->set(origin - boundsDiag, origin + boundsDiag);
}