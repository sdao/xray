#include <optix.h>
#include <optix_cuda.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

#include "math.cuh"

using namespace optix;

rtDeclareVariable(float3, origin, , );
rtDeclareVariable(float, radius, , );
rtDeclareVariable(int, invertMode, , );

rtDeclareVariable(Ray, ray, rtCurrentRay, );

//rtDeclareVariable(float3, isectNormal, attribute isectNormal, );

rtDeclareVariable(float3, texcoord, attribute texcoord, ); 
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 

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

        //isectNormal = normal;
				texcoord = make_float3( 0.0f );
				shading_normal = geometric_normal = normal;
        rtReportIntersection(0);
      }
    } else if (math::isPositive(resPos)) {
      if (rtPotentialIntersection(resPos)) {
        float3 pt = math::at(ray, resPos);
        float3 normal = normalize(inverted ? origin - pt : pt - origin);
      
        //isectNormal = normal;
				texcoord = make_float3( 0.0f );
				shading_normal = geometric_normal = normal;
        rtReportIntersection(0);
      }
    }
  }

  // Either no isect was found or it was behind us.
}

RT_PROGRAM void sphereBounds(int, float result[6]) {
  bool inverted = invertMode != 0;

  float3 boundsDiag = make_float3(radius);
	Aabb* aabb = (Aabb*) result;
  aabb->set(origin - boundsDiag, origin + boundsDiag);
}