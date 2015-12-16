#include <optix.h>
#include <optix_cuda.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

using namespace optix;

rtDeclareVariable(int, id, , );
rtBuffer<float3> vertexBuffer;     
rtBuffer<float3> normalBuffer;
rtBuffer<int3> faceIndices; // Index into vertex/normalBuffer for each face.

rtDeclareVariable(Ray, ray, rtCurrentRay, );

rtDeclareVariable(float3, isectNormal, attribute isectNormal, );
rtDeclareVariable(int, isectHitId, attribute isectHitId, );

RT_PROGRAM void meshIntersect(int primIdx)
{
  int3 face = faceIndices[primIdx];
  if (face.x < 0 | face.y < 0 | face.z < 0) {
    return;
  }

  float3 p0 = vertexBuffer[face.x];
  float3 p1 = vertexBuffer[face.y];
  float3 p2 = vertexBuffer[face.z];

  // Intersect ray with triangle
  float3 n;
  float t, beta, gamma;
  if (intersect_triangle(ray, p0, p1, p2, n, t, beta, gamma)) {
    if (rtPotentialIntersection(t)) {
      float3 n0 = normalBuffer[face.x];
      float3 n1 = normalBuffer[face.y];
      float3 n2 = normalBuffer[face.z];
      
      isectNormal = normalize(n1 * beta + n2 * gamma + n0 * (1.0f - beta - gamma));
      isectHitId = id;
      rtReportIntersection(0);
    }
  }
}

RT_PROGRAM void meshBounds(int primIdx, float result[6])
{
  int3 face = faceIndices[primIdx];

  const float3 v0 = vertexBuffer[face.x];
  const float3 v1 = vertexBuffer[face.y];
  const float3 v2 = vertexBuffer[face.z];
  const float area = length(cross(v1 - v0, v2 - v0));

  optix::Aabb* aabb = (optix::Aabb*) result;
  
  if (area > 0.0f & !isinf(area)) {
    aabb->m_min = fminf(fminf(v0, v1), v2);
    aabb->m_max = fmaxf(fmaxf(v0, v1), v2);
  } else {
    aabb->invalidate();
  }
}

