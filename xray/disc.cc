#include "disc.h"
#include "cuda/shared.cuh"

Disc::Disc(Xray* xray, optix::float3 origin, optix::float3 normal, float radiusOuter, float radiusInner)
  : Geom(xray->getContext()), _origin(origin), _normal(normal), _radiusOuter(radiusOuter), _radiusInner(radiusInner) {
  _geom["origin"]->set3fv(&_origin.x);
  _geom["normal"]->set3fv(&_normal.x);
  _geom["radiusOuter"]->setFloat(_radiusOuter);
  _geom["radiusInner"]->setFloat(_radiusInner);

  freeze();
}

Disc* Disc::make(Xray* xray, const Node& n) {
  return new Disc(
    xray,
    n.getFloat3("origin"),
    n.getFloat3("normal"),
    n.getFloat("radiusOuter"),
    n.getFloat("radiusInner")
  );
}

unsigned Disc::getPrimitiveCount() const {
  return 1u;
}

std::string Disc::getPtxFile() const {
  return "ptx/disc.cu.ptx";
}

std::string Disc::getIsectProgram() const {
  return "discIntersect";
}

std::string Disc::getBoundsProgram() const {
  return "discBounds";
}

optix::Aabb Disc::getBoundingBox() const {
  optix::float3 tangent;
  optix::float3 binormal;
  shared::coordSystem(_normal, &tangent, &binormal);

  optix::float3 tr = tangent * _radiusOuter;
  optix::float3 br = binormal * _radiusOuter;

	optix::Aabb aabb(optix::make_float3(0), optix::make_float3(0));
  aabb.include(_origin + tr + br);
  aabb.include(_origin - tr - br);
  aabb.include(_origin + tr - br);
  aabb.include(_origin - tr + br);

  return aabb;
}

void Disc::getBoundingSphere(optix::float3* origin, float* radius) const {
  *origin = _origin;
  *radius = _radiusOuter;
}
