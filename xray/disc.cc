#include "disc.h"

Disc::Disc(optix::float3 origin, optix::float3 normal, float radiusOuter, float radiusInner)
  : _origin(origin), _normal(normal), _radiusOuter(radiusOuter), _radiusInner(radiusInner) {}

optix::Geometry Disc::makeOptixGeometry(optix::Context ctx) const {
  optix::Geometry geom = ctx->createGeometry();
  geom->setPrimitiveCount(1u);
  geom->setIntersectionProgram(ctx->createProgramFromPTXFile(getPtxFileName("disc.cu"), "discIntersect"));
  geom->setBoundingBoxProgram(ctx->createProgramFromPTXFile(getPtxFileName("disc.cu"), "discBounds"));
  geom["origin"]->set3fv(&_origin.x);
  geom["normal"]->set3fv(&_normal.x);
  geom["radiusOuter"]->setFloat(_radiusOuter);
  geom["radiusInner"]->setFloat(_radiusInner);
  return geom;
}