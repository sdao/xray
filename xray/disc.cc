#include "disc.h"

Disc::Disc(optix::Context ctx, optix::float3 origin, optix::float3 normal, float radiusOuter, float radiusInner)
  : Geom(ctx), _origin(origin), _normal(normal), _radiusOuter(radiusOuter), _radiusInner(radiusInner) {}

optix::Geometry Disc::makeOptixGeometry() const {
  optix::Geometry geom = _ctx->createGeometry();
  geom->setPrimitiveCount(1u);
  geom->setIntersectionProgram(_ctx->createProgramFromPTXFile(getPtxFileName("disc.cu"), "discIntersect"));
  geom->setBoundingBoxProgram(_ctx->createProgramFromPTXFile(getPtxFileName("disc.cu"), "discBounds"));
  geom["origin"]->set3fv(&_origin.x);
  geom["normal"]->set3fv(&_normal.x);
  geom["radiusOuter"]->setFloat(_radiusOuter);
  geom["radiusInner"]->setFloat(_radiusInner);
  return geom;
}