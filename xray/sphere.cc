#include "sphere.h"

Sphere::Sphere(optix::Context ctx, optix::float3 origin, float radius, bool inverted)
  : Geom(ctx), _origin(origin), _radius(radius), _inverted(inverted) {}

optix::Geometry Sphere::makeOptixGeometry() const {
  optix::Geometry geom = _ctx->createGeometry();
  geom->setPrimitiveCount(1u);
  geom->setIntersectionProgram(_ctx->createProgramFromPTXFile(getPtxFileName("sphere.cu"), "sphereIntersect"));
  geom->setBoundingBoxProgram(_ctx->createProgramFromPTXFile(getPtxFileName("sphere.cu"), "sphereBounds"));
  geom["origin"]->set3fv(&_origin.x);
  geom["radius"]->setFloat(_radius);
  geom["invertMode"]->setInt(_inverted ? 1 : 0);
  return geom;
}
