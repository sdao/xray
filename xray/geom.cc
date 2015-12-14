#include "geom.h"

Geom::Geom(optix::Context ctx) : _ctx(ctx), _geom(ctx->createGeometry()), _frozen(false) {}

Geom::~Geom() {}

void Geom::freeze() {
  if (_frozen) {
    return;
  }

  _geom->setPrimitiveCount(getPrimitiveCount());
  _geom->setIntersectionProgram(_ctx->createProgramFromPTXFile(getPtxFile(), getIsectProgram()));
  _geom->setBoundingBoxProgram(_ctx->createProgramFromPTXFile(getPtxFile(), getBoundsProgram()));
  _frozen = true;
}

void Geom::getBoundingSphere(optix::float3* origin, float* radius) const {
  optix::Aabb b = getBoundingBox();
  *origin = b.center();
  *radius = length(b.extent()) * 0.5f;
}

optix::Geometry Geom::getGeometry() const {
  if (_frozen) {
    return _geom;
  }

  return nullptr;
}
