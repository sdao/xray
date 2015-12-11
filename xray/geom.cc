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

optix::Geometry Geom::getGeometry() const {
  if (_frozen) {
    return _geom;
  }

  return nullptr;
}
