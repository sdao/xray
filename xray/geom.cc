#include "geom.h"

Geom::Geom(optix::Context ctx) :  _ctx(ctx), _material(nullptr) {}

Geom::~Geom() {}

std::string Geom::getPtxFileName(std::string cuFile) const {
  return str(boost::format("PTX_files/%1%.ptx") % cuFile);
}

optix::GeometryInstance Geom::makeInstance() const {
  optix::Geometry geom = makeOptixGeometry();
  optix::GeometryInstance instance = _ctx->createGeometryInstance();
  instance->setGeometry(geom);
  instance->setMaterialCount(1u);
  instance->setMaterial(0u, _material);
  return instance;
}