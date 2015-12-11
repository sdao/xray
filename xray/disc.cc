#include "disc.h"

Disc::Disc(Xray xray, optix::float3 origin, optix::float3 normal, float radiusOuter, float radiusInner)
  : Geom(xray.getContext()), _origin(origin), _normal(normal), _radiusOuter(radiusOuter), _radiusInner(radiusInner) {
  _geom["origin"]->set3fv(&_origin.x);
  _geom["normal"]->set3fv(&_normal.x);
  _geom["radiusOuter"]->setFloat(_radiusOuter);
  _geom["radiusInner"]->setFloat(_radiusInner);

  freeze();
}

Disc* Disc::make(Xray xray, const Node& n) {
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
  return "PTX_files/disc.cu.ptx";
}

std::string Disc::getIsectProgram() const {
  return "discIntersect";
}

std::string Disc::getBoundsProgram() const {
  return "discBounds";
}
