#include "sphere.h"

Sphere::Sphere(Xray* xray, optix::float3 origin, float radius, bool inverted)
  : Geom(xray->getContext()), _origin(origin), _radius(radius), _inverted(inverted) {
  _geom["origin"]->set3fv(&_origin.x);
  _geom["radius"]->setFloat(_radius);
  _geom["invertMode"]->setInt(_inverted ? 1 : 0);

  freeze();
}

Sphere* Sphere::make(Xray* xray, const Node& n) {
  return new Sphere(
    xray,
    n.getFloat3("origin"),
    n.getFloat("radius"),
    n.getBool("inverted")
  );
}

unsigned Sphere::getPrimitiveCount() const {
  return 1u;
}

std::string Sphere::getPtxFile() const {
  return "PTX_files/sphere.cu.ptx";
}

std::string Sphere::getIsectProgram() const {
  return "sphereIntersect";
}

std::string Sphere::getBoundsProgram() const {
  return "sphereBounds";
}

optix::Aabb Sphere::getBoundingBox() const {
  optix::float3 boundsDiag = optix::make_float3(_radius);
  return optix::Aabb(_origin - boundsDiag, _origin + boundsDiag);
}


void Sphere::getBoundingSphere(optix::float3* origin, float* radius) const {
  *origin = _origin;
  *radius = _radius;
}
