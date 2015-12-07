#pragma once
#define NOMINMAX
#include <optix.h>
#include <optix_world.h>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>

class Geom {
  optix::Material _material;

protected:
  mutable optix::Context _ctx;
  std::string getPtxFileName(std::string cuFile) const;

public:
  Geom(optix::Context ctx);
  virtual ~Geom();
  virtual optix::Geometry makeOptixGeometry() const = 0;
  optix::GeometryInstance makeInstance() const;
};
