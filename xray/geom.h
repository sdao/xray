#pragma once
#define NOMINMAX
#include <optix.h>
#include <optix_world.h>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>

class Geom {
  bool _frozen;

protected:
  mutable optix::Context _ctx;
  optix::Geometry _geom;

  virtual unsigned getPrimitiveCount() const = 0;
  virtual std::string getPtxFile() const = 0;
  virtual std::string getIsectProgram() const = 0;
  virtual std::string getBoundsProgram() const = 0;
  void freeze();

public:
  Geom(optix::Context ctx);
  virtual ~Geom();
  optix::Geometry getGeometry() const;
};
