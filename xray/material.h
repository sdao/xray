#pragma once
#define NOMINMAX
#include <optix.h>
#include <optix_world.h>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>

class Material
{
  bool _frozen;

protected:
  optix::Context _ctx;
  optix::Material _material;

  virtual std::string getClosestHitPtxFile() const = 0;
  virtual std::string getClosestHitProgram() const = 0;
  virtual std::string getAnyHitPtxFile() const;
  virtual std::string getAnyHitProgram() const;
  void freeze();

public:
  Material(optix::Context ctx);
  virtual ~Material();
  optix::Material getMaterial() const;
};

