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

  virtual std::string getPtxFile() const = 0;
  virtual std::string getEvalBSDFLocalProgram() const;
  virtual std::string getEvalPDFLocalProgram() const;
  virtual std::string getSampleLocalProgram() const;
  virtual bool doesReflect() const = 0;
  virtual bool shouldDirectIlluminate() const = 0;
  void freeze();

public:
  Material(optix::Context ctx);
  virtual ~Material();
  optix::Material getMaterial() const;
};

