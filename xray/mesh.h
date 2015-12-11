#pragma once
#include "geom.h"
#include <vector>
#include "node.h"
#include "xray.h"

class Mesh : public Geom {
  optix::Buffer _vertices;
  optix::Buffer _normals;
  optix::Buffer _faces;
  optix::float3 _origin;
  int _numFaces;
  
  void readPolyModel(std::string name);

protected:
  virtual unsigned getPrimitiveCount() const override;
  virtual std::string getPtxFile() const override;
  virtual std::string getIsectProgram() const override;
  virtual std::string getBoundsProgram() const override;

public:
  Mesh(Xray xray, optix::float3 origin, std::string name);
  static Mesh* make(Xray xray, const Node& n);
};

