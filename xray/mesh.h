#pragma once
#include "geom.h"
#include <vector>

class Mesh : public Geom {
  optix::Buffer _vertices;
  optix::Buffer _normals;
  optix::Buffer _faces;
  optix::float3 _origin;
  int _numFaces;

public:
  Mesh(optix::Context ctx, optix::float3 origin, std::string name);
  virtual optix::Geometry makeOptixGeometry() const override;
  void readPolyModel(std::string name);
};

