#pragma once
#include "geom.h"
#include <vector>
#include "node.h"
#include "xray.h"

/** A triangle mesh with per-vertex normals. */
class Mesh : public Geom {
  optix::Buffer _vertices;
  optix::Buffer _normals;
  optix::Buffer _faces;
  optix::float3 _origin;
  optix::Aabb _bounds;
  int _numFaces;
  
  void readPolyModel(std::string name);

protected:
  virtual unsigned getPrimitiveCount() const override;
  virtual std::string getPtxFile() const override;
  virtual std::string getIsectProgram() const override;
  virtual std::string getBoundsProgram() const override;

public:
  /**
   * Constructs a triangle mesh by loading it from an OBJ file.
   *
   * @param xray   the global Xray state to attach
   * @param origin an offset with which to shift the mesh's origin
   * @param name   the name of the OBJ file from which to load the mesh
   */
  Mesh(Xray* xray, optix::float3 origin, std::string name);

  /** Makes a triangle mesh from the given node. */
  static Mesh* make(Xray* xray, const Node& n);

  virtual optix::Aabb getBoundingBox() const override;
};

