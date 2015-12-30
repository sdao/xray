#pragma once
#include "geom.h"
#include "node.h"
#include "xray.h"

/** A sphere with a movable origin. */
class Sphere : public Geom {
  optix::float3 _origin;
  float _radius;
  bool _inverted;

protected:
  virtual unsigned getPrimitiveCount() const override;
  virtual std::string getPtxFile() const override;
  virtual std::string getIsectProgram() const override;
  virtual std::string getBoundsProgram() const override;

public:
  /**
   * Constructs a sphere.
   *
   * @param xray     the global Xray state to attach
   * @param origin   the origin of the sphere in world space
   * @param radius   the radius of the sphere
   * @param inverted whether the normals of the sphere point inwards;
   *                 mainly useful for controlling illumination of area lights
   */
  Sphere(Xray* xray, optix::float3 origin, float radius, bool inverted = false);

  /** Makes a sphere from the given node. */
  static Sphere* make(Xray* xray, const Node& n);

  virtual optix::Aabb getBoundingBox() const override;
  virtual void getBoundingSphere(
    optix::float3* origin,
    float* radius
  ) const override;
};

