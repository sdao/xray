#pragma once
#include "geom.h"
#include "node.h"
#include "xray.h"

/** A disc or annulus. */
class Disc : public Geom {
  optix::float3 _origin;
  optix::float3 _normal;
  float _radiusOuter;
  float _radiusInner;
  
protected:
  virtual unsigned getPrimitiveCount() const override;
  virtual std::string getPtxFile() const override;
  virtual std::string getIsectProgram() const override;
  virtual std::string getBoundsProgram() const override;

public:
  /**
   * Constructs a disc.
   *
   * @param xray        the global Xray state to attach
   * @param origin      the center (origin) of the disc
   * @param normal      the normal vector perpendicular to the disc's plane
   * @param radiusOuter the outer radius of the disc
   * @param radiusInner the inner radius of the disc (the radius of its hole)
   */
  Disc(
    Xray* xray,
    optix::float3 origin,
    optix::float3 normal,
    float radiusOuter,
    float radiusInner
  );

  /** Makes a disc from the given node. */
  static Disc* make(Xray* xray, const Node& n);

  virtual optix::Aabb getBoundingBox() const override;
  virtual void getBoundingSphere(
    optix::float3* origin,
    float* radius
  ) const override;
};

