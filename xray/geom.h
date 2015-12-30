#pragma once
#define NOMINMAX
#include <optix.h>
#include <optix_world.h>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>

/**
 * The base interface for geometry objects. This host representation is a
 * wrapper around optix::Geometry, which in turn can correspond to multiple
 * geometry instances on the device.
 */
class Geom {
  bool _frozen;

protected:
  /** The OptiX context for the geometry. */
  mutable optix::Context _ctx;
  /** The OptiX geometry object. Subclasses can set their own properties. */
  optix::Geometry _geom;

  /** Returns the number of primitives in this geometry. */
  virtual unsigned getPrimitiveCount() const = 0;
  /** Gets the PTX file for this geometry's intersection/bounds programs. */
  virtual std::string getPtxFile() const = 0;
  /** Gets the name of the intersection program. */
  virtual std::string getIsectProgram() const = 0;
  /** Gets the name of the bounds program. */
  virtual std::string getBoundsProgram() const = 0;
  /**
   * Freezes the geometry object. No users may access the getGeometry() function
   * until after the object is frozen. In turn, subclasses must promise not to
   * modify the _geom member after freezing the object.
   *
   * Preferably, you should call freeze() in a subclass constructor once the
   * subclass has set its own properties on the _geom member, but you may also
   * choose to call freeze elsewhere. It is your responsibility to not call
   * getGeometry() until freeze() has been called.
   */
  void freeze();

public:
  /** Creates a new geometry object connected to the given context. */
  Geom(optix::Context ctx);
  virtual ~Geom();

  /** Gets the bounding box of the geometry. */
  virtual optix::Aabb getBoundingBox() const = 0;

  /**
   * Gets the bounding sphere of the geometry. The default implementation
   * returns the tightest sphere around the bounding box.
   */
  virtual void getBoundingSphere(optix::float3* origin, float* radius) const;

  /**
   * Gets the underlying OptiX object wrapped by this geometry.
   */
  optix::Geometry getGeometry() const;
};
