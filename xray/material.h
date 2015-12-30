#pragma once
#define NOMINMAX
#include <optix.h>
#include <optix_world.h>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>

/**
 * The base interface for material objects. Materials wrap optix::Material.
 * All materials use the same basic BSDF closest-hit programs on the device,
 * relying on different callable programs to give different sampling behavior.
 */
class Material
{
  bool _frozen;

protected:
  /** The OptiX context for the material. */
  optix::Context _ctx;
  /** The OptiX material object. Subclasses can set their own properties. */
  optix::Material _material;
  
  /** Gets the PTX file for this material's callable functions. */
  virtual std::string getPtxFile() const = 0;
  /** Gets the name of the BSDF callable function. */
  virtual std::string getEvalBSDFLocalProgram() const;
  /** Gets the name of the PDF callable function. */
  virtual std::string getEvalPDFLocalProgram() const;
  /** Gets the name of the sampling callable function. */
  virtual std::string getSampleLocalProgram() const;
  /** Whether the current material causes a ray to be bounced. */
  virtual bool doesReflect() const = 0;
  /** Whether direct illumination (next event estimation) can be performed. */
  virtual bool shouldDirectIlluminate() const = 0;
  /**
   * Freezes the material object. No users may access the getMaterial() function
   * until after the object is frozen. In turn, subclasses must promise not to
   * modify the _material member after freezing the object.
   *
   * Preferably, you should call freeze() in a subclass constructor once the
   * subclass has set its own properties on the _material member, but you may
   * also choose to call freeze elsewhere. It is your responsibility to not call
   * getMaterial() until freeze() has been called.
   */
  void freeze();

public:
  /**
   * Creates a new material object connected to the given context.
   */
  Material(optix::Context ctx);
  virtual ~Material();

  /**
   * Gets the underlying OptiX object wrapped by this material.
   */
  optix::Material getMaterial() const;
};

