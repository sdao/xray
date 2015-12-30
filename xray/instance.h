#pragma once
#define NOMINMAX
#include <optix.h>
#include <optix_world.h>
#include "geom.h"
#include "material.h"
#include "light.h"
#include "xray.h"

/**
 * An instance object represents an object to be rendered in the scene.
 * It is a combination of a geometry instance, a light instance, and a material
 * reference.
 * When rendering with next-event estimation enabled, the light instance
 * needs to be collected for direct sampling. Otherwise, only the geometry
 * instance needs to be collected and the light instance can be ignored.
 */
class Instance {
  int _id;
  optix::GeometryInstance _instance;
  Light* _light;

  Instance(
    Xray* xray,
    const Geom* g,
    const Material* m,
    const AreaLight* l,
    int id
  );

public:
  ~Instance();

  /**
   * Creates a scene object instance.
   * Note: It is safe to delete any pointer passed into the make() method after
   * the method returns because geometry instances do not keep references to
   * their source geometry, material, or light.
   *
   * @param xray the global Xray state to attach
   * @param g    the geometry for the instance
   * @param m    the material for the instance
   * @param l    the light for the instance
   * @returns    a pointer to a geometry instance; the caller owns the pointer
   */
  static Instance* make(
    Xray* xray,
    const Geom* g,
    const Material* m,
    const AreaLight* l
  );

  /**
   * Gets the geometry instance component of this instance.
   */
  optix::GeometryInstance getGeometryInstance() const;

  /**
   * Gets the light instance component of this instance.
   */
  Light* getLightInstance() const;
};
