#pragma once
#define NOMINMAX
#include <optix.h>
#include <optix_world.h>
#include <vector>
#include <map>
#include <boost/property_tree/ptree.hpp>

class AreaLight;
class Material;
class Instance;
class Scene;

/**
 * Wrapper class around a boost::property_tree::ptree to help
 * facilitate data access.
 */
class Node {
  /** Translator to convert a string to its vector representation. */
  struct NodeFloat3Translator {
    typedef std::string internal_type;
    typedef optix::float3 external_type;

    NodeFloat3Translator();
    boost::optional<external_type> get_value(const internal_type& data) const;
  };

  /** Translator to lookup the object referenced by a string. */
  template <typename T, bool allowNull = true>
  struct NodeLookupTranslator {
    typedef std::string internal_type;
    typedef T external_type;

    const std::map<internal_type, external_type>& lookup;
    NodeLookupTranslator(const std::map<internal_type, external_type>& l);
    boost::optional<external_type> get_value(const internal_type& data) const;
  };

  /** Property tree wrapped by this node. */
  const boost::property_tree::ptree& attributes;
  /** Scene associated with this node. */
  const Scene& container;

public:
  /**
   * Creates a node from the given property tree and scene. They will be stored
   * by reference, and must be kept alive as long as this node is in use.
   */
  Node(const boost::property_tree::ptree& attr, const Scene& cont);

  /** Gets the string property at the given key. */
  std::string getString(std::string key) const;
  /** Gets the integer property at the given key. */
  int getInt(std::string key) const;
  /** Gets the boolean property at the given key. */
  bool getBool(std::string key) const;
  /** Gets the float property at the given key. */
  float getFloat(std::string key) const;
  /** Gets the 3D vector property at the given key. */
  optix::float3 getFloat3(std::string key) const;
  /** Gets the light pointer referenced by the given key. */
  const AreaLight* getLight(std::string key) const;
  /** Gets the material pointer referenced by the given key. */
  const Material* getMaterial(std::string key) const;
  /** Gets the geom instance pointer referenced by the given key. */
  const Instance* getGeomInstance(std::string key) const;
  /** Gets the instance pointers referenced in the list with the given key. */
  std::vector<const Instance*> getGeomInstanceList(std::string key) const;
};
