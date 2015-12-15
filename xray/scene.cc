#include "scene.h"
#include "camera.h"
#include "material.h"
#include "normaltest.h"
#include "dielectric.h"
#include "lambert.h"
#include "phong.h"
#include "geom.h"
#include "disc.h"
#include "sphere.h"
#include "mesh.h"
#include "light.h"
#include "node.h"
#include <exception>
#include <boost/format.hpp>

using boost::property_tree::ptree;
using boost::format;

Scene::Scene(std::string jsonFile)
  : _xray(), materials(), instances(), cameras()
{
  try {
    ptree pt;
    read_json(jsonFile, pt);
    
    readLights(pt);
    readMats(pt);
    readGeomInstances(pt);
    readCameras(pt);
  } catch (...) {
    cleanUp();
    throw;
  }
}

Scene::~Scene() {
  cleanUp();
}

void Scene::cleanUp() {
  for (auto& pair : materials) {
    delete pair.second;
  }

  for (auto& pair : instances) {
    delete pair.second;
  }

  for (auto& pair : cameras) {
    delete pair.second;
  }
}

template<typename T>
void Scene::readMultiple(
  const boost::property_tree::ptree& root,
  const std::string& prefix,
  const std::function<T(Xray* xray, const Node&, std::string type)> lookup,
  std::map<std::string, T>& storage
) {
  const auto& children = root.get_child(prefix);

  int count = 0;
  for (const auto& child : children) {
    const std::string name = child.first;

    try {
      const Node node(child.second, *this);
      const std::string type = node.getString("type");

      if (name.length() == 0) {
        throw std::runtime_error("No name");
      } else if (storage.count(name) != 0) {
        throw std::runtime_error("Name was reused");
      }

      storage[name] = lookup(&_xray, node, type);
    } catch (std::runtime_error err) {
      throw std::runtime_error(
        str(format("Error parsing node (%1%.[%2%]%3%):\n%4%") % prefix % count % name % err.what())
      );
    }

    count++;
  }
}

void Scene::readLights(const ptree& root) {
  static auto lookup = [](Xray* xray, const Node& n, std::string type) -> const AreaLight* {
    if (type == "area") {
      return AreaLight::make(xray, n);
    } else {
      throw std::runtime_error(type + " is not a recognized type");
    }
  };

  readMultiple<const AreaLight*>(root, "lights", lookup, lights);
}

void Scene::readMats(const ptree& root) {
  static auto lookup = [](Xray* xray, const Node& n, std::string type) -> const Material* {
    if (type == "dielectric") {
      return Dielectric::make(xray, n);
    } else if (type == "lambert") {
      return Lambert::make(xray, n);
    } else if (type == "phong") {
      return Phong::make(xray, n);
    } else {
      throw std::runtime_error(type + " is not a recognized type");
    }
  };

  readMultiple<const Material*>(root, "materials", lookup, materials);
}

void Scene::readGeomInstances(const ptree& root) {
  static auto lookup = [](Xray* xray, const Node& n, std::string type) -> const Instance* {
    const Geom* g;
    if (type == "disc") {
      g = Disc::make(xray, n);
    } else if (type == "sphere") {
      g = Sphere::make(xray, n);
    } else if (type == "mesh") {
      g = Mesh::make(xray, n);
    } else {
      throw std::runtime_error(type + " is not a recognized type");
    }
    const Instance* instance = Instance::make(xray, g, n.getMaterial("mat"), n.getLight("light"));
    delete g;
    return instance;
  };

  readMultiple<const Instance*>(root, "geometry", lookup, instances);
}

void Scene::readCameras(const ptree& root) {
  static auto lookup = [](Xray* xray, const Node& n, std::string type) -> Camera* {
    if (type == "persp") {
      return Camera::make(xray, n);
    } else {
      throw std::runtime_error(type + " is not a recognized type");
    }
  };

  readMultiple<Camera*>(root, "cameras", lookup, cameras);
}

Camera* Scene::defaultCamera() const {
  if (cameras.count("default") == 0) {
    throw std::runtime_error("Scene contains no default camera");
  }

  return cameras.at("default");
}

optix::Context Scene::getContext() const {
  return _xray.getContext();
}
