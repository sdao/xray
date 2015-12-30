#include "node.h"
#include "scene.h"
#include <exception>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>

using boost::format;

Node::Node(const boost::property_tree::ptree& attr, const Scene& cont)
  : attributes(attr), container(cont) {}

std::string Node::getString(std::string key) const {
  auto result = attributes.get_optional<std::string>(key);

  if (!result) {
    throw std::runtime_error(
      str(format("Cannot read string property '%1%'") % key)
    );
  }

  return *result;
}

int Node::getInt(std::string key) const {
  auto result = attributes.get_optional<int>(key);

  if (!result) {
    throw std::runtime_error(
      str(format("Cannot read integer property '%1%'") % key)
    );
  }

  return *result;
}

bool Node::getBool(std::string key) const {
  auto result = attributes.get_optional<bool>(key);

  if (!result) {
    throw std::runtime_error(
      str(format("Cannot read boolean property '%1%'") % key)
    );
  }

  return *result;
}

float Node::getFloat(std::string key) const {
  auto result = attributes.get_optional<float>(key);

  if (!result) {
    throw std::runtime_error(
      str(format("Cannot read float property '%1%'") % key)
    );
  }

  return *result;
}

optix::float3 Node::getFloat3(std::string key) const {
  const NodeFloat3Translator t;
  auto result = attributes.get_optional<
    optix::float3,
    NodeFloat3Translator
  >(key, t);

  if (!result) {
    throw std::runtime_error(
      str(format("Cannot read vector property '%1%'") % key)
    );
  }

  return *result;
}

const AreaLight* Node::getLight(std::string key) const {
  const Node::NodeLookupTranslator<const AreaLight*> t(container.lights);
  auto result = attributes.get_optional<
    const AreaLight*,
    Node::NodeLookupTranslator<const AreaLight*>
  >(key, t);

  if (!result) {
    const std::string itemName = attributes.get<std::string>(key);
    const std::string msg =
      "Cannot resolve light reference '%1%' in property '%2%'";
    throw std::runtime_error(str(format(msg) % itemName % key));
  }

  return *result;
}

const Material* Node::getMaterial(std::string key) const {
  const Node::NodeLookupTranslator<const Material*> t(container.materials);
  auto result = attributes.get_optional<
    const Material*,
    Node::NodeLookupTranslator<const Material*>
  >(key, t);

  if (!result) {
    const std::string itemName = attributes.get<std::string>(key);
    const std::string msg =
      "Cannot resolve material reference '%1%' in property '%2%'";
    throw std::runtime_error(str(format(msg) % itemName % key));
  }

  return *result;
}

const Instance* Node::getGeomInstance(std::string key) const {
  const Node::NodeLookupTranslator<const Instance*> t(container.instances);
  auto result = attributes.get_optional<
    const Instance*,
    Node::NodeLookupTranslator<const Instance*>
  >(key, t);

  if (!result) {
    const std::string itemName = attributes.get<std::string>(key);
    const std::string msg =
      "Cannot resolve geometry reference '%1%' in property '%2%'";
    throw std::runtime_error(str(format(msg) % itemName % key));
  }

  return *result;
}

std::vector<const Instance*> Node::getGeomInstanceList(std::string key) const {
  const Node::NodeLookupTranslator<
    const Instance*,
    false
  > t(container.instances);
  const auto& listRoot = attributes.get_child_optional(key);

  if (!listRoot) {
    throw std::runtime_error(str(format("Cannot read list '%1%'") % key));
  }

  std::vector<const Instance*> result;
  for (const auto& listItem : *listRoot) {
    const auto item = listItem.second.get_value_optional<const Instance*>(t);

    if (!item) {
      const std::string itemName = listItem.second.get_value<std::string>();
      const std::string msg =
        "Cannot resolve geometry reference '%1%' in list '%2%'";
      throw std::runtime_error(str(format(msg) % itemName % key));
    }

    result.push_back(*item);
  }

  return result;
}

Node::NodeFloat3Translator::NodeFloat3Translator() {}

boost::optional<optix::float3> Node::NodeFloat3Translator::get_value(
  const std::string& data
) const {;
  std::vector<std::string> tokens;
  boost::algorithm::split(
    tokens,
    data,
    boost::is_space(),
    boost::token_compress_on
  );

  if (tokens.size() != 3) {
    return boost::optional<optix::float3>();
  }

  optix::float3 result;
  try {
    result.x = std::stof(tokens[0]);
    result.y = std::stof(tokens[1]);
    result.z = std::stof(tokens[2]);
  } catch (...) {
    return boost::optional<optix::float3>();
  }

  return boost::optional<optix::float3>(result);
}

template <typename T, bool allowNull>
Node::NodeLookupTranslator<T, allowNull>::NodeLookupTranslator(
  const std::map<std::string, T>& l
) : lookup(l) {}

template <typename T, bool allowNull>
boost::optional<T> Node::NodeLookupTranslator<T, allowNull>::get_value(
  const std::string& data
) const {
  if (data.length() == 0 && allowNull) {
    return boost::optional<T>((T) nullptr);
  } else if (lookup.count(data) != 0) {
    return boost::optional<T>(lookup.at(data));
  }

  return boost::optional<T>();
}
