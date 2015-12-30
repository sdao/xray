#pragma once
#define NOMINMAX
#include <optix.h>
#include <optix_world.h>

/**
 * Wraps global state for the Xray renderer and owns all OptiX objects.
 * When this object is destroyed, all Xray objects, including the context,
 * will be destroyed as well.
 */
class Xray {
  int _nextID;
  optix::Context _ctx;

public:
  Xray();
  ~Xray();

  /** Gets the OptiX context. Each Xray instance owns a separate context. */
  optix::Context getContext() const;

  /** Gets a monotonically increasing, 0-based unique ID. */
  int getNextID();
};
