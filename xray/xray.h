#pragma once
#define NOMINMAX
#include <optix.h>
#include <optix_world.h>

class Xray {
  optix::Context _ctx;

public:
  Xray();
  ~Xray();

  optix::Context getContext() const;
};