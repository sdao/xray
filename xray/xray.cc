#include "xray.h"

Xray::Xray() : _nextID(0) {
  _ctx = optix::Context::create();
  _ctx->setRayTypeCount(1);
  _ctx->setEntryPointCount(3);
}

Xray::~Xray() {}

optix::Context Xray::getContext() const {
  return _ctx;
}

int Xray::getNextID() {
  return _nextID++;
}
