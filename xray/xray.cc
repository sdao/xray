#include "xray.h"
#include "camera.h"

Xray::Xray() : _nextID(0) {
  _ctx = optix::Context::create();
  _ctx->setRayTypeCount(RayTypes::RAY_TYPE_COUNT);
  _ctx->setEntryPointCount(CameraMode::CAMERA_MODE_COUNT);
}

Xray::~Xray() {
  _ctx->destroy();
}

optix::Context Xray::getContext() const {
  return _ctx;
}

int Xray::getNextID() {
  return _nextID++;
}
