#include "camera.h"

Camera::Camera(
  optix::Context ctx,
  optix::Matrix4x4 xform,
  int ww,
  int hh,
  float fov,
  float len,
  float fStop
) : _ctx(ctx),
    _focalLength(len),
    _lensRadius((len / fStop) * 0.5f), // Diameter = focalLength / fStop.
    _camToWorldXform(xform),
    _w(ww), _h(hh)
{
  // Calculate ray-tracing vectors.
  float halfFocalPlaneUp;
  float halfFocalPlaneRight;

  if (_w > _h) {
    halfFocalPlaneUp = _focalLength * tanf(0.5f * fov);
    halfFocalPlaneRight = halfFocalPlaneUp * float(_w) / float(_h);
  } else {
    halfFocalPlaneRight = _focalLength * tanf(0.5f * fov);
    halfFocalPlaneUp = halfFocalPlaneRight * float(_h) / float(_w);
  }

  _focalPlaneUp = -2.0f * halfFocalPlaneUp;
  _focalPlaneRight = 2.0f * halfFocalPlaneRight;
  _focalPlaneOrigin = optix::make_float3(-halfFocalPlaneRight, halfFocalPlaneUp, -_focalLength);
}

optix::Program Camera::getProgram() const {
  optix::Program cam = _ctx->createProgramFromPTXFile("PTX_files/camera.cu.ptx", "camera");
  cam["xform"]->setMatrix4x4fv(false, _camToWorldXform.getData());
  cam["focalPlaneOrigin"]->set3fv(&_focalPlaneOrigin.x);
  cam["focalPlaneRight"]->setFloat(_focalPlaneRight);
  cam["focalPlaneUp"]->setFloat(_focalPlaneUp);
  return cam;
}