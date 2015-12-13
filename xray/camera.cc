#include "camera.h"
#include "math.h"

Camera::Camera(
  Xray xray,
  optix::Matrix4x4 xform,
  const std::vector<const Instance*>& objs,
  int ww,
  int hh,
  float fov,
  float len,
  float fStop
) : _ctx(xray.getContext()),
    _focalLength(len),
    _lensRadius((len / fStop) * 0.5f), // Diameter = focalLength / fStop.
    _camToWorldXform(xform),
    _w(ww), _h(hh),
    _objs(objs),
    _frame(0)
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
  
  // Set up OptiX image buffers.
  _raw = _ctx->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT3, 2, ww, hh);
  _accum = _ctx->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, ww, hh);
  _image = _ctx->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, ww, hh);
  _rng = _ctx->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_INT, ww, hh);

  // Set up OptiX ray and miss programs.
  _cam = _ctx->createProgramFromPTXFile("PTX_files/camera.cu.ptx", "camera");
  _cam["xform"]->setMatrix4x4fv(false, _camToWorldXform.getData());
  _cam["focalPlaneOrigin"]->set3fv(&_focalPlaneOrigin.x);
  _cam["focalPlaneRight"]->setFloat(_focalPlaneRight);
  _cam["focalPlaneUp"]->setFloat(_focalPlaneUp);
  _cam["lensRadius"]->setFloat(_lensRadius);

  _miss = _ctx->createProgramFromPTXFile("PTX_files/camera.cu.ptx", "miss");
  _miss["backgroundColor"]->setFloat(0, 0, 0);

  _commit = _ctx->createProgramFromPTXFile("PTX_files/camera.cu.ptx", "commit");
  _init = _ctx->createProgramFromPTXFile("PTX_files/camera.cu.ptx", "init");
}

Camera::~Camera() {
  _ctx->destroy();
}

Camera* Camera::make(Xray xray, const Node& n) {
  return new Camera(
    xray,
    shared::rotationThenTranslation(n.getFloat("rotateAngle"), n.getFloat3("rotateAxis"), n.getFloat3("translate")),
    n.getGeomInstanceList("objects"),
    n.getInt("width"), n.getInt("height"),
    n.getFloat("fov"), n.getFloat("focalLength"),
    n.getFloat("fStop")
  );
}

optix::Buffer Camera::imageBuffer() {
  return _image;
}

int Camera::pixelWidth() const {
  return _w;
}

int Camera::pixelHeight() const {
  return _h;
}

unsigned int Camera::frameNumber() const {
  return _frame;
}

void Camera::prepare() {
  // Associate camera programs/buffers.
  _ctx["rawBuffer"]->setBuffer(_raw);
  _ctx["accumBuffer"]->setBuffer(_accum);
  _ctx["imageBuffer"]->setBuffer(_image);
  _ctx["rngBuffer"]->setBuffer(_rng);
  _ctx->setRayGenerationProgram(CAMERA_TRACE, _cam);
  _ctx->setRayGenerationProgram(CAMERA_COMMIT, _commit);
  _ctx->setRayGenerationProgram(CAMERA_INIT, _init);
  _ctx->setMissProgram(0, _miss);

  // Set up acceleration structures.
  optix::GeometryGroup group = _ctx->createGeometryGroup();
  group->setChildCount(_objs.size());
  for (int i = 0; i < _objs.size(); ++i) {
    group->setChild(i, _objs[i]->getGeometryInstance());
  }

  optix::Acceleration accel = _ctx->createAcceleration("Trbvh", "Bvh");
  group->setAcceleration(accel);
  accel->markDirty();

  _ctx["sceneRoot"]->set(group);

  // Validate, compile, and run.
  _ctx->validate();
  _ctx->compile();
  _ctx->launch(CAMERA_INIT, _w, _h);
}

void Camera::render() {
  _ctx["frameNumber"]->setUint(_frame);
  _ctx->launch(CAMERA_TRACE, _w, _h);
  _ctx->launch(CAMERA_COMMIT, _w, _h);

  optix::float3* raw = static_cast<optix::float3*>(_raw->map());

  _raw->unmap();

  _frame++;
}