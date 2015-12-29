#include "camera.h"
#include "math.h"
#include "light.h"
#include <random>

Camera::Camera(
  Xray* xray,
  optix::Matrix4x4 xform,
  const std::vector<const Instance*>& objs,
  int ww,
  int hh,
  float fov,
  float len,
  float fStop
) : _ctx(xray->getContext()),
    _focalLength(len),
    _lensRadius((len / fStop) * 0.5f), // Diameter = focalLength / fStop.
    _camToWorldXform(xform),
    _w(ww), _h(hh),
    _objs(objs),
    _frame(0),
    _needReset(true)
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

  std::mt19937 rng(42);
  std::uniform_int_distribution<unsigned> unsignedDist;
  unsigned int* rngMapped = static_cast<unsigned int*>(_rng->map());
  for (int i = 0; i < ww * hh; ++i) {
    rngMapped[i] = unsignedDist(rng);
  }
  _rng->unmap();

  // Set up OptiX ray and miss programs.
  _cam = _ctx->createProgramFromPTXFile("ptx/camera.cu.ptx", "camera_nodirect");
  _cam["xform"]->setMatrix4x4fv(false, _camToWorldXform.getData());
  _cam["focalPlaneOrigin"]->set3fv(&_focalPlaneOrigin.x);
  _cam["focalPlaneRight"]->setFloat(_focalPlaneRight);
  _cam["focalPlaneUp"]->setFloat(_focalPlaneUp);
  _cam["lensRadius"]->setFloat(_lensRadius);
  
  _camNee = _ctx->createProgramFromPTXFile("ptx/camera.cu.ptx", "camera_direct");
  _camNee["xform"]->setMatrix4x4fv(false, _camToWorldXform.getData());
  _camNee["focalPlaneOrigin"]->set3fv(&_focalPlaneOrigin.x);
  _camNee["focalPlaneRight"]->setFloat(_focalPlaneRight);
  _camNee["focalPlaneUp"]->setFloat(_focalPlaneUp);
  _camNee["lensRadius"]->setFloat(_lensRadius);

  _miss = _ctx->createProgramFromPTXFile("ptx/camera.cu.ptx", "miss");
  _miss["backgroundColor"]->setFloat(0, 0, 0);

  _commit = _ctx->createProgramFromPTXFile("ptx/camera.cu.ptx", "commit");
  _init = _ctx->createProgramFromPTXFile("ptx/camera.cu.ptx", "init");
}

Camera::~Camera() {
  _ctx->destroy();
}

Camera* Camera::make(Xray* xray, const Node& n) {
  return new Camera(
    xray,
    shared::rotationThenTranslation(n.getFloat("rotateAngle"), n.getFloat3("rotateAxis"), n.getFloat3("translate")),
    n.getGeomInstanceList("objects"),
    n.getInt("width"), n.getInt("height"),
    n.getFloat("fov"), n.getFloat("focalLength"),
    n.getFloat("fStop")
  );
}

optix::Buffer Camera::accumBuffer() {
  return _accum;
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
  _ctx["randBuffer"]->setBuffer(_rng);
  _ctx->setRayGenerationProgram(CAMERA_TRACE_NORMAL, _cam);
  _ctx->setRayGenerationProgram(CAMERA_TRACE_NEXT_EVENT_ESTIMATION, _camNee);
  _ctx->setRayGenerationProgram(CAMERA_COMMIT, _commit);
  _ctx->setRayGenerationProgram(CAMERA_INIT, _init);
  _ctx->setMissProgram(CAMERA_TRACE_NORMAL, _miss);
  _ctx->setMissProgram(CAMERA_TRACE_NEXT_EVENT_ESTIMATION, _miss);

  // Set up acceleration structures.
  std::vector<Light*> lightPtrs;
  optix::GeometryGroup group = _ctx->createGeometryGroup();
  group->setChildCount(unsigned(_objs.size()));
  for (int i = 0; i < _objs.size(); ++i) {
    const Instance* inst = _objs[i];
    optix::GeometryInstance g = inst->getGeometryInstance();
    Light* l = inst->getLightInstance();

    group->setChild(i, inst->getGeometryInstance());
    if (l->id != -1) {
      lightPtrs.push_back(l);
    }
  }

  _lights = _ctx->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, lightPtrs.size());
  _lights->setElementSize(sizeof(Light));
  Light* lightsMapped = static_cast<Light*>(_lights->map());
  for (int i = 0; i < lightPtrs.size(); ++i) {
    lightsMapped[i] = *lightPtrs[i];
  }
  _lights->unmap();
  _ctx["lightsBuffer"]->setBuffer(_lights);
  _ctx["numLights"]->setUint(unsigned(lightPtrs.size()));

  optix::Acceleration accel = _ctx->createAcceleration("Trbvh", "Bvh");
  group->setAcceleration(accel);
  accel->markDirty();

  _ctx["sceneRoot"]->set(group);

  // Validate and compile.
  _ctx->validate();
  _ctx->compile();
}

void Camera::render(bool nextEventEstimation) {
  if (_needReset) {
    _ctx->launch(CAMERA_INIT, _w, _h);
    _needReset = false;
  }

  _commit["commitWeight"]->setFloat(nextEventEstimation ? 1.0f : 0.25f);
  _ctx->launch(nextEventEstimation ? CAMERA_TRACE_NEXT_EVENT_ESTIMATION : CAMERA_TRACE_NORMAL, _w, _h);
  _ctx->launch(CAMERA_COMMIT, _w, _h);
  _frame++;
}

void Camera::translate(optix::float3 v) {
  _camToWorldXform = optix::Matrix4x4::translate(v) * _camToWorldXform;
  _cam["xform"]->setMatrix4x4fv(false, _camToWorldXform.getData());
  _camNee["xform"]->setMatrix4x4fv(false, _camToWorldXform.getData());
  _needReset = true;
}
