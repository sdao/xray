#include "camera.h"
#include "math.h"
#include "light.h"
#include "util.h"

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

  _focalPlaneOrigin =
    optix::make_float3(-halfFocalPlaneRight, halfFocalPlaneUp, -_focalLength);
  _focalPlaneSize =
    optix::make_float2(2.0f * halfFocalPlaneRight, -2.0f * halfFocalPlaneUp);
  
  // Set up OptiX image buffers.
  _raw =
    _ctx->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT3, 2, ww, hh);
  _accum =
    _ctx->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, ww, hh);
  _image =
    _ctx->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, ww, hh);
  _rng =
    _ctx->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_INT, ww, hh);
  util::withMappedBuffer<unsigned int>(_rng, util::fillRandom);

  // Set up OptiX ray and miss programs.
  _cam = _ctx->createProgramFromPTXFile(
    "ptx/camera.cu.ptx",
    "camera_nodirect"
  );
  _camNee = _ctx->createProgramFromPTXFile(
    "ptx/camera.cu.ptx",
    "camera_direct"
  );

  optix::Program cams[2] = {_cam, _camNee};
  for (optix::Program c : cams) {
    c["xform"]->setMatrix4x4fv(false, _camToWorldXform.getData());
    c["focalPlaneOrigin"]->set3fv(&_focalPlaneOrigin.x);
    c["focalPlaneSize"]->set2fv(&_focalPlaneSize.x);
    c["lensRadius"]->setFloat(_lensRadius);
  }

  _miss = _ctx->createProgramFromPTXFile("ptx/camera.cu.ptx", "miss");
  _miss["backgroundColor"]->setFloat(0, 0, 0);

  _commit = _ctx->createProgramFromPTXFile("ptx/camera.cu.ptx", "commit");
  _init = _ctx->createProgramFromPTXFile("ptx/camera.cu.ptx", "init");
}

Camera::~Camera() {}

Camera* Camera::make(Xray* xray, const Node& n) {
  return new Camera(
    xray,
    shared::rotationThenTranslation(
      n.getFloat("rotateAngle"),
      n.getFloat3("rotateAxis"),
      n.getFloat3("translate")
    ),
    n.getInstanceList("objects"),
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
  _ctx->setRayGenerationProgram(CAMERA_TRACE_NO_NEXT_EVENT_ESTIMATION, _cam);
  _ctx->setRayGenerationProgram(CAMERA_TRACE_NEXT_EVENT_ESTIMATION, _camNee);
  _ctx->setRayGenerationProgram(CAMERA_COMMIT, _commit);
  _ctx->setRayGenerationProgram(CAMERA_INIT, _init);
  _ctx->setMissProgram(CAMERA_TRACE_NO_NEXT_EVENT_ESTIMATION, _miss);
  _ctx->setMissProgram(CAMERA_TRACE_NEXT_EVENT_ESTIMATION, _miss);

  // Collect instances and set up acceleration structures.
  std::vector<optix::GeometryInstance> geomPtrs;
  std::vector<Light*> lightPtrs;
  for (const Instance* inst : _objs) {
    optix::GeometryInstance g = inst->getGeometryInstance();
    geomPtrs.push_back(g);

    Light* l = inst->getLightInstance();
    if (l->id != -1) {
      lightPtrs.push_back(l);
    }
  }

  optix::GeometryGroup group =
    _ctx->createGeometryGroup(geomPtrs.begin(), geomPtrs.end());
  optix::Acceleration accel = _ctx->createAcceleration("Trbvh", "Bvh");
  group->setAcceleration(accel);
  accel->markDirty();
  _ctx["sceneRoot"]->set(group);

  _ctx["lightsBuffer"]->setBuffer(util::putUserBuffer<Light>(_ctx, lightPtrs));
  _ctx["numLights"]->setUint(unsigned(lightPtrs.size()));

  // Validate and compile.
  _ctx->validate();
  _ctx->compile();
}

void Camera::render(bool nextEventEstimation) {
  if (_needReset) {
    _ctx->launch(CAMERA_INIT, _w, _h);
    _needReset = false;
  }

  if (nextEventEstimation) {
    _ctx->launch(CAMERA_TRACE_NEXT_EVENT_ESTIMATION, _w, _h);
  } else {
    _ctx->launch(CAMERA_TRACE_NO_NEXT_EVENT_ESTIMATION, _w, _h);
  }

  _commit["commitWeight"]->setFloat(nextEventEstimation ? 1.0f : 0.25f);
  _ctx->launch(CAMERA_COMMIT, _w, _h);

  _frame++;
}

void Camera::translate(optix::float3 v) {
  _camToWorldXform = optix::Matrix4x4::translate(v) * _camToWorldXform;
  _cam["xform"]->setMatrix4x4fv(false, _camToWorldXform.getData());
  _camNee["xform"]->setMatrix4x4fv(false, _camToWorldXform.getData());
  _needReset = true;
}
