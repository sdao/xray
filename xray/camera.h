#pragma once
#define NOMINMAX
#include <optix.h>
#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include "CUDA_files/shared.cuh"
#include "instance.h"
#include "node.h"
#include "xray.h"

enum CameraMode {
  CAMERA_TRACE = 0,
  CAMERA_COMMIT = 1,
  CAMERA_INIT = 2
};

class Camera {
  mutable optix::Context _ctx;
  optix::Program _cam;
  optix::Program _miss;
  optix::Program _commit;
  optix::Program _init;
  optix::Buffer _raw;
  optix::Buffer _accum;
  optix::Buffer _image;
  optix::Buffer _rng;

  const std::vector<const Instance*> _objs;

  int _w;
  int _h;

  unsigned int _frame;

  const float _focalLength; /**< The distance from the eye to the focal plane. */
  const float _lensRadius; /**< The radius of the lens opening. */
  const optix::Matrix4x4 _camToWorldXform; /**< Transform from camera to world space. */

  float _focalPlaneUp; /**< The height of the focal plane. */
  float _focalPlaneRight; /**< The width of the focal plane. */
  optix::float3 _focalPlaneOrigin; /**< The origin (corner) of the focal plane. */

public:
  Camera(
    Xray xray,
    optix::Matrix4x4 xform,
    const std::vector<const Instance*>& objs,
    int ww,
    int hh,
    float fov = XRAY_PI_4,
    float len = 50.0f,
    float fStop = 16.0f
  );
  ~Camera();

  static Camera* make(Xray xray, const Node& n);

  optix::Buffer imageBuffer();
  int pixelWidth() const;
  int pixelHeight() const;
  unsigned int frameNumber() const;

  void prepare();
  void render();
};

