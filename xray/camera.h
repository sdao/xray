#pragma once
#define NOMINMAX
#include <optix.h>
#include <optix_world.h>
#include "CUDA_files/math.cuh"
#include "instance.h"
#include "node.h"
#include "xray.h"

class Camera {
  mutable optix::Context _ctx;
  optix::Program _cam;
  optix::Program _miss;
  optix::Buffer _raw;
  optix::Buffer _image;

  const std::vector<const Instance*> _objs;

  int _w;
  int _h;

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

  optix::Buffer getImageBuffer();
  void render();
};

