#pragma once
#define NOMINMAX
#include <optix.h>
#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include "cuda/shared.cuh"
#include "instance.h"
#include "node.h"
#include "xray.h"

enum CameraMode {
  CAMERA_TRACE_NO_NEXT_EVENT_ESTIMATION = 0,
  CAMERA_TRACE_NEXT_EVENT_ESTIMATION = 1,
  CAMERA_COMMIT = 2,
  CAMERA_INIT = 3,
  CAMERA_MODE_COUNT = 4
};

/**
 * Represents a pinhole camera. Contains functionality for building
 * acceleration structures for a scene and for rendering incremental iterations.
 */
class Camera {
  mutable optix::Context _ctx; /**< OptiX context for this camera. */
  optix::Program _cam;         /**< Camera program without NEE. */
  optix::Program _camNee;      /**< Camera program with NEE. */
  optix::Program _miss;        /**< Miss (constant black) program. */
  optix::Program _commit;      /**< Program for committing iteration samples. */
  optix::Program _init;        /**< Program for clearing committed samples. */
  optix::Buffer _raw;          /**< Buffer with current raw samples. */
  optix::Buffer _accum;        /**< Buffer with committed samples. */
  optix::Buffer _image;        /**< Integer BGRA image buffer for preview. */
  optix::Buffer _rng;          /**< Buffer with per-pixel RNG seeds. */

  const int _w;                /**< Rendered image width. */
  const int _h;                /**< Rendered image height. */

  /** Vector with all instances in the scene to render. */
  const std::vector<const Instance*> _objs;

  /** The distance from the eye to the focal plane. */
  const float _focalLength;
  /** The radius of the lens opening. */
  const float _lensRadius;
  /** Transform from camera to world space. */
  optix::Matrix4x4 _camToWorldXform;
  /** The origin (corner) of the focal plane. */
  optix::float3 _focalPlaneOrigin;
  /** The 2D size of the focal plane. */
  optix::float2 _focalPlaneSize;
  
  /** Last frame index (1-based). Frame 0 means no frames rendered yet. */
  unsigned int _frame;
  /** Flag indicating if the accumulated committed samples should be cleared. */
  bool _needReset;

public:
  /**
   * Constructs a camera.
   *
   * @param xray   the global Xray state to attach
   * @param xform  the transformation from camera space to world space
   * @param objs   the objects to render
   * @param ww     the width of the output image, in pixels
   * @param hh     the height of the output image, in pixels
   * @param fov    the field of view (horizontal or vertical, whichever is
   *               smaller), in radians
   * @param len    the focal length of the lens
   * @param fStop the f-stop (aperture) of the lens
   */
  Camera(
    Xray* xray,
    optix::Matrix4x4 xform,
    const std::vector<const Instance*>& objs,
    int ww,
    int hh,
    float fov = XRAY_PI_4,
    float len = 50.0f,
    float fStop = 16.0f
  );
  ~Camera();

  /**
   * Makes a camera from the given node.
   */
  static Camera* make(Xray* xray, const Node& n);

  optix::Buffer accumBuffer();
  optix::Buffer imageBuffer();
  int pixelWidth() const;
  int pixelHeight() const;
  unsigned int frameNumber() const;

  /**
   * Prepares the acceleration structures and binds the camera to the current
   * context in preparation for rendering. This needs to be called before the
   * render() method, and needs to be called again if any other camera is bound
   * using prepare.
   * Might take a long time to complete depending on the acceleration structure
   * building process.
   */
  void prepare();

  /**
   * Renders one iteration of the image, accumulating onto older iterations.
   * You need to call prepare() before any call to render().
   *
   * @param nextEventEstimation whether to perform next-event estimation (direct
   *                            lighting step).
   */
  void render(bool nextEventEstimation = false);

  /**
   * Translates the camera's transformation matrix by the given vector, and
   * flags the accumulated sample buffer to be reset on the next render pass.
   */
  void translate(optix::float3 v);
};

