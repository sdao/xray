/*
 * Copyright (c) 2008 - 2013 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#include <optix_world.h>

using namespace optix;

// This structure will be passed along with each ray as a user-defined custom data structure
struct PerRayData_radiance
{
  float3 result;
  float  importance;
  int    depth;
};

// Retrieve the eye, U, V, W variables
rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(float,         scene_epsilon, , );
// Retrieve the output buffer and treats it as a uchar4 output_buffer[WIDTH][HEIGHT] array to facilitate access.
// Notice that the []operator has been overloaded so output_buffer[launch_index] provides immediate access to the
// ray's corresponding image pixel.
// The difference between variables and buffers is that variables are set by the host once before the program's
// execution while buffers are set by the program to communicate with the host
rtBuffer<uchar4, 2>              output_buffer;
rtDeclareVariable(rtObject,      top_object, , ); // This variable will be set at last and represents the node from which
// the entire tree of objects, transformations, geometries to be rendered will start to be drawn
rtDeclareVariable(unsigned int,  radiance_ray_type, , );

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );


// A convenience device-only function
// Convert a float3 in [0,1) (3 bytes) to a uchar4 in [0,255] (4 bytes) -- 4th channel is set to 255 (fully opaque)
#ifdef __CUDACC__
static __device__ __inline__ optix::uchar4 make_color(const optix::float3& c)
{
    return optix::make_uchar4( static_cast<unsigned char>(__saturatef(c.z)*255.99f),  /* B */
                               static_cast<unsigned char>(__saturatef(c.y)*255.99f),  /* G */
                               static_cast<unsigned char>(__saturatef(c.x)*255.99f),  /* R */
                               255u);                                                 /* A */
}
#endif


RT_PROGRAM void pinhole_camera()
{
  // This line allows the 'd' ray variable to be in the range [-1,1] depending on where the ray starts (ray at pixel 0;0 gets -1;-1, ray at pixel WIDTH;HEIGHT gets 1;1)
  float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f;
  float3 ray_origin = eye;
  // Create a "gradient-like" field, each ray has a vector shot from a pin-hole camera directly into the scene. The ray at 0;0 (the center of the screen)
  // gets a ray perfectly perpendicular to the screen plane
  // E.g.
  //				 /
  //				|->
  // =camera=>		|->
  //				|->
  //		  		 \
  
  float3 ray_direction = normalize(d.x*U + d.y*V + W);
  
  // Create a ray in the desired direction from the specified origin
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);

  PerRayData_radiance prd;
  prd.importance = 1.f;
  prd.depth = 0;

  // Trace the ray by starting the rendering process on the node specified by top_object. The prd object will be passed across all hit and shading programs
  // and will ultimately be set to the result of the shading process
  rtTrace(top_object, ray, prd);

  output_buffer[launch_index] = make_color( make_float3(prd.result.z, prd.result.x, prd.result.y) );
}

RT_PROGRAM void exception()
{
  const unsigned int code = rtGetExceptionCode();
  rtPrintf( "Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y );
  output_buffer[launch_index] = make_color( bad_color );
}
