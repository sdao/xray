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
#pragma once

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "phong.h"

using namespace optix;

// These variables are set by the host and are coefficients necessary to the phong shading
rtDeclareVariable(float3, Ka, , );
rtDeclareVariable(float3, Kd, , );
rtDeclareVariable(float3, Ks, , );
rtDeclareVariable(float3, reflectivity, , );
rtDeclareVariable(float,  phong_exp, , );

// These variables are passed directly from the intersection program, that's why they're attributes
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 


// Compute a simple shadow
RT_PROGRAM void any_hit_shadow()
{
	phongShadowed();
}

// This will be executed just for the first hit and it's the primary color of the object directly hit by a ray
RT_PROGRAM void closest_hit_radiance()
{
	// Our normal is just expressed in object coordinates (relative to the object) and without any notion of the world of the scene
	// the rtTransformNormal takes the combination of each transformation on the transformation stack for this "tree" of nodes to be rendered
	// and ultimately applies the inverse transpose of this matrix to the normal. Why the inverse transpose? Take a look around on the internet
	// and search for "normal transformation": normals don't get transformed like every other point otherwise their direction would not be preserved.
	float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
	float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );

	// Remember that in this example world_shading_normal and world_geometric_normal are the same.. this function
	// makes sure to get a normal oriented away from the entering ray
	float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );
	// Use the phong shading equation by supplying it common parameters
	phongShade( Kd, Ka, Ks, ffnormal, phong_exp, reflectivity );
  //prd.result = make_float3(1, 0, 0);
}
