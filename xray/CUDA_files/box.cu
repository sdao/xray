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
#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

using namespace optix;

// Declare some variables set by the host
rtDeclareVariable(float3, boxmin, , );
rtDeclareVariable(float3, boxmax, , );
// Get an internal variable
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
// Attributes are variables with a specific function: they must be passed from this intersection program to
// the shader program (in case there's something that needs to be shared). This is somewhat similar to the
// openGL attribute variables passed to shaders
rtDeclareVariable(float3, texcoord, attribute texcoord, ); 
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 

// This function calculates the normal vector to the box's plane intersected by the ray with parameter t
// Notice that it is not a RT_PROGRAM
static __device__ float3 boxnormal(float t)
{
	float3 t0 = (boxmin - ray.origin)/ray.direction;
	float3 t1 = (boxmax - ray.origin)/ray.direction;
	float3 neg = make_float3(t==t0.x?1:0, t==t0.y?1:0, t==t0.z?1:0);
	float3 pos = make_float3(t==t1.x?1:0, t==t1.y?1:0, t==t1.z?1:0);
	return pos-neg;
}


RT_PROGRAM void box_intersect(int)
{
	// Does a ray-box intersection (these formulas have been obtained from the well-known ray-box intersection case)
	float3 t0 = (boxmin - ray.origin)/ray.direction;
	float3 t1 = (boxmax - ray.origin)/ray.direction;
	// Get a vector of 3 floats with the minimum values and the maximum ones
	float3 near = fminf(t0, t1);
	float3 far = fmaxf(t0, t1);
	// Choose absolute max and min
	float tmin = fmaxf( near );
	float tmax = fminf( far );

	if(tmin <= tmax) 
	{
		bool check_second = true;
		// Every time there's a potential intersection and the 't' parameter is calculated, the rtPotentialIntersection function
		// must be called to verify if 't' is in the acceptable range for the ray
		if( rtPotentialIntersection(tmin) ) 
		{
			// If it is, we have a valid intersection so calculate normals for further shading
			texcoord = make_float3( 0.0f );
			shading_normal = geometric_normal = boxnormal( tmin );
			// Confirm the intersection and assigns it material 0
			if(rtReportIntersection(0))
				check_second = false; // Don't check the exit point of the ray, we already have what we need
		} 
		if(check_second) // If the first intersection hadn't a valid distance, try the second one (that normally is the ray's exit point from the object)
		{
			if( rtPotentialIntersection( tmax ) ) 
			{
				texcoord = make_float3( 0.0f );
				shading_normal = geometric_normal = boxnormal( tmax );
				rtReportIntersection(0);
			}
		}
	}
}

// Just returns the minimum and maximum value for the intersection
RT_PROGRAM void box_bounds (int, float result[6])
{
	optix::Aabb* aabb = (optix::Aabb*)result;
	aabb->set(boxmin, boxmax);
}
