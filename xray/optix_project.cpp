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
#include "optix_helpers.h"

#define WIDTH   512
#define HEIGHT  512
#define PATH_TO_MY_PTX_FILES  "PTX_files" // The directory relative to this executable where PTX
                                          // files are stored. WARNING: if this is missing, the
                                          // executable will crash

// ----------------------------------- OptiX Routines -----------------------------------
sUtilWrapper *sUtil;
void createContext(RTcontext* context, RTbuffer* output_buffer_obj);
void createGeometry(RTcontext context, RTgeometry* box);
void createMaterial(RTcontext context, RTmaterial* material);
void createInstances(RTcontext context, RTgeometry box, RTmaterial material);
// --------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
  // Load a wrapper to sUtil.dll and other utility functions
  try { sUtil = new sUtilWrapper(); } catch(char *err) { cout << err; return 1; }
  
  // First initialize freeglut to create a gl context where to draw on
  sUtil->initGlut(&argc, argv);

  // -----> OptiX rendering process
  
  // This object will hold the resulting scene
  RTbuffer output_buffer_obj;

  // Step 1 - create an OptiX context and indicate the resulting buffer where the scene will be drawn
  RTcontext context;
  createContext(&context, &output_buffer_obj);

  // Step 2 - create the geometry of the scene
  RTgeometry box;
  createGeometry(context, &box);

  // Step 3 - create materials for the objects in the scene
  RTmaterial material;
  createMaterial(context, &material);

  // Step 4 - create geometry nodes, transformation nodes and attach them to the OptiX tree hierarchy
  createInstances( context, box, material );

  // Step 5 - run the context!
  rtContextValidate( context );
  rtContextCompile( context );
  rtContextLaunch2D( context, 0, WIDTH, HEIGHT );

  /* Finally Display image */
  sUtil->DisplayBufferInGlutWindow( context, argv[0], output_buffer_obj );

  // <-----

  // Destroy the context for a graceful cleanup
  rtContextDestroy(context); 
  // Free the wrapper
  delete sUtil;

  return 0;
}


// In order for Optix to work, a context and some entry points (objects that contain a ray generation program and a miss program)
// must be created. If these programs have variables associated with them, they need to be created (and defined) as well.
void createContext(RTcontext* context, RTbuffer* output_buffer_obj)
{
  // Create an OptiX context. A context is necessary to ray trace a scene and contains all materials, geometries, textures, etc...
  // It is the fundamental part of an OptiX scene
  rtContextCreate(context);

  // We will be using 2 kind of rays in this example, a shadow ray and a radiance ray
  rtContextSetRayTypeCount(*context, 2);

  // One entry point, an entry point is a couple of
  // - a ray generation program in ptx (this is where a ray is shot)
  // - an exception program (this is called in case a ray cannot be shot for some reasons)
  rtContextSetEntryPointCount(*context, 1);

  // Declare some variables that will be used
  RTvariable output_buffer;
  RTvariable light_buffer;
  RTvariable radiance_ray_type;
  RTvariable shadow_ray_type;
  RTvariable epsilon;
  RTvariable max_depth;
  rtContextDeclareVariable( *context, "output_buffer", &output_buffer );
  rtContextDeclareVariable( *context, "lights", &light_buffer );
  rtContextDeclareVariable( *context, "max_depth", &max_depth );
  rtContextDeclareVariable( *context, "radiance_ray_type", &radiance_ray_type );
  rtContextDeclareVariable( *context, "shadow_ray_type", &shadow_ray_type );
  rtContextDeclareVariable( *context, "scene_epsilon", &epsilon );

  // Set the value of some of these variables, 1i means "set 1 integer variable", 
  // likewise for 1u (unsigned int) and 1f (float)
  rtVariableSet1i( max_depth, 10u ) ; // A maximum depth of 10
  rtVariableSet1ui( radiance_ray_type, 0u ) ;
  rtVariableSet1ui( shadow_ray_type, 1u ) ;
  rtVariableSet1f( epsilon, 1.e-4f ) ;

  // Set an output buffer to draw the scene on

  // Create an output buffer for this context and set it to RGBA format
  rtBufferCreate( *context, RT_BUFFER_OUTPUT, output_buffer_obj ) ;
  rtBufferSetFormat( *output_buffer_obj, RT_FORMAT_UNSIGNED_BYTE4 ) ;
  rtBufferSetSize2D( *output_buffer_obj, WIDTH, HEIGHT ) ;
  // Binds a program variable to an OptiX object (the scene to be rendered) so it can be accessed
  rtVariableSetObject( output_buffer, *output_buffer_obj ) ;

  // Time to add a light to the scene!

  // Declare a buffer to store the light object just defined in the device (GPU) memory
  RTbuffer   light_buffer_obj;
  // RT BUFFER INPUT specifies that the host may only write to the buffer and the device may only read from the buffer
  rtBufferCreate( *context, RT_BUFFER_INPUT, &light_buffer_obj ) ;
  // Specifies that this buffer will contain data that the programmer knows what it is (and what format it does have)
  rtBufferSetFormat( light_buffer_obj, RT_FORMAT_USER ) ; // rtBufferSetFormat sets a predefined or custom data format for a given buffer
  // Sets the dimension of the buffer to that of the structure (32 bytes)
  rtBufferSetElementSize( light_buffer_obj, sizeof(BasicLight) ) ;
  // Sets this buffer to be a one-dimension array of one element (just our structure, one time only)
  rtBufferSetSize1D( light_buffer_obj, 1 );
  // Declare a simple mono-color light with the totally-arbitrarily-structure defined by us
  BasicLight light;
  light.color.x = 0.9f;
  light.color.y = 0.9f;
  light.color.z = 0.9f;
  light.pos.x   = 0.0f;
  light.pos.y   = 20.0f;
  light.pos.z   = 20.0f;
  light.casts_shadow = 1;
  light.padding      = 0u;
  // Maps the light buffer on the device to a pointer in host space so we can write to it
  void* light_buffer_data;
  rtBufferMap( light_buffer_obj, &light_buffer_data ) ;
  // And actually writes our defined-light to that host pointer (remember that it's a double pointer)
  ((BasicLight*)light_buffer_data)[0] = light;
  // Data has been transferred, stop the host <=> device binding
  rtBufferUnmap( light_buffer_obj );
  // Assigns a variable to this buffer so the device can read/use it
  rtVariableSetObject( light_buffer, light_buffer_obj ) ;

  // Now it's time to set up the ptx programs for this context!
  char path_to_ptx[512];
  RTprogram  ray_gen_program;
  RTprogram  miss_program;
  sprintf_s( path_to_ptx, 512, "%s/%s", PATH_TO_MY_PTX_FILES, "pinhole_camera.cu.ptx" );
  rtProgramCreateFromPTXFile( *context, path_to_ptx, "pinhole_camera", &ray_gen_program ) ;
  // Sets the eye of the spectator (the camera if you want) to be at 0;0;5. Like in openGL's convention we're looking
  // at the Z axis on its negative side. It also sets three vectors for a left-handed base
  RTvariable eye;
  RTvariable U;
  RTvariable V;
  RTvariable W;
  rtProgramDeclareVariable( ray_gen_program, "eye", &eye );
  rtProgramDeclareVariable( ray_gen_program, "U", &U );
  rtProgramDeclareVariable( ray_gen_program, "V", &V );
  rtProgramDeclareVariable( ray_gen_program, "W", &W );
  rtVariableSet3f( eye, 0.0f, 0.0f, 5.0f );
  // U, V and W form a left-handed base
  rtVariableSet3f( U, 1.0f, 0.0f, 0.0f );
  rtVariableSet3f( V, 0.0f, 1.0f, 0.0f );
  rtVariableSet3f( W, 0.0f, 0.0f, -1.0f );
  // Set the ray generation program for entry point 0 (associate the entry point 0 with this program)
  rtContextSetRayGenerationProgram( *context, 0, ray_gen_program ) ;

  // Set the miss program
  sprintf_s( path_to_ptx,512,  "%s/%s", PATH_TO_MY_PTX_FILES, "constantbg.cu.ptx" );
  RTvariable color;
  rtProgramCreateFromPTXFile( *context, path_to_ptx, "miss", &miss_program ) ;
  rtProgramDeclareVariable( miss_program, "bg_color" , &color) ;
  rtVariableSet3f( color, .3f, 0.1f, 0.2f ) ;
  rtContextSetMissProgram( *context, 0, miss_program ) ;
}

// A geometry node (at least one) is necessary to draw a scene. Here we will create a geometry (the box) and its bounding box (that we will
// make the same as the box), we will create the geometry node afterwards and assign it this geometry. Each geometry has also an intersection and bounding box program.
void createGeometry(RTcontext context, RTgeometry* box)
{
  RTprogram  box_intersection_program;
  RTprogram  box_bounding_box_program;
  RTvariable box_min_var;
  RTvariable box_max_var;

  float     box_min[3];
  float     box_max[3];

  // OptiX proceeds by creating nodes for geometries, acceleration structures, etc.. this allows to share one of these objects
  // among other nodes, here a single geometry node for this context is created
  rtGeometryCreate(context, box) ;
  // Sets just one primitive on this geometry node
  rtGeometrySetPrimitiveCount( *box, 1u ) ;

  char path_to_ptx[512];
  sprintf_s(path_to_ptx, 512,"%s/%s", PATH_TO_MY_PTX_FILES, "box.cu.ptx");
  rtProgramCreateFromPTXFile( context, path_to_ptx, "box_bounds", &box_bounding_box_program );
  // Each geometry node must have a bounding box program, this program computes the axis-aligned bounding box for everything inside this node
  rtGeometrySetBoundingBoxProgram( *box, box_bounding_box_program );
  rtProgramCreateFromPTXFile( context, path_to_ptx, "box_intersect", &box_intersection_program );
  rtGeometrySetIntersectionProgram( *box, box_intersection_program );

  // Specify a simple object-space bounding box from the minimum point to the maximum point of the cube
  box_min[0] = box_min[1] = box_min[2] = -0.5f;
  box_max[0] = box_max[1] = box_max[2] =  0.5f;

  // Make these variables available to the gpu and store the chosen value
  rtGeometryDeclareVariable( *box, "boxmin", &box_min_var ) ;
  rtGeometryDeclareVariable( *box, "boxmax", &box_max_var ) ;
  rtVariableSet3fv( box_min_var, box_min ) ;
  rtVariableSet3fv( box_max_var, box_max ) ;
}

// Each time a ray hits a geometry and (possibly) calculates normals/shading parameters, a material needs to be associated with that geometry
// and the material defines the shading equation (in this case, a simple phong shading)
void createMaterial(RTcontext context, RTmaterial* material)
{
  RTprogram closest_hit_program;
  RTprogram any_hit_program;

  // Create the hit programs, closest hit is executed for the ray's closest hit while any hit is executed each time an intersection for a ray is found
  // These programs are shared among all materials
  char path_to_ptx[512];
  sprintf_s(path_to_ptx, 512, "%s/%s", PATH_TO_MY_PTX_FILES, "phong.cu.ptx");
  rtProgramCreateFromPTXFile( context, path_to_ptx, "closest_hit_radiance", &closest_hit_program );
  rtProgramCreateFromPTXFile( context, path_to_ptx, "any_hit_shadow", &any_hit_program );

  // Create a new material, we will be using this each time a ray reaches a hit against a geometry node associated with this material (and we will execute
  // our shading programs then)
  rtMaterialCreate( context, material );

  /* Note that we are leaving anyHitProgram[0] and closestHitProgram[1] as NULL.
  * This is because our radiance rays only need closest_hit and shadow rays only
  * need any_hit */
  // Associate the closest_hit_program and any_hit_program with this material
  // Furthermore closest_hit_program is associated with ray type 0 (we chose ray type 0 to be radiance ray, the lighting ray)
  // while any_hit_program is associated with ray type 1.
  // To summarize we have the following:
  //
  //	Ray type			|	0 (radiance)			1 (shadow)
  //	Material			|	*material				*material
  //	Closest hit program	|	closest_hit_program		NONE
  //	Any hit program		|	NONE					any_hit_program
  //
  rtMaterialSetClosestHitProgram( *material, 0, closest_hit_program );
  rtMaterialSetAnyHitProgram( *material, 1, any_hit_program );
}

void createInstances(RTcontext context, RTgeometry box, RTmaterial material)
{
#define NUM_BOXES 6

  // Each box will be slightly shifted with respect to the others
  RTtransform     transforms[NUM_BOXES];
  // These will be used to set up the scene
  RTgroup         top_level_group;
  RTvariable      top_object;
  RTvariable      top_shadower;
  RTacceleration  top_level_acceleration;
  //
  int i;

  // This matrix is a no-operation transform. We will set just one field of this matrix (m[3])
  // to shift each box to the left or right
  float m[16];
  m[ 0] = 1.0f;  m[ 1] = 0.0f;  m[ 2] = 0.0f;  m[ 3] = 0.0f;
  m[ 4] = 0.0f;  m[ 5] = 1.0f;  m[ 6] = 0.0f;  m[ 7] = 0.0f;
  m[ 8] = 0.0f;  m[ 9] = 0.0f;  m[10] = 1.0f;  m[11] = 0.0f;
  m[12] = 0.0f;  m[13] = 0.0f;  m[14] = 0.0f;  m[15] = 1.0f;

  for ( i = 0; i < NUM_BOXES; ++i ) // For each box in the scene (we want 6 boxes)
  {
    float kd_slider;
    RTgeometrygroup geometrygroup;
    RTgeometryinstance instance;
    RTacceleration acceleration;
    RTvariable kd;
    RTvariable ks;
    RTvariable ka;
    RTvariable reflectivity;
    RTvariable expv;
    RTvariable ambient;

    // Actually create the geometry node in the OptiX context's tree
    rtGeometryInstanceCreate( context, &instance );
    // Assign a geometry to this geometry node
    rtGeometryInstanceSetGeometry( instance, box );
    // We will be using just one material for this geometry node
    rtGeometryInstanceSetMaterialCount( instance, 1 );
    // Indicate the phong shaded material we just created (this contains the programs needed to shade it)
    rtGeometryInstanceSetMaterial( instance, 0, material );

    // Access and set variables to be consumed by the material for this geometry instance
    // These are the parameters in the phong shading equation and some other stuff
    kd_slider = (float)i / (float)(NUM_BOXES-1); // Needed to obtain a [0;1] value for each box
    rtGeometryInstanceDeclareVariable( instance, "Kd", &kd ); // Notice that rtGeometryInstanceDeclareVariable is different from rtGeometryDeclareVariable, 
    // we are accessing a variable defined in a material program associated with a geometry instance node. rtGeometryDeclareVariable just accesses a variable
    // defined in a geometry program.
    rtGeometryInstanceDeclareVariable( instance, "Ks", &ks );
    rtGeometryInstanceDeclareVariable( instance, "Ka", &ka );
    rtGeometryInstanceDeclareVariable( instance, "phong_exp", &expv );
    rtGeometryInstanceDeclareVariable( instance, "reflectivity", &reflectivity );
    rtGeometryInstanceDeclareVariable( instance, "ambient_light_color", &ambient);
    rtVariableSet3f( kd, kd_slider, 0.0f, 1.0f-kd_slider );
    rtVariableSet3f( ks, 0.5f, 0.5f, 0.5f );
    rtVariableSet3f( ka, 0.8f, 0.8f, 0.8f );
    rtVariableSet3f( reflectivity, 0.8f, 0.8f, 0.8f );
    rtVariableSet1f( expv, 10.0f );
    rtVariableSet3f( ambient, 0.2f, 0.2f, 0.2f );

    // A geometry group node may contain one or more [geometry node, acceleration structure]
    rtGeometryGroupCreate( context, &geometrygroup );
    rtGeometryGroupSetChildCount( geometrygroup, 1 );
    rtGeometryGroupSetChild( geometrygroup, 0, instance );

    // Create an acceleration object for group and specify some build hints
    rtAccelerationCreate(context,&acceleration);
    rtAccelerationSetBuilder(acceleration,"NoAccel"); // Dummy acceleration structure, this actually scroll through each node
    rtAccelerationSetTraverser(acceleration,"NoAccel");
    rtGeometryGroupSetAcceleration( geometrygroup, acceleration);

    // mark acceleration as dirty (will be rebuilt at next rtContextLaunch)
    rtAccelerationMarkDirty( acceleration );

    // Add a transform node to the context's tree
    rtTransformCreate( context, &transforms[i] );
    // And make this geometry group (which in turn contains the geometry node (which in turn contains the geometry AND the material) the AND the acceleration structure) a child
    // of that transform (so the transformation is first applied)
    rtTransformSetChild( transforms[i], geometrygroup );
    // Set a slight translation for this transformation node
    m[3] = i*1.5f - (NUM_BOXES-1)*0.75f;
    rtTransformSetMatrix( transforms[i], 0, m, 0 );
  }

  // Now place these transform nodes (and all their geometrygroup children) as children of the top level object
  rtGroupCreate( context, &top_level_group ); // Create a top level group, this just keeps other nodes together
  rtGroupSetChildCount( top_level_group, NUM_BOXES );
  for ( i = 0; i < NUM_BOXES; ++i ) 
  {
    rtGroupSetChild( top_level_group, i, transforms[i] );
  }
  rtContextDeclareVariable( context, "top_object", &top_object ); // Declare a context variable, these are the top-level variables of the context
  rtVariableSetObject( top_object, top_level_group ); // Binds the top_level_group node to the top_object variable
  rtContextDeclareVariable( context, "top_shadower", &top_shadower ); // Does the same thing for the top_shadower variable
  rtVariableSetObject( top_shadower, top_level_group );

  // Create a dummy acceleration structure for the top level group node too
  rtAccelerationCreate( context, &top_level_acceleration );
  rtAccelerationSetBuilder(top_level_acceleration,"NoAccel");
  rtAccelerationSetTraverser(top_level_acceleration,"NoAccel");
  rtGroupSetAcceleration( top_level_group, top_level_acceleration);
  rtAccelerationMarkDirty( top_level_acceleration );
}