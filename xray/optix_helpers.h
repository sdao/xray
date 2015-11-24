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
#include <windows.h>
#include <iostream>
#include <string.h>
#include <sutil.h>  // OptiX utility library
#include <optix.h>  // Main OptiX header
#include <optixu/optixu_vector_types.h> // OptiX common vector types

using namespace std;

/* assumes that there is no context, just print to stderr */
#define SWRAPPER_CHECK_ERROR_NO_CONTEXT( func )                    \
  do {                                                             \
    RTresult code = func;                                          \
    if( code != RT_SUCCESS )                                       \
      f_sutilHandleErrorNoContext(code, __FILE__, __LINE__ );      \
  } while(0)

#define SWRAPPER_RT_CHECK_ERROR( func )                            \
  do {                                                             \
    RTresult code = func;                                          \
    if( code != RT_SUCCESS )                                       \
      f_sutilHandleError( context, code, __FILE__, __LINE__ );     \
  } while(0)

class sUtilWrapper
{
public:
  sUtilWrapper()
  {
    // Load OptiX and OptiX sUtil library
    string optixLib(m_supportDir);
    optixLib.append("\\optix.1.dll");
    h_optix_library = LoadLibrary(optixLib.c_str());
    string optixuLib(m_supportDir);
    optixuLib.append("\\optixu.1.dll");
    h_optixu_library = LoadLibrary(optixuLib.c_str());
    string freeglutLib(m_supportDir);
    freeglutLib.append("\\freeglut.dll");
    h_freeglut_library = LoadLibrary(freeglutLib.c_str());
    string sutilLib(m_supportDir);
    sutilLib.append("\\sutil.dll");
    h_sutil_library = LoadLibrary(sutilLib.c_str());
    if (h_optix_library == NULL || h_optixu_library == NULL || h_freeglut_library == NULL || h_sutil_library == NULL)
    {
      throw "sUtilWrapper error - could not load sutil.dll";
      return;
    }

    // Get function addresses
    f_sutilInitGlut = (RTresult (*)(int*, char**)) GetProcAddress(h_sutil_library, "sutilInitGlut");
    f_sutilDisplayBufferInGlutWindow = (RTresult (*)(const char*, RTbuffer)) GetProcAddress(h_sutil_library, "sutilDisplayBufferInGlutWindow");
  }
  ~sUtilWrapper()
  {
    FreeLibrary(h_sutil_library);
  }

  void initGlut(int* argc, char** argv)
  {
    SWRAPPER_CHECK_ERROR_NO_CONTEXT(f_sutilInitGlut(argc, argv));
  }

  void DisplayBufferInGlutWindow(RTcontext context, const char* window_title, RTbuffer buffer)
  {
    SWRAPPER_RT_CHECK_ERROR(f_sutilDisplayBufferInGlutWindow(window_title, buffer));
  }
private:
  static string m_supportDir; // OptiX supporting binaries path
  HINSTANCE h_optix_library, h_optixu_library, h_freeglut_library, h_sutil_library;  // Handles to OptiX, freeglut and sUtil library binaries that ship with OptiX SDK

  // Function addresses
  RTresult (*f_sutilInitGlut)(int* argc, char** argv);
  void (*f_sutilHandleErrorNoContext)(RTresult code, const char* file, int line);
  void (*f_sutilHandleError)(RTcontext context, RTresult code, const char* file, int line);
  RTresult (*f_sutilDisplayBufferInGlutWindow)(const char* window_title, RTbuffer buffer);
};

// Location of OptiX binaries and supporting dynamic libraries
string sUtilWrapper::m_supportDir = "C:\\ProgramData\\NVIDIA Corporation\\OptiX SDK 3.8.0\\SDK-precompiled-samples";

typedef struct struct_BasicLight
{
  typedef optix::float3 float3;

  float3 pos; // 3*4 bytes = 12 bytes
  float3 color; // 3*4 bytes = 12 bytes
  int    casts_shadow; // 4 bytes
  int    padding;      // This isn't an used field, but it's here to make this structure 32 bytes -- powers of two objects can have better memory alignment
} BasicLight;