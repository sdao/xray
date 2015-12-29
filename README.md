Xray
===========

Little global illumination Monte Carlo path tracer in Nvidia OptiX

[OptiX](https://developer.nvidia.com/optix) is a CUDA-based ray tracing system
that runs ray-tracing programs on Nvidia GPUs.

See [this repo](https://github.com/sdao/path-tracer) for an Intel Embree-based
version of this project that is ready for both Windows and Linux.

The build environment is set to compile using Visual Studio 2012 on Windows.
It will probably compile on Linux but may require some tweaking.
This repository is cloned from my internal Perforce depot.

Dependencies
------------
These are the dependencies:
* [Open Asset Import Library](http://assimp.sourceforge.net/)
  (automatically retrieved from NuGet)
* [Boost](http://www.boost.org/)
  (automatically retrieved from NuGet)
* [TinyExr](https://github.com/syoyo/tinyexr)
  (included as source in the `tinyexr` folder; see `tinyexr/REVISION` for more
    details)
* [Nvidia Optix and CUDA](https://developer.nvidia.com/optix)
  (tested with OptiX 3.9 and CUDA 7.5; requires Nvidia GPU; must be installed
    separately from Nvidia's website)

Windows Build (Visual Studio)
-----------------------------
This was tested using Visual Studio 2012 and isn't guaranteed to work on
other versions (although newer versions will probably work).

_Boost_ and the _Open Asset Import Library_ can be obtained from
NuGet; the project should automatically update the packages on first load.
_TinyExr_ is included in the project as source in the `tinyexr` folder.

You will need to install CUDA and OptiX from Nvidia by registering for their
[developer program](https://developer.nvidia.com/optix). The Visual Studio
project references OptiX 3.9 and CUDA 7.5 in the default install directories.
If you use a different version or install in a different directory, you will
need to edit the VS project file.

Running
-------
Run the `xray` executable for command-line options. Make sure to run from the
working directory `xray` (the directory containing the `cuda` or `ptx` folders).
