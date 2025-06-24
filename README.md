Tinker-GPU: Next Generation of Tinker with GPU Support
===================================================
[//]: # (Badges)
[![Docs Status](https://readthedocs.org/projects/tinker9-manual/badge/?version=latest&style=flat)](https://tinker9-manual.readthedocs.io)
[![Doxygen Page Status](https://github.com/tinkertools/tinker-gpu/actions/workflows/doxygen_gh_pages.yaml/badge.svg)](https://tinkertools.github.io/tinker-gpu/)


<h2>Introduction</h2>

Tinker-GPU is a complete rewrite and extension of the canonical Tinker software. Tinker-GPU is implemented as C++ code with OpenACC directives and CUDA kernels providing excellent performance on GPUs. At present, Tinker-GPU builds against the object library from Tinker, and provides GPU versions of the Tinker ANALYZE, BAR, DYNAMIC, MINIMIZE and TESTGRAD programs. Existing Tinker file formats and force field parameter files are fully compatible with Tinker-GPU, and nearly all Tinker keywords function identically in Tinker-GPU. Over time we plan to port much or all of the remaining portions of Fortran Tinker to the C++ Tinker-GPU code base.


<h2><a href="https://github.com/TinkerTools/tinker-gpu/releases">
Release Notes
</a></h2>


<h2><a href="https://hub.docker.com/r/tinkertools/tinker9">
Docker Images
</a></h2>

The executables included in these images were compiled on a recent computer. It is known that they will not run on the machines with very old CPUs. If this is a problem for you, please write a new issue and provide us with more details.


<h2>Installation Steps</h2>

Use `Git` to retrieve the source code. If you want to specify a release version,
use command `git clone --depth 1 --branch $VERSION https://github.com/TinkerTools/tinker9`.
Do not download the zip or tarball file from the release page.

   1. [Prerequisites](doc/manual/m/install/preq.rst)
   2. [Download the Canonical Tinker (important to get the REQUIRED version)](doc/manual/m/install/tinker.rst)
   3. [Build Tinker9 with CMake](doc/manual/m/install/buildwithcmake.rst)

Examples of other successful builds are shown [here](https://github.com/TinkerTools/tinker9/discussions/121).

<h2>User Manual (in progress)</h2>

The HTML version is hosted on [readthedocs](https://tinker9-manual.readthedocs.io)
and the [PDF](https://tinker9-manual.readthedocs.io/_/downloads/en/latest/pdf/)
version is accessible from the same webpage.

We are trying to merge this documentation into [tinkerdoc.](https://tinkerdoc.readthedocs.io)

[C++ and CUDA Code Documentation (Doxygen)](https://tinkertools.github.io/tinker9/)

<h2>Issues and Discussions</h2>

Please use [GitHub Issues](https://github.com/TinkerTools/tinker-gpu/issues) for bug tracking and
[GitHub Discussions](https://github.com/TinkerTools/tinker-gpu/discussions) for general discussions.
