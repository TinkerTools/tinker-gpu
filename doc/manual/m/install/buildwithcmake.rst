Build Tinker-GPU with CMake
===========================

Quick Start
-----------

For a GPU card with compute capability 7.0,
an example to compile the GPU code without OpenACC:

.. code-block:: bash

   cd tinker-gpu && mkdir build
   FC=gfortran compute_capability=70 gpu_lang=cuda cmake ..
   make
   make test

Assuming separate CUDA and NVHPC are properly installed,
another example to compile the GPU code with both OpenACC and CUDA:

.. code-block:: bash

   cd tinker-gpu && mkdir build
   cmake -DCMAKE_Fortran_COMPILER=gfortran -DCOMPUTE_CAPABILITY=70 ..
   make
   make test

For the options of other GPU devices and features,
please refer to the subsequent sections.

Configure CMake
---------------
You can skip this section if you are familar with CMake.

Suppose the current working directory is */home/tinker-gpu* and we
want to create a build directory called *build* in
*/home/tinker-gpu*. We can do *mkdir build* then *cd build*.
Because the top-level CMakeLists.txt file is in the parent directory,
if there was nothing else to configure, command *cmake ..* would generate
the Makefile. The alternative way is to specify the build and source
directories to CMake, e.g.,

.. code-block:: bash

   cmake -B /home/tinker9/build -S /home/tinker-gpu

Some CMake installations also provide a command line gui *ccmake* and a
simple gui program *cmake-gui* that can replace *cmake* in the commands
above.

Configure Compilers
-------------------
If we are lucky, we do not need to specify compilers in the *cmake*
configuration. However, specifying these compilers is preferred because
programs are not always installed the way we wanted.
Set *CXX=...*, *CUDACXX=...*, and *FC=...* to specify the non-default C++,
CUDA, and Fortran compilers, respectively. These environmental variables
are supported by *cmake*.
Do not use nvfortran.

This cmake script checks a custom environmental variable *ACC=...*
*only* for the OpenACC GPU code.
If not set, the building script will take a guess at the OpenACC compiler.
*ACC* will also be used as the C++ compiler. The value of *CXX* (if set)
will be neglected.

The only place where a C compiler may be used in Tinker-gpu is on the old Mac
computers that had Nvidia support. clang is hardwired in the cmake scripts
to compile the Objective-C and C source files. Thus, *CC=...* is not worth
setting in the cmake configuration.

Configure Tinker-GPU
--------------------
The following options are passed to CMake program with their default
values (if there is one). **-D** is prefixed to the options. CMake provides
two standard ways to let users customize the values:

- Change their values interactively in the *ccmake* command line gui;
- Pass the new value to CMake via command line arguments
  *cmake -DOPTION=NewValue*.

In addition to these two canonical methods, default value can also be set
by its corresponding environmental variable, documented as **(env)** here.
Note that there is no **-D** prefix for the environmental variables.

**-DCMAKE_BUILD_TYPE (opt) = Release**

Standard *CMAKE_BUILD_TYPE* option. Build type is case insensitive and
can be *Debug*, *Release*, *RelWithDebInfo* (release with debug info),
and *MinSizeRel* (minimum size release).

**-DCMAKE_INSTALL_PREFIX (install) = [NO DEFAULT VALUE]**

Install the executables under *${CMAKE_INSTALL_PREFIX}*. If this option is
not set, *make install* is configured not to install anything, which is
different from the default cmake behavior to install the program under */usr/local*.

**-DSTD (std) = AUTO**

C++ syntax standard. The source code is c++11-compliant, and should have no
problems compiled with c++14. If set to *14* here, users should make sure
the compilers are c++14-compliant.
In general, users should not worry about the C++ standard for Tinker9.
Using a more recent C++ standard to write the source code is unlikely
to speed up the performance of Tinker9 and may harm the availability of
Tinker9 to older machines.

**-DPREC (prec) = mixed**

Precision of the floating-point numbers. With flag *double*/*d*, all of the
floating-point numbers are treated as real\*8/double values,
or real\*4/single values if with flag *single*/*s*. Mixed precision flag *mixed*/*m* will
use real\*4 or real\*8 numbers in different places. Note that this flag will
not change the precision of the variables hard-coded as *float* or *double*
types.

**-DDETERMINISTIC_FORCE (deterministic_force) = AUTO**

Flag to use deterministic force.
This feature will be implicitly enabled by mixed and single precisions, but
can be explicitly disabled by setting the flag to *OFF* (or 0),
and can be explicitly enabled by value *ON* (or 1).

In general, evaluating energy, forces etc. twice, we don't expect to get
two identical answers, but we may not care as much because the difference
is usually negligible. (See
`Why is cos(x) != cos(y)? <https://isocpp.org/wiki/faq/newbie#floating-point-arith2>`_)
Whereas in MD, two simulations with the same initial configurations can
easily diverge due to the accumulated difference. If, for whatever reason,
you are willing to elongate the process of the inevitable divergence at the
cost of slightly slower simulation speed, a more "deterministic" force
(using fixed-point arithmetic) can help.

**-DHOST (host) = OFF**

Flag to compile to GPU (with value 0 or OFF) or CPU (with value 1 or ON)
version.

**-DGPU_LANG (gpu_lang) = OPENACC**

If set to *CUDA*, the GPU code will only use the cuda source files.
And the program will crash at runtime if it falls into an OpenACC code path.

**-DCOMPUTE_CAPABILITY (compute_capability) = AUTO**

GPU code only.

CUDA compute capability (multiplied by 10) of GPU.
Valid values (noninclusive) are 35, 50, 60, 70, 75, etc., and can be
comma-separated, e.g. 35,60.
Multiple compute capabilites will increase the size of executables.
If left unspecified, the script will attempt to detect the GPU,
although the detection may fail due to different reasons, which would
then require this option to be specified explicitly.

If new cards are released but the newer compute capabilities
are not supported, please inform us.

The full list of compute capabilities can be found on the
`NVIDIA website. <https://developer.nvidia.com/cuda-gpus>`_

**-DCUDA_DIR (cuda_dir) = /usr/local/cuda**

Nvidia GPU code only.

Top-level CUDA installation directory, under which directories *include*,
*lib* or *lib64* can be found.
This option will supersede the CUDA installation identified by the official
*CUDACXX* environmental variable.

Sometimes the PGI compiler and the NVCC compiler are not "compatible." For
instance, although PGI 19.4 supports CUDA 9.2, 10.0, 10.1, but the default
CUDA version configured in PGI 19.4 may be 9.2 and the external NVCC version
is 10.1. One solution is to pass *CUDA_HOME=${cuda_dir}* to the PGI
compiler, in which case, **cuda_dir** should be set to
*/usr/local/cuda-10.1*.

**-DFFTW_DIR (fftw_dir) = ${CMAKE_BINARY_DIR}/fftw**

CPU code only.

Top-level FFTW3 installation, under which
*include/fftw3.h* and *lib/libfftw3* static libraries are expected to be found.

Make Tinker-GPU
---------------
The following Makefile targets will be generated by CMake.
Run *make -j* for the default target(s) and *make TARGET(S) -j* for others.

**tinker9**

Compile and link the *tinker9* executable.

**all.tests**

Compile and link the *all.tests* executable.

**default**

Make two targets: *tinker9* and *all.tests* executables.

**all**

Same as the default target.

**test**

Run unit tests in a random order. Exit on the first error.

**man**

Generate user manual.

**doc**

Generate developer guides.

