Prerequisites
=============

**Hardware**

A relatively recent NVIDIA GPU is mandatory for the GPU code.
The oldest NVIDIA GPU that Tinker-GPU has been tested on is GeForce GTX 675MX (compute capability 3.0).

**Operating Systems and Compilers**

=================  ===========================
OS and Toolchain   Version
=================  ===========================
OS                 Linux, WSL2, macOS <= 10.13
CMake              >= 3.18
Fortran            GNU or Intel
C++                [a]
CUDA/nvcc          [b]
OpenACC/NVHPC/PGI  [c]
=================  ===========================

- [a] Recent C++ compiler that supports C++11 syntax.
- [b] GPU code only. Version >= 9.0.
- [c] Optional for the GPU code. A recent `NVIDIA HPC SDK <https://www.developer.nvidia.com/hpc-sdk>`_ is preferred.
- [d] We have successfully built Tinker-GPU on Windows WSL2 Ubuntu with CUDA 11.0 and NVHPC 20.9. Please `check this link <https://docs.nvidia.com/cuda/wsl-user-guide/index.html>`_ for more details.

**Using NVIDIA HPC SDK on Clusters**

Prior to rebranding, the current NVIDIA HPC SDK was known as the PGI compiler
suite. During Jan. 2020 we worked on a cluster that was still running
Red Hat with gcc 4.8.5 by default without root privilege. Although several
more recent gcc and PGI versions were available via the *module* program,
the most recent PGI compiler (2019) was still configured with gcc 4.8.5
by default, which had a very bad support for C++11.
Without root privilege on the cluster, we had to use
a custom *localrc* file by running the following command to
reconfigure PGI compiler with gcc 7.4.0:

.. code-block:: bash

   makelocalrc $PGI/linux86-64-llvm/2019 \
   -gcc /usr/local/gcc-7.4.0/bin/gcc \
   -gpp /usr/local/gcc-7.4.0/bin/g++ \
   -g77 /usr/local/gcc-7.4.0/bin/gfortran \
   -d /dir_for_new_config -x

then added *export PGI_LOCALRC=/dir_for_new_config/localrc* to my bash resource file.

(Updated in Apr. 2021)

Compilers with the new brand name NVHPC now have a different
directory structure. The new command will look like

.. code-block:: bash

   makelocalrc $NVHPC/Linux_x86_64/2020/compilers \
   -gcc /usr/local/gcc-7.4.0/bin/gcc \
   -gpp /usr/local/gcc-7.4.0/bin/g++ \
   -g77 /usr/local/gcc-7.4.0/bin/gfortran \
   -d /dir_for_new_config -x

then add *export NVLOCALRC=/dir_for_new_config/localrc* to the shell resource file.

**FFTW Libraries**

Canonical Tinker requires FFTW libraries because by default it is compiled with OpenMP.
Otherwise, Tinker will use *FFTPACK*.
In Tinker-GPU, the underlying *libtinker.a* will be compiled without OpenMP,
therefore FFTW libraries are no longer mandatory for GPU code.

However, FFTW libraries are required by CPU code.
Two prebuilt FFTW libraries, *libfftw3* and *libfftw3_threads* are needed by
the double precision CPU code.
The other two FFTW libraries, *libfftw3f* and *libfftw3f_threads* are needed by
the mixed and single precision CPU code.

**Other Nonmandatory Utilities**

- `ClangFormat <https://clang.llvm.org/docs/ClangFormat.html>`_:
  to format the source code.

- `Doxygen <https://www.doxygen.nl>`_: to generate developer guides.

- `Sphinx <https://www.sphinx-doc.org>`_: to generate user manual.

   - PDF version also depends on `TeX <https://www.tug.org/begin.html>`_.

.. code-block:: bash

   python3 -m venv env-tinker9doc
   source env-tinker9doc/bin/activate
   pip3 install -r path_to/doc/manual/requirements.txt
