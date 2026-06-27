Parallelization
===============

.. include:: ../replace.rst

**CUDA-DEVICE [integer]**

.. index:: CUDA-DEVICE
.. index:: CUDA_DEVICE

Followed by an integer value starting from 0, sets the CUDA-enabled
GPU device for the program. Value will be overwritten by environment variable
*CUDA_DEVICE*.
For instance, a node has four CUDA devices, and the *CUDA_VISIBLE_DEVICES*
environment variable (part of CUDA library) has been set to
*CUDA_VISIBLE_DEVICES=1,3*. This means only two CUDA devices are avaiable
here, thus the valid values for *CUDA-DEVICE* are 0 and 1.

**GPU-PACKAGE [CUDA / OPENACC]** |not8|

.. index:: GPU-PACKAGE
.. index:: GPU_PACKAGE

Selects code paths for some GPU algorithms where both CUDA and
OpenACC versions have been implemented.
The default value is CUDA. Value will be overwritten by environment variable
*GPU_PACKAGE*.
