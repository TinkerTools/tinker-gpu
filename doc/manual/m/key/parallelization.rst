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

**LIST-PRUNE [ON / OFF]** |not8|

.. index:: LIST-PRUNE

Controls pruning of the CUDA neighbor lists. The lists are built with the
*LIST-BUFFER* (or *USOLVE-BUFFER*) added to the cutoff so they can be reused
for several steps; when pruning is on, every coordinate update also drops the
neighbors that are now beyond the cutoff, so the pairwise kernels skip them.
Only the multipole list and the preconditioner list are pruned, since the
polarization solver reads them many times per step.
Results are unchanged up to round-off. The default value is ON; a value of
OFF, NONE, NO or FALSE turns pruning off.
