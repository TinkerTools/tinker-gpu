#pragma once

/// \def SEQ_ROUTINE
/// \ingroup cuda_syntax
/// Expands to \c __device__ in CUDA source files.
#define SEQ_ROUTINE __device__

/// \def SEQ_CUDA
/// \ingroup cuda_syntax
/// Expands to \c __device__ in CUDA source files.
/// Used in CUDA kernel templates.
#define SEQ_CUDA __device__

/// \def TINKER_GRID_CONSTANT
/// \ingroup cuda_syntax
/// Expands to \c __grid_constant__ when every target is sm_70 or newer and
/// nvcc is 11.7 or newer (TINKER_CUDA_GRID_CONSTANT, set by CMake). A kernel
/// parameter struct indexed at run time is then read from the parameter space
/// instead of a copy in local memory per thread; measured faster for echgtrn
/// and emplarast (3-5%), but 1% slower for empoleChgpenDlmda.
#ifdef TINKER_CUDA_GRID_CONSTANT
#define TINKER_GRID_CONSTANT __grid_constant__
#else
#define TINKER_GRID_CONSTANT
#endif
