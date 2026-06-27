#pragma once
#include "ff/precision.h"
#include "seq/seq.h"

#include <cmath>
#include <cstdlib>

// erfc
extern "C"
{
   // The helper function for the complementary error function.
   // \param ex2  exp(-x*x)
   SEQ_ROUTINE
   inline float tinkerErfcfHastings2Args(float x, float ex2)
   {
      float t = 1.0f / (1.0f + 0.3275911f * x);
      return (0.254829592f + (-0.284496736f + (1.421413741f + (-1.453152027f + 1.061405429f * t) * t) * t) * t) * t * ex2;
   }

   // The complementary error function.
   // C. Hastings. Approximations for Digital Computers.
   // Princeton Univ. Press. (1955) p. 167.
   SEQ_ROUTINE
   inline float tinkerErfcfHastings(float x)
   {
      float exp2a = expf(-x * x);
      return tinkerErfcfHastings2Args(x, exp2a);
   }
}

// abs functions
#if TINKER_CUDART
#define INT_ABS abs

#if TINKER_REAL_SIZE == 8
#define REAL_ABS fabs
#endif

#if TINKER_REAL_SIZE == 4
#define REAL_ABS fabsf
#endif
#else
#define INT_ABS std::abs

#if TINKER_REAL_SIZE == 8
#define REAL_ABS std::fabs
#endif

#if TINKER_REAL_SIZE == 4
// Bug. Cannot use std::fabsf. See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=79700
#define REAL_ABS ::fabsf
#endif
#endif

// double
#if TINKER_REAL_SIZE == 8
#define REAL_SQRT     sqrt
#define REAL_EXP      exp
#define REAL_FLOOR    floor
#define REAL_POW      pow
#define REAL_RECIP(x) (1 / static_cast<double>(x))
#define REAL_RSQRT(x) (1 / sqrt(x))
#define REAL_COS      cos
#define REAL_SIN      sin
#define REAL_ACOS     acos
#define REAL_ASIN     asin
#define REAL_SINH     sinh
#define REAL_COSH     cosh
#define REAL_ERF      erf
#define REAL_ERFC(x)  (1 - erf(x))
#define REAL_MIN      fmin
#define REAL_MAX      fmax
#define REAL_SIGN     copysign
// #   define REAL_ERFC     erfc
#define REAL_ERFC_V2(x, expterm) (1 - erf(x))
#define REAL_LOG                 log
#endif

// float
#if TINKER_REAL_SIZE == 4
#define REAL_SQRT     sqrtf
#define REAL_EXP      expf
#define REAL_FLOOR    floorf
#define REAL_POW      powf
#define REAL_RECIP(x) (1 / static_cast<float>(x))
#define REAL_RSQRT(x) (1 / sqrtf(x))
#define REAL_COS      cosf
#define REAL_SIN      sinf
#define REAL_ACOS     acosf
#define REAL_ASIN     asinf
#define REAL_SINH     sinhf
#define REAL_COSH     coshf
#define REAL_ERF      erff
#define REAL_ERFC     tinkerErfcfHastings
#define REAL_MIN      fminf
#define REAL_MAX      fmaxf
#define REAL_SIGN     copysignf
// #   define REAL_ERFC     erfcf
// #   define REAL_ERFC(x)  (1 - erff(x))
#define REAL_ERFC_V2(x, expterm) tinkerErfcfHastings2Args((x), (expterm))
#define REAL_LOG                 logf
#endif

#ifdef _OPENACC
#include <accelmath.h>

#pragma acc routine(abs) seq

#pragma acc routine(sqrt) seq
#pragma acc routine(exp) seq
#pragma acc routine(floor) seq
#pragma acc routine(fabs) seq
#pragma acc routine(pow) seq
#pragma acc routine(cos) seq
#pragma acc routine(sin) seq
#pragma acc routine(acos) seq
#pragma acc routine(asin) seq
#pragma acc routine(erf) seq
#pragma acc routine(erfc) seq
#pragma acc routine(fmin) seq
#pragma acc routine(fmax) seq
#pragma acc routine(copysign) seq

#pragma acc routine(sqrtf) seq
#pragma acc routine(expf) seq
#pragma acc routine(floorf) seq
#pragma acc routine(fabsf) seq
#pragma acc routine(powf) seq
#pragma acc routine(cosf) seq
#pragma acc routine(sinf) seq
#pragma acc routine(acosf) seq
#pragma acc routine(asinf) seq
#pragma acc routine(erff) seq
#pragma acc routine(erfcf) seq
#pragma acc routine(fminf) seq
#pragma acc routine(fmaxf) seq
#pragma acc routine(copysignf) seq
#endif
