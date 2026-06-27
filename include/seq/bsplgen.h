#pragma once
#include "math/libfunc.h"

namespace tinker {
#ifdef __CUDACC__
/**
 * \ingroup pme
 * \brief B-spline coefficients and derivatives for a single %PME atomic site
 * along a particular direction. See also subroutine `bsplgen` in `pmestuf.f`
 * file.
 *
 * \tparam bsorder Desired B-spline order; `MAX_BSORDER` is hard-coded to 5.
 * \tparam LEVEL   Flag to control the results in `thetai`, and must be:
 *                    - greater than 0.
 *                    - less than `bsorder`.
 * \param w        Fractional distance to the reference %PME grid point along
 *                 a particular direction.
 * \param thetai   Output array of size `thetai[bsorder][MAX_BSORDER-1]` in C
 *                 or `thetai(MAX_BSORDER-1,bsorder)` in Fortran. Based on the
 *                 value of `LEVEL`, the output is as follows,
 *                    - `LEVEL=1`, B-spline coefficients are in `thetai(1,:)`;
 *                    - `LEVEL=2`, first derivatives are in `thetai(2,:)`;
 *                    - `LEVEL=3`, second derivatives are in `thetai(3,:)`;
 *                    - `LEVEL=4`, third derivatives are in `thetai(4,:)`.
 * \param bsbuild_ A CUDA working array of size `MAX_BSORDER*MAX_BSORDER`.
 */
template <int LEVEL, int bsorder>
__device__
void bsplgen(real w, real* restrict thetai, volatile real* restrict bsbuild_)
#else
#pragma acc routine seq
template <int LEVEL>
void bsplgen(real w, real* restrict thetai, int bsorder)
#endif
{
#ifndef __CUDACC__
   real bsbuild_[5 * 5];
#endif

// Fortran 2D array syntax
#define bsbuild(j, i) bsbuild_[((i)-1) * bsorder + (j)-1]

   // initialization to get to 2nd order recursion
   bsbuild(2, 2) = w;
   bsbuild(2, 1) = 1 - w;

   // perform one pass to get to 3rd order recursion
   bsbuild(3, 3) = 0.5f * w * bsbuild(2, 2);
   bsbuild(3, 2) = 0.5f * ((1 + w) * bsbuild(2, 1) + (2 - w) * bsbuild(2, 2));
   bsbuild(3, 1) = 0.5f * (1 - w) * bsbuild(2, 1);

   // compute standard B-spline recursion to desired order
   for (int i = 4; i <= bsorder; ++i) {
      int k = i - 1;
      real denom = REAL_RECIP(k);
      bsbuild(i, i) = denom * w * bsbuild(k, k);
      for (int j = 1; j <= i - 2; j++) {
         bsbuild(i, i - j) = denom * ((w + j) * bsbuild(k, i - j - 1) + (i - j - w) * bsbuild(k, i - j));
      }
      bsbuild(i, 1) = denom * (1 - w) * bsbuild(k, 1);
   }

   // get coefficients for the B-spline first derivative
   if CONSTEXPR (LEVEL >= 2) {
      int k = bsorder - 1;
      bsbuild(k, bsorder) = bsbuild(k, bsorder - 1);
      for (int i = bsorder - 1; i >= 2; --i) {
         bsbuild(k, i) = bsbuild(k, i - 1) - bsbuild(k, i);
      }
      bsbuild(k, 1) = -bsbuild(k, 1);
   }

   // get coefficients for the B-spline second derivative
   if CONSTEXPR (LEVEL >= 3) {
      int k = bsorder - 2;
      bsbuild(k, bsorder - 1) = bsbuild(k, bsorder - 2);
      for (int i = bsorder - 2; i >= 2; --i) {
         bsbuild(k, i) = bsbuild(k, i - 1) - bsbuild(k, i);
      }
      bsbuild(k, 1) = -bsbuild(k, 1);
      bsbuild(k, bsorder) = bsbuild(k, bsorder - 1);
      for (int i = bsorder - 1; i >= 2; --i) {
         bsbuild(k, i) = bsbuild(k, i - 1) - bsbuild(k, i);
      }
      bsbuild(k, 1) = -bsbuild(k, 1);
   }

   // get coefficients for the B-spline third derivative
   if CONSTEXPR (LEVEL == 4) {
      int k = bsorder - 3;
      bsbuild(k, bsorder - 2) = bsbuild(k, bsorder - 3);
      for (int i = bsorder - 3; i >= 2; --i) {
         bsbuild(k, i) = bsbuild(k, i - 1) - bsbuild(k, i);
      }
      bsbuild(k, 1) = -bsbuild(k, 1);
      bsbuild(k, bsorder - 1) = bsbuild(k, bsorder - 2);
      for (int i = bsorder - 2; i >= 2; --i) {
         bsbuild(k, i) = bsbuild(k, i - 1) - bsbuild(k, i);
      }
      bsbuild(k, 1) = -bsbuild(k, 1);
      bsbuild(k, bsorder) = bsbuild(k, bsorder - 1);
      for (int i = bsorder - 1; i >= 2; --i)
         bsbuild(k, i) = bsbuild(k, i - 1) - bsbuild(k, i);
      bsbuild(k, 1) = -bsbuild(k, 1);
   }

   // copy coefficients from temporary to permanent storage
   for (int i = 1; i <= bsorder; ++i) {
      // keep this line to work on Tesla
      #pragma acc loop seq
      for (int j = 1; j <= LEVEL; ++j) {
         thetai[4 * (i - 1) + (j - 1)] = bsbuild(bsorder - j + 1, i);
      }
   }
#undef bsbuild
}

#ifdef __CUDACC__
template <int LEVEL, int bsorder>
__device__
void bsplgen2(real w, real* restrict thetai, int k, int padded_n, volatile real* restrict bsbuild_)
{
// Fortran 2D array syntax
#define bsbuild(j, i) bsbuild_[((i)-1) * bsorder + (j)-1]

   // initialization to get to 2nd order recursion
   bsbuild(2, 2) = w;
   bsbuild(2, 1) = 1 - w;

   // perform one pass to get to 3rd order recursion
   bsbuild(3, 3) = 0.5f * w * bsbuild(2, 2);
   bsbuild(3, 2) = 0.5f * ((1 + w) * bsbuild(2, 1) + (2 - w) * bsbuild(2, 2));
   bsbuild(3, 1) = 0.5f * (1 - w) * bsbuild(2, 1);

   // compute standard B-spline recursion to desired order
   for (int i = 4; i <= bsorder; ++i) {
      int k = i - 1;
      real denom = REAL_RECIP(k);
      bsbuild(i, i) = denom * w * bsbuild(k, k);
      for (int j = 1; j <= i - 2; j++) {
         bsbuild(i, i - j) = denom * ((w + j) * bsbuild(k, i - j - 1) + (i - j - w) * bsbuild(k, i - j));
      }
      bsbuild(i, 1) = denom * (1 - w) * bsbuild(k, 1);
   }

   // get coefficients for the B-spline first derivative
   if CONSTEXPR (LEVEL >= 2) {
      int k = bsorder - 1;
      bsbuild(k, bsorder) = bsbuild(k, bsorder - 1);
      for (int i = bsorder - 1; i >= 2; --i) {
         bsbuild(k, i) = bsbuild(k, i - 1) - bsbuild(k, i);
      }
      bsbuild(k, 1) = -bsbuild(k, 1);
   }

   // get coefficients for the B-spline second derivative
   if CONSTEXPR (LEVEL >= 3) {
      int k = bsorder - 2;
      bsbuild(k, bsorder - 1) = bsbuild(k, bsorder - 2);
      for (int i = bsorder - 2; i >= 2; --i) {
         bsbuild(k, i) = bsbuild(k, i - 1) - bsbuild(k, i);
      }
      bsbuild(k, 1) = -bsbuild(k, 1);
      bsbuild(k, bsorder) = bsbuild(k, bsorder - 1);
      for (int i = bsorder - 1; i >= 2; --i) {
         bsbuild(k, i) = bsbuild(k, i - 1) - bsbuild(k, i);
      }
      bsbuild(k, 1) = -bsbuild(k, 1);
   }

   // get coefficients for the B-spline third derivative
   if CONSTEXPR (LEVEL == 4) {
      int k = bsorder - 3;
      bsbuild(k, bsorder - 2) = bsbuild(k, bsorder - 3);
      for (int i = bsorder - 3; i >= 2; --i) {
         bsbuild(k, i) = bsbuild(k, i - 1) - bsbuild(k, i);
      }
      bsbuild(k, 1) = -bsbuild(k, 1);
      bsbuild(k, bsorder - 1) = bsbuild(k, bsorder - 2);
      for (int i = bsorder - 2; i >= 2; --i) {
         bsbuild(k, i) = bsbuild(k, i - 1) - bsbuild(k, i);
      }
      bsbuild(k, 1) = -bsbuild(k, 1);
      bsbuild(k, bsorder) = bsbuild(k, bsorder - 1);
      for (int i = bsorder - 1; i >= 2; --i)
         bsbuild(k, i) = bsbuild(k, i - 1) - bsbuild(k, i);
      bsbuild(k, 1) = -bsbuild(k, 1);
   }

   // copy coefficients from temporary to permanent storage
   for (int i = 1; i <= bsorder; ++i) {
      for (int j = 1; j <= LEVEL; ++j) {
         int offset = (4 * (i - 1) + (j - 1)) * padded_n + k;
         thetai[offset] = bsbuild(bsorder - j + 1, i);
      }
   }
#undef bsbuild
}
#endif
}
