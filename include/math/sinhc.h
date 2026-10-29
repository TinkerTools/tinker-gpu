#pragma once
#include "math/libfunc.h"
#include "seq/seq.h"
#include "tool/macro.h"

// sinhc on host
namespace tinker {
/// \ingroup math
/// \brief \f$ \sinh(x)/x \f$
template <class T>
T sinhc(T x)
{
   T y = std::fabs(x);
   if (y <= (T)1.0e-8)
      return (T)1;
   else
      return std::sinh(y) / y;
}
}

// sinhc on device
namespace tinker {
inline namespace v1 {
#pragma acc routine seq
template <int N>
SEQ_CUDA
inline void fsinhcCheck()
{
   static_assert(1 <= N && N <= 7, "1 <= N <= 7 is violated.");
}

#pragma acc routine seq
template <int N, class T>
SEQ_CUDA
inline T fsinhcPade44(T x2)
{
   fsinhcCheck<N>();
   constexpr int M = N - 1;
   // Pade coefficients
   // clang-format off
   constexpr T c[][2][3] = {
   {{1./1     , 53./396       , 551./166320      },{1.,-13./396 , 5./11088 }},
   {{1./3     ,211./8580      ,2647./6486480     },{1.,-15./572 ,17./61776 }},
   {{1./15    ,271./81900     ,   7./171600      },{1.,-17./780 ,19./102960}},
   {{1./105   ,113./321300    ,1889./551350800   },{1.,-19./1020, 7./53040 }},
   {{1./945   , 83./2686068   ,7789./31426995600 },{1.,-21./1292,23./232560}},
   {{1./10395 ,499./215675460 ,3461./219988969200},{1.,-23./1596,25./325584}},
   {{1./135135,197./1305404100, 409./459976935600},{1.,-25./1932, 3./48944 }}};
   // clang-format on
   return (c[M][0][0] + x2 * (c[M][0][1] + x2 * c[M][0][2])) / (c[M][1][0] + x2 * (c[M][1][1] + x2 * c[M][1][2]));
}

#pragma acc routine seq
template <int N, class T>
SEQ_CUDA
inline T fsinhcTaylor(T x2)
{
   fsinhcCheck<N>();
   constexpr int M = N - 1;
   // clang-format off
   constexpr T c[][5] = {
      {1. / 1     , 1. / 6 , 1. / 20, 1. / 42 , 1. / 72 },
      {1. / 3     , 1. / 10, 1. / 28, 1. / 54 , 1. / 88 },
      {1. / 15    , 1. / 14, 1. / 36, 1. / 66 , 1. / 104},
      {1. / 105   , 1. / 18, 1. / 44, 1. / 78 , 1. / 120},
      {1. / 945   , 1. / 22, 1. / 52, 1. / 90 , 1. / 136},
      {1. / 10395 , 1. / 26, 1. / 60, 1. / 102, 1. / 152},
      {1. / 135135, 1. / 30, 1. / 68, 1. / 114, 1. / 168}};
   return c[M][0]*(1+x2*c[M][1]*(1+x2*c[M][2]*(1+x2*c[M][3]*(1+x2*c[M][4]))));
   // clang-format on
}

#pragma acc routine seq
template <int N, class T>
SEQ_CUDA
inline T fsinhcSeries(T x2)
{
   return fsinhcTaylor<N>(x2);
   // return fsinhcPade44<N>(x2);
}

#pragma acc routine seq
template <class T>
SEQ_CUDA
inline T fsinhcAnalyt7(T d, T d13, T y /* exp(-d) */, T z /* exp(+d) */)
{
   T cy, cz;
   cy = d * (d * (d * (d * (d * (d + 21) + 210) + 1260) + 4725) + 10395) + 10395;
   cy = -cy;
   cz = d * (d * (d * (d * (d * (d - 21) + 210) - 1260) + 4725) - 10395) + 10395;
   return (cy * y + cz * z) / (2 * d13);
}

#pragma acc routine seq
template <class T>
SEQ_CUDA
inline T fsinhcAnalyt6(T d, T d11, T y /* exp(-d) */, T z /* exp(+d) */)
{
   T cy, cz;
   cy = d * (d * (d * (d * (d + 15) + 105) + 420) + 945) + 945;
   cz = d * (d * (d * (d * (d - 15) + 105) - 420) + 945) - 945;
   return (cy * y + cz * z) / (2 * d11);
}

#pragma acc routine seq
template <class T>
SEQ_CUDA
inline T fsinhcAnalyt5(T d, T d9, T y /* exp(-d) */, T z /* exp(+d) */)
{
   T cy, cz;
   cy = d * (d * (d * (d + 10) + 45) + 105) + 105;
   cy = -cy;
   cz = d * (d * (d * (d - 10) + 45) - 105) + 105;
   return (cy * y + cz * z) / (2 * d9);
}

#pragma acc routine seq
template <class T>
SEQ_CUDA
inline T fsinhcAnalyt4(T d, T d7, T y /* exp(-d) */, T z /* exp(+d) */)
{
   T cy, cz;
   cy = d * (d * (d + 6) + 15) + 15;
   cz = d * (d * (d - 6) + 15) - 15;
   return (cy * y + cz * z) / (2 * d7);
}

#pragma acc routine seq
template <class T>
SEQ_CUDA
inline T fsinhcAnalyt3(T d, T d5, T y /* exp(-d) */, T z /* exp(+d) */)
{
   T cy, cz;
   cy = (d + 3) * d + 3;
   cy = -cy;
   cz = (d - 3) * d + 3;
   return (cy * y + cz * z) / (2 * d5);
}

#pragma acc routine seq
template <class T>
SEQ_CUDA
inline T fsinhcAnalyt2(T d, T d3, T y /* exp(-d) */, T z /* exp(+d) */)
{
   T cy, cz;
   cy = d + 1;
   cz = d - 1;
   return (cy * y + cz * z) / (2 * d3);
}

#pragma acc routine seq
template <class T>
SEQ_CUDA
inline T fsinhcAnalyt1(T d, T y /* exp(-d) */, T z /* exp(+d) */)
{
   T cy, cz;
   cy = -1;
   cz = 1;
   return (cy * y + cz * z) / (2 * d);
}

#pragma acc routine seq
template <int N, class T>
SEQ_CUDA
inline void fsinhcImpl(T d, T& restrict f1d, T& restrict f2d, T& restrict f3d, T& restrict f4d, T& restrict f5d, T& restrict f6d, T& restrict f7d)
{
   fsinhcCheck<N>();

   // Each fk takes the Taylor series up to its own |d| threshold and the
   // analytic form above it, so fk does not depend on how many orders are
   // requested; the energy-only and gradient versions of a kernel ask for
   // different orders and must agree. A threshold minimizes the largest
   // relative error of fk over |d| <= 8 against the exact series
   // fk = sum_n d^(2n) / ((2n)!! (2k+2n-1)!!); the analytic forms cancel
   // badly at small |d|, more so for higher k.
   //
   //        float           double
   // f1   (0.89, 1.6e-7)   (0.15, 5e-16)
   // f2   (0.95, 2.1e-7)   (0.26, 2e-14)
   // f3   (1.64, 4.2e-7)   (0.41, 3e-13)
   // f4   (2.23, 1.9e-6)   (0.61, 6e-12)
   // f5   (2.79, 9.4e-6)   (0.84, 6e-11)
   // f6   (3.56, 4.2e-5)   (1.09, 6e-10)
   // f7   (4.22, 1.2e-4)   (1.42, 4e-9)
   double epsd[] = {0.15, 0.26, 0.41, 0.61, 0.84, 1.09, 1.42};
   float epsfl[] = {0.89, 0.95, 1.64, 2.23, 2.79, 3.56, 4.22};
   T absd, eps[7], expmd, exppd;
   if CONSTEXPR (sizeof(T) == sizeof(float)) {
      absd = fabsf(d);
      for (int i = 0; i < 7; ++i)
         eps[i] = epsfl[i];
      expmd = expf(-d), exppd = expf(+d);
   } else {
      absd = fabs(d);
      for (int i = 0; i < 7; ++i)
         eps[i] = epsd[i];
      expmd = exp(-d), exppd = exp(+d);
   }

   T d2, d4;
   T d3, d5, d7, d9, d11, d13;
   if CONSTEXPR (N >= 1) {
      d2 = d * d;
      if (absd > eps[0]) {
         f1d = fsinhcAnalyt1(d, expmd, exppd);
      } else {
         f1d = fsinhcSeries<1>(d2);
      }
   }
   if CONSTEXPR (N >= 2) {
      d3 = d * d2;
      if (absd > eps[1]) {
         f2d = fsinhcAnalyt2(d, d3, expmd, exppd);
      } else {
         f2d = fsinhcSeries<2>(d2);
      }
   }
   if CONSTEXPR (N >= 3) {
      d5 = d2 * d3;
      if (absd > eps[2]) {
         f3d = fsinhcAnalyt3(d, d5, expmd, exppd);
      } else {
         f3d = fsinhcSeries<3>(d2);
      }
   }
   if CONSTEXPR (N >= 4) {
      d4 = d2 * d2;
      d7 = d3 * d4;
      if (absd > eps[3]) {
         f4d = fsinhcAnalyt4(d, d7, expmd, exppd);
      } else {
         f4d = fsinhcSeries<4>(d2);
      }
   }
   if CONSTEXPR (N >= 5) {
      d9 = d3 * d3 * d3;
      if (absd > eps[4]) {
         f5d = fsinhcAnalyt5(d, d9, expmd, exppd);
      } else {
         f5d = fsinhcSeries<5>(d2);
      }
   }
   if CONSTEXPR (N >= 6) {
      d11 = d3 * d4 * d4;
      if (absd > eps[5]) {
         f6d = fsinhcAnalyt6(d, d11, expmd, exppd);
      } else {
         f6d = fsinhcSeries<6>(d2);
      }
   }
   if CONSTEXPR (N >= 7) {
      d13 = d3 * d3 * d3 * d4;
      if (absd > eps[6]) {
         f7d = fsinhcAnalyt7(d, d13, expmd, exppd);
      } else {
         f7d = fsinhcSeries<7>(d2);
      }
   }
}
}

/// \ingroup math
/// \{
#pragma acc routine seq
template <class T>
SEQ_CUDA
inline void fsinhc1(T d, T& restrict f1d)
{
   T f2d, f3d, f4d, f5d, f6d, f7d;
   fsinhcImpl<1, T>(d, f1d, f2d, f3d, f4d, f5d, f6d, f7d);
}

#pragma acc routine seq
template <class T>
SEQ_CUDA
inline void fsinhc2(T d, T& restrict f1d, T& restrict f2d)
{
   T f3d, f4d, f5d, f6d, f7d;
   fsinhcImpl<2, T>(d, f1d, f2d, f3d, f4d, f5d, f6d, f7d);
}

#pragma acc routine seq
template <class T>
SEQ_CUDA
inline void fsinhc3(T d, T& restrict f1d, T& restrict f2d, T& restrict f3d)
{
   T f4d, f5d, f6d, f7d;
   fsinhcImpl<3, T>(d, f1d, f2d, f3d, f4d, f5d, f6d, f7d);
}

#pragma acc routine seq
template <class T>
SEQ_CUDA
inline void fsinhc4(T d, T& restrict f1d, T& restrict f2d, T& restrict f3d, T& restrict f4d)
{
   T f5d, f6d, f7d;
   fsinhcImpl<4, T>(d, f1d, f2d, f3d, f4d, f5d, f6d, f7d);
}

#pragma acc routine seq
template <class T>
SEQ_CUDA
inline void fsinhc5(T d, T& restrict f1d, T& restrict f2d, T& restrict f3d, T& restrict f4d, T& restrict f5d)
{
   T f6d, f7d;
   fsinhcImpl<5, T>(d, f1d, f2d, f3d, f4d, f5d, f6d, f7d);
}

#pragma acc routine seq
template <class T>
SEQ_CUDA
inline void fsinhc6(T d, T& restrict f1d, T& restrict f2d, T& restrict f3d, T& restrict f4d, T& restrict f5d, T& restrict f6d)
{
   T f7d;
   fsinhcImpl<6, T>(d, f1d, f2d, f3d, f4d, f5d, f6d, f7d);
}

#pragma acc routine seq
template <class T>
SEQ_CUDA
inline void fsinhc7(T d, T& restrict f1d, T& restrict f2d, T& restrict f3d, T& restrict f4d, T& restrict f5d, T& restrict f6d, T& restrict f7d)
{
   fsinhcImpl<7, T>(d, f1d, f2d, f3d, f4d, f5d, f6d, f7d);
}
/// \}
}
