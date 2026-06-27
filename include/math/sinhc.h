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
   constexpr int M = N - 1;

   // (x, Approx. |Analyt - Taylor|)
   //
   // f1   (0.90, 1e-8)   (0.35, 1e-12)   (0.28, 1e-13)
   // f2   (1.15, 1e-8)   (0.45, 1e-12)   (0.37, 1e-13)
   // f3   (1.5,  1e-8)   (0.6,  1e-12)   (0.48, 1e-13)
   // f4   (2.0,  1e-8)   (0.8,  1e-12)   (0.64, 1e-13)
   // f5   (2.7,  1e-8)   (1.1,  1e-12)   (0.87, 1e-13)
   // f6   (3.7,  1e-8)   (1.5,  1e-12)   (1.16, 1e-13)
   // f7   (5.0,  1e-8)   (2.0,  1e-12)   (1.60, 1e-13)

   // double epsd[] = {0.35, 0.45, 0.60, 0.80, 1.10, 1.50, 2.00};
   // double epsd[] = {0.28, 0.37, 0.48, 0.64, 0.87, 1.16, 1.60};
   // float epsfl[] = {0.90, 1.15, 1.50, 2.00, 2.70, 3.70, 5.00};

   // (x, Approx. relative error)
   //
   // f1   (0.92, 1e-8)   (0.28, 1e-13)
   // f2   (1.06, 1e-8)   (0.33, 1e-13)
   // f3   (1.19, 1e-8)   (0.37, 1e-13)
   // f4   (1.30, 1e-8)   (0.40, 1e-13)
   // f5   (1.40, 1e-8)   (0.43, 1e-13)
   // f6   (1.49, 1e-8)   (0.46, 1e-13)
   // f7   (1.58, 1e-8)   (0.49, 1e-13)

   double epsd[] = {0.28, 0.33, 0.37, 0.40, 0.43, 0.46, 0.49};
   float epsfl[] = {0.92, 1.06, 1.19, 1.30, 1.40, 1.49, 1.58};
   T absd, eps, expmd, exppd;
   if CONSTEXPR (sizeof(T) == sizeof(float)) {
      absd = fabsf(d), eps = epsfl[M];
      expmd = expf(-d), exppd = expf(+d);
   } else {
      absd = fabs(d), eps = epsd[M];
      expmd = exp(-d), exppd = exp(+d);
   }

   T d2, d4;
   T d3, d5, d7, d9, d11, d13;
   if CONSTEXPR (N >= 1) {
      d2 = d * d;
      if (absd > eps) {
         f1d = fsinhcAnalyt1(d, expmd, exppd);
      } else {
         f1d = fsinhcSeries<1>(d2);
      }
   }
   if CONSTEXPR (N >= 2) {
      d3 = d * d2;
      if (absd > eps) {
         f2d = fsinhcAnalyt2(d, d3, expmd, exppd);
      } else {
         f2d = fsinhcSeries<2>(d2);
      }
   }
   if CONSTEXPR (N >= 3) {
      d5 = d2 * d3;
      if (absd > eps) {
         f3d = fsinhcAnalyt3(d, d5, expmd, exppd);
      } else {
         f3d = fsinhcSeries<3>(d2);
      }
   }
   if CONSTEXPR (N >= 4) {
      d4 = d2 * d2;
      d7 = d3 * d4;
      if (absd > eps) {
         f4d = fsinhcAnalyt4(d, d7, expmd, exppd);
      } else {
         f4d = fsinhcSeries<4>(d2);
      }
   }
   if CONSTEXPR (N >= 5) {
      d9 = d3 * d3 * d3;
      if (absd > eps) {
         f5d = fsinhcAnalyt5(d, d9, expmd, exppd);
      } else {
         f5d = fsinhcSeries<5>(d2);
      }
   }
   if CONSTEXPR (N >= 6) {
      d11 = d3 * d4 * d4;
      if (absd > eps) {
         f6d = fsinhcAnalyt6(d, d11, expmd, exppd);
      } else {
         f6d = fsinhcSeries<6>(d2);
      }
   }
   if CONSTEXPR (N >= 7) {
      d13 = d3 * d3 * d3 * d4;
      if (absd > eps) {
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
