#pragma once
#include "ff/amoeba/mpole.h"
#include "ff/energybuffer.h"
#include "seq/add.h"

namespace tinker {
__device__
inline real empoleSelfEnergyAtomI(int i, const real (*restrict rpole)[10], real fterm, real aewald_sq_2)
{
   real ci = rpole[i][MPL_PME_0];
   real dix = rpole[i][MPL_PME_X];
   real diy = rpole[i][MPL_PME_Y];
   real diz = rpole[i][MPL_PME_Z];
   real qixx = rpole[i][MPL_PME_XX];
   real qixy = rpole[i][MPL_PME_XY];
   real qixz = rpole[i][MPL_PME_XZ];
   real qiyy = rpole[i][MPL_PME_YY];
   real qiyz = rpole[i][MPL_PME_YZ];
   real qizz = rpole[i][MPL_PME_ZZ];

   real cii = ci * ci;
   real dii = dix * dix + diy * diy + diz * diz;
   real qii = 2 * (qixy * qixy + qixz * qixz + qiyz * qiyz) + qixx * qixx + qiyy * qiyy + qizz * qizz;

   return fterm * (cii + aewald_sq_2 * (dii / 3 + 2 * aewald_sq_2 * qii * (real)0.2));
}

template <bool do_a>
__global__
void empoleSelf_cu(CountBuffer restrict nem, EnergyBuffer restrict em, const real (*restrict rpole)[10], int n, real f, real aewald)
{
   real aewald_sq_2 = 2 * aewald * aewald;
   real fterm = -f * aewald * 0.5f * (real)(M_2_SQRTPI);

   for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
      int offset = threadIdx.x + blockIdx.x * blockDim.x;
      real e = empoleSelfEnergyAtomI(i, rpole, fterm, aewald_sq_2);
      atomic_add(e, em, offset);
      if CONSTEXPR (do_a)
         atomic_add(1, nem, offset);
   }
}

template <class Ver>
__global__
void empoleSelfDlmda_cu(CountBuffer restrict nem, EnergyBuffer restrict em, EnergyBuffer restrict demdl,
   EnergyBuffer restrict d2emdl2, const real (*restrict rpole)[10], const int* restrict mut, int n, real f,
   real aewald, real elambda, real deldl, real d2eldl2)
{
   constexpr bool do_a = Ver::a;
   constexpr bool do_dl1 = Ver::e_dlmda1;
   constexpr bool do_dl2 = Ver::e_dlmda2;

   real aewald_sq_2 = 2 * aewald * aewald;
   real fterm = -f * aewald * 0.5f * (real)(M_2_SQRTPI);

   for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
      int offset = threadIdx.x + blockIdx.x * blockDim.x;
      real e = empoleSelfEnergyAtomI(i, rpole, fterm, aewald_sq_2);
      if (mut[i]) {
         if CONSTEXPR (do_dl1)
            atomic_add(2 * elambda * deldl * e, demdl, offset);
         if CONSTEXPR (do_dl2)
            atomic_add((2 * deldl * deldl + 2 * elambda * d2eldl2) * e, d2emdl2, offset);
         e *= elambda * elambda;
      }
      atomic_add(e, em, offset);
      if CONSTEXPR (do_a)
         atomic_add(1, nem, offset);
   }
}
}
