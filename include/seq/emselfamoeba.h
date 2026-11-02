#pragma once
#include "ff/amoeba/mpole.h"
#include "ff/atom.h"
#include "ff/dlmda.h"
#include "ff/energybuffer.h"
#include "ff/modamoeba.h"
#include "seq/add.h"
#include "seq/launch.h"

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

/// The multipole self energy of a lambda run, each site against itself taking
/// the scale of its own group pair, with its lambda derivatives. With \c CFLX
/// (HIPPO, AMOEBA+) also the charge flux potential of the self term at the
/// group's charge scale \c esc and, with the gradient lambda derivatives, its
/// lambda derivative in \c dlpot; without it \c esc, \c pot and \c dlpot go
/// unused.
template <class Ver, bool CFLX>
__global__
void empoleSelfDlmda_cu(CountBuffer restrict nem,
                        EnergyBuffer restrict em,
                        EnergyBuffer restrict demdl,
                        EnergyBuffer restrict d2emdl2,
                        const real (*restrict rpole)[10],
                        const int* restrict grp,
                        GrpScale esc,
                        GrpSelfScale ss,
                        real* restrict pot,
                        real* restrict dlpot,
                        int n,
                        real f,
                        real aewald)
{
   constexpr bool do_a = Ver::a;
   constexpr bool do_e = Ver::e;
   constexpr bool do_dl1 = Ver::e_dlmda1;
   constexpr bool do_dl2 = Ver::e_dlmda2;
   constexpr bool do_gdl = Ver::g_dlmda;

   real aewald_sq_2 = 2 * aewald * aewald;
   real fterm = -f * aewald * 0.5f * (real)(M_2_SQRTPI);

   for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
      int offset = threadIdx.x + blockIdx.x * blockDim.x;
      int g = grp[i];
      if CONSTEXPR (do_e) {
         // a site against itself takes the scale of its own group pair
         real e = empoleSelfEnergyAtomI(i, rpole, fterm, aewald_sq_2);
         real dl = ss.dl[g], d2l = ss.d2l[g];
         if (dl != 0 or d2l != 0) {
            if CONSTEXPR (do_dl1)
               atomic_add(dl * e, demdl, offset);
            if CONSTEXPR (do_dl2)
               atomic_add(d2l * e, d2emdl2, offset);
         }
         atomic_add(ss.sc[g] * e, em, offset);
         if CONSTEXPR (do_a)
            atomic_add(1, nem, offset);
      }
      if CONSTEXPR (CFLX) {
         real cfl_term = 2 * fterm * rpole[i][MPL_PME_0];
         atomic_add(esc.s[g] * cfl_term, pot, i);
         if CONSTEXPR (do_gdl)
            atomic_add(esc.ds[g] * cfl_term, dlpot, i);
      }
   }
}

/// The AMOEBA multipole self energy of a lambda run, which has no charge flux,
/// into the multipole energy and lambda buffers.
template <class Ver>
void empoleSelfDlmdaAmoeba(CountBuffer nem, EnergyBuffer em, real f, real aewald)
{
   launch_k1b(g::s0, n, empoleSelfDlmda_cu<Ver, false>, //
              nem, em, demdl_buf, d2emdl2_buf, rpole, emGroup(), GrpScale{}, grpSelfScale(elecPairScale()), nullptr, nullptr, n, f, aewald);
}
}
