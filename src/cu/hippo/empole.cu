#include "ff/image.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
#include "ff/pme.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "seq/emselfhippo.h"
#include "seq/launch.h"
#include "seq/pair_mpole_chgpen.h"
#include "seq/pairmpoleaplus.h"
#include "seq/triangle.h"
#include <cassert>

namespace tinker {
#include "empoleChgpen_cu1.cc"

template <class Ver, class ETYP, Chgpen CP, int CFLX>
static void empoleChgpen_cu()
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;

   const auto& st = *mspatial_v2_unit;
   real off;
   if CONSTEXPR (eq<ETYP, EWALD>())
      off = switchOff(Switch::EWALD);
   else
      off = switchOff(Switch::MPOLE);

   const real f = electric / dielec;
   real aewald = 0;
   if CONSTEXPR (eq<ETYP, EWALD>()) {
      PMEUnit pu = epme_unit;
      aewald = pu->aewald;
      launch_k1b(g::s0, n, empoleChgpenSelf_cu<do_a, do_e, CFLX>, //
         nem, em, rpole, pot, n, f, aewald);
   }

   int ngrid = gpuGridSize(BLOCK_DIM);
   empoleChgpen_cu1<Ver, ETYP, CP, CFLX><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, nem, em, vir_em, demx,
      demy, demz, off, st.si1.bit0, nmdwexclude, mdwexclude, mdwexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl,
      st.iakpl, st.niak, st.iak, st.lst, trqx, trqy, trqz, pot, rpole, pcore, pval, palpha, aewald, f);
}

void empoleChgpenNonEwald_cu(int vers, int use_cf)
{
   assert(pentyp == Chgpen::GORDON1);
   constexpr auto CP = Chgpen::GORDON1;
   if (use_cf) {
      if (vers == calc::v0) {
         // empoleChgpen_cu<calc::V0, NON_EWALD, CP, 1>();
         assert(false && "CFLX must compute gradient.");
      } else if (vers == calc::v1) {
         empoleChgpen_cu<calc::V1, NON_EWALD, CP, 1>();
      } else if (vers == calc::v3) {
         // empoleChgpen_cu<calc::V3, NON_EWALD, CP, 1>();
         assert(false && "CFLX must compute gradient.");
      } else if (vers == calc::v4) {
         empoleChgpen_cu<calc::V4, NON_EWALD, CP, 1>();
      } else if (vers == calc::v5) {
         empoleChgpen_cu<calc::V5, NON_EWALD, CP, 1>();
      } else if (vers == calc::v6) {
         empoleChgpen_cu<calc::V6, NON_EWALD, CP, 1>();
      }
   } else {
      if (vers == calc::v0) {
         empoleChgpen_cu<calc::V0, NON_EWALD, CP, 0>();
      } else if (vers == calc::v1) {
         empoleChgpen_cu<calc::V1, NON_EWALD, CP, 0>();
      } else if (vers == calc::v3) {
         empoleChgpen_cu<calc::V3, NON_EWALD, CP, 0>();
      } else if (vers == calc::v4) {
         empoleChgpen_cu<calc::V4, NON_EWALD, CP, 0>();
      } else if (vers == calc::v5) {
         empoleChgpen_cu<calc::V5, NON_EWALD, CP, 0>();
      } else if (vers == calc::v6) {
         empoleChgpen_cu<calc::V6, NON_EWALD, CP, 0>();
      }
   }
}

void empoleChgpenEwaldRealSelf_cu(int vers, int use_cf)
{
   assert(pentyp == Chgpen::GORDON1);
   constexpr auto CP = Chgpen::GORDON1;
   if (use_cf) {
      if (vers == calc::v0) {
         // empoleChgpen_cu<calc::V0, EWALD, CP, 1>();
         assert(false && "CFLX must compute gradient.");
      } else if (vers == calc::v1) {
         empoleChgpen_cu<calc::V1, EWALD, CP, 1>();
      } else if (vers == calc::v3) {
         // empoleChgpen_cu<calc::V3, EWALD, CP, 1>();
         assert(false && "CFLX must compute gradient.");
      } else if (vers == calc::v4) {
         empoleChgpen_cu<calc::V4, EWALD, CP, 1>();
      } else if (vers == calc::v5) {
         empoleChgpen_cu<calc::V5, EWALD, CP, 1>();
      } else if (vers == calc::v6) {
         empoleChgpen_cu<calc::V6, EWALD, CP, 1>();
      }
   } else {
      if (vers == calc::v0) {
         empoleChgpen_cu<calc::V0, EWALD, CP, 0>();
      } else if (vers == calc::v1) {
         empoleChgpen_cu<calc::V1, EWALD, CP, 0>();
      } else if (vers == calc::v3) {
         empoleChgpen_cu<calc::V3, EWALD, CP, 0>();
      } else if (vers == calc::v4) {
         empoleChgpen_cu<calc::V4, EWALD, CP, 0>();
      } else if (vers == calc::v5) {
         empoleChgpen_cu<calc::V5, EWALD, CP, 0>();
      } else if (vers == calc::v6) {
         empoleChgpen_cu<calc::V6, EWALD, CP, 0>();
      }
   }
}

void empoleAplusNonEwald_cu(int vers, int use_cf)
{
   assert(pentyp == Chgpen::GORDON2);
   constexpr auto CP = Chgpen::GORDON2;
   if (use_cf) {
      if (vers == calc::v0) {
         // empoleChgpen_cu<calc::V0, NON_EWALD, CP, 1>();
         assert(false && "CFLX must compute gradient.");
      } else if (vers == calc::v1) {
         empoleChgpen_cu<calc::V1, NON_EWALD, CP, 1>();
      } else if (vers == calc::v3) {
         // empoleChgpen_cu<calc::V3, NON_EWALD, CP, 1>();
         assert(false && "CFLX must compute gradient.");
      } else if (vers == calc::v4) {
         empoleChgpen_cu<calc::V4, NON_EWALD, CP, 1>();
      } else if (vers == calc::v5) {
         empoleChgpen_cu<calc::V5, NON_EWALD, CP, 1>();
      } else if (vers == calc::v6) {
         empoleChgpen_cu<calc::V6, NON_EWALD, CP, 1>();
      }
   } else {
      if (vers == calc::v0) {
         empoleChgpen_cu<calc::V0, NON_EWALD, CP, 0>();
      } else if (vers == calc::v1) {
         empoleChgpen_cu<calc::V1, NON_EWALD, CP, 0>();
      } else if (vers == calc::v3) {
         empoleChgpen_cu<calc::V3, NON_EWALD, CP, 0>();
      } else if (vers == calc::v4) {
         empoleChgpen_cu<calc::V4, NON_EWALD, CP, 0>();
      } else if (vers == calc::v5) {
         empoleChgpen_cu<calc::V5, NON_EWALD, CP, 0>();
      } else if (vers == calc::v6) {
         empoleChgpen_cu<calc::V6, NON_EWALD, CP, 0>();
      }
   }
}

void empoleAplusEwaldRealSelf_cu(int vers, int use_cf)
{
   assert(pentyp == Chgpen::GORDON2);
   constexpr auto CP = Chgpen::GORDON2;
   if (use_cf) {
      if (vers == calc::v0) {
         // empoleChgpen_cu<calc::V0, EWALD, CP, 1>();
         assert(false && "CFLX must compute gradient.");
      } else if (vers == calc::v1) {
         empoleChgpen_cu<calc::V1, EWALD, CP, 1>();
      } else if (vers == calc::v3) {
         // empoleChgpen_cu<calc::V3, EWALD, CP, 1>();
         assert(false && "CFLX must compute gradient.");
      } else if (vers == calc::v4) {
         empoleChgpen_cu<calc::V4, EWALD, CP, 1>();
      } else if (vers == calc::v5) {
         empoleChgpen_cu<calc::V5, EWALD, CP, 1>();
      } else if (vers == calc::v6) {
         empoleChgpen_cu<calc::V6, EWALD, CP, 1>();
      }
   } else {
      if (vers == calc::v0) {
         empoleChgpen_cu<calc::V0, EWALD, CP, 0>();
      } else if (vers == calc::v1) {
         empoleChgpen_cu<calc::V1, EWALD, CP, 0>();
      } else if (vers == calc::v3) {
         empoleChgpen_cu<calc::V3, EWALD, CP, 0>();
      } else if (vers == calc::v4) {
         empoleChgpen_cu<calc::V4, EWALD, CP, 0>();
      } else if (vers == calc::v5) {
         empoleChgpen_cu<calc::V5, EWALD, CP, 0>();
      } else if (vers == calc::v6) {
         empoleChgpen_cu<calc::V6, EWALD, CP, 0>();
      }
   }
}
}

namespace tinker {
__global__
void empoleEwaldRecipGenericAddVirM_cu(size_t size, VirialBuffer restrict vir_em, const VirialBuffer restrict vir_m)
{
   for (size_t i = ITHREAD; i < size; i += STRIDE)
      vir_em[0][i] += vir_m[0][i];
}

template <bool do_e, bool do_g, bool do_v, int CFLX>
__global__
void empoleEwaldRecipGeneric_cu1(int n, real f,                                  //
   EnergyBuffer restrict em, VirialBuffer restrict vir_em,                       //
   grad_prec* restrict demx, grad_prec* restrict demy, grad_prec* restrict demz, //
   real* restrict trqx, real* restrict trqy, real* restrict trqz,                //
   real* restrict pot,                                                           //
   const real (*restrict cmp)[10], const real (*restrict fmp)[10],               //
   const real (*restrict cphi)[10], const real (*restrict fphi)[20],             //
   int nfft1, int nfft2, int nfft3, TINKER_IMAGE_PARAMS)
{
   int ithread = ITHREAD;
   for (int i = ithread; i < n; i += STRIDE) {
      constexpr int deriv1[] = {2, 5, 8, 9, 11, 16, 18, 14, 15, 20};
      constexpr int deriv2[] = {3, 8, 6, 10, 14, 12, 19, 16, 20, 17};
      constexpr int deriv3[] = {4, 9, 10, 7, 15, 17, 13, 20, 18, 19};

      real e = 0;
      real f1 = 0;
      real f2 = 0;
      real f3 = 0;

      for (int k = 0; k < 10; ++k) {
         if CONSTEXPR (do_e)
            e += fmp[i][k] * fphi[i][k];
         if CONSTEXPR (do_g) {
            f1 += fmp[i][k] * fphi[i][deriv1[k] - 1];
            f2 += fmp[i][k] * fphi[i][deriv2[k] - 1];
            f3 += fmp[i][k] * fphi[i][deriv3[k] - 1];
         }
      } // end for (int k)

      // increment the permanent multipole energy and gradient

      if CONSTEXPR (do_e)
         atomic_add(0.5f * e * f, em, ithread);

      if CONSTEXPR (do_g) {
         f1 *= nfft1;
         f2 *= nfft2;
         f3 *= nfft3;

         real h1 = recipa.x * f1 + recipb.x * f2 + recipc.x * f3;
         real h2 = recipa.y * f1 + recipb.y * f2 + recipc.y * f3;
         real h3 = recipa.z * f1 + recipb.z * f2 + recipc.z * f3;

         atomic_add(h1 * f, demx, i);
         atomic_add(h2 * f, demy, i);
         atomic_add(h3 * f, demz, i);

         // resolve site torques then increment forces and virial

         real tem1 = cmp[i][3] * cphi[i][2] - cmp[i][2] * cphi[i][3] + 2 * (cmp[i][6] - cmp[i][5]) * cphi[i][9]
            + cmp[i][8] * cphi[i][7] + cmp[i][9] * cphi[i][5] - cmp[i][7] * cphi[i][8] - cmp[i][9] * cphi[i][6];
         real tem2 = cmp[i][1] * cphi[i][3] - cmp[i][3] * cphi[i][1] + 2 * (cmp[i][4] - cmp[i][6]) * cphi[i][8]
            + cmp[i][7] * cphi[i][9] + cmp[i][8] * cphi[i][6] - cmp[i][8] * cphi[i][4] - cmp[i][9] * cphi[i][7];
         real tem3 = cmp[i][2] * cphi[i][1] - cmp[i][1] * cphi[i][2] + 2 * (cmp[i][5] - cmp[i][4]) * cphi[i][7]
            + cmp[i][7] * cphi[i][4] + cmp[i][9] * cphi[i][8] - cmp[i][7] * cphi[i][5] - cmp[i][8] * cphi[i][9];
         tem1 *= f;
         tem2 *= f;
         tem3 *= f;

         atomic_add(tem1, trqx, i);
         atomic_add(tem2, trqy, i);
         atomic_add(tem3, trqz, i);

         if CONSTEXPR (do_v) {
            real vxx = -cmp[i][1] * cphi[i][1] - 2 * cmp[i][4] * cphi[i][4] - cmp[i][7] * cphi[i][7]
               - cmp[i][8] * cphi[i][8];
            real vxy = -0.5f * (cmp[i][2] * cphi[i][1] + cmp[i][1] * cphi[i][2]) - (cmp[i][4] + cmp[i][5]) * cphi[i][7]
               - 0.5f * cmp[i][7] * (cphi[i][4] + cphi[i][5])
               - 0.5f * (cmp[i][8] * cphi[i][9] + cmp[i][9] * cphi[i][8]);
            real vxz = -0.5f * (cmp[i][3] * cphi[i][1] + cmp[i][1] * cphi[i][3]) - (cmp[i][4] + cmp[i][6]) * cphi[i][8]
               - 0.5f * cmp[i][8] * (cphi[i][4] + cphi[i][6])
               - 0.5f * (cmp[i][7] * cphi[i][9] + cmp[i][9] * cphi[i][7]);
            real vyy = -cmp[i][2] * cphi[i][2] - 2 * cmp[i][5] * cphi[i][5] - cmp[i][7] * cphi[i][7]
               - cmp[i][9] * cphi[i][9];
            real vyz = -0.5f * (cmp[i][3] * cphi[i][2] + cmp[i][2] * cphi[i][3]) - (cmp[i][5] + cmp[i][6]) * cphi[i][9]
               - 0.5f * cmp[i][9] * (cphi[i][5] + cphi[i][6])
               - 0.5f * (cmp[i][7] * cphi[i][8] + cmp[i][8] * cphi[i][7]);
            real vzz = -cmp[i][3] * cphi[i][3] - 2 * cmp[i][6] * cphi[i][6] - cmp[i][8] * cphi[i][8]
               - cmp[i][9] * cphi[i][9];
            vxx *= f;
            vxy *= f;
            vxz *= f;
            vyy *= f;
            vyz *= f;
            vzz *= f;

            atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, vir_em, ithread);
         } // end if (do_v)
         if CONSTEXPR (CFLX) {
            atomic_add(f * cphi[i][0], pot, i);
         }
      } // end if (do_g)
   }
}

template <class Ver, int CFLX>
static void empoleEwaldRecipGeneric_cu()
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;

   const PMEUnit pu = epme_unit;
   cmpToFmp(pu, cmp, fmp);
   gridMpole(pu, fmp);
   fftfront(pu);
   if CONSTEXPR (do_v) {
      if (vir_m) {
         pmeConv(pu, vir_m);
         auto size = bufferSize() * VirialBufferTraits::value;
         launch_k1s(g::s0, size, empoleEwaldRecipGenericAddVirM_cu, size, vir_em, vir_m);
      } else {
         pmeConv(pu, vir_em);
      }
   } else {
      pmeConv(pu);
   }
   fftback(pu);
   fphiMpole(pu);
   fphiToCphi(pu, fphi, cphi);

   auto& st = *pu;
   const int nfft1 = st.nfft1;
   const int nfft2 = st.nfft2;
   const int nfft3 = st.nfft3;
   const real f = electric / dielec;

   launch_k1b(g::s0, n, empoleEwaldRecipGeneric_cu1<do_e, do_g, do_v, CFLX>, //
      n, f, em, vir_em, demx, demy, demz, trqx, trqy, trqz, pot,             //
      cmp, fmp, cphi, fphi,                                                  //
      nfft1, nfft2, nfft3, TINKER_IMAGE_ARGS);
}

void empoleChgpenEwaldRecip_cu(int vers, int use_cf)
{
   if (use_cf) {
      if (vers == calc::v0)
         // empoleEwaldRecipGeneric_cu<calc::V0, 1>();
         assert(false && "CFLX must compute gradient.");
      else if (vers == calc::v1)
         empoleEwaldRecipGeneric_cu<calc::V1, 1>();
      else if (vers == calc::v3)
         // empoleEwaldRecipGeneric_cu<calc::V3, 1>();
         assert(false && "CFLX must compute gradient.");
      else if (vers == calc::v4)
         empoleEwaldRecipGeneric_cu<calc::V4, 1>();
      else if (vers == calc::v5)
         empoleEwaldRecipGeneric_cu<calc::V5, 1>();
      else if (vers == calc::v6)
         empoleEwaldRecipGeneric_cu<calc::V6, 1>();
   } else {
      if (vers == calc::v0)
         empoleEwaldRecipGeneric_cu<calc::V0, 0>();
      else if (vers == calc::v1)
         empoleEwaldRecipGeneric_cu<calc::V1, 0>();
      else if (vers == calc::v3)
         empoleEwaldRecipGeneric_cu<calc::V3, 0>();
      else if (vers == calc::v4)
         empoleEwaldRecipGeneric_cu<calc::V4, 0>();
      else if (vers == calc::v5)
         empoleEwaldRecipGeneric_cu<calc::V5, 0>();
      else if (vers == calc::v6)
         empoleEwaldRecipGeneric_cu<calc::V6, 0>();
   }
}
}
