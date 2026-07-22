#include "ff/egvop.h"
#include "ff/image.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
#include "ff/pme.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "seq/emrecip.h"
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
      real e, f1, f2, f3;
      emrecipEnergyForceAtomI<do_e, do_g>(i, fmp, fphi, e, f1, f2, f3);

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

         real tem[3];
         emrecipTorqueAtomI(i, cmp, cphi, tem);
         atomic_add(tem[0] * f, trqx, i);
         atomic_add(tem[1] * f, trqy, i);
         atomic_add(tem[2] * f, trqz, i);

         if CONSTEXPR (do_v) {
            real v[6];
            emrecipVirialAtomI(i, cmp, cphi, v);
            atomic_add(v[0] * f, v[1] * f, v[2] * f, v[3] * f, v[4] * f, v[5] * f, vir_em, ithread);
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
         sumVirialBuffer(size, vir_em, vir_m);
      } else {
         pmeConv(pu, vir_em);
      }
   } else {
      pmeConv(pu);
   }
   fftback(pu);
   fphiMpole(pu, fphi);
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
