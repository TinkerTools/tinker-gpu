#include "ff/cumodamoeba.h"
#include "ff/dlmda.h"
#include "ff/evdw.h"
#include "ff/image.h"
#include "ff/modamoeba.h"
#include "ff/pme.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "seq/epolartorque.h"
#include "seq/launch.h"
#include "seq/pair_polar.h"
#include "seq/triangle.h"
#include <tinker/detail/extfld.hh>

namespace tinker {
__global__
static void polarState_cu1(int n, real* restrict polarity, real* restrict polarity_inv,
   const real* restrict polarityorig, RdtMask mask, const int* restrict group, real factor)
{
   constexpr real polmin = 1.0e-16;
   unsigned active_mask = static_cast<unsigned>(mask);
   unsigned env = static_cast<unsigned>(RdtMask::ENV);
   for (int i = ITHREAD; i < n; i += STRIDE) {
      unsigned atom_mask = env;
      if (group[i] == 1)
         atom_mask = static_cast<unsigned>(RdtMask::LIGA);
      else if (group[i] == 2)
         atom_mask = static_cast<unsigned>(RdtMask::LIGB);

      auto pol = (active_mask & atom_mask) ? polarityorig[i] : real(0);

      if (atom_mask != env)
         pol *= factor;
      polarity[i] = pol;
      polarity_inv[i] = real(1) / (pol > polmin ? pol : polmin);
   }
}

void polarState_cu(RdtMask mask, const int* group, real factor)
{
   launch_k1s(g::s0, n, polarState_cu1, n, polarity, polarity_inv, polarityorig, mask, group, factor);
}

__global__
static void epolar0DotProd_cu1(int n, real f, EnergyBuffer restrict ep, const real (*restrict gpu_uind)[3],
   const real (*restrict gpu_udirp)[3], const real* restrict polarity_inv)
{
   int ithread = ITHREAD;
   for (int i = ithread; i < n; i += STRIDE) {
      real e = polarity_inv[i]
         * (gpu_uind[i][0] * gpu_udirp[i][0] + gpu_uind[i][1] * gpu_udirp[i][1] + gpu_uind[i][2] * gpu_udirp[i][2]);
      atomic_add(f * e, ep, ithread);
   }
}

void epolar0DotProd_cu(const real (*gpu_uind)[3], const real (*gpu_udirp)[3], EnergyBuffer eout)
{
   const real f = -0.5 * electric / dielec;
   launch_k1b(g::s0, n, epolar0DotProd_cu1, n, f, eout, gpu_uind, gpu_udirp, polarity_inv);
}

__global__
static void epolarAstDeriv_cu1(int n, real f, EnergyBuffer restrict depdl,
   const real (*restrict uind)[3], const real (*restrict uinp)[3], //
   const real (*restrict dfd)[3], const real (*restrict dfp)[3],   //
   const real (*restrict f0d)[3], const real (*restrict f0p)[3],   //
   const real (*restrict ufd)[3], const real (*restrict ufp)[3],   //
   const real* restrict polarity_inv, const real* restrict polarityorig, const int* restrict mut)
{
   int ithread = ITHREAD;
   for (int i = ithread; i < n; i += STRIDE) {
      real term = uinp[i][0] * dfd[i][0] + uinp[i][1] * dfd[i][1] + uinp[i][2] * dfd[i][2]
         + uind[i][0] * dfp[i][0] + uind[i][1] * dfp[i][1] + uind[i][2] * dfp[i][2];

      if (mut[i] and polarityorig[i] != 0) {
         real fdx, fdy, fdz, fpx, fpy, fpz;
         if (f0d) {
            fdx = f0d[i][0] + ufd[i][0], fdy = f0d[i][1] + ufd[i][1], fdz = f0d[i][2] + ufd[i][2];
            fpx = f0p[i][0] + ufp[i][0], fpy = f0p[i][1] + ufp[i][1], fpz = f0p[i][2] + ufp[i][2];
         } else {
            real pinv = polarity_inv[i];
            fdx = uind[i][0] * pinv, fdy = uind[i][1] * pinv, fdz = uind[i][2] * pinv;
            fpx = uinp[i][0] * pinv, fpy = uinp[i][1] * pinv, fpz = uinp[i][2] * pinv;
         }
         term += polarityorig[i] * (fdx * fpx + fdy * fpy + fdz * fpz);
      }

      atomic_add(f * term, depdl, ithread);
   }
}

void epolarAstDeriv_cu(EnergyBuffer depdl, const real (*dfd)[3], const real (*dfp)[3], //
   const real (*f0d)[3], const real (*f0p)[3], const real (*ufd)[3], const real (*ufp)[3])
{
   const real f = -0.5 * (electric / dielec) * dpldlmda;
   launch_k1b(g::s0, n, epolarAstDeriv_cu1, n, f, depdl, uind, uinp, dfd, dfp, f0d, f0p, ufd, ufp,
      polarity_inv, polarityorig, mut);
}

__global__
static void epolarPairwiseExtfield_cu1(CountBuffer restrict nep, EnergyBuffer restrict ep, const real (*uind)[3], int n,
   real f, real ex1, real ex2, real ex3)
{
   int ithread = ITHREAD;
   for (int i = ithread; i < n; i += STRIDE) {
      real e = uind[i][0] * ex1 + uind[i][1] * ex2 + uind[i][2] * ex3;
      atomic_add(f * e, ep, ithread);
      if (e != 0)
         atomic_add(1, nep, ithread);
   }
}

void epolarPairwiseExtfield_cu(const real (*uind)[3])
{
   const real f = -0.5 * electric / dielec;
   real ex1 = extfld::texfld[0];
   real ex2 = extfld::texfld[1];
   real ex3 = extfld::texfld[2];
   launch_k1b(g::s0, n, epolarPairwiseExtfield_cu1, nep, ep, uind, n, f, ex1, ex2, ex3);
}
}

namespace tinker {
#include "epolar_cu1.cc"

template <class Ver, class ETYP>
static void epolar_cu(const real (*uind)[3], const real (*uinp)[3])
{
   constexpr bool do_g = Ver::g;

   const auto& st = *mspatial_v2_unit;
   real off;
   if CONSTEXPR (eq<ETYP, EWALD>())
      off = switchOff(Switch::EWALD);
   else
      off = switchOff(Switch::MPOLE);

   const real f = 0.5f * electric / dielec;
   real aewald = 0;
   if CONSTEXPR (eq<ETYP, EWALD>()) {
      PMEUnit pu = ppme_unit;
      aewald = pu->aewald;
   }

   if CONSTEXPR (do_g)
      darray::zero(g::q0, n, ufld, dufld);
   int ngrid = gpuGridSize(BLOCK_DIM);
   epolar_cu1<Ver, ETYP><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, nep, ep, vir_ep, depx, depy, depz,
      off, st.si1.bit0, nmdpuexclude, mdpuexclude, mdpuexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl,
      st.niak, st.iak, st.lst, ufld, dufld, rpole, uind, uinp, f, aewald);

   // torque
   if CONSTEXPR (do_g) {
      launch_k1s(g::s0, n, epolarTorque_cu, //
         trqx, trqy, trqz, n, rpole, ufld, dufld);
   }
}

void epolarNonEwald_cu(int vers, const real (*uind)[3], const real (*uinp)[3])
{
   if (vers == calc::v0) {
      epolar_cu<calc::V0, NON_EWALD>(uind, uinp);
   } else if (vers == calc::v1) {
      epolar_cu<calc::V1, NON_EWALD>(uind, uinp);
   } else if (vers == calc::v3) {
      epolar_cu<calc::V3, NON_EWALD>(uind, uinp);
   } else if (vers == calc::v4) {
      epolar_cu<calc::V4, NON_EWALD>(uind, uinp);
   } else if (vers == calc::v5) {
      epolar_cu<calc::V5, NON_EWALD>(uind, uinp);
   } else if (vers == calc::v6) {
      epolar_cu<calc::V6, NON_EWALD>(uind, uinp);
   }
}

void epolarEwaldReal_cu(int vers, const real (*uind)[3], const real (*uinp)[3])
{
   if (vers == calc::v0) {
      epolar_cu<calc::V0, EWALD>(uind, udirp);
   } else if (vers == calc::v1) {
      epolar_cu<calc::V1, EWALD>(uind, uinp);
   } else if (vers == calc::v3) {
      epolar_cu<calc::V3, EWALD>(uind, uinp);
   } else if (vers == calc::v4) {
      epolar_cu<calc::V4, EWALD>(uind, uinp);
   } else if (vers == calc::v5) {
      epolar_cu<calc::V5, EWALD>(uind, uinp);
   } else if (vers == calc::v6) {
      epolar_cu<calc::V6, EWALD>(uind, uinp);
   }
}
}
