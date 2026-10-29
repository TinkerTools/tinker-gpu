#include "ff/cumodamoeba.h"
#include "ff/dlmda.h"
#include "ff/elec.h"
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
   const real* restrict polarityorig, RdtMask mask, const int* restrict group, real factor, bool expol,
   real* restrict prepep, const real* restrict prepeporig)
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

      // the exchange polarization size follows the polarizability
      if (expol) {
         real siz = (active_mask & atom_mask) ? prepeporig[i] : real(0);
         if (atom_mask != env)
            siz *= factor;
         prepep[i] = siz;
      }
   }
}

void polarState_cu(RdtMask mask, const int* group, real factor, bool expol, real* prepep, const real* prepeporig)
{
   launch_k1s(g::s0, n, polarState_cu1, n, polarity, polarity_inv, polarityorig, mask, group, factor, expol, prepep,
      prepeporig);
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
   const real* restrict polarity_inv, const real* restrict polarityorig, const int* restrict grp, GrpScale sc,
   bool expol, const real (*restrict polinv)[3][3], const real (*restrict dpscdl)[3][3])
{
   int ithread = ITHREAD;
   for (int i = ithread; i < n; i += STRIDE) {
      real term = uinp[i][0] * dfd[i][0] + uinp[i][1] * dfd[i][1] + uinp[i][2] * dfd[i][2]
         + uind[i][0] * dfp[i][0] + uind[i][1] * dfp[i][1] + uind[i][2] * dfp[i][2];

      // only the charging ligand carries a polarizability derivative
      if (sc.ds[grp[i]] != 0 and polarityorig[i] != 0) {
         real fdx, fdy, fdz, fpx, fpy, fpz;
         if (f0d) {
            fdx = f0d[i][0] + ufd[i][0], fdy = f0d[i][1] + ufd[i][1], fdz = f0d[i][2] + ufd[i][2];
            fpx = f0p[i][0] + ufp[i][0], fpy = f0p[i][1] + ufp[i][1], fpz = f0p[i][2] + ufp[i][2];
         } else {
            real pinv = polarity_inv[i];
            fdx = uind[i][0] * pinv, fdy = uind[i][1] * pinv, fdz = uind[i][2] * pinv;
            fpx = uinp[i][0] * pinv, fpy = uinp[i][1] * pinv, fpz = uinp[i][2] * pinv;
         }
         // With exchange polarization the dipoles already carry polinv, so only
         // the permanent field at zero lambda needs it, applied as induce does.
         if (expol and f0d) {
            const real(&p)[3][3] = polinv[i];
            real dx = p[0][0] * fdx + p[1][0] * fdy + p[2][0] * fdz;
            real dy = p[0][1] * fdx + p[1][1] * fdy + p[2][1] * fdz;
            real dz = p[0][2] * fdx + p[1][2] * fdy + p[2][2] * fdz;
            real px = p[0][0] * fpx + p[1][0] * fpy + p[2][0] * fpz;
            real py = p[0][1] * fpx + p[1][1] * fpy + p[2][1] * fpz;
            real pz = p[0][2] * fpx + p[1][2] * fpy + p[2][2] * fpz;
            fdx = dx, fdy = dy, fdz = dz;
            fpx = px, fpy = py, fpz = pz;
         }
         term += polarityorig[i] * (fdx * fpx + fdy * fpy + fdz * fpz);
      }

      // the overlap of every site with the charging ligand changes its
      // polarizability tensor as well
      if (expol and polarityorig[i] != 0) {
         const real(&d)[3][3] = dpscdl[i];
         real ux = uind[i][0], uy = uind[i][1], uz = uind[i][2];
         real px = uinp[i][0], py = uinp[i][1], pz = uinp[i][2];
         real s = ux * (d[0][0] * px + d[0][1] * py + d[0][2] * pz) + uy * (d[1][0] * px + d[1][1] * py + d[1][2] * pz)
            + uz * (d[2][0] * px + d[2][1] * py + d[2][2] * pz);
         term -= s / polarityorig[i];
      }

      atomic_add(f * term, depdl, ithread);
   }
}

void epolarAstDeriv_cu(EnergyBuffer depdl, const real (*ud)[3], const real (*up)[3], //
   const real (*dfd)[3], const real (*dfp)[3], const real (*f0d)[3], const real (*f0p)[3],  //
   const real (*ufd)[3], const real (*ufp)[3], bool expol, const real (*polinv)[3][3], const real (*dpscdl)[3][3])
{
   const real f = -0.5 * (electric / dielec) * dpldlmda;
   launch_k1b(g::s0, n, epolarAstDeriv_cu1, n, f, depdl, ud, up, dfd, dfp, f0d, f0p, ufd, ufp,
      polarity_inv, polarityorig, emGroup(), grpScale(plam), expol, polinv, dpscdl);
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
      st.niak, st.iak, st.lst, ufld, dufld, rpole, uind, uinp, f, aewald, elecDeadGroups(), elecLiveMask());

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
