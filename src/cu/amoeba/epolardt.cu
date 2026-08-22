#include "ff/amoeba/epolar.h"
#include "ff/cumodamoeba.h"
#include "ff/dlmda.h"
#include "ff/elec.h"
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

// One dual topology subsystem of the polarization energy. Unlike the permanent
// multipoles, the two endpoints cannot be fused: the induced dipoles solve a
// global linear system over the masked polarity, so each subsystem has its own
// uind and its own pair energies. What the endpoints do share is that the
// masking is already baked into rpole and polarity before the pass runs, so
// every interaction in a pass carries the same two weights,
//
//     energy, virial, gradient, torque  <- wa times the subsystem result
//     their lambda derivatives          <- wb times the same
//
// and the kernel needs none of the per-pair group machinery empoledt uses. The
// caller accumulates all the passes into the global buffers and converts the
// torque once at the end. See DtCoef and dtPassWeights().
//
// The energy travels with its own lambda derivatives, and for polarization all
// three ride the dot product rather than this kernel -- the only version that
// takes its energy from here is calc::v3, which carries no derivatives at all.
// So unlike empoledt_cu1 this kernel has no energy channels beyond the analysis
// breakdown. See epolarEnergyFromDotProd().

namespace tinker {
#include "epolardt_cu1.cc"

template <class Ver, class ETYP>
static void epolardt_cu(const real (*uind)[3], const real (*uinp)[3], real wa, real wb)
{
   constexpr bool do_g = Ver::g;
   // The lambda torque is the one shared input: the lambda gradient resolves
   // from it, and so does the torque part of the lambda virial.
   constexpr bool do_tdl = Ver::g_dlmda or Ver::v_dlmda;

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
   epolardt_cu1<Ver, ETYP><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, nep, ep,
      vir_ep, dvirdl_buf, depx, depy, depz, dfdlx, dfdly, dfdlz, off, st.si1.bit0, nmdpuexclude, mdpuexclude,
      mdpuexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst, ufld, dufld,
      rpole, uind, uinp, f, aewald, wa, wb);

   // torque
   if CONSTEXPR (do_g) {
      launch_k1s(g::s0, n, epolarTorqueDt_cu, //
         trqx, trqy, trqz, dltrqx, dltrqy, dltrqz, n, rpole, ufld, dufld, wa, wb, do_tdl);
   }
}

template <class ETYP>
static void epolardtVers_cu(int vers, const real (*uind)[3], const real (*uinp)[3], real wa, real wb)
{
   if (vers == calc::v0)
      epolardt_cu<calc::V0, ETYP>(uind, uinp, wa, wb);
   else if (vers == calc::v1)
      epolardt_cu<calc::V1, ETYP>(uind, uinp, wa, wb);
   else if (vers == calc::v3)
      epolardt_cu<calc::V3, ETYP>(uind, uinp, wa, wb);
   else if (vers == calc::v4)
      epolardt_cu<calc::V4, ETYP>(uind, uinp, wa, wb);
   else if (vers == calc::v5)
      epolardt_cu<calc::V5, ETYP>(uind, uinp, wa, wb);
   else if (vers == calc::v6)
      epolardt_cu<calc::V6, ETYP>(uind, uinp, wa, wb);
   else if (vers == calc::v7)
      epolardt_cu<calc::V7, ETYP>(uind, uinp, wa, wb);
   else if (vers == calc::v8)
      epolardt_cu<calc::V8, ETYP>(uind, uinp, wa, wb);
   else if (vers == calc::v9)
      epolardt_cu<calc::V9, ETYP>(uind, uinp, wa, wb);
   else if (vers == calc::v10)
      epolardt_cu<calc::V10, ETYP>(uind, uinp, wa, wb);
}

void epolarNonEwaldDt_cu(int vers, const real (*uind)[3], const real (*uinp)[3], real wa, real wb)
{
   epolardtVers_cu<NON_EWALD>(vers, uind, uinp, wa, wb);
}

void epolarEwaldRealDt_cu(int vers, const real (*uind)[3], const real (*uinp)[3], real wa, real wb)
{
   epolardtVers_cu<EWALD>(vers, uind, uinp, wa, wb);
}
}

namespace tinker {
__global__
static void epolar0DotProdDt_cu1(int n, real f, EnergyBuffer restrict ep, EnergyBuffer restrict epdl,
   EnergyBuffer restrict d2epdl2, const real (*restrict gpu_uind)[3], const real (*restrict gpu_udirp)[3],
   const real* restrict polarity_inv, real wa, real wb, real wc, bool do_dl1, bool do_dl2)
{
   int ithread = ITHREAD;
   for (int i = ithread; i < n; i += STRIDE) {
      real e = polarity_inv[i]
         * (gpu_uind[i][0] * gpu_udirp[i][0] + gpu_uind[i][1] * gpu_udirp[i][1] + gpu_uind[i][2] * gpu_udirp[i][2]);
      e *= f;
      atomic_add(wa * e, ep, ithread);
      if (do_dl1)
         atomic_add(wb * e, epdl, ithread);
      if (do_dl2)
         atomic_add(wc * e, d2epdl2, ithread);
   }
}

// The dot product owns all three energy channels of a dual topology pass: the
// energy itself and both of its lambda derivatives.
void epolar0DotProdDt_cu(int vers, const real (*gpu_uind)[3], const real (*gpu_udirp)[3], real wa, real wb, real wc)
{
   const real f = -0.5 * electric / dielec;
   const bool do_dl1 = vers & calc::energy_dlmda1;
   const bool do_dl2 = vers & calc::energy_dlmda2;
   launch_k1b(g::s0, n, epolar0DotProdDt_cu1, n, f, ep, depdl_buf, d2epdl2_buf, gpu_uind, gpu_udirp,
      polarity_inv, wa, wb, wc, do_dl1, do_dl2);
}

__global__
static void epolarPairwiseExtfieldDt_cu1(CountBuffer restrict nep, EnergyBuffer restrict ep,
   const real (*uind)[3], int n, real f, real ex1, real ex2, real ex3, real wa)
{
   int ithread = ITHREAD;
   for (int i = ithread; i < n; i += STRIDE) {
      real e = uind[i][0] * ex1 + uind[i][1] * ex2 + uind[i][2] * ex3;
      atomic_add(wa * f * e, ep, ithread);
      if (e != 0)
         atomic_add(1, nep, ithread);
   }
}

// Only an analysis run reaches this, and lmdaDerivVers() leaves calc::v3 alone,
// so there is no lambda derivative channel to feed.
void epolarPairwiseExtfieldDt_cu(const real (*uind)[3], real wa)
{
   const real f = -0.5 * electric / dielec;
   real ex1 = extfld::texfld[0];
   real ex2 = extfld::texfld[1];
   real ex3 = extfld::texfld[2];
   launch_k1b(g::s0, n, epolarPairwiseExtfieldDt_cu1, nep, ep, uind, n, f, ex1, ex2, ex3, wa);
}
}
