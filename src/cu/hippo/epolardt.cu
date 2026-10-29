#include "ff/amoeba/mpolestate.h"
#include "ff/cumodamoeba.h"
#include "ff/dlmda.h"
#include "ff/elec.h"
#include "ff/image.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
#include "ff/pme.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "seq/epolartorque.h"
#include "seq/launch.h"
#include "seq/pair_polar_chgpen.h"
#include "seq/pairpolaraplus.h"
#include "seq/triangle.h"
#include <cassert>
#include <tinker/detail/polpot.hh>

namespace tinker {
#include "epolarAplusDt_cu1.cc"
#include "epolarChgpenDt_cu1.cc"

// One dual topology pass of HIPPO or AMOEBA+ polarization, the models with one
// set of induced dipoles: the pass adds wa times its result into the term and
// wb times it into the lambda derivative sinks, as epolardt_cu does for AMOEBA.
// The flux potential stays unweighted, for the caller's chain rule of the pass.
template <class Ver, class ETYP, int CFLX>
static void epolar1Dt_cu(const real (*uind)[3], real wa, real wb)
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
   if (polpot::use_tholed) {
      epolarAplusDt_cu1<Ver, ETYP, CFLX><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, nep, ep, vir_ep,
         dvirdl_buf, depx, depy, depz, dfdlx, dfdly, dfdlz, off, st.si2.bit0, nmdpuexclude, mdpuexclude,
         mdpuexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst, ufld, dufld,
         uind, pot, rpole, pdamp, aewald, f, wa, wb, elecDeadGroups(), elecLiveMask());
   } else {
      epolarChgpenDt_cu1<Ver, ETYP, CFLX><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, nep, ep, vir_ep,
         dvirdl_buf, depx, depy, depz, dfdlx, dfdly, dfdlz, off, st.si1.bit0, nmdwexclude, mdwexclude,
         mdwexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst, ufld, dufld,
         uind, pot, rpole, mpoleCore(), mpoleVal(), palpha, aewald, f, wa, wb, elecDeadGroups(), elecLiveMask());
   }

   if CONSTEXPR (do_g) {
      launch_k1s(g::s0, n, epolarTorqueDt_cu, //
         trqx, trqy, trqz, dltrqx, dltrqy, dltrqz, n, rpole, ufld, dufld, wa, wb, do_tdl);
   }
}

// The flux potentials are gathered with the gradient only, so CFLX leaves out
// the versions without one.
template <class ETYP, int CFLX>
static void epolar1DtVers_cu(int vers, const real (*uind)[3], real wa, real wb)
{
   if (vers == calc::v0 or vers == calc::v3) {
      if CONSTEXPR (CFLX)
         assert(false && "CFLX must compute gradient.");
      else if (vers == calc::v0)
         epolar1Dt_cu<calc::V0, ETYP, 0>(uind, wa, wb);
      else
         epolar1Dt_cu<calc::V3, ETYP, 0>(uind, wa, wb);
   } else if (vers == calc::v1)
      epolar1Dt_cu<calc::V1, ETYP, CFLX>(uind, wa, wb);
   else if (vers == calc::v4)
      epolar1Dt_cu<calc::V4, ETYP, CFLX>(uind, wa, wb);
   else if (vers == calc::v5)
      epolar1Dt_cu<calc::V5, ETYP, CFLX>(uind, wa, wb);
   else if (vers == calc::v6)
      epolar1Dt_cu<calc::V6, ETYP, CFLX>(uind, wa, wb);
   else if (vers == calc::v7)
      epolar1Dt_cu<calc::V7, ETYP, CFLX>(uind, wa, wb);
   else if (vers == calc::v8)
      epolar1Dt_cu<calc::V8, ETYP, CFLX>(uind, wa, wb);
   else if (vers == calc::v9)
      epolar1Dt_cu<calc::V9, ETYP, CFLX>(uind, wa, wb);
   else if (vers == calc::v10)
      epolar1Dt_cu<calc::V10, ETYP, CFLX>(uind, wa, wb);
}

void epolar1NonEwaldDt_cu(int vers, int use_cf, const real (*uind)[3], real wa, real wb)
{
   if (use_cf)
      epolar1DtVers_cu<NON_EWALD, 1>(vers, uind, wa, wb);
   else
      epolar1DtVers_cu<NON_EWALD, 0>(vers, uind, wa, wb);
}

void epolar1EwaldRealDt_cu(int vers, int use_cf, const real (*uind)[3], real wa, real wb)
{
   if (use_cf)
      epolar1DtVers_cu<EWALD, 1>(vers, uind, wa, wb);
   else
      epolar1DtVers_cu<EWALD, 0>(vers, uind, wa, wb);
}
}
