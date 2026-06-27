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

namespace tinker {
#include "epolarChgpen_cu1.cc"

template <class Ver, class ETYP, int CFLX>
static void epolarChgpen_cu(const real (*uind)[3])
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
   epolarChgpen_cu1<Ver, ETYP, CFLX><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, nep, ep, vir_ep, depx,
      depy, depz, off, st.si1.bit0, nmdwexclude, mdwexclude, mdwexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl,
      st.iakpl, st.niak, st.iak, st.lst, ufld, dufld, uind, pot, rpole, pcore, pval, palpha, aewald, f);

   // torque
   if CONSTEXPR (do_g) {
      launch_k1s(g::s0, n, epolarTorque_cu, //
         trqx, trqy, trqz, n, rpole, ufld, dufld);
   }
}

void epolarChgpenNonEwald_cu(int vers, int use_cf, const real (*uind)[3])
{
   if (use_cf) {
      if (vers == calc::v0) {
         // epolarChgpen_cu<calc::V0, NON_EWALD, 1>(uind);
         assert(false && "CFLX must compute gradient.");
      } else if (vers == calc::v1) {
         epolarChgpen_cu<calc::V1, NON_EWALD, 1>(uind);
      } else if (vers == calc::v3) {
         // epolarChgpen_cu<calc::V3, NON_EWALD, 1>(uind);
         assert(false && "CFLX must compute gradient.");
      } else if (vers == calc::v4) {
         epolarChgpen_cu<calc::V4, NON_EWALD, 1>(uind);
      } else if (vers == calc::v5) {
         epolarChgpen_cu<calc::V5, NON_EWALD, 1>(uind);
      } else if (vers == calc::v6) {
         epolarChgpen_cu<calc::V6, NON_EWALD, 1>(uind);
      }
   } else {
      if (vers == calc::v0) {
         epolarChgpen_cu<calc::V0, NON_EWALD, 0>(uind);
      } else if (vers == calc::v1) {
         epolarChgpen_cu<calc::V1, NON_EWALD, 0>(uind);
      } else if (vers == calc::v3) {
         epolarChgpen_cu<calc::V3, NON_EWALD, 0>(uind);
      } else if (vers == calc::v4) {
         epolarChgpen_cu<calc::V4, NON_EWALD, 0>(uind);
      } else if (vers == calc::v5) {
         epolarChgpen_cu<calc::V5, NON_EWALD, 0>(uind);
      } else if (vers == calc::v6) {
         epolarChgpen_cu<calc::V6, NON_EWALD, 0>(uind);
      }
   }
}

void epolarChgpenEwaldReal_cu(int vers, int use_cf, const real (*uind)[3])
{
   if (use_cf) {
      if (vers == calc::v0) {
         // epolarChgpen_cu<calc::V0, EWALD, 1>(uind);
         assert(false && "CFLX must compute gradient.");
      } else if (vers == calc::v1) {
         epolarChgpen_cu<calc::V1, EWALD, 1>(uind);
      } else if (vers == calc::v3) {
         // epolarChgpen_cu<calc::V3, EWALD, 1>(uind);
         assert(false && "CFLX must compute gradient.");
      } else if (vers == calc::v4) {
         epolarChgpen_cu<calc::V4, EWALD, 1>(uind);
      } else if (vers == calc::v5) {
         epolarChgpen_cu<calc::V5, EWALD, 1>(uind);
      } else if (vers == calc::v6) {
         epolarChgpen_cu<calc::V6, EWALD, 1>(uind);
      }
   } else {
      if (vers == calc::v0) {
         epolarChgpen_cu<calc::V0, EWALD, 0>(uind);
      } else if (vers == calc::v1) {
         epolarChgpen_cu<calc::V1, EWALD, 0>(uind);
      } else if (vers == calc::v3) {
         epolarChgpen_cu<calc::V3, EWALD, 0>(uind);
      } else if (vers == calc::v4) {
         epolarChgpen_cu<calc::V4, EWALD, 0>(uind);
      } else if (vers == calc::v5) {
         epolarChgpen_cu<calc::V5, EWALD, 0>(uind);
      } else if (vers == calc::v6) {
         epolarChgpen_cu<calc::V6, EWALD, 0>(uind);
      }
   }
}
}

namespace tinker {
#include "epolarAplus_cu1.cc"

template <class Ver, class ETYP, int CFLX>
static void epolarAplus_cu(const real (*uind)[3])
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

   if CONSTEXPR (do_g) {
      darray::zero(g::q0, n, ufld, dufld);
   }
   int ngrid = gpuGridSize(BLOCK_DIM);
   epolarAplus_cu1<Ver, ETYP, CFLX><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, nep, ep, vir_ep, depx,
      depy, depz, off, st.si2.bit0, nmdpuexclude, mdpuexclude, mdpuexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl,
      st.iakpl, st.niak, st.iak, st.lst, ufld, dufld, uind, pot, rpole, pdamp, thole, dirdamp, aewald, f);

   // torque
   if CONSTEXPR (do_g) {
      launch_k1s(g::s0, n, epolarTorque_cu, //
         trqx, trqy, trqz, n, rpole, ufld, dufld);
   }
}

void epolarAplusNonEwald_cu(int vers, int use_cf, const real (*uind)[3])
{
   if (use_cf) {
      if (vers == calc::v0) {
         epolarAplus_cu<calc::V0, NON_EWALD, 1>(uind);
      } else if (vers == calc::v1) {
         epolarAplus_cu<calc::V1, NON_EWALD, 1>(uind);
      } else if (vers == calc::v3) {
         epolarAplus_cu<calc::V3, NON_EWALD, 1>(uind);
      } else if (vers == calc::v4) {
         epolarAplus_cu<calc::V4, NON_EWALD, 1>(uind);
      } else if (vers == calc::v5) {
         epolarAplus_cu<calc::V5, NON_EWALD, 1>(uind);
      } else if (vers == calc::v6) {
         epolarAplus_cu<calc::V6, NON_EWALD, 1>(uind);
      }
   } else {
      if (vers == calc::v0) {
         epolarAplus_cu<calc::V0, NON_EWALD, 0>(uind);
      } else if (vers == calc::v1) {
         epolarAplus_cu<calc::V1, NON_EWALD, 0>(uind);
      } else if (vers == calc::v3) {
         epolarAplus_cu<calc::V3, NON_EWALD, 0>(uind);
      } else if (vers == calc::v4) {
         epolarAplus_cu<calc::V4, NON_EWALD, 0>(uind);
      } else if (vers == calc::v5) {
         epolarAplus_cu<calc::V5, NON_EWALD, 0>(uind);
      } else if (vers == calc::v6) {
         epolarAplus_cu<calc::V6, NON_EWALD, 0>(uind);
      }
   }
}

void epolarAplusEwaldReal_cu(int vers, int use_cf, const real (*uind)[3])
{
   if (use_cf) {
      if (vers == calc::v0) {
         epolarAplus_cu<calc::V0, EWALD, 1>(uind);
         // assert(false && "CFLX must compute gradient.");
      } else if (vers == calc::v1) {
         epolarAplus_cu<calc::V1, EWALD, 1>(uind);
      } else if (vers == calc::v3) {
         epolarAplus_cu<calc::V3, EWALD, 1>(uind);
         // assert(false && "CFLX must compute gradient.");
      } else if (vers == calc::v4) {
         epolarAplus_cu<calc::V4, EWALD, 1>(uind);
      } else if (vers == calc::v5) {
         epolarAplus_cu<calc::V5, EWALD, 1>(uind);
      } else if (vers == calc::v6) {
         epolarAplus_cu<calc::V6, EWALD, 1>(uind);
      }
   } else {
      if (vers == calc::v0) {
         epolarAplus_cu<calc::V0, EWALD, 0>(uind);
      } else if (vers == calc::v1) {
         epolarAplus_cu<calc::V1, EWALD, 0>(uind);
      } else if (vers == calc::v3) {
         epolarAplus_cu<calc::V3, EWALD, 0>(uind);
      } else if (vers == calc::v4) {
         epolarAplus_cu<calc::V4, EWALD, 0>(uind);
      } else if (vers == calc::v5) {
         epolarAplus_cu<calc::V5, EWALD, 0>(uind);
      } else if (vers == calc::v6) {
         epolarAplus_cu<calc::V6, EWALD, 0>(uind);
      }
   }
}
}
