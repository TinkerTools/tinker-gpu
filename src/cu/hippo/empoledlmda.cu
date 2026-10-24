#include "ff/amoeba/mpolestate.h"
#include "ff/dlmda.h"
#include "ff/elec.h"
#include "ff/image.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
#include "ff/pme.h"
#include "ff/potent.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "seq/emselfhippo.h"
#include "seq/launch.h"
#include "seq/pair_mpole_chgpen.h"
#include "seq/pairmpoleaplus.h"
#include "seq/triangle.h"
#include <cassert>

namespace tinker {
#include "empoleChgpenDlmda_cu1.cc"

// Charge penetration multipoles scaled on the fly from their unscaled values,
// with their lambda derivatives (empole4.f). Under charge flux the valence
// charges carry the unscaled flux change, and with CFLX the kernels gather the
// flux potentials of the scaled multipoles and of their lambda derivative.
template <class Ver, class ETYP, Chgpen CP, int CFLX>
static void empoleChgpenDlmda_cu()
{
   const auto& st = *mspatial_v2_unit;
   real off;
   if CONSTEXPR (eq<ETYP, EWALD>())
      off = switchOff(Switch::EWALD);
   else
      off = switchOff(Switch::MPOLE);

   const real f = electric / dielec;
   const int* grp = emGroup();
   const GrpScale esc = grpScale(elam);
   const GrpPairScale ps = elecPairScale();
   real* dlp = Ver::g_dlmda ? dlpot : nullptr;
   const real* dq = use(Potent::CHGFLX) ? pdelta : nullptr;

   real aewald = 0;
   if CONSTEXPR (eq<ETYP, EWALD>()) {
      PMEUnit pu = epme_unit;
      aewald = pu->aewald;
      launch_k1b(g::s0, n, empoleChgpenSelfDlmda_cu<Ver, CFLX>, //
         nem, em, demdl_buf, d2emdl2_buf, rpole, grp, esc, grpSelfScale(ps), pot, dlp, n, f, aewald);
   }

   int ngrid = gpuGridSize(BLOCK_DIM);
   empoleChgpenDlmda_cu1<Ver, ETYP, CP, CFLX><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, nem, em,
      demdl_buf, d2emdl2_buf, vir_em, dvirdl_buf, demx, demy, demz, dfdlx, dfdly, dfdlz, off, st.si1.bit0, nmdwexclude,
      mdwexclude, mdwexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst, trqx,
      trqy, trqz, dltrqx, dltrqy, dltrqz, pot, dlp, rpole, pcoreorig, pvalorig, dq, palpha, grp, aewald, f, esc, ps);
}

// The flux potentials are gathered with the gradient only, so CFLX leaves out
// the energy-only versions v0 and v3.
template <class ETYP, Chgpen CP, int CFLX>
static void empoleChgpenDlmda_cu2(int vers)
{
   if (vers == calc::v0 or vers == calc::v3) {
      if CONSTEXPR (CFLX)
         assert(false && "CFLX must compute gradient.");
      else if (vers == calc::v0)
         empoleChgpenDlmda_cu<calc::V0, ETYP, CP, 0>();
      else
         empoleChgpenDlmda_cu<calc::V3, ETYP, CP, 0>();
   } else if (vers == calc::v1)
      empoleChgpenDlmda_cu<calc::V1, ETYP, CP, CFLX>();
   else if (vers == calc::v4)
      empoleChgpenDlmda_cu<calc::V4, ETYP, CP, CFLX>();
   else if (vers == calc::v5)
      empoleChgpenDlmda_cu<calc::V5, ETYP, CP, CFLX>();
   else if (vers == calc::v6)
      empoleChgpenDlmda_cu<calc::V6, ETYP, CP, CFLX>();
   else if (vers == calc::v7)
      empoleChgpenDlmda_cu<calc::V7, ETYP, CP, CFLX>();
   else if (vers == calc::v8)
      empoleChgpenDlmda_cu<calc::V8, ETYP, CP, CFLX>();
   else if (vers == calc::v9)
      empoleChgpenDlmda_cu<calc::V9, ETYP, CP, CFLX>();
   else if (vers == calc::v10)
      empoleChgpenDlmda_cu<calc::V10, ETYP, CP, CFLX>();
}

template <class ETYP>
static void empoleChgpenDlmda_cu3(int vers, int use_cf)
{
   if (pentyp == Chgpen::GORDON1) {
      if (use_cf)
         empoleChgpenDlmda_cu2<ETYP, Chgpen::GORDON1, 1>(vers);
      else
         empoleChgpenDlmda_cu2<ETYP, Chgpen::GORDON1, 0>(vers);
   } else if (pentyp == Chgpen::GORDON2) {
      if (use_cf)
         empoleChgpenDlmda_cu2<ETYP, Chgpen::GORDON2, 1>(vers);
      else
         empoleChgpenDlmda_cu2<ETYP, Chgpen::GORDON2, 0>(vers);
   }
}

void empoleChgpenDlmdaNonEwald_cu(int vers, int use_cf)
{
   empoleChgpenDlmda_cu3<NON_EWALD>(vers, use_cf);
}

void empoleChgpenDlmdaEwaldRealSelf_cu(int vers, int use_cf)
{
   empoleChgpenDlmda_cu3<EWALD>(vers, use_cf);
}
}
