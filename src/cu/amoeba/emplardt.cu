#include "ff/amoeba/empole.h"
#include "ff/amoeba/epolar.h"
#include "ff/cumodamoeba.h"
#include "ff/dlmda.h"
#include "ff/elec.h"
#include "ff/image.h"
#include "ff/modamoeba.h"
#include "ff/pme.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "seq/damp.h"
#include "seq/emselfamoeba.h"
#include "seq/launch.h"
#include "seq/pair_mplar.h"
#include "seq/triangle.h"

// One dual topology subsystem of the fused multipole/polarization kernel. The
// pass masks rpole and polarity before it runs, so every interaction shares one
// set of weights and the accumulators are scaled where they are flushed rather
// than pair by pair. The caller adds every pass into the global buffers and
// converts the torque once at the end.
//
// The energy here is the permanent multipole part only, self term included; the
// polarization energy and its lambda derivatives come from the induced dipole
// dot product, as they do in the ordinary emplar path.

namespace tinker {
#include "emplardt_cu1.cc"

template <class Ver>
__global__
static void emplarSelfDt_cu(EnergyBuffer restrict em, EnergyBuffer restrict demdl,
   EnergyBuffer restrict d2emdl2, const real (*restrict rpole)[10], int n, real f, real aewald, real wa,
   real wb, real wc)
{
   constexpr bool do_dl1 = Ver::e_dlmda1;
   constexpr bool do_dl2 = Ver::e_dlmda2;

   real aewald_sq_2 = 2 * aewald * aewald;
   real fterm = -f * aewald * 0.5f * (real)(M_2_SQRTPI);

   // rpole is masked down to this subsystem, so an atom outside it carries no
   // multipole and contributes nothing.
   for (int i = ITHREAD; i < n; i += STRIDE) {
      int offset = ITHREAD;
      real e = empoleSelfEnergyAtomI(i, rpole, fterm, aewald_sq_2);
      atomic_add(wa * e, em, offset);
      if CONSTEXPR (do_dl1)
         atomic_add(wb * e, demdl, offset);
      if CONSTEXPR (do_dl2)
         atomic_add(wc * e, d2emdl2, offset);
   }
}

template <class Ver, class ETYP>
static void emplardt_cu(const real (*uind)[3], const real (*uinp)[3], real wa, real wb, real wc)
{
   const auto& st = *mspatial_v2_unit;
   real off;
   if CONSTEXPR (eq<ETYP, EWALD>())
      off = switchOff(Switch::EWALD);
   else
      off = switchOff(Switch::MPOLE);

   const real f = electric / dielec;
   real aewald = 0;
   if CONSTEXPR (eq<ETYP, EWALD>()) {
      assert(epme_unit == ppme_unit);
      PMEUnit pu = epme_unit;
      aewald = pu->aewald;

      if CONSTEXPR (Ver::e) {
         launch_k1b(g::s0, n, emplarSelfDt_cu<Ver>, //
            em, demdl_buf, d2emdl2_buf, rpole, n, f, aewald, wa, wb, wc);
      }
   }

   int ngrid = gpuGridSize(BLOCK_DIM);
   auto kera = emplardt_cu1a<Ver, ETYP>;
   kera<<<ngrid, BLOCK_DIM, 0, g::s0>>>(TINKER_IMAGE_ARGS, em, demdl_buf, d2emdl2_buf, vir_em, dvirdl_buf,
      demx, demy, demz, dfdlx, dfdly, dfdlz, off, trqx, trqy, trqz, dltrqx, dltrqy, dltrqz, rpole, uind, uinp,
      f, aewald, wa, wb, wc, //
      st.sorted, st.niak, st.iak, st.lst);
   auto kerb = emplardt_cu1b<Ver, ETYP>;
   kerb<<<ngrid, BLOCK_DIM, 0, g::s0>>>(TINKER_IMAGE_ARGS, em, demdl_buf, d2emdl2_buf, vir_em, dvirdl_buf,
      demx, demy, demz, dfdlx, dfdly, dfdlz, off, trqx, trqy, trqz, dltrqx, dltrqy, dltrqz, rpole, uind, uinp,
      f, aewald, wa, wb, wc, //
      st.sorted, st.n, st.nakpl, st.iakpl);
   auto kerc = emplardt_cu1c<Ver, ETYP>;
   kerc<<<ngrid, BLOCK_DIM, 0, g::s0>>>(TINKER_IMAGE_ARGS, em, demdl_buf, d2emdl2_buf, vir_em, dvirdl_buf,
      demx, demy, demz, dfdlx, dfdly, dfdlz, off, trqx, trqy, trqz, dltrqx, dltrqy, dltrqz, rpole, uind, uinp,
      f, aewald, wa, wb, wc, //
      nmdpuexclude, mdpuexclude, mdpuexclude_scale, st.x, st.y, st.z);
}

template <class ETYP>
static void emplardtVers_cu(int vers, const real (*uind)[3], const real (*uinp)[3], real wa, real wb, real wc)
{
   // calc::v3 is unreachable: emplarDecide() turns emplar down under analysis.
   if (vers == calc::v0)
      emplardt_cu<calc::V0, ETYP>(uind, uinp, wa, wb, wc);
   else if (vers == calc::v1)
      emplardt_cu<calc::V1, ETYP>(uind, uinp, wa, wb, wc);
   else if (vers == calc::v4)
      emplardt_cu<calc::V4, ETYP>(uind, uinp, wa, wb, wc);
   else if (vers == calc::v5)
      emplardt_cu<calc::V5, ETYP>(uind, uinp, wa, wb, wc);
   else if (vers == calc::v6)
      emplardt_cu<calc::V6, ETYP>(uind, uinp, wa, wb, wc);
   else if (vers == calc::v7)
      emplardt_cu<calc::V7, ETYP>(uind, uinp, wa, wb, wc);
   else if (vers == calc::v8)
      emplardt_cu<calc::V8, ETYP>(uind, uinp, wa, wb, wc);
   else if (vers == calc::v9)
      emplardt_cu<calc::V9, ETYP>(uind, uinp, wa, wb, wc);
   else if (vers == calc::v10)
      emplardt_cu<calc::V10, ETYP>(uind, uinp, wa, wb, wc);
}

void emplarDt_cu(int vers, real wa, real wb, real wc)
{
   if (useEwald())
      emplardtVers_cu<EWALD>(vers, uind, uinp, wa, wb, wc);
   else
      emplardtVers_cu<NON_EWALD>(vers, uind, uinp, wa, wb, wc);
}
}
