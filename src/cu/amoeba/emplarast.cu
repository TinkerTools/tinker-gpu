#include "ff/amoeba/empole.h"
#include "ff/amoeba/epolar.h"
#include "ff/cumodamoeba.h"
#include "ff/dlmda.h"
#include "ff/elec.h"
#include "ff/evdw.h"
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

// The absolute single topology flavour of the fused multipole/polarization
// kernel. rpole holds the unscaled multipoles here, and the pair kernel applies
// elambda to the permanent side itself -- that is what lets it hand back the
// permanent energy derivative without dividing by a lambda that may be zero.
// The induced dipoles arrive already solved against the lambda-scaled system,
// so they need no scaling of their own.
//
// The energy is the permanent multipole part only, self term included; the
// polarization energy comes from the induced dipole dot product, as it does in
// the ordinary emplar path, and its lambda derivative from epolarAstDeriv.

namespace tinker {
#include "emplarast_cu1.cc"

template <class Ver, class ETYP>
static void emplarast_cu(const real (*uind)[3], const real (*uinp)[3])
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

      // The self term is one site against itself, so it scales as lambda
      // squared and empoleSelfDlmda_cu already differentiates it.
      if CONSTEXPR (Ver::e) {
         auto ker0 = empoleSelfDlmda_cu<Ver>;
         launch_k1b(g::s0, n, ker0, //
            nullptr, em, demdl_buf, d2emdl2_buf, rpole, mut, n, f, aewald, elam, deldlmda, d2eldlmda2);
      }
   }
   int ngrid = gpuGridSize(BLOCK_DIM);
   auto kera = emplarast_cu1a<Ver, ETYP>;
   kera<<<ngrid, BLOCK_DIM, 0, g::s0>>>(TINKER_IMAGE_ARGS, em, demdl_buf, vir_em, demx, demy, demz, off, trqx, trqy,
      trqz, rpole, uind, uinp, mut, f, aewald, elam, deldlmda, //
      st.sorted, st.niak, st.iak, st.lst);
   auto kerb = emplarast_cu1b<Ver, ETYP>;
   kerb<<<ngrid, BLOCK_DIM, 0, g::s0>>>(TINKER_IMAGE_ARGS, em, demdl_buf, vir_em, demx, demy, demz, off, trqx, trqy,
      trqz, rpole, uind, uinp, mut, f, aewald, elam, deldlmda, //
      st.sorted, st.n, st.nakpl, st.iakpl);
   auto kerc = emplarast_cu1c<Ver, ETYP>;
   kerc<<<ngrid, BLOCK_DIM, 0, g::s0>>>(TINKER_IMAGE_ARGS, em, demdl_buf, vir_em, demx, demy, demz, off, trqx, trqy,
      trqz, rpole, uind, uinp, mut, f, aewald, elam, deldlmda, //
      nmdpuexclude, mdpuexclude, mdpuexclude_scale, st.x, st.y, st.z);
}

template <class ETYP>
static void emplarastVers_cu(int vers, const real (*uind)[3], const real (*uinp)[3])
{
   // calc::v3 is unreachable: emplarDecide() turns emplar down under analysis.
   // v9 and v10 are unreachable too -- use_epast pins the reduced dispatch, and
   // epolarData() refuses OST outright, so v1 and v4 only ever become v7 and v8.
   if (vers == calc::v0)
      emplarast_cu<calc::V0, ETYP>(uind, uinp);
   else if (vers == calc::v1)
      emplarast_cu<calc::V1, ETYP>(uind, uinp);
   else if (vers == calc::v4)
      emplarast_cu<calc::V4, ETYP>(uind, uinp);
   else if (vers == calc::v5)
      emplarast_cu<calc::V5, ETYP>(uind, uinp);
   else if (vers == calc::v6)
      emplarast_cu<calc::V6, ETYP>(uind, uinp);
   else if (vers == calc::v7)
      emplarast_cu<calc::V7, ETYP>(uind, uinp);
   else if (vers == calc::v8)
      emplarast_cu<calc::V8, ETYP>(uind, uinp);
}

void emplarAst_cu(int vers)
{
   if (useEwald())
      emplarastVers_cu<EWALD>(vers, uind, uinp);
   else
      emplarastVers_cu<NON_EWALD>(vers, uind, uinp);
}
}
