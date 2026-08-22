#include "ff/amoeba/empole.h"
#include "ff/amoeba/epolar.h"
#include "ff/amoeba/induce.h"
#include "ff/cumodamoeba.h"
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

namespace tinker {
#include "emplar_cu1.cc"

template <class Ver, class ETYP>
static void emplar_cu(const real (*uind)[3], const real (*uinp)[3])
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
         auto ker0 = empoleSelf_cu<Ver::a>;
         launch_k1b(g::s0, n, ker0, //
            nullptr, em, rpole, n, f, aewald);
      }
   }
   int ngrid = gpuGridSize(BLOCK_DIM);
   auto kera = emplar_cu1a<Ver, ETYP>;
   kera<<<ngrid, BLOCK_DIM, 0, g::s0>>>(TINKER_IMAGE_ARGS, em, vir_em, demx, demy, demz, off, trqx, trqy, trqz, rpole,
      uind, uinp, f,
      aewald, //
      st.sorted, st.niak, st.iak, st.lst);
   auto kerb = emplar_cu1b<Ver, ETYP>;
   kerb<<<ngrid, BLOCK_DIM, 0, g::s0>>>(TINKER_IMAGE_ARGS, em, vir_em, demx, demy, demz, off, trqx, trqy, trqz, rpole,
      uind, uinp, f,
      aewald, //
      st.sorted, st.n, st.nakpl, st.iakpl);
   auto kerc = emplar_cu1c<Ver, ETYP>;
   kerc<<<ngrid, BLOCK_DIM, 0, g::s0>>>(TINKER_IMAGE_ARGS, em, vir_em, demx, demy, demz, off, trqx, trqy, trqz, rpole,
      uind, uinp, f,
      aewald, //
      nmdpuexclude, mdpuexclude, mdpuexclude_scale, st.x, st.y, st.z);
}

template <class Ver>
static void emplarEwald_cu()
{
   // induce
   induce(uind, uinp);

   // empole real self; epolar real without epolar energy
   emplar_cu<Ver, EWALD>(uind, uinp);
   // empole recip
   empoleEwaldRecip(Ver::value);
   // epolar recip self; must toggle off the calc::energy flag
   AccumRef out = em_buf.ref();
   epolarEwaldRecipSelf(Ver::value & ~calc::energy, out.e, out.v, out.gx, out.gy, out.gz);

   // epolar energy
   if CONSTEXPR (Ver::e)
      epolar0DotProd(uind, udirp, em_buf.ref().e);
}

template <class Ver>
static void emplarNonEwald_cu()
{
   // induce
   induce(uind, uinp);

   // empole and epolar
   emplar_cu<Ver, NON_EWALD>(uind, uinp);
   if CONSTEXPR (Ver::e)
      epolar0DotProd(uind, udirp, em_buf.ref().e);
}

void emplar_cu(int vers)
{
   if (useEwald()) {
      if (vers == calc::v0)
         emplarEwald_cu<calc::V0>();
      else if (vers == calc::v1)
         emplarEwald_cu<calc::V1>();
      // else if (vers == calc::v3)
      //    emplarEwald_cu<calc::V3>();
      else if (vers == calc::v4)
         emplarEwald_cu<calc::V4>();
      else if (vers == calc::v5)
         emplarEwald_cu<calc::V5>();
      else if (vers == calc::v6)
         emplarEwald_cu<calc::V6>();
   } else {
      if (vers == calc::v0)
         emplarNonEwald_cu<calc::V0>();
      else if (vers == calc::v1)
         emplarNonEwald_cu<calc::V1>();
      // else if (vers == calc::v3)
      //    emplarNonEwald_cu<calc::V3>();
      else if (vers == calc::v4)
         emplarNonEwald_cu<calc::V4>();
      else if (vers == calc::v5)
         emplarNonEwald_cu<calc::V5>();
      else if (vers == calc::v6)
         emplarNonEwald_cu<calc::V6>();
   }
}
}
