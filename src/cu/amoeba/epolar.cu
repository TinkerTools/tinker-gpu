#include "ff/cumodamoeba.h"
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

void epolar0DotProd_cu(const real (*gpu_uind)[3], const real (*gpu_udirp)[3])
{
   const real f = -0.5 * electric / dielec;
   launch_k1b(g::s0, n, epolar0DotProd_cu1, n, f, ep, gpu_uind, gpu_udirp, polarity_inv);
}

__global__
static void epolarPairwiseExtfield_cu1(EnergyBuffer restrict ep, const real (*uind)[3], int n, real f, real ex1,
   real ex2, real ex3)
{
   int ithread = ITHREAD;
   for (int i = ithread; i < n; i += STRIDE) {
      real e = uind[i][0] * ex1 + uind[i][1] * ex2 + uind[i][2] * ex3;
      atomic_add(f * e, ep, ithread);
   }
}

void epolarPairwiseExtfield_cu(const real (*uind)[3])
{
   const real f = -0.5 * electric / dielec;
   real ex1 = extfld::exfld[0];
   real ex2 = extfld::exfld[1];
   real ex3 = extfld::exfld[2];
   launch_k1b(g::s0, n, epolarPairwiseExtfield_cu1, ep, uind, n, f, ex1, ex2, ex3);
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
