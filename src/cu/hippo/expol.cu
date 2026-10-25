#include "ff/dlmda.h"
#include "ff/elec.h"
#include "ff/image.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "seq/add.h"
#include "seq/launch.h"
#include "seq/pair_alterpol.h"
#include "seq/triangle.h"

namespace tinker {
#include "alterpol_cu1.cc"

__global__
static void alterpolInit_cu1(int n, real (*restrict polscale)[3][3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      polscale[i][0][0] = 1.f;
      polscale[i][0][1] = 0.f;
      polscale[i][0][2] = 0.f;
      polscale[i][1][0] = 0.f;
      polscale[i][1][1] = 1.f;
      polscale[i][1][2] = 0.f;
      polscale[i][2][0] = 0.f;
      polscale[i][2][1] = 0.f;
      polscale[i][2][2] = 1.f;
   }
}

__global__
static void alterpolInvert_cu1(int n, real (*restrict polscale)[3][3], real (*restrict polinv)[3][3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      real det;
      real(&ps)[3][3] = polscale[i];
      det = ps[0][0] * (ps[1][1] * ps[2][2] - ps[1][2] * ps[2][1])
         - ps[1][0] * (ps[0][1] * ps[2][2] - ps[2][1] * ps[0][2])
         + ps[2][0] * (ps[0][1] * ps[1][2] - ps[1][1] * ps[0][2]);
      polinv[i][0][0] = (ps[1][1] * ps[2][2] - ps[1][2] * ps[2][1]) / det;
      polinv[i][1][0] = (ps[2][0] * ps[1][2] - ps[1][0] * ps[2][2]) / det;
      polinv[i][2][0] = (ps[1][0] * ps[2][1] - ps[2][0] * ps[1][1]) / det;
      polinv[i][0][1] = (ps[2][1] * ps[0][2] - ps[0][1] * ps[2][2]) / det;
      polinv[i][1][1] = (ps[0][0] * ps[2][2] - ps[2][0] * ps[0][2]) / det;
      polinv[i][2][1] = (ps[0][1] * ps[2][0] - ps[0][0] * ps[2][1]) / det;
      polinv[i][0][2] = (ps[0][1] * ps[1][2] - ps[0][2] * ps[1][1]) / det;
      polinv[i][1][2] = (ps[0][2] * ps[1][0] - ps[0][0] * ps[1][2]) / det;
      polinv[i][2][2] = (ps[0][0] * ps[1][1] - ps[0][1] * ps[1][0]) / det;
   }
}

void alterpol_cu(real (*polscale)[3][3], real (*polinv)[3][3])
{
   const auto& st = *mspatial_v2_unit;
   real cut = switchCut(Switch::REPULS);
   real off = switchOff(Switch::REPULS);

   launch_k1s(g::s0, n, alterpolInit_cu1, //
      n, polscale);

   int ngrid = gpuGridSize(BLOCK_DIM);
#define ALTERPOL_CU1_ARGS                                                                                             \
   n, TINKER_IMAGE_ARGS, cut, off, st.si1.bit0, nmdwexclude, mdwexclude, mdwexclude_scale, st.x, st.y, st.z,          \
      st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst, reinterpret_cast<real(*)[9]>(polscale), kpep, prepep,   \
      dmppep, lpep, scrtyp, reinterpret_cast<real(*)[9]>(dpscdl), prepeporig, emGroup(), grpScale(plam)
   if (expol_mutpsc) {
      // the single topology lambda derivative also wants the overlap with the
      // charging ligand (epolar4.f:epolar4s)
      darray::zero(g::q0, n, dpscdl);
      alterpol_cu1<1><<<ngrid, BLOCK_DIM, 0, g::s0>>>(ALTERPOL_CU1_ARGS);
   } else {
      alterpol_cu1<0><<<ngrid, BLOCK_DIM, 0, g::s0>>>(ALTERPOL_CU1_ARGS);
   }
#undef ALTERPOL_CU1_ARGS

   launch_k1s(g::s0, n, alterpolInvert_cu1, //
      n, polscale, polinv);
}

#include "dexpol_cu1.cc"
void dexpol_cu(int vers, const real (*uind)[3], grad_prec* depx, grad_prec* depy, grad_prec* depz, VirialBuffer vir_ep,
   real wa, grad_prec* dgx, grad_prec* dgy, grad_prec* dgz, VirialBuffer dvir, real wb)
{
   const auto& st = *mspatial_v2_unit;
   real cut = switchCut(Switch::REPULS);
   real off = switchOff(Switch::REPULS);

   real f = 0.5f * electric / dielec;

   int ngrid = gpuGridSize(BLOCK_DIM);

#define DEXPOL_CU1_ARGS                                                                                               \
   n, TINKER_IMAGE_ARGS, vir_ep, dvir, depx, depy, depz, dgx, dgy, dgz, cut, off, st.si1.bit0, nmdwexclude,         \
      mdwexclude, mdwexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst,        \
      polarity, uind, kpep, prepep, dmppep, lpep, scrtyp, f, wa, wb

   // the lambda gradient comes with the lambda virial whenever the virial is
   // on (lmdaDerivMask), so V9 and V10 cover both
   const bool do_gdl = vers & calc::grad_dlmda;
   assert(not do_gdl or (bool)(vers & calc::virial) == (bool)(vers & calc::virial_dlmda));
   if ((vers & calc::virial) and do_gdl) {
      dexpol_cu1<calc::V9><<<ngrid, BLOCK_DIM, 0, g::s0>>>(DEXPOL_CU1_ARGS);
   } else if (vers & calc::virial) {
      dexpol_cu1<calc::V6><<<ngrid, BLOCK_DIM, 0, g::s0>>>(DEXPOL_CU1_ARGS);
   } else if ((vers & calc::grad) and do_gdl) {
      dexpol_cu1<calc::V10><<<ngrid, BLOCK_DIM, 0, g::s0>>>(DEXPOL_CU1_ARGS);
   } else if (vers & calc::grad) {
      dexpol_cu1<calc::V5><<<ngrid, BLOCK_DIM, 0, g::s0>>>(DEXPOL_CU1_ARGS);
   } else {
      assert(false && "This function should not have been called if neither gradient nor virial is calculated.");
   }

#undef DEXPOL_CU1_ARGS
}
}
