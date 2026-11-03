#include "ff/dlmda.h"
#include "ff/evdw.h"
#include "ff/image.h"
#include "ff/potent.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "math/switch.h"
#include "seq/add.h"
#include "seq/launch.h"
#include "seq/pair_lj.h"
#include "seq/triangle.h"
#include <type_traits>

namespace tinker {
#include "elj_cu1.cc"

// The vdw14 pairs, which the pair kernel took with the plain radius and well
// depth scaled by v4scale: the difference the vdw14 ones make, at the same soft
// core and with its lambda derivatives (elj1.f). It counts no interactions, the
// pair kernel having counted these pairs.
template <class Ver>
__global__
void elj_cu2(EnergyBuffer restrict ebuf, EnergyBuffer restrict dlbuf,
   EnergyBuffer restrict d2lbuf, VirialBuffer restrict vbuf, VirialBuffer restrict dlvbuf, grad_prec* restrict gx,
   grad_prec* restrict gy, grad_prec* restrict gz, grad_prec* restrict dlx, grad_prec* restrict dly,
   grad_prec* restrict dlz, TINKER_IMAGE_PARAMS, real cut, real off, const real* restrict x, const real* restrict y,
   const real* restrict z, //
   int njvdw, const real* restrict radmin, const real* restrict epsilon, const int* restrict jvdw,
   const int* restrict mutg, int mutcls, SoftCoreLmda lp1, SoftCoreLmda lp2, //
   real v4scale, int nvdw14, const int (*restrict vdw14ik)[2], const real* restrict radmin4,
   const real* restrict epsilon4)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;
   constexpr bool do_dl1 = Ver::e_dlmda1;
   constexpr bool do_dl2 = Ver::e_dlmda2;
   constexpr bool do_gdl = Ver::g_dlmda;
   constexpr bool do_vdl = Ver::v_dlmda;

   using ebuf_prec = EnergyBufferTraits::type;
   using vbuf_prec = VirialBufferTraits::type;
   ebuf_prec etl, dl1tl, dl2tl;
   vbuf_prec vtlxx, vtlyx, vtlzx, vtlyy, vtlzy, vtlzz;
   vbuf_prec dvtlxx, dvtlyx, dvtlzx, dvtlyy, dvtlzy, dvtlzz;
   if CONSTEXPR (do_e) {
      etl = 0;
   }
   if CONSTEXPR (do_dl1) {
      dl1tl = 0;
   }
   if CONSTEXPR (do_dl2) {
      dl2tl = 0;
   }
   if CONSTEXPR (do_v) {
      vtlxx = 0;
      vtlyx = 0;
      vtlzx = 0;
      vtlyy = 0;
      vtlzy = 0;
      vtlzz = 0;
   }
   if CONSTEXPR (do_vdl) {
      dvtlxx = 0;
      dvtlyx = 0;
      dvtlzx = 0;
      dvtlyy = 0;
      dvtlzy = 0;
      dvtlzz = 0;
   }

   int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   for (int ii = ithread; ii < nvdw14; ii += blockDim.x * gridDim.x) {
      int i = vdw14ik[ii][0];
      int k = vdw14ik[ii][1];

      int mc = pair_mutcls(mutcls, mutg[i], mutg[k]);
      if (mc < 0)
         continue;

      int jit = jvdw[i];
      real xi = x[i];
      real yi = y[i];
      real zi = z[i];

      int jkt = jvdw[k];
      real xr = xi - x[k];
      real yr = yi - y[k];
      real zr = zi - z[k];

      real r2 = image2(xr, yr, zr);
      if (r2 > off * off)
         continue;

      int pos = jit * njvdw + jkt;
      real rv = radmin[pos];
      real eps = epsilon[pos];
      real rv4 = radmin4[pos];
      real eps4 = epsilon4[pos];

      real r = REAL_SQRT(r2);
      real invr = REAL_RECIP(r);

      const bool mutik = mc > 0;
      const SoftCoreLmda lp = pair_vdw_softcore<true>(mc, lp1, lp2);
      real e, de, e4, de4;
      PairVdwLambda dl = {0, 0, 0}, dl4 = {0, 0, 0};
      pair_lj_v3<do_g, true, 0, do_dl1, do_dl2, do_gdl>(r, invr, lp, v4scale, rv, eps, cut, off, e, de, &dl, mutik);
      pair_lj_v3<do_g, true, 0, do_dl1, do_dl2, do_gdl>(
         r, invr, lp, v4scale, rv4, eps4, cut, off, e4, de4, &dl4, mutik);
      e = e4 - e;
      if CONSTEXPR (do_g)
         de = de4 - de;

      if CONSTEXPR (do_e) {
         // if CONSTEXPR (do_a) {}
         etl += floatTo<ebuf_prec>(e);
      }
      if (mutik) {
         if CONSTEXPR (do_dl1)
            dl1tl += floatTo<ebuf_prec>(dl4.dedl - dl.dedl);
         if CONSTEXPR (do_dl2)
            dl2tl += floatTo<ebuf_prec>(dl4.d2edl2 - dl.d2edl2);
      }
      // Two sites at the same point have no direction to push along.
      real rinv = pair_lj_rinv(invr);
      if CONSTEXPR (do_g) {
         de *= rinv;
         real dedx = de * xr;
         real dedy = de * yr;
         real dedz = de * zr;
         if CONSTEXPR (do_v) {
            vtlxx += floatTo<vbuf_prec>(xr * dedx);
            vtlyx += floatTo<vbuf_prec>(yr * dedx);
            vtlzx += floatTo<vbuf_prec>(zr * dedx);
            vtlyy += floatTo<vbuf_prec>(yr * dedy);
            vtlzy += floatTo<vbuf_prec>(zr * dedy);
            vtlzz += floatTo<vbuf_prec>(zr * dedz);
         }
         atomic_add(dedx, gx, i);
         atomic_add(dedy, gy, i);
         atomic_add(dedz, gz, i);
         atomic_add(-dedx, gx, k);
         atomic_add(-dedy, gy, k);
         atomic_add(-dedz, gz, k);
      }
      if CONSTEXPR (do_gdl) {
         if (mutik) {
            real dlde = (dl4.dlde - dl.dlde) * rinv;
            real dldedx = dlde * xr;
            real dldedy = dlde * yr;
            real dldedz = dlde * zr;
            atomic_add(dldedx, dlx, i);
            atomic_add(dldedy, dly, i);
            atomic_add(dldedz, dlz, i);
            atomic_add(-dldedx, dlx, k);
            atomic_add(-dldedy, dly, k);
            atomic_add(-dldedz, dlz, k);
            if CONSTEXPR (do_vdl) {
               dvtlxx += floatTo<vbuf_prec>(xr * dldedx);
               dvtlyx += floatTo<vbuf_prec>(yr * dldedx);
               dvtlzx += floatTo<vbuf_prec>(zr * dldedx);
               dvtlyy += floatTo<vbuf_prec>(yr * dldedy);
               dvtlzy += floatTo<vbuf_prec>(zr * dldedy);
               dvtlzz += floatTo<vbuf_prec>(zr * dldedz);
            }
         }
      }
   }

   if CONSTEXPR (do_e) {
      atomic_add(etl, ebuf, ithread);
   }
   if CONSTEXPR (do_dl1) {
      atomic_add(dl1tl, dlbuf, ithread);
   }
   if CONSTEXPR (do_dl2) {
      atomic_add(dl2tl, d2lbuf, ithread);
   }
   if CONSTEXPR (do_v) {
      atomic_add(vtlxx, vtlyx, vtlzx, vtlyy, vtlzy, vtlzz, vbuf, ithread);
   }
   if CONSTEXPR (do_vdl) {
      atomic_add(dvtlxx, dvtlyx, dvtlzx, dvtlyy, dvtlzy, dvtlzz, dlvbuf, ithread);
   }
}

template <class Ver>
static void elj_cu4(const SoftCoreLmda& lp1, const SoftCoreLmda& lp2)
{
   const auto& st = *cspatial_v2_unit;
   const real cut = switchCut(Switch::VDW);
   const real off = switchOff(Switch::VDW);
   // A relative calculation labels atoms by ligand group; ligand 2 couples as
   // the complement of vlambda.
   const int* grp = emGroup();
   // only a mutation, or the lambda derivatives, pay for the soft core
   const bool mutated = use(Potent::MUTATE);

   int ngrid = gpuGridSize(BLOCK_DIM);
   boxShapeDispatch(box_shape, [&](auto img) {
      using IMG = decltype(img);
      auto ker = elj_cu1<Ver, IMG, true>;
      if (not mutated)
         ker = elj_cu1<Ver, IMG, softCoreUnmutated(Ver::value)>;
      ker<<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, nev, ev, devdl_buf, d2evdl2_buf, vir_ev,
         dvirdl_buf, devx, devy, devz, dfdlx, dfdly, dfdlz, cut, off, st.si2.bit0, nvexclude, vexclude, vexclude_scale,
         st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst, st.akc, njvdw, radmin, epsilon, jvdw,
         grp, vdwMutclsTable(), lp1, lp2);
   });
}

template <class Ver>
static void elj_cu5(const SoftCoreLmda& lp1, const SoftCoreLmda& lp2)
{
   if (nvdw14 > 0) {
      const auto& st = *cspatial_v2_unit;
      const real cut = switchCut(Switch::VDW);
      const real off = switchOff(Switch::VDW);
      // the analysis version adds no count here, so it takes the plain one
      using V = typename std::conditional<eq<Ver, calc::V3>(), calc::V0, Ver>::type;

      launch_k1b(g::s0, nvdw14, elj_cu2<V>,                                                 //
         ev, devdl_buf, d2evdl2_buf, vir_ev, dvirdl_buf, devx, devy, devz, dfdlx, dfdly, dfdlz, //
         TINKER_IMAGE_ARGS, cut, off, st.x, st.y, st.z,                                     //
         njvdw, radmin, epsilon, jvdw, emGroup(), vdwMutclsTable(), lp1, lp2,               //
         v4scale, nvdw14, vdw14ik, radmin4, epsilon4);
   }
}

template <class Ver>
static void elj_cu5()
{
   // nothing to set up without 1-4 pairs
   if (nvdw14 == 0)
      return;
   SoftCoreLmda lp1, lp2;
   grpSoftCores(ljSoft, lmdaDodl(Ver::value), lp1, lp2);
   elj_cu5<Ver>(lp1, lp2);
}

template <class Ver>
static void elj_cu6()
{
   SoftCoreLmda lp1, lp2;
   grpSoftCores(ljSoft, lmdaDodl(Ver::value), lp1, lp2);
   elj_cu4<Ver>(lp1, lp2);
   elj_cu5<Ver>(lp1, lp2);
}

// One kernel takes both the plain versions and the lambda derivative ones, so a
// run whose van der Waals map is flat takes the plain path (elj1.f).
void elj14_cu(int vers)
{
   if (vers == calc::v0)
      elj_cu5<calc::V0>();
   else if (vers == calc::v1)
      elj_cu5<calc::V1>();
   else if (vers == calc::v3)
      elj_cu5<calc::V3>();
   else if (vers == calc::v4)
      elj_cu5<calc::V4>();
   else if (vers == calc::v5)
      elj_cu5<calc::V5>();
   else if (vers == calc::v6)
      elj_cu5<calc::V6>();
   else if (vers == calc::v7)
      elj_cu5<calc::V7>();
   else if (vers == calc::v8)
      elj_cu5<calc::V8>();
   else if (vers == calc::v9)
      elj_cu5<calc::V9>();
   else if (vers == calc::v10)
      elj_cu5<calc::V10>();
}

void elj_cu(int vers)
{
   if (vers == calc::v0)
      elj_cu6<calc::V0>();
   else if (vers == calc::v1)
      elj_cu6<calc::V1>();
   else if (vers == calc::v3)
      elj_cu6<calc::V3>();
   else if (vers == calc::v4)
      elj_cu6<calc::V4>();
   else if (vers == calc::v5)
      elj_cu6<calc::V5>();
   else if (vers == calc::v6)
      elj_cu6<calc::V6>();
   else if (vers == calc::v7)
      elj_cu6<calc::V7>();
   else if (vers == calc::v8)
      elj_cu6<calc::V8>();
   else if (vers == calc::v9)
      elj_cu6<calc::V9>();
   else if (vers == calc::v10)
      elj_cu6<calc::V10>();
}
}
