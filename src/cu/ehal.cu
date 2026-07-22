#include "ff/dlmda.h"
#include "ff/evdw.h"
#include "ff/image.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "math/switch.h"
#include "seq/add.h"
#include "seq/launch.h"
#include "seq/pair_hal.h"
#include "seq/triangle.h"

namespace tinker {
__global__
void ehalReduceXyz_cu1(int n, const int* restrict ired, const real* restrict kred, const real* restrict x,
   const real* restrict y, const real* restrict z, real* restrict xred, real* restrict yred, real* restrict zred)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      auto iv = ired[i];
      auto rdn = kred[i];
      auto xi = x[i], xiv = x[iv];
      auto yi = y[i], yiv = y[iv];
      auto zi = z[i], ziv = z[iv];
      xred[i] = rdn * (xi - xiv) + xiv;
      yred[i] = rdn * (yi - yiv) + yiv;
      zred[i] = rdn * (zi - ziv) + ziv;
   }
}

void ehalReduceXyz_cu()
{
   launch_k1s(g::s0, n, ehalReduceXyz_cu1, n, ired, kred, x, y, z, xred, yred, zred);
}

__global__
void ehalResolveGradient_cu1(int n, const int* restrict ired, const real* restrict kred,
   const grad_prec* restrict gxred, const grad_prec* restrict gyred, const grad_prec* restrict gzred,
   grad_prec* restrict devx, grad_prec* restrict devy, grad_prec* restrict devz)
{
   for (int ii = ITHREAD; ii < n; ii += STRIDE) {
      auto grx = gxred[ii];
      auto gry = gyred[ii];
      auto grz = gzred[ii];
      auto iv = ired[ii];
      if (ii == iv) {
         atomic_add(grx, devx, ii);
         atomic_add(gry, devy, ii);
         atomic_add(grz, devz, ii);
      } else {
         auto fx = toFloatGrad<real>(grx);
         auto fy = toFloatGrad<real>(gry);
         auto fz = toFloatGrad<real>(grz);
         auto redii = kred[ii];
         auto rediv = 1 - redii;
         atomic_add(fx * redii, devx, ii);
         atomic_add(fy * redii, devy, ii);
         atomic_add(fz * redii, devz, ii);
         atomic_add(fx * rediv, devx, iv);
         atomic_add(fy * rediv, devy, iv);
         atomic_add(fz * rediv, devz, iv);
      }
   }
}

void ehalResolveGradient_cu(const grad_prec* gxred, const grad_prec* gyred, const grad_prec* gzred, grad_prec* devx,
   grad_prec* devy, grad_prec* devz)
{
   launch_k1s(g::s0, n, ehalResolveGradient_cu1, //
      n, ired, kred, gxred, gyred, gzred, devx, devy, devz);
}
}

/**
 * Overheads:
 *    - Different vcouple methods.
 *    - PBC type in image().
 *    - Random access to the "i" parameters and gradients.
 *    - (If not hard-coded) ghal, dhal, scexp, scalpha.
 */

/**
 * Kernel ehal_cu1 on GTX 1070 for DHFR2
 * 7 angstroms cutoff and 10 % buffer
 *
 * unsorted (a) | generic image (b) | decouple vlambda (c) | 1e-6 s
 * --------------------------------------------------------------------
 * -            | -                 | -                    | 342
 * +            | -                 | -                    | 348
 * +            | -                 | +                    | 366
 * +            | +                 | -                    | 373
 * +            | +                 | +                    | 382
 * --------------------------------------------------------------------
 * (a) - assuming sorted[i].unsorted == i, no random memory access for the "i"
 *     parameters and gradients; + the original code.
 * (b) - hard-coded orthogonal image routine; + generic image routine.
 * (c) - hard-coded decouple method; + generic vlambda method.
 */

namespace tinker {
#if 1
#define GHAL    (real)0.12
#define DHAL    (real)0.07
#define SCEXP   5
#define SCALPHA (real)0.7
#elif 0
#define GHAL    ghal
#define DHAL    dhal
#define SCEXP   scexp
#define SCALPHA scalpha
#endif
#include "ehal_cu1.cc"
#include "ehaldlmda_cu1.cc"

template <class Ver>
static void ehal_cu3()
{
   constexpr bool do_g = Ver::g;

   const auto& st = *vspatial_v2_unit;
   const real cut = switchCut(Switch::VDW);
   const real off = switchOff(Switch::VDW);

   if CONSTEXPR (do_g) {
      darray::zero(g::q0, n, gxred, gyred, gzred);
      if (use_evast)
         darray::zero(g::q0, n, gxred_dlmda, gyred_dlmda, gzred_dlmda);
   }

   int ngrid = gpuGridSize(BLOCK_DIM);
   if (use_evast) {
      auto ker1 = ehaldlmda_cu1<Ver>;
      ker1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, nev, ev, devdl_buf, d2evdl2_buf, vir_ev,
         devvirdl_buf, gxred, gyred, gzred, gxred_dlmda, gyred_dlmda, gzred_dlmda, cut, off, st.si1.bit0, nvexclude,
         vexclude, vexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst, njvdw,
         vlam, vcouple, radmin, epsilon, jvdw, mut);
   } else {
      auto ker1 = ehal_cu1<Ver>;
      ker1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, nev, ev, vir_ev, gxred, gyred, gzred, cut, off,
         st.si1.bit0, nvexclude, vexclude, vexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak,
         st.iak, st.lst, njvdw, vlam, vcouple, radmin, epsilon, jvdw, mut);
   }

   if CONSTEXPR (do_g) {
      ehalResolveGradient(gxred, gyred, gzred, devx, devy, devz);
      if (use_evast)
         ehalResolveGradient(gxred_dlmda, gyred_dlmda, gzred_dlmda, dfvdlx, dfvdly, dfvdlz);
   }
}

void ehal_cu(int vers)
{
   if (vers == calc::v0)
      ehal_cu3<calc::V0>();
   else if (vers == calc::v1)
      ehal_cu3<calc::V1>();
   else if (vers == calc::v3)
      ehal_cu3<calc::V3>();
   else if (vers == calc::v4)
      ehal_cu3<calc::V4>();
   else if (vers == calc::v5)
      ehal_cu3<calc::V5>();
   else if (vers == calc::v6)
      ehal_cu3<calc::V6>();
}

}
