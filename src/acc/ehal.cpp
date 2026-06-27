#include "ff/atom.h"
#include "ff/evdw.h"
#include "ff/image.h"
#include "ff/nblist.h"
#include "ff/switch.h"
#include "math/switch.h"
#include "seq/add.h"
#include "seq/pair_hal.h"
#include "tool/gpucard.h"

namespace tinker {
void ehalReduceXyz_acc()
{
   #pragma acc parallel loop independent async\
               deviceptr(x,y,z,ired,kred,xred,yred,zred)
   for (int i = 0; i < n; ++i) {
      int iv = ired[i];
      real rdn = kred[i];
      xred[i] = rdn * (x[i] - x[iv]) + x[iv];
      yred[i] = rdn * (y[i] - y[iv]) + y[iv];
      zred[i] = rdn * (z[i] - z[iv]) + z[iv];
   }
}

void ehalResolveGradient_acc()
{
   #pragma acc parallel loop independent async\
               deviceptr(ired,kred,gxred,gyred,gzred,devx,devy,devz)
   for (int ii = 0; ii < n; ++ii) {
      int iv = ired[ii];
      real fx = toFloatGrad<real>(gxred[ii]);
      real fy = toFloatGrad<real>(gyred[ii]);
      real fz = toFloatGrad<real>(gzred[ii]);
      if (ii == iv) {
         atomic_add(fx, devx, ii);
         atomic_add(fy, devy, ii);
         atomic_add(fz, devz, ii);
      } else {
         real redii = kred[ii];
         real rediv = 1 - redii;
         atomic_add(fx * redii, devx, ii);
         atomic_add(fy * redii, devy, ii);
         atomic_add(fz * redii, devz, ii);
         atomic_add(fx * rediv, devx, iv);
         atomic_add(fy * rediv, devy, iv);
         atomic_add(fz * rediv, devz, iv);
      }
   }
}

#define DEVICE_PTRS xred, yred, zred, gxred, gyred, gzred, jvdw, radmin, epsilon, mut, nev, ev, vir_ev
template <class Ver>
static void ehal_acc1()
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;

   const real cut = switchCut(Switch::VDW);
   const real off = switchOff(Switch::VDW);
   const real cut2 = cut * cut;
   const real off2 = off * off;
   const int maxnlst = vlist_unit->maxnlst;
   const auto* vlst = vlist_unit.deviceptr();

   auto bufsize = bufferSize();

   if CONSTEXPR (do_g)
      darray::zero(g::q0, n, gxred, gyred, gzred);

   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
   #pragma acc parallel async num_gangs(GRID_DIM) vector_length(BLOCK_DIM)\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(DEVICE_PTRS,vlst)
   #pragma acc loop gang independent
   for (int i = 0; i < n; ++i) {
      int it = jvdw[i];
      real xi = xred[i];
      real yi = yred[i];
      real zi = zred[i];
      int mut1 = mut[i];
      MAYBE_UNUSED real gxi = 0, gyi = 0, gzi = 0;

      int nvlsti = vlst->nlst[i];
      #pragma acc loop vector independent reduction(+:gxi,gyi,gzi)
      for (int kk = 0; kk < nvlsti; ++kk) {
         int offset = (kk + i * n) & (bufsize - 1);
         int k = vlst->lst[i * maxnlst + kk];
         int kt = jvdw[k];
         real xr = xi - xred[k];
         real yr = yi - yred[k];
         real zr = zi - zred[k];
         int mut2 = mut[k];

         real vlambda = 1;
         if (vcouple == Vdw::DECOUPLE) {
            vlambda = (mut1 == mut2 ? 1 : vlam);
         }
         if (vcouple == Vdw::ANNIHILATE) {
            vlambda = (mut1 || mut2 ? vlam : 1);
         }

         real rik2 = image2(xr, yr, zr);
         if (rik2 <= off2) {
            real rik = REAL_SQRT(rik2);
            real rv = radmin[it * njvdw + kt];
            real eps = epsilon[it * njvdw + kt];

            MAYBE_UNUSED real e, de;
            pair_hal<do_g>(rik, rv, eps, 1, vlambda, //
               ghal, dhal, scexp, scalpha,           //
               e, de);

            if (rik2 > cut2) {
               real taper, dtaper;
               switchTaper5<do_g>(rik, cut, off, taper, dtaper);
               if CONSTEXPR (do_g)
                  de = e * dtaper + de * taper;
               if CONSTEXPR (do_e)
                  e = e * taper;
            }

            // Increment the energy, gradient, and virial.

            if CONSTEXPR (do_a)
               atomic_add(1, nev, offset);
            if CONSTEXPR (do_e)
               atomic_add(e, ev, offset);
            if CONSTEXPR (do_g) {
               de *= REAL_RECIP(rik);
               real dedx = de * xr;
               real dedy = de * yr;
               real dedz = de * zr;

               gxi += dedx;
               gyi += dedy;
               gzi += dedz;
               atomic_add(-dedx, gxred, k);
               atomic_add(-dedy, gyred, k);
               atomic_add(-dedz, gzred, k);

               if CONSTEXPR (do_v) {
                  real vxx = xr * dedx;
                  real vyx = yr * dedx;
                  real vzx = zr * dedx;
                  real vyy = yr * dedy;
                  real vzy = zr * dedy;
                  real vzz = zr * dedz;

                  atomic_add(vxx, vyx, vzx, vyy, vzy, vzz, vir_ev, offset);
               } // end if (do_v)
            }    // end if (do_g)
         }
      } // end for (int kk)

      if CONSTEXPR (do_g) {
         atomic_add(gxi, gxred, i);
         atomic_add(gyi, gyred, i);
         atomic_add(gzi, gzred, i);
      }
   } // end for (int i)

   #pragma acc parallel async present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(DEVICE_PTRS,vexclude,vexclude_scale)
   #pragma acc loop independent
   for (int ii = 0; ii < nvexclude; ++ii) {
      int offset = ii & (bufsize - 1);

      int i = vexclude[ii][0];
      int k = vexclude[ii][1];
      real vscale = vexclude_scale[ii] - 1;

      int it = jvdw[i];
      real xi = xred[i];
      real yi = yred[i];
      real zi = zred[i];
      int mut1 = mut[i];

      int kt = jvdw[k];
      real xr = xi - xred[k];
      real yr = yi - yred[k];
      real zr = zi - zred[k];
      int mut2 = mut[k];

      real vlambda = 1;
      if (vcouple == Vdw::DECOUPLE) {
         vlambda = (mut1 == mut2 ? 1 : vlam);
      }
      if (vcouple == Vdw::ANNIHILATE) {
         vlambda = (mut1 || mut2 ? vlam : 1);
      }

      real rik2 = image2(xr, yr, zr);
      if (rik2 <= off2) {
         real rik = REAL_SQRT(rik2);
         real rv = radmin[it * njvdw + kt];
         real eps = epsilon[it * njvdw + kt];

         MAYBE_UNUSED real e, de;
         pair_hal<do_g>(rik, rv, eps, vscale, vlambda, //
            ghal, dhal, scexp, scalpha,                //
            e, de);

         if (rik2 > cut2) {
            real taper, dtaper;
            switchTaper5<do_g>(rik, cut, off, taper, dtaper);
            if CONSTEXPR (do_g)
               de = e * dtaper + de * taper;
            if CONSTEXPR (do_e)
               e = e * taper;
         }

         if CONSTEXPR (do_a)
            if (vscale == -1)
               atomic_add(-1, nev, offset);
         if CONSTEXPR (do_e)
            atomic_add(e, ev, offset);
         if CONSTEXPR (do_g) {
            de *= REAL_RECIP(rik);
            real dedx = de * xr;
            real dedy = de * yr;
            real dedz = de * zr;

            atomic_add(dedx, gxred, i);
            atomic_add(dedy, gyred, i);
            atomic_add(dedz, gzred, i);
            atomic_add(-dedx, gxred, k);
            atomic_add(-dedy, gyred, k);
            atomic_add(-dedz, gzred, k);

            if CONSTEXPR (do_v) {
               real vxx = xr * dedx;
               real vyx = yr * dedx;
               real vzx = zr * dedx;
               real vyy = yr * dedy;
               real vzy = zr * dedy;
               real vzz = zr * dedz;

               atomic_add(vxx, vyx, vzx, vyy, vzy, vzz, vir_ev, offset);
            } // end if (do_v)
         }    // end if (do_g)
      }
   } // end for (int ii)

   if CONSTEXPR (do_g)
      ehalResolveGradient();
}

void ehal_acc(int vers)
{
   if (vers == calc::v0)
      ehal_acc1<calc::V0>();
   else if (vers == calc::v1)
      ehal_acc1<calc::V1>();
   else if (vers == calc::v3)
      ehal_acc1<calc::V3>();
   else if (vers == calc::v4)
      ehal_acc1<calc::V4>();
   else if (vers == calc::v5)
      ehal_acc1<calc::V5>();
   else if (vers == calc::v6)
      ehal_acc1<calc::V6>();
}
}
