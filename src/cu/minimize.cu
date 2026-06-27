#include "ff/atom.h"
#include "seq/launch.h"
#include "tool/dvector.h"

namespace tinker {
static dvector<double> d_xx, d_sc;

__global__
static void xMinimizeSetXxByPos_cu1(int n, double* restrict dxx, const double* restrict dsc,
   const pos_prec* restrict xpos, const pos_prec* restrict ypos, const pos_prec* restrict zpos)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      int j = 3 * i;
      dxx[j + 0] = xpos[i] * dsc[j + 0];
      dxx[j + 1] = ypos[i] * dsc[j + 1];
      dxx[j + 2] = zpos[i] * dsc[j + 2];
   }
}

void xMinimizeSetXxByPos_cu(int n, double* xx, const double* scale)
{
   size_t n3 = 3 * n;
   d_xx.reserve(n3);
   d_sc.reserve(n3);

   // copyin scale
   darray::copyin(g::q0, n3, d_sc.data(), scale);

   launch_k1s(g::s0, n, xMinimizeSetXxByPos_cu1, //
      n, d_xx.data(), d_sc.data(), xpos, ypos, zpos);

   // copyout xx
   darray::copyout(g::q0, n3, xx, d_xx.data());

   waitFor(g::q0);
}

__global__
static void xMinimizeSetPos_cu1(int n, const double* restrict dxx, const double* restrict dsc, pos_prec* restrict xpos,
   pos_prec* restrict ypos, pos_prec* restrict zpos)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      int j = 3 * i;
      xpos[i] = dxx[j + 0] / dsc[j + 0];
      ypos[i] = dxx[j + 1] / dsc[j + 1];
      zpos[i] = dxx[j + 2] / dsc[j + 2];
   }
}

void xMinimizeSetPos_cu(int n, const double* xx, const double* scale)
{
   size_t n3 = 3 * n;
   d_xx.reserve(n3);
   d_sc.reserve(n3);

   // copyin scale and xx
   darray::copyin(g::q0, n3, d_sc.data(), scale);
   darray::copyin(g::q0, n3, d_xx.data(), xx);

   launch_k1s(g::s0, n, xMinimizeSetPos_cu1, //
      n, d_xx.data(), d_sc.data(), xpos, ypos, zpos);

   waitFor(g::q0);
}
}
