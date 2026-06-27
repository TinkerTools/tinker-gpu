#pragma once
#include "ff/amoeba/mpole.h"
#include "ff/hippo/erepel.h"
#include "math/libfunc.h"
#include "seq/seq.h"

namespace tinker {
SEQ_ROUTINE
inline void chkpoleAtomI(int i, real (*restrict pole)[MPL_TOTAL], LocalFrame* zaxis, const real* restrict x, const real* restrict y, const real* restrict z)
{
   int polaxe = zaxis[i].polaxe;
   bool check = ((polaxe != LFRM_Z_THEN_X) or (zaxis[i].yaxis) == 0) ? false : true;
   if (check) {
      int k = zaxis[i].yaxis;
      int ia = i;
      int ib = zaxis[i].zaxis;
      int ic = zaxis[i].xaxis;
      int id = INT_ABS(k) - 1;

      // compute the signed parallelpiped volume at chiral site

      real xad = x[ia] - x[id];
      real yad = y[ia] - y[id];
      real zad = z[ia] - z[id];
      real xbd = x[ib] - x[id];
      real ybd = y[ib] - y[id];
      real zbd = z[ib] - z[id];
      real xcd = x[ic] - x[id];
      real ycd = y[ic] - y[id];
      real zcd = z[ic] - z[id];
      real c1 = ybd * zcd - zbd * ycd;
      real c2 = ycd * zad - zcd * yad;
      real c3 = yad * zbd - zad * ybd;
      real vol = xad * c1 + xbd * c2 + xcd * c3;

      // invert atomic multipole components involving the y-axis

      if ((k < 0 && vol > 0) or (k > 0 && vol < 0)) {
         zaxis[i].yaxis = -k;
         // y -> -y
         pole[i][MPL_PME_Y] = -pole[i][MPL_PME_Y];
         // xy -> -xy
         // yz -> -yz
         pole[i][MPL_PME_XY] = -pole[i][MPL_PME_XY];
         pole[i][MPL_PME_YZ] = -pole[i][MPL_PME_YZ];
      }
   }
}
}

namespace tinker {
SEQ_ROUTINE
inline void rotpoleNorm(real* a)
{
   real a1 = REAL_RSQRT(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
   a[0] *= a1;
   a[1] *= a1;
   a[2] *= a1;
}

SEQ_ROUTINE
inline void rotpoleAddBy(real* restrict a, const real* restrict b)
{
   a[0] += b[0];
   a[1] += b[1];
   a[2] += b[2];
}

SEQ_ROUTINE
inline void rotpoleAddBy2(real* restrict a, const real* restrict b, const real* restrict c)
{
   a[0] += (b[0] + c[0]);
   a[1] += (b[1] + c[1]);
   a[2] += (b[2] + c[2]);
}

SEQ_ROUTINE
inline void rotsite(int isite, const real (*restrict a)[3], real (*restrict rpole)[10], const real (*restrict pole)[10])
{
   static_assert(MPL_TOTAL == 10, "");

   // charge
   rpole[isite][0] = pole[isite][0];
   // dipole
   rpole[isite][1] = 0;
   rpole[isite][2] = 0;
   rpole[isite][3] = 0;
#if _OPENACC
#pragma acc loop seq collapse(2)
#endif
   for (int i = 1; i < 4; ++i)
      for (int j = 1; j < 4; ++j)
         rpole[isite][i] += pole[isite][j] * a[j - 1][i - 1];

   // quadrupole
   real rp[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
   real mp[3][3];
   mp[0][0] = pole[isite][MPL_PME_XX];
   mp[0][1] = pole[isite][MPL_PME_XY];
   mp[0][2] = pole[isite][MPL_PME_XZ];
   mp[1][0] = pole[isite][MPL_PME_YX];
   mp[1][1] = pole[isite][MPL_PME_YY];
   mp[1][2] = pole[isite][MPL_PME_YZ];
   mp[2][0] = pole[isite][MPL_PME_ZX];
   mp[2][1] = pole[isite][MPL_PME_ZY];
   mp[2][2] = pole[isite][MPL_PME_ZZ];
#if _OPENACC
   #pragma acc loop seq
#endif
   for (int i = 0; i < 3; ++i) {
#if _OPENACC
#pragma acc loop seq
#endif
      for (int j = 0; j <= i; ++j) {
         // if (j < i) {
         //  rp[j][i] = rp[i][j];
         // } else {
#if _OPENACC
#pragma acc loop seq collapse(2)
#endif
         for (int k = 0; k < 3; ++k)
            for (int m = 0; m < 3; ++m)
               rp[j][i] += a[k][i] * a[m][j] * mp[m][k];
         // }
      }
   }
   rpole[isite][MPL_PME_XX] = rp[0][0];
   rpole[isite][MPL_PME_XY] = rp[0][1];
   rpole[isite][MPL_PME_XZ] = rp[0][2];
   rpole[isite][MPL_PME_YY] = rp[1][1];
   rpole[isite][MPL_PME_YZ] = rp[1][2];
   rpole[isite][MPL_PME_ZZ] = rp[2][2];
}
}

namespace tinker {
SEQ_ROUTINE
inline void rotpoleAtomI(int i,
                         real (*restrict rpole)[MPL_TOTAL],
                         const real (*restrict pole)[MPL_TOTAL],
                         const LocalFrame* restrict zaxis,
                         const real* restrict x,
                         const real* restrict y,
                         const real* restrict z)
{
   // rotmat routine
   real xi = x[i];
   real yi = y[i];
   real zi = z[i];
   int iz = zaxis[i].zaxis;
   int ix = zaxis[i].xaxis;
   int iy = INT_ABS(zaxis[i].yaxis) - 1;
   int polaxe = zaxis[i].polaxe;
   // the default identity matrix
   real a[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

   if (polaxe != LFRM_NONE) {
      real* restrict xx = &a[0][0];
      real* restrict yy = &a[1][0];
      real* restrict zz = &a[2][0];

      // STEP 1: PICK Z AND NORM Z
      // pick z
      zz[0] = x[iz] - xi;
      zz[1] = y[iz] - yi;
      zz[2] = z[iz] - zi;
      // norm z
      rotpoleNorm(zz);

      // STEP 2: PICK X AND NORM X
      // even if it is not needef for z then x)
      if (polaxe == LFRM_Z_ONLY) {
         // pick x
         int okay = !(REAL_ABS(zz[0]) > 0.866f);
         xx[0] = (okay ? 1 : 0);
         xx[1] = (okay ? 0 : 1);
         xx[2] = 0;
      } else {
         // pick x
         xx[0] = x[ix] - xi;
         xx[1] = y[ix] - yi;
         xx[2] = z[ix] - zi;
         rotpoleNorm(xx);
      }

      // STEP 3: PICK Y AND NORM Y
      // only for z biscector and 3 fold
      if (polaxe == LFRM_Z_BISECT || polaxe == LFRM_3_FOLD) {
         yy[0] = x[iy] - xi;
         yy[1] = y[iy] - yi;
         yy[2] = z[iy] - zi;
         rotpoleNorm(yy);
      }

      // STEP 4
      if (polaxe == LFRM_BISECTOR) {
         rotpoleAddBy(zz, xx);
         rotpoleNorm(zz);
      } else if (polaxe == LFRM_Z_BISECT) {
         rotpoleAddBy(xx, yy);
         rotpoleNorm(xx);
      } else if (polaxe == LFRM_3_FOLD) {
         rotpoleAddBy2(zz, xx, yy);
         rotpoleNorm(zz);
      }

      // STEP 5
      // x -= (x.z) z
      real dotxz = xx[0] * zz[0] + xx[1] * zz[1] + xx[2] * zz[2];
      xx[0] -= dotxz * zz[0];
      xx[1] -= dotxz * zz[1];
      xx[2] -= dotxz * zz[2];
      // norm x
      rotpoleNorm(xx);
      // y = z cross x
      a[1][0] = a[0][2] * a[2][1] - a[0][1] * a[2][2];
      a[1][1] = a[0][0] * a[2][2] - a[0][2] * a[2][0];
      a[1][2] = a[0][1] * a[2][0] - a[0][0] * a[2][1];
   } // end if (.not. LFRM_NONE)

   // rotsite routine
   rotsite(i, a, rpole, pole);
}
}
