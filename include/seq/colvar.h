#pragma once
#include "ff/colvar.h"
#include "seq/add.h"
#include "seq/geom.h"

namespace tinker {
/// \ingroup ff
/// The share of entry \c j of #Colvar::satom in the center of its set \c s:
/// its mass-weighted offset from the first atom of the set (#geomGroupOffset),
/// zero for the first atom itself.
#pragma acc routine seq
SEQ_ROUTINE
inline void colvarSetTerm(int& restrict s,
                          real& restrict dx,
                          real& restrict dy,
                          real& restrict dz,
                          int j,
                          const int* restrict sown,
                          const int (*restrict srange)[2],
                          const int* restrict satom,
                          const real* restrict sminv,
                          const real* restrict x,
                          const real* restrict y,
                          const real* restrict z,
                          const double* restrict mass,
                          const int* restrict molec,
                          TINKER_IMAGE_PARAMS)
{
   s = sown[j];
   const int j1 = srange[s][0];
   dx = 0, dy = 0, dz = 0;
   if (j != j1) {
      const int k = satom[j];
      geomGroupOffset(dx, dy, dz, k, satom[j1], (real)mass[k] * sminv[s], x, y, z, molec, TINKER_IMAGE_ARGS);
   }
}

/// \ingroup ff
/// Center of set \c s from the summed offsets of its atoms.
#pragma acc routine seq
SEQ_ROUTINE
inline void colvarSetCenter(real& restrict cx,
                            real& restrict cy,
                            real& restrict cz,
                            int s,
                            const grad_prec (*restrict ssum)[3],
                            const int (*restrict srange)[2],
                            const int* restrict satom,
                            const real* restrict x,
                            const real* restrict y,
                            const real* restrict z)
{
   const int k1 = satom[srange[s][0]];
   cx = x[k1] + toFloatGrad<real>(ssum[s][0]);
   cy = y[k1] + toFloatGrad<real>(ssum[s][1]);
   cz = z[k1] + toFloatGrad<real>(ssum[s][2]);
}

/// \ingroup ff
/// DISTANCE, slot \c g of CV \c i: the distance between the centers of sets A
/// and B, imaged if their first atoms are in different molecules. The slots
/// are the atoms of A, then the atoms of B; a zero distance has a zero
/// gradient. The first slot also writes q and W.
#pragma acc routine seq
SEQ_ROUTINE
inline void colvarDistanceSlot(int g,
                               int i,
                               real* restrict val,
                               real (*restrict w)[9],
                               real (*restrict grad)[3],
                               const grad_prec (*restrict ssum)[3],
                               const int (*restrict iarg)[4],
                               const int (*restrict slot)[2],
                               const int (*restrict srange)[2],
                               const int* restrict satom,
                               const real* restrict sminv,
                               const real* restrict x,
                               const real* restrict y,
                               const real* restrict z,
                               const double* restrict mass,
                               const int* restrict molec,
                               TINKER_IMAGE_PARAMS)
{
   const int sa = iarg[i][0], sb = iarg[i][1];
   real xa, ya, za, xb, yb, zb;
   colvarSetCenter(xa, ya, za, sa, ssum, srange, satom, x, y, z);
   colvarSetCenter(xb, yb, zb, sb, ssum, srange, satom, x, y, z);
   real xr = xa - xb, yr = ya - yb, zr = za - zb;
   const int ja1 = srange[sa][0], ja2 = srange[sa][1], jb1 = srange[sb][0];
   if (molec[satom[ja1]] != molec[satom[jb1]])
      image(xr, yr, zr);
   const real r = REAL_SQRT(xr * xr + yr * yr + zr * zr);
   const real rinv = (r == 0 ? 0 : REAL_RECIP(r));
   const real ux = xr * rinv, uy = yr * rinv, uz = zr * rinv;

   const int g0 = slot[i][0];
   if (g == g0) {
      val[i] = r;
      w[i][0] = xr * ux, w[i][1] = xr * uy, w[i][2] = xr * uz;
      w[i][3] = yr * ux, w[i][4] = yr * uy, w[i][5] = yr * uz;
      w[i][6] = zr * ux, w[i][7] = zr * uy, w[i][8] = zr * uz;
   }

   const int local = g - g0;
   const int na = ja2 - ja1;
   const int j = (local < na ? ja1 + local : jb1 + local - na);
   const real wt = (real)mass[satom[j]] * (local < na ? sminv[sa] : -sminv[sb]);
   grad[g][0] = wt * ux;
   grad[g][1] = wt * uy;
   grad[g][2] = wt * uz;
}

/// \ingroup ff
/// Gradient slot \c g, by the type of its CV.
#pragma acc routine seq
SEQ_ROUTINE
inline void colvarSlot1(int g,
                        const int* restrict gcv,
                        const int* restrict type,
                        real* restrict val,
                        real (*restrict w)[9],
                        real (*restrict grad)[3],
                        const grad_prec (*restrict ssum)[3],
                        const int (*restrict iarg)[4],
                        const int (*restrict slot)[2],
                        const int (*restrict srange)[2],
                        const int* restrict satom,
                        const real* restrict sminv,
                        const real* restrict x,
                        const real* restrict y,
                        const real* restrict z,
                        const double* restrict mass,
                        const int* restrict molec,
                        TINKER_IMAGE_PARAMS)
{
   const int i = gcv[g];
   if (type[i] == (int)ColvarType::DISTANCE)
      colvarDistanceSlot(g, i, val, w, grad, ssum, iarg, slot, srange, satom, sminv, x, y, z, mass, molec, TINKER_IMAGE_ARGS);
}
}
