#pragma once
#include "ff/evalence.h"
#include "math/libfunc.h"
#include "seq/add.h"
#include "seq/seq.h"

namespace tinker {
#pragma acc routine seq
template <class Ver>
SEQ_CUDA
void dk_bond(real& restrict e,
             real& restrict vxx,
             real& restrict vyx,
             real& restrict vzx,
             real& restrict vyy,
             real& restrict vzy,
             real& restrict vzz,

             grad_prec* restrict debx,
             grad_prec* restrict deby,
             grad_prec* restrict debz,

             Bond bndtyp,
             real bndunit,
             int i,
             const int (*restrict ibnd)[2],
             const real* restrict bl,
             const real* restrict bk,
             real cbnd,
             real qbnd,

             const real* restrict x,
             const real* restrict y,
             const real* restrict z)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;

   int ia = ibnd[i][0];
   int ib = ibnd[i][1];
   real ideal = bl[i];
   real force = bk[i];

   real xab = x[ia] - x[ib];
   real yab = y[ia] - y[ib];
   real zab = z[ia] - z[ib];

   real rab = REAL_SQRT(xab * xab + yab * yab + zab * zab);
   real dt = rab - ideal;

   real deddt;
   if (bndtyp == Bond::HARMONIC) {
      real dt2 = dt * dt;
      if CONSTEXPR (do_e)
         e = bndunit * force * dt2 * (1 + cbnd * dt + qbnd * dt2);
      if CONSTEXPR (do_g)
         deddt = 2 * bndunit * force * dt * (1 + 1.5f * cbnd * dt + 2 * qbnd * dt2);
   } else if (bndtyp == Bond::MORSE) {
      real expterm = REAL_EXP(-2 * dt);
      real bde = 0.25f * bndunit * force;
      if CONSTEXPR (do_e)
         e = bde * (1 - expterm) * (1 - expterm);
      if CONSTEXPR (do_g)
         deddt = 4 * bde * (1 - expterm) * expterm;
   }

   if CONSTEXPR (do_g) {
      real de = deddt * REAL_RECIP(rab);
      real dedx = de * xab;
      real dedy = de * yab;
      real dedz = de * zab;
      atomic_add(dedx, debx, ia);
      atomic_add(dedy, deby, ia);
      atomic_add(dedz, debz, ia);
      atomic_add(-dedx, debx, ib);
      atomic_add(-dedy, deby, ib);
      atomic_add(-dedz, debz, ib);

      if CONSTEXPR (do_v) {
         vxx = xab * dedx;
         vyx = yab * dedx;
         vzx = zab * dedx;
         vyy = yab * dedy;
         vzy = zab * dedy;
         vzz = zab * dedz;
      }
   }
}
}
