#pragma once
#include "math/libfunc.h"
#include "seq/add.h"
#include "seq/seq.h"

namespace tinker {
#pragma acc routine seq
template <class Ver>
SEQ_CUDA
void dk_urey(real& restrict e,
             real& restrict vxx,
             real& restrict vyx,
             real& restrict vzx,
             real& restrict vyy,
             real& restrict vzy,
             real& restrict vzz,

             grad_prec* restrict deubx,
             grad_prec* restrict deuby,
             grad_prec* restrict deubz,

             real ureyunit,
             int i,
             const int (*restrict iury)[3],
             const real* restrict uk,
             const real* restrict ul,
             real cury,
             real qury,

             const real* restrict x,
             const real* restrict y,
             const real* restrict z)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;

   const int ia = iury[i][0];
   const int ic = iury[i][2];
   const real ideal = ul[i];
   const real force = uk[i];

   real xac = x[ia] - x[ic];
   real yac = y[ia] - y[ic];
   real zac = z[ia] - z[ic];

   real rac = REAL_SQRT(xac * xac + yac * yac + zac * zac);
   real dt = rac - ideal;
   real dt2 = dt * dt;

   if CONSTEXPR (do_e) {
      e = ureyunit * force * dt2 * (1 + cury * dt + qury * dt2);
   }

   if CONSTEXPR (do_g) {
      real deddt = 2 * ureyunit * force * dt * (1 + 1.5f * cury * dt + 2 * qury * dt2);
      real de = deddt * REAL_RECIP(rac);
      real dedx = de * xac;
      real dedy = de * yac;
      real dedz = de * zac;

      atomic_add(dedx, deubx, ia);
      atomic_add(dedy, deuby, ia);
      atomic_add(dedz, deubz, ia);
      atomic_add(-dedx, deubx, ic);
      atomic_add(-dedy, deuby, ic);
      atomic_add(-dedz, deubz, ic);

      if CONSTEXPR (do_v) {
         vxx = xac * dedx;
         vyx = yac * dedx;
         vzx = zac * dedx;
         vyy = yac * dedy;
         vzy = zac * dedy;
         vzz = zac * dedz;
      }
   }
}
}
