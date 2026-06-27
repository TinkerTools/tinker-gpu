#pragma once
#include "math/const.h"
#include "math/libfunc.h"
#include "seq/add.h"
#include "seq/seq.h"

namespace tinker {
#pragma acc routine seq
template <class Ver>
SEQ_CUDA
void dk_strbnd(real& restrict e,
               real& restrict vxx,
               real& restrict vyx,
               real& restrict vzx,
               real& restrict vyy,
               real& restrict vzy,
               real& restrict vzz,

               grad_prec* restrict debax,
               grad_prec* restrict debay,
               grad_prec* restrict debaz,

               real stbnunit,
               int istrbnd,
               const int (*restrict isb)[3],
               const real (*restrict sbk)[2],

               const real* restrict bl,
               const int (*restrict iang)[4],
               const real* restrict anat,

               const real* restrict x,
               const real* restrict y,
               const real* restrict z)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;
   if CONSTEXPR (do_e)
      e = 0;
   if CONSTEXPR (do_v)
      vxx = 0, vyx = 0, vzx = 0, vyy = 0, vzy = 0, vzz = 0;

   int i = isb[istrbnd][0];
   int j = isb[istrbnd][1];
   int k = isb[istrbnd][2];
   int ia = iang[i][0];
   int ib = iang[i][1];
   int ic = iang[i][2];
   real force1 = sbk[istrbnd][0];
   real force2 = sbk[istrbnd][1];

   real xia = x[ia];
   real yia = y[ia];
   real zia = z[ia];
   real xib = x[ib];
   real yib = y[ib];
   real zib = z[ib];
   real xic = x[ic];
   real yic = y[ic];
   real zic = z[ic];

   real xab = xia - xib;
   real yab = yia - yib;
   real zab = zia - zib;
   real xcb = xic - xib;
   real ycb = yic - yib;
   real zcb = zic - zib;

   real rab2 = xab * xab + yab * yab + zab * zab;
   real rcb2 = xcb * xcb + ycb * ycb + zcb * zcb;

   if (REAL_MIN(rab2, rcb2) != 0) {
      real rab = REAL_SQRT(rab2);
      real rcb = REAL_SQRT(rcb2);
      real xp = ycb * zab - zcb * yab;
      real yp = zcb * xab - xcb * zab;
      real zp = xcb * yab - ycb * xab;
      real rp = REAL_SQRT(xp * xp + yp * yp + zp * zp);
      rp = REAL_MAX(rp, (real)(0.0001));
      real dot = xab * xcb + yab * ycb + zab * zcb;
      real cosine = dot * REAL_RECIP(rab * rcb);
      cosine = REAL_MIN((real)1, REAL_MAX((real)-1, cosine));
      real angle = radian * REAL_ACOS(cosine);

      real dt = angle - anat[i];
      real term1 = -radian * REAL_RECIP(rab2 * rp);
      real term2 = radian * REAL_RECIP(rcb2 * rp);
      real ddtdxia = term1 * (yab * zp - zab * yp);
      real ddtdyia = term1 * (zab * xp - xab * zp);
      real ddtdzia = term1 * (xab * yp - yab * xp);
      real ddtdxic = term2 * (ycb * zp - zcb * yp);
      real ddtdyic = term2 * (zcb * xp - xcb * zp);
      real ddtdzic = term2 * (xcb * yp - ycb * xp);

      real dr1 = rab - bl[j];
      term1 = REAL_RECIP(rab);
      real dr2 = rcb - bl[k];
      term2 = REAL_RECIP(rcb);
      real ddrdxia = term1 * xab;
      real ddrdyia = term1 * yab;
      real ddrdzia = term1 * zab;
      real ddrdxic = term2 * xcb;
      real ddrdyic = term2 * ycb;
      real ddrdzic = term2 * zcb;

      term1 = stbnunit * force1;
      term2 = stbnunit * force2;
      real termr = term1 * dr1 + term2 * dr2;

      if CONSTEXPR (do_e)
         e = termr * dt;

      if CONSTEXPR (do_g) {
         real term1t = term1 * dt;
         real term2t = term2 * dt;
         real dedxia = term1t * ddrdxia + termr * ddtdxia;
         real dedyia = term1t * ddrdyia + termr * ddtdyia;
         real dedzia = term1t * ddrdzia + termr * ddtdzia;
         real dedxic = term2t * ddrdxic + termr * ddtdxic;
         real dedyic = term2t * ddrdyic + termr * ddtdyic;
         real dedzic = term2t * ddrdzic + termr * ddtdzic;
         real dedxib = -dedxia - dedxic;
         real dedyib = -dedyia - dedyic;
         real dedzib = -dedzia - dedzic;

         atomic_add(dedxia, debax, ia);
         atomic_add(dedyia, debay, ia);
         atomic_add(dedzia, debaz, ia);
         atomic_add(dedxib, debax, ib);
         atomic_add(dedyib, debay, ib);
         atomic_add(dedzib, debaz, ib);
         atomic_add(dedxic, debax, ic);
         atomic_add(dedyic, debay, ic);
         atomic_add(dedzic, debaz, ic);

         if CONSTEXPR (do_v) {
            vxx = xab * dedxia + xcb * dedxic;
            vyx = yab * dedxia + ycb * dedxic;
            vzx = zab * dedxia + zcb * dedxic;
            vyy = yab * dedyia + ycb * dedyic;
            vzy = zab * dedyia + zcb * dedyic;
            vzz = zab * dedzia + zcb * dedzic;
         }
      }
   }
}
}
