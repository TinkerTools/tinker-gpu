#pragma once
#include "ff/evalence.h"
#include "math/const.h"
#include "math/libfunc.h"
#include "seq/add.h"
#include "seq/seq.h"

namespace tinker {
/**
 * Comments
 * Zhi Wang, Jun 25, 2019
 *
 * The original implementation in Tinker uses ACOS(cosine) to calculate the
 * out-of-plane angle, which is the major source of error in the single
 * precision mode.
 *
 * These angles (theta) are usually very small (e.g. 0.001 rad), so the value of
 * variable cosine is very close to 1 (cosine = SQRT(1 - eps**2)). As a result,
 * it is much more accurate to use theta = ASIN(eps) instead of theta =
 * ACOS(cosine) to calculate the angles.
 */
#pragma acc routine seq
template <class Ver>
SEQ_CUDA
void dk_opbend(real& restrict e,
               real& restrict vxx,
               real& restrict vyx,
               real& restrict vzx,
               real& restrict vyy,
               real& restrict vzy,
               real& restrict vzz,

               grad_prec* restrict deopbx,
               grad_prec* restrict deopby,
               grad_prec* restrict deopbz,

               OPBend opbtyp,
               real opbunit,
               int iopbend,
               const int* restrict iopb,
               const real* restrict opbk,
               const int (*restrict iang)[4],
               real copb,
               real qopb,
               real popb,
               real sopb,

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

   const real force = opbk[iopbend];
   const int i = iopb[iopbend];
   const int ia = iang[i][0];
   const int ib = iang[i][1];
   const int ic = iang[i][2];
   const int id = iang[i][3];

   real xia = x[ia];
   real yia = y[ia];
   real zia = z[ia];
   real xib = x[ib];
   real yib = y[ib];
   real zib = z[ib];
   real xic = x[ic];
   real yic = y[ic];
   real zic = z[ic];
   real xid = x[id];
   real yid = y[id];
   real zid = z[id];

   real xab = xia - xib;
   real yab = yia - yib;
   real zab = zia - zib;
   real xcb = xic - xib;
   real ycb = yic - yib;
   real zcb = zic - zib;
   real xdb = xid - xib;
   real ydb = yid - yib;
   real zdb = zid - zib;
   real xad = xia - xid;
   real yad = yia - yid;
   real zad = zia - zid;
   real xcd = xic - xid;
   real ycd = yic - yid;
   real zcd = zic - zid;

   real rab2, rad2;
   real rcb2, rcd2;
   real dot;
   real cc;
   if (opbtyp == OPBend::WDC) {

      // W-D-C angle between A-B-C plane and B-D vector for D-B<AC

      rab2 = xab * xab + yab * yab + zab * zab;
      rcb2 = xcb * xcb + ycb * ycb + zcb * zcb;
      cc = rab2 * rcb2 - (xab * xcb + yab * ycb + zab * zcb) * (xab * xcb + yab * ycb + zab * zcb);
      if CONSTEXPR (do_g)
         dot = xab * xcb + yab * ycb + zab * zcb;
   } else if (opbtyp == OPBend::ALLINGER) {

      // Allinger angle between A-C-D plane and D-B vector for D-B<AC

      rad2 = xad * xad + yad * yad + zad * zad;
      rcd2 = xcd * xcd + ycd * ycd + zcd * zcd;
      cc = rad2 * rcd2 - (xad * xcd + yad * ycd + zad * zcd) * (xad * xcd + yad * ycd + zad * zcd);
      if CONSTEXPR (do_g)
         dot = xad * xcd + yad * ycd + zad * zcd;
   }

   // find the out-of-plane angle bending energy

   real ee = xdb * (yab * zcb - zab * ycb) + ydb * (zab * xcb - xab * zcb) + zdb * (xab * ycb - yab * xcb);
   real rdb2 = xdb * xdb + ydb * ydb + zdb * zdb;
   rdb2 = REAL_MAX(rdb2, (real)0.0001);

   if (cc != 0) {
      real sine = REAL_ABS(ee) * REAL_RSQRT(cc * rdb2);
      sine = REAL_MIN((real)1, sine);
      real angle = radian * REAL_ASIN(sine);
      real dt = angle;
      real dt2 = dt * dt;
      real dt3 = dt2 * dt;
      real dt4 = dt2 * dt2;

      if CONSTEXPR (do_e)
         e = opbunit * force * dt2 * (1 + copb * dt + qopb * dt2 + popb * dt3 + sopb * dt4);

      if CONSTEXPR (do_g) {
         real deddt = opbunit * force * dt * radian * (2 + 3 * copb * dt + 4 * qopb * dt2 + 5 * popb * dt3 + 6 * sopb * dt4);
         real dedcos = -deddt * REAL_SIGN((real)1, ee) * REAL_RSQRT(cc * rdb2 - ee * ee);
         real term = ee * REAL_RECIP(cc);
         real dccdxia, dccdyia, dccdzia;
         real dccdxic, dccdyic, dccdzic;
         real dccdxid, dccdyid, dccdzid;
         if (opbtyp == OPBend::WDC) {
            dccdxia = (xab * rcb2 - xcb * dot) * term;
            dccdyia = (yab * rcb2 - ycb * dot) * term;
            dccdzia = (zab * rcb2 - zcb * dot) * term;
            dccdxic = (xcb * rab2 - xab * dot) * term;
            dccdyic = (ycb * rab2 - yab * dot) * term;
            dccdzic = (zcb * rab2 - zab * dot) * term;
            dccdxid = 0;
            dccdyid = 0;
            dccdzid = 0;
         } else if (opbtyp == OPBend::ALLINGER) {
            dccdxia = (xad * rcd2 - xcd * dot) * term;
            dccdyia = (yad * rcd2 - ycd * dot) * term;
            dccdzia = (zad * rcd2 - zcd * dot) * term;
            dccdxic = (xcd * rad2 - xad * dot) * term;
            dccdyic = (ycd * rad2 - yad * dot) * term;
            dccdzic = (zcd * rad2 - zad * dot) * term;
            dccdxid = -dccdxia - dccdxic;
            dccdyid = -dccdyia - dccdyic;
            dccdzid = -dccdzia - dccdzic;
         }

         term = ee * REAL_RECIP(rdb2);
         real deedxia = ydb * zcb - zdb * ycb;
         real deedyia = zdb * xcb - xdb * zcb;
         real deedzia = xdb * ycb - ydb * xcb;
         real deedxic = yab * zdb - zab * ydb;
         real deedyic = zab * xdb - xab * zdb;
         real deedzic = xab * ydb - yab * xdb;
         real deedxid = ycb * zab - zcb * yab + xdb * term;
         real deedyid = zcb * xab - xcb * zab + ydb * term;
         real deedzid = xcb * yab - ycb * xab + zdb * term;

         // compute first derivative components for this angle

         real dedxia = dedcos * (dccdxia + deedxia);
         real dedyia = dedcos * (dccdyia + deedyia);
         real dedzia = dedcos * (dccdzia + deedzia);
         real dedxic = dedcos * (dccdxic + deedxic);
         real dedyic = dedcos * (dccdyic + deedyic);
         real dedzic = dedcos * (dccdzic + deedzic);
         real dedxid = dedcos * (dccdxid + deedxid);
         real dedyid = dedcos * (dccdyid + deedyid);
         real dedzid = dedcos * (dccdzid + deedzid);
         real dedxib = -dedxia - dedxic - dedxid;
         real dedyib = -dedyia - dedyic - dedyid;
         real dedzib = -dedzia - dedzic - dedzid;

         atomic_add(dedxia, deopbx, ia);
         atomic_add(dedyia, deopby, ia);
         atomic_add(dedzia, deopbz, ia);
         atomic_add(dedxib, deopbx, ib);
         atomic_add(dedyib, deopby, ib);
         atomic_add(dedzib, deopbz, ib);
         atomic_add(dedxic, deopbx, ic);
         atomic_add(dedyic, deopby, ic);
         atomic_add(dedzic, deopbz, ic);
         atomic_add(dedxid, deopbx, id);
         atomic_add(dedyid, deopby, id);
         atomic_add(dedzid, deopbz, id);

         if CONSTEXPR (do_v) {
            vxx = xab * dedxia + xcb * dedxic + xdb * dedxid;
            vyx = yab * dedxia + ycb * dedxic + ydb * dedxid;
            vzx = zab * dedxia + zcb * dedxic + zdb * dedxid;
            vyy = yab * dedyia + ycb * dedyic + ydb * dedyid;
            vzy = zab * dedyia + zcb * dedyic + zdb * dedyid;
            vzz = zab * dedzia + zcb * dedzic + zdb * dedzid;
         }
      }
   }
}
}
