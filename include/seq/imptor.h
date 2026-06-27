#pragma once
#include "math/libfunc.h"
#include "seq/add.h"
#include "seq/seq.h"

namespace tinker {
#pragma acc routine seq
template <class Ver>
SEQ_CUDA
void dk_imptor(real& restrict e,
               real& restrict vxx,
               real& restrict vyx,
               real& restrict vzx,
               real& restrict vyy,
               real& restrict vzy,
               real& restrict vzz,

               grad_prec* restrict deitx,
               grad_prec* restrict deity,
               grad_prec* restrict deitz,

               real itorunit,
               int i,
               const int (*restrict iitors)[4],
               const real (*restrict itors1)[4],
               const real (*restrict itors2)[4],
               const real (*restrict itors3)[4],

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

   const int ia = iitors[i][0];
   const int ib = iitors[i][1];
   const int ic = iitors[i][2];
   const int id = iitors[i][3];

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
   real xba = xib - xia;
   real yba = yib - yia;
   real zba = zib - zia;
   real xcb = xic - xib;
   real ycb = yic - yib;
   real zcb = zic - zib;
   real xdc = xid - xic;
   real ydc = yid - yic;
   real zdc = zid - zic;

   real xt = yba * zcb - ycb * zba;
   real yt = zba * xcb - zcb * xba;
   real zt = xba * ycb - xcb * yba;
   real xu = ycb * zdc - ydc * zcb;
   real yu = zcb * xdc - zdc * xcb;
   real zu = xcb * ydc - xdc * ycb;
   real xtu = yt * zu - yu * zt;
   real ytu = zt * xu - zu * xt;
   real ztu = xt * yu - xu * yt;
   real rt2 = xt * xt + yt * yt + zt * zt;
   real ru2 = xu * xu + yu * yu + zu * zu;
   real rtru = REAL_SQRT(rt2 * ru2);

   if (rtru != 0) {
      real rcb = REAL_SQRT(xcb * xcb + ycb * ycb + zcb * zcb);
      real cosine = (xt * xu + yt * yu + zt * zu) * REAL_RECIP(rtru);
      real sine = (xcb * xtu + ycb * ytu + zcb * ztu) * REAL_RECIP(rcb * rtru);

      real v1 = itors1[i][0];
      real c1 = itors1[i][2];
      real s1 = itors1[i][3];
      real v2 = itors2[i][0];
      real c2 = itors2[i][2];
      real s2 = itors2[i][3];
      real v3 = itors3[i][0];
      real c3 = itors3[i][2];
      real s3 = itors3[i][3];

      real cosine2 = cosine * cosine - sine * sine;
      real sine2 = 2 * cosine * sine;
      real cosine3 = cosine * cosine2 - sine * sine2;
      real sine3 = cosine * sine2 + sine * cosine2;
      real phi1 = 1 + (cosine * c1 + sine * s1);
      real phi2 = 1 + (cosine2 * c2 + sine2 * s2);
      real phi3 = 1 + (cosine3 * c3 + sine3 * s3);

      if CONSTEXPR (do_e) {
         e = itorunit * (v1 * phi1 + v2 * phi2 + v3 * phi3);
      }

      if CONSTEXPR (do_g) {
         real dphi1 = (cosine * s1 - sine * c1);
         real dphi2 = 2 * (cosine2 * s2 - sine2 * c2);
         real dphi3 = 3 * (cosine3 * s3 - sine3 * c3);
         real dedphi = itorunit * (v1 * dphi1 + v2 * dphi2 + v3 * dphi3);

         real xca = xic - xia;
         real yca = yic - yia;
         real zca = zic - zia;
         real xdb = xid - xib;
         real ydb = yid - yib;
         real zdb = zid - zib;

         real inv1 = REAL_RECIP(rt2 * rcb);
         real inv2 = REAL_RECIP(ru2 * rcb);
         real dedxt = dedphi * (yt * zcb - ycb * zt) * inv1;
         real dedyt = dedphi * (zt * xcb - zcb * xt) * inv1;
         real dedzt = dedphi * (xt * ycb - xcb * yt) * inv1;
         real dedxu = -dedphi * (yu * zcb - ycb * zu) * inv2;
         real dedyu = -dedphi * (zu * xcb - zcb * xu) * inv2;
         real dedzu = -dedphi * (xu * ycb - xcb * yu) * inv2;

         real dedxia = zcb * dedyt - ycb * dedzt;
         real dedyia = xcb * dedzt - zcb * dedxt;
         real dedzia = ycb * dedxt - xcb * dedyt;
         real dedxib = yca * dedzt - zca * dedyt + zdc * dedyu - ydc * dedzu;
         real dedyib = zca * dedxt - xca * dedzt + xdc * dedzu - zdc * dedxu;
         real dedzib = xca * dedyt - yca * dedxt + ydc * dedxu - xdc * dedyu;
         real dedxic = zba * dedyt - yba * dedzt + ydb * dedzu - zdb * dedyu;
         real dedyic = xba * dedzt - zba * dedxt + zdb * dedxu - xdb * dedzu;
         real dedzic = yba * dedxt - xba * dedyt + xdb * dedyu - ydb * dedxu;
         real dedxid = zcb * dedyu - ycb * dedzu;
         real dedyid = xcb * dedzu - zcb * dedxu;
         real dedzid = ycb * dedxu - xcb * dedyu;

         atomic_add(dedxia, deitx, ia);
         atomic_add(dedyia, deity, ia);
         atomic_add(dedzia, deitz, ia);
         atomic_add(dedxib, deitx, ib);
         atomic_add(dedyib, deity, ib);
         atomic_add(dedzib, deitz, ib);
         atomic_add(dedxic, deitx, ic);
         atomic_add(dedyic, deity, ic);
         atomic_add(dedzic, deitz, ic);
         atomic_add(dedxid, deitx, id);
         atomic_add(dedyid, deity, id);
         atomic_add(dedzid, deitz, id);

         if CONSTEXPR (do_v) {
            vxx = xcb * (dedxic + dedxid) - xba * dedxia + xdc * dedxid;
            vyx = ycb * (dedxic + dedxid) - yba * dedxia + ydc * dedxid;
            vzx = zcb * (dedxic + dedxid) - zba * dedxia + zdc * dedxid;
            vyy = ycb * (dedyic + dedyid) - yba * dedyia + ydc * dedyid;
            vzy = zcb * (dedyic + dedyid) - zba * dedyia + zdc * dedyid;
            vzz = zcb * (dedzic + dedzid) - zba * dedzia + zdc * dedzid;
         }
      }
   }
}
}
