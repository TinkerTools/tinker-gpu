#pragma once
#include "math/libfunc.h"
#include "seq/add.h"
#include "seq/seq.h"

namespace tinker {
#pragma acc routine seq
template <class Ver>
SEQ_CUDA
void dk_strtor(real& restrict e,
               real& restrict vxx,
               real& restrict vyx,
               real& restrict vzx,
               real& restrict vyy,
               real& restrict vzy,
               real& restrict vzz,

               grad_prec* restrict debtx,
               grad_prec* restrict debty,
               grad_prec* restrict debtz,

               real storunit,
               int istrtor,
               const int (*restrict ist)[4],
               const real (*restrict kst)[9],

               const real* restrict bl,
               const int (*restrict itors)[4],
               const real (*restrict tors1)[4],
               const real (*restrict tors2)[4],
               const real (*restrict tors3)[4],

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

   const int i = ist[istrtor][0];
   const int ia = itors[i][0];
   const int ib = itors[i][1];
   const int ic = itors[i][2];
   const int id = itors[i][3];

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

   real rba = REAL_SQRT(xba * xba + yba * yba + zba * zba);
   real rcb = REAL_SQRT(xcb * xcb + ycb * ycb + zcb * zcb);
   real rdc = REAL_SQRT(xdc * xdc + ydc * ydc + zdc * zdc);
   real rmin = REAL_MIN(rba, rcb);
   rmin = REAL_MIN(rmin, rdc);
   if (rmin == 0)
      return;

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
   rt2 = REAL_MAX(rt2, (real)0.000001);
   real ru2 = xu * xu + yu * yu + zu * zu;
   ru2 = REAL_MAX(ru2, (real)0.000001);
   real rtru = REAL_SQRT(rt2 * ru2);

   real xca = xic - xia;
   real yca = yic - yia;
   real zca = zic - zia;
   real xdb = xid - xib;
   real ydb = yid - yib;
   real zdb = zid - zib;

   real cosine = (xt * xu + yt * yu + zt * zu) * REAL_RECIP(rtru);
   real sine = (xcb * xtu + ycb * ytu + zcb * ztu) * REAL_RECIP(rcb * rtru);

   real c1 = tors1[i][2];
   real s1 = tors1[i][3];
   real c2 = tors2[i][2];
   real s2 = tors2[i][3];
   real c3 = tors3[i][2];
   real s3 = tors3[i][3];
   real cosine2 = cosine * cosine - sine * sine;
   real sine2 = 2 * cosine * sine;
   real cosine3 = cosine * cosine2 - sine * sine2;
   real sine3 = cosine * sine2 + sine * cosine2;
   real phi1 = 1 + (cosine * c1 + sine * s1);
   real phi2 = 1 + (cosine2 * c2 + sine2 * s2);
   real phi3 = 1 + (cosine3 * c3 + sine3 * s3);

   real dphi1, dphi2, dphi3;
   if CONSTEXPR (do_g) {
      dphi1 = cosine * s1 - sine * c1;
      dphi2 = 2 * (cosine2 * s2 - sine2 * c2);
      dphi3 = 3 * (cosine3 * s3 - sine3 * c3);
   }

   int k;
   real v1, v2, v3, dr, e1, e2, e3;
   MAYBE_UNUSED real dedxia, dedyia, dedzia;
   MAYBE_UNUSED real dedxib, dedyib, dedzib;
   MAYBE_UNUSED real dedxic, dedyic, dedzic;
   MAYBE_UNUSED real dedxid, dedyid, dedzid;

   v1 = kst[istrtor][0];
   v2 = kst[istrtor][1];
   v3 = kst[istrtor][2];
   k = ist[istrtor][1];
   dr = rba - bl[k];
   e1 = storunit * dr * (v1 * phi1 + v2 * phi2 + v3 * phi3);
   if CONSTEXPR (do_g) {
      real dedphi = storunit * dr * (v1 * dphi1 + v2 * dphi2 + v3 * dphi3);
      real ddr = storunit * (v1 * phi1 + v2 * phi2 + v3 * phi3) / rba;
      real ddrdx = xba * ddr;
      real ddrdy = yba * ddr;
      real ddrdz = zba * ddr;
      real dedxt = dedphi * (yt * zcb - ycb * zt) / (rt2 * rcb);
      real dedyt = dedphi * (zt * xcb - zcb * xt) / (rt2 * rcb);
      real dedzt = dedphi * (xt * ycb - xcb * yt) / (rt2 * rcb);
      real dedxu = -dedphi * (yu * zcb - ycb * zu) / (ru2 * rcb);
      real dedyu = -dedphi * (zu * xcb - zcb * xu) / (ru2 * rcb);
      real dedzu = -dedphi * (xu * ycb - xcb * yu) / (ru2 * rcb);

      dedxia = zcb * dedyt - ycb * dedzt - ddrdx;
      dedyia = xcb * dedzt - zcb * dedxt - ddrdy;
      dedzia = ycb * dedxt - xcb * dedyt - ddrdz;
      dedxib = yca * dedzt - zca * dedyt + zdc * dedyu - ydc * dedzu + ddrdx;
      dedyib = zca * dedxt - xca * dedzt + xdc * dedzu - zdc * dedxu + ddrdy;
      dedzib = xca * dedyt - yca * dedxt + ydc * dedxu - xdc * dedyu + ddrdz;
      dedxic = zba * dedyt - yba * dedzt + ydb * dedzu - zdb * dedyu;
      dedyic = xba * dedzt - zba * dedxt + zdb * dedxu - xdb * dedzu;
      dedzic = yba * dedxt - xba * dedyt + xdb * dedyu - ydb * dedxu;
      dedxid = zcb * dedyu - ycb * dedzu;
      dedyid = xcb * dedzu - zcb * dedxu;
      dedzid = ycb * dedxu - xcb * dedyu;
   }

   v1 = kst[istrtor][3];
   v2 = kst[istrtor][4];
   v3 = kst[istrtor][5];
   k = ist[istrtor][2];
   dr = rcb - bl[k];
   e2 = storunit * dr * (v1 * phi1 + v2 * phi2 + v3 * phi3);
   if CONSTEXPR (do_g) {
      real dedphi = storunit * dr * (v1 * dphi1 + v2 * dphi2 + v3 * dphi3);
      real ddr = storunit * (v1 * phi1 + v2 * phi2 + v3 * phi3) / rcb;
      real ddrdx = xcb * ddr;
      real ddrdy = ycb * ddr;
      real ddrdz = zcb * ddr;
      real dedxt = dedphi * (yt * zcb - ycb * zt) / (rt2 * rcb);
      real dedyt = dedphi * (zt * xcb - zcb * xt) / (rt2 * rcb);
      real dedzt = dedphi * (xt * ycb - xcb * yt) / (rt2 * rcb);
      real dedxu = -dedphi * (yu * zcb - ycb * zu) / (ru2 * rcb);
      real dedyu = -dedphi * (zu * xcb - zcb * xu) / (ru2 * rcb);
      real dedzu = -dedphi * (xu * ycb - xcb * yu) / (ru2 * rcb);

      dedxia += zcb * dedyt - ycb * dedzt;
      dedyia += xcb * dedzt - zcb * dedxt;
      dedzia += ycb * dedxt - xcb * dedyt;
      dedxib += yca * dedzt - zca * dedyt + zdc * dedyu - ydc * dedzu - ddrdx;
      dedyib += zca * dedxt - xca * dedzt + xdc * dedzu - zdc * dedxu - ddrdy;
      dedzib += xca * dedyt - yca * dedxt + ydc * dedxu - xdc * dedyu - ddrdz;
      dedxic += zba * dedyt - yba * dedzt + ydb * dedzu - zdb * dedyu + ddrdx;
      dedyic += xba * dedzt - zba * dedxt + zdb * dedxu - xdb * dedzu + ddrdy;
      dedzic += yba * dedxt - xba * dedyt + xdb * dedyu - ydb * dedxu + ddrdz;
      dedxid += zcb * dedyu - ycb * dedzu;
      dedyid += xcb * dedzu - zcb * dedxu;
      dedzid += ycb * dedxu - xcb * dedyu;
   }

   v1 = kst[istrtor][6];
   v2 = kst[istrtor][7];
   v3 = kst[istrtor][8];
   k = ist[istrtor][3];
   dr = rdc - bl[k];
   e3 = storunit * dr * (v1 * phi1 + v2 * phi2 + v3 * phi3);
   if CONSTEXPR (do_g) {
      real dedphi = storunit * dr * (v1 * dphi1 + v2 * dphi2 + v3 * dphi3);
      real ddr = storunit * (v1 * phi1 + v2 * phi2 + v3 * phi3) / rdc;
      real ddrdx = xdc * ddr;
      real ddrdy = ydc * ddr;
      real ddrdz = zdc * ddr;
      real dedxt = dedphi * (yt * zcb - ycb * zt) / (rt2 * rcb);
      real dedyt = dedphi * (zt * xcb - zcb * xt) / (rt2 * rcb);
      real dedzt = dedphi * (xt * ycb - xcb * yt) / (rt2 * rcb);
      real dedxu = -dedphi * (yu * zcb - ycb * zu) / (ru2 * rcb);
      real dedyu = -dedphi * (zu * xcb - zcb * xu) / (ru2 * rcb);
      real dedzu = -dedphi * (xu * ycb - xcb * yu) / (ru2 * rcb);

      dedxia += zcb * dedyt - ycb * dedzt;
      dedyia += xcb * dedzt - zcb * dedxt;
      dedzia += ycb * dedxt - xcb * dedyt;
      dedxib += yca * dedzt - zca * dedyt + zdc * dedyu - ydc * dedzu;
      dedyib += zca * dedxt - xca * dedzt + xdc * dedzu - zdc * dedxu;
      dedzib += xca * dedyt - yca * dedxt + ydc * dedxu - xdc * dedyu;
      dedxic += zba * dedyt - yba * dedzt + ydb * dedzu - zdb * dedyu - ddrdx;
      dedyic += xba * dedzt - zba * dedxt + zdb * dedxu - xdb * dedzu - ddrdy;
      dedzic += yba * dedxt - xba * dedyt + xdb * dedyu - ydb * dedxu - ddrdz;
      dedxid += zcb * dedyu - ycb * dedzu + ddrdx;
      dedyid += xcb * dedzu - zcb * dedxu + ddrdy;
      dedzid += ycb * dedxu - xcb * dedyu + ddrdz;
   }

   if CONSTEXPR (do_e)
      e = e1 + e2 + e3;
   if CONSTEXPR (do_g) {
      atomic_add(dedxia, debtx, ia);
      atomic_add(dedyia, debty, ia);
      atomic_add(dedzia, debtz, ia);
      atomic_add(dedxib, debtx, ib);
      atomic_add(dedyib, debty, ib);
      atomic_add(dedzib, debtz, ib);
      atomic_add(dedxic, debtx, ic);
      atomic_add(dedyic, debty, ic);
      atomic_add(dedzic, debtz, ic);
      atomic_add(dedxid, debtx, id);
      atomic_add(dedyid, debty, id);
      atomic_add(dedzid, debtz, id);
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
