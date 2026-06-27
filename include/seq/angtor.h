#pragma once
#include "math/const.h"
#include "math/libfunc.h"
#include "seq/add.h"
#include "seq/seq.h"

namespace tinker {
#pragma acc routine seq
template <class Ver>
SEQ_CUDA
void dk_angtor(real& restrict e,
               real& restrict vxx,
               real& restrict vyx,
               real& restrict vzx,
               real& restrict vyy,
               real& restrict vzy,
               real& restrict vzz,

               grad_prec* restrict deatx,
               grad_prec* restrict deaty,
               grad_prec* restrict deatz,

               real atorunit,
               int iangtor,
               const int (*restrict iat)[3],
               const real (*restrict kant)[6],

               const real* restrict anat,
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

   const int i = iat[iangtor][0];
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

   real rba2 = xba * xba + yba * yba + zba * zba;
   real rcb2 = xcb * xcb + ycb * ycb + zcb * zcb;
   real rdc2 = xdc * xdc + ydc * ydc + zdc * zdc;
   real rmin = REAL_MIN(rba2, rcb2);
   rmin = REAL_MIN(rmin, rdc2);
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
   real rcb = REAL_SQRT(rcb2);

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

   real e1, e2;
   MAYBE_UNUSED real dedxia, dedyia, dedzia;
   MAYBE_UNUSED real dedxib, dedyib, dedzib;
   MAYBE_UNUSED real dedxic, dedyic, dedzic;
   MAYBE_UNUSED real dedxid, dedyid, dedzid;

   {
      real v1 = kant[iangtor][0];
      real v2 = kant[iangtor][1];
      real v3 = kant[iangtor][2];
      int k = iat[iangtor][1];
      real dot = xba * xcb + yba * ycb + zba * zcb;
      real cosang = -dot * REAL_RSQRT(rba2 * rcb2);
      real angle = radian * REAL_ACOS(cosang);
      real dt = angle - anat[k];
      e1 = atorunit * dt * (v1 * phi1 + v2 * phi2 + v3 * phi3);
      if CONSTEXPR (do_g) {
         real dedphi = atorunit * dt * (v1 * dphi1 + v2 * dphi2 + v3 * dphi3);
         real ddt = atorunit * radian * (v1 * phi1 + v2 * phi2 + v3 * phi3);
         real dedxt = dedphi * (zcb * yt - ycb * zt) / (rt2 * rcb);
         real dedyt = dedphi * (xcb * zt - zcb * xt) / (rt2 * rcb);
         real dedzt = dedphi * (ycb * xt - xcb * yt) / (rt2 * rcb);
         real dedxu = dedphi * (ycb * zu - zcb * yu) / (ru2 * rcb);
         real dedyu = dedphi * (zcb * xu - xcb * zu) / (ru2 * rcb);
         real dedzu = dedphi * (xcb * yu - ycb * xu) / (ru2 * rcb);

         real terma = -ddt / (rba2 * REAL_SQRT(rt2));
         real termc = ddt / (rcb2 * REAL_SQRT(rt2));
         dedxia = terma * (zba * yt - yba * zt) + zcb * dedyt - ycb * dedzt;
         dedyia = terma * (xba * zt - zba * xt) + xcb * dedzt - zcb * dedxt;
         dedzia = terma * (yba * xt - xba * yt) + ycb * dedxt - xcb * dedyt;
         dedxib = terma * (yba * zt - zba * yt) + termc * (zcb * yt - ycb * zt) + yca * dedzt - zca * dedyt + zdc * dedyu - ydc * dedzu;
         dedyib = terma * (zba * xt - xba * zt) + termc * (xcb * zt - zcb * xt) + zca * dedxt - xca * dedzt + xdc * dedzu - zdc * dedxu;
         dedzib = terma * (xba * yt - yba * xt) + termc * (ycb * xt - xcb * yt) + xca * dedyt - yca * dedxt + ydc * dedxu - xdc * dedyu;
         dedxic = termc * (ycb * zt - zcb * yt) + zba * dedyt - yba * dedzt + ydb * dedzu - zdb * dedyu;
         dedyic = termc * (zcb * xt - xcb * zt) + xba * dedzt - zba * dedxt + zdb * dedxu - xdb * dedzu;
         dedzic = termc * (xcb * yt - ycb * xt) + yba * dedxt - xba * dedyt + xdb * dedyu - ydb * dedxu;
         dedxid = zcb * dedyu - ycb * dedzu;
         dedyid = xcb * dedzu - zcb * dedxu;
         dedzid = ycb * dedxu - xcb * dedyu;
      }
   }

   {
      real v1 = kant[iangtor][3];
      real v2 = kant[iangtor][4];
      real v3 = kant[iangtor][5];
      int k = iat[iangtor][2];
      real dot = xcb * xdc + ycb * ydc + zcb * zdc;
      real cosang = -dot * REAL_RSQRT(rcb2 * rdc2);
      real angle = radian * REAL_ACOS(cosang);
      real dt = angle - anat[k];
      e2 = atorunit * dt * (v1 * phi1 + v2 * phi2 + v3 * phi3);
      if CONSTEXPR (do_g) {
         real dedphi = atorunit * dt * (v1 * dphi1 + v2 * dphi2 + v3 * dphi3);
         real ddt = atorunit * radian * (v1 * phi1 + v2 * phi2 + v3 * phi3);
         real dedxt = dedphi * (zcb * yt - ycb * zt) / (rt2 * rcb);
         real dedyt = dedphi * (xcb * zt - zcb * xt) / (rt2 * rcb);
         real dedzt = dedphi * (ycb * xt - xcb * yt) / (rt2 * rcb);
         real dedxu = dedphi * (ycb * zu - zcb * yu) / (ru2 * rcb);
         real dedyu = dedphi * (zcb * xu - xcb * zu) / (ru2 * rcb);
         real dedzu = dedphi * (xcb * yu - ycb * xu) / (ru2 * rcb);

         real termb = -ddt / (rcb2 * REAL_SQRT(ru2));
         real termd = ddt / (rdc2 * REAL_SQRT(ru2));
         dedxia += zcb * dedyt - ycb * dedzt;
         dedyia += xcb * dedzt - zcb * dedxt;
         dedzia += ycb * dedxt - xcb * dedyt;
         dedxib += termb * (zcb * yu - ycb * zu) + yca * dedzt - zca * dedyt + zdc * dedyu - ydc * dedzu;
         dedyib += termb * (xcb * zu - zcb * xu) + zca * dedxt - xca * dedzt + xdc * dedzu - zdc * dedxu;
         dedzib += termb * (ycb * xu - xcb * yu) + xca * dedyt - yca * dedxt + ydc * dedxu - xdc * dedyu;
         dedxic += termb * (ycb * zu - zcb * yu) + termd * (zdc * yu - ydc * zu) + zba * dedyt - yba * dedzt + ydb * dedzu - zdb * dedyu;
         dedyic += termb * (zcb * xu - xcb * zu) + termd * (xdc * zu - zdc * xu) + xba * dedzt - zba * dedxt + zdb * dedxu - xdb * dedzu;
         dedzic += termb * (xcb * yu - ycb * xu) + termd * (ydc * xu - xdc * yu) + yba * dedxt - xba * dedyt + xdb * dedyu - ydb * dedxu;
         dedxid += termd * (ydc * zu - zdc * yu) + zcb * dedyu - ycb * dedzu;
         dedyid += termd * (zdc * xu - xdc * zu) + xcb * dedzu - zcb * dedxu;
         dedzid += termd * (xdc * yu - ydc * xu) + ycb * dedxu - xcb * dedyu;
      }
   }

   if CONSTEXPR (do_e) {
      e = e1 + e2;
   }
   if CONSTEXPR (do_g) {
      atomic_add(dedxia, deatx, ia);
      atomic_add(dedyia, deaty, ia);
      atomic_add(dedzia, deatz, ia);
      atomic_add(dedxib, deatx, ib);
      atomic_add(dedyib, deaty, ib);
      atomic_add(dedzib, deatz, ib);
      atomic_add(dedxic, deatx, ic);
      atomic_add(dedyic, deaty, ic);
      atomic_add(dedzic, deatz, ic);
      atomic_add(dedxid, deatx, id);
      atomic_add(dedyid, deaty, id);
      atomic_add(dedzid, deatz, id);
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
