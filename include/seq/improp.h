#pragma once
#include "math/const.h"
#include "math/libfunc.h"
#include "seq/add.h"
#include "seq/seq.h"

namespace tinker {
// Computing angle from sine instead of cosine may have a higher precision.
#pragma acc routine seq
template <class Ver>
SEQ_CUDA
void dk_improp(real& restrict e,
               real& restrict vxx,
               real& restrict vyx,
               real& restrict vzx,
               real& restrict vyy,
               real& restrict vzy,
               real& restrict vzz,

               grad_prec* restrict deidx,
               grad_prec* restrict deidy,
               grad_prec* restrict deidz,

               real idihunit,
               int i,
               const int (*restrict iiprop)[4],
               const real* restrict kprop,
               const real* restrict vprop,

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

   const int ia = iiprop[i][0];
   const int ib = iiprop[i][1];
   const int ic = iiprop[i][2];
   const int id = iiprop[i][3];
   real ideal = vprop[i];
   real force = kprop[i];

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
      real sine = (xcb * xtu + ycb * ytu + zcb * ztu) * REAL_RECIP(rcb * rtru);
      real angle = radian * REAL_ASIN(sine);

      if (REAL_ABS(angle + ideal) < REAL_ABS(angle - ideal))
         ideal = -ideal;
      real dt = angle - ideal;
      if (dt > 180)
         dt -= 360;
      if (dt < -180)
         dt += 360;

      if CONSTEXPR (do_e)
         e = idihunit * force * dt * dt;

      if CONSTEXPR (do_g) {
         real dedphi = 2 * idihunit * force * dt * radian;
         real xca = xic - xia;
         real yca = yic - yia;
         real zca = zic - zia;
         real xdb = xid - xib;
         real ydb = yid - yib;
         real zdb = zid - zib;

         real dedxt = dedphi * (yt * zcb - ycb * zt) * REAL_RECIP(rt2 * rcb);
         real dedyt = dedphi * (zt * xcb - zcb * xt) * REAL_RECIP(rt2 * rcb);
         real dedzt = dedphi * (xt * ycb - xcb * yt) * REAL_RECIP(rt2 * rcb);
         real dedxu = -dedphi * (yu * zcb - ycb * zu) * REAL_RECIP(ru2 * rcb);
         real dedyu = -dedphi * (zu * xcb - zcb * xu) * REAL_RECIP(ru2 * rcb);
         real dedzu = -dedphi * (xu * ycb - xcb * yu) * REAL_RECIP(ru2 * rcb);

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

         atomic_add(dedxia, deidx, ia);
         atomic_add(dedyia, deidy, ia);
         atomic_add(dedzia, deidz, ia);
         atomic_add(dedxib, deidx, ib);
         atomic_add(dedyib, deidy, ib);
         atomic_add(dedzib, deidz, ib);
         atomic_add(dedxic, deidx, ic);
         atomic_add(dedyic, deidy, ic);
         atomic_add(dedzic, deidz, ic);
         atomic_add(dedxid, deidx, id);
         atomic_add(dedyid, deidy, id);
         atomic_add(dedzid, deidz, id);

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
