#pragma once
#include "math/libfunc.h"
#include "seq/add.h"
#include "seq/seq.h"

namespace tinker {
#pragma acc routine seq
template <class Ver>
SEQ_CUDA
void dk_pitors(real& restrict e,
               real& restrict vxx,
               real& restrict vyx,
               real& restrict vzx,
               real& restrict vyy,
               real& restrict vzy,
               real& restrict vzz,

               grad_prec* restrict deptx,
               grad_prec* restrict depty,
               grad_prec* restrict deptz,

               real ptorunit,
               int i,
               const int (*restrict ipit)[6],
               const real* restrict kpit,

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

   const int ia = ipit[i][0];
   const int ib = ipit[i][1];
   const int ic = ipit[i][2];
   const int id = ipit[i][3];
   const int ie = ipit[i][4];
   const int ig = ipit[i][5];

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
   real xie = x[ie];
   real yie = y[ie];
   real zie = z[ie];
   real xig = x[ig];
   real yig = y[ig];
   real zig = z[ig];
   real xad = xia - xid;
   real yad = yia - yid;
   real zad = zia - zid;
   real xbd = xib - xid;
   real ybd = yib - yid;
   real zbd = zib - zid;
   real xec = xie - xic;
   real yec = yie - yic;
   real zec = zie - zic;
   real xgc = xig - xic;
   real ygc = yig - yic;
   real zgc = zig - zic;

   real xip = yad * zbd - ybd * zad + xic;
   real yip = zad * xbd - zbd * xad + yic;
   real zip = xad * ybd - xbd * yad + zic;
   real xiq = yec * zgc - ygc * zec + xid;
   real yiq = zec * xgc - zgc * xec + yid;
   real ziq = xec * ygc - xgc * yec + zid;
   real xcp = xic - xip;
   real ycp = yic - yip;
   real zcp = zic - zip;
   real xdc = xid - xic;
   real ydc = yid - yic;
   real zdc = zid - zic;
   real xqd = xiq - xid;
   real yqd = yiq - yid;
   real zqd = ziq - zid;

   real xt = ycp * zdc - ydc * zcp;
   real yt = zcp * xdc - zdc * xcp;
   real zt = xcp * ydc - xdc * ycp;
   real xu = ydc * zqd - yqd * zdc;
   real yu = zdc * xqd - zqd * xdc;
   real zu = xdc * yqd - xqd * ydc;
   real xtu = yt * zu - yu * zt;
   real ytu = zt * xu - zu * xt;
   real ztu = xt * yu - xu * yt;
   real rt2 = xt * xt + yt * yt + zt * zt;
   real ru2 = xu * xu + yu * yu + zu * zu;
   real rtru = REAL_SQRT(rt2 * ru2);

   if (rtru != 0) {
      real rdc = REAL_SQRT(xdc * xdc + ydc * ydc + zdc * zdc);
      real cosine = (xt * xu + yt * yu + zt * zu) * REAL_RECIP(rtru);
      real sine = (xdc * xtu + ydc * ytu + zdc * ztu) * REAL_RECIP(rdc * rtru);

      // set the pi-system torsion parameters for this angle

      real v2 = kpit[i];
      real c2 = -1;
      real s2 = 0;

      // compute the multiple angle trigonometry and the phase terms

      real cosine2 = cosine * cosine - sine * sine;
      real sine2 = 2 * cosine * sine;
      real phi2 = 1 + (cosine2 * c2 + sine2 * s2);

      // calculate the pi-system torsion energy for this angle

      if CONSTEXPR (do_e)
         e = ptorunit * v2 * phi2;

      if CONSTEXPR (do_g) {
         real dphi2 = 2 * (cosine2 * s2 - sine2 * c2);
         real dedphi = ptorunit * v2 * dphi2;

         // chain rule terms for first derivative components

         real xdp = xid - xip;
         real ydp = yid - yip;
         real zdp = zid - zip;
         real xqc = xiq - xic;
         real yqc = yiq - yic;
         real zqc = ziq - zic;
         real rt2rdc_inv = REAL_RECIP(rt2 * rdc);
         real ru2rdc_inv = REAL_RECIP(ru2 * rdc);
         real dedxt = dedphi * (yt * zdc - ydc * zt) * rt2rdc_inv;
         real dedyt = dedphi * (zt * xdc - zdc * xt) * rt2rdc_inv;
         real dedzt = dedphi * (xt * ydc - xdc * yt) * rt2rdc_inv;
         real dedxu = -dedphi * (yu * zdc - ydc * zu) * ru2rdc_inv;
         real dedyu = -dedphi * (zu * xdc - zdc * xu) * ru2rdc_inv;
         real dedzu = -dedphi * (xu * ydc - xdc * yu) * ru2rdc_inv;

         // compute first derivative components for pi-system angle

         real dedxip = zdc * dedyt - ydc * dedzt;
         real dedyip = xdc * dedzt - zdc * dedxt;
         real dedzip = ydc * dedxt - xdc * dedyt;
         real dedxic = ydp * dedzt - zdp * dedyt + zqd * dedyu - yqd * dedzu;
         real dedyic = zdp * dedxt - xdp * dedzt + xqd * dedzu - zqd * dedxu;
         real dedzic = xdp * dedyt - ydp * dedxt + yqd * dedxu - xqd * dedyu;
         real dedxid = zcp * dedyt - ycp * dedzt + yqc * dedzu - zqc * dedyu;
         real dedyid = xcp * dedzt - zcp * dedxt + zqc * dedxu - xqc * dedzu;
         real dedzid = ycp * dedxt - xcp * dedyt + xqc * dedyu - yqc * dedxu;
         real dedxiq = zdc * dedyu - ydc * dedzu;
         real dedyiq = xdc * dedzu - zdc * dedxu;
         real dedziq = ydc * dedxu - xdc * dedyu;

         // compute first derivative components for individual atoms

         real dedxia = ybd * dedzip - zbd * dedyip;
         real dedyia = zbd * dedxip - xbd * dedzip;
         real dedzia = xbd * dedyip - ybd * dedxip;
         real dedxib = zad * dedyip - yad * dedzip;
         real dedyib = xad * dedzip - zad * dedxip;
         real dedzib = yad * dedxip - xad * dedyip;
         real dedxie = ygc * dedziq - zgc * dedyiq;
         real dedyie = zgc * dedxiq - xgc * dedziq;
         real dedzie = xgc * dedyiq - ygc * dedxiq;
         real dedxig = zec * dedyiq - yec * dedziq;
         real dedyig = xec * dedziq - zec * dedxiq;
         real dedzig = yec * dedxiq - xec * dedyiq;

         dedxic += (dedxip - dedxie - dedxig);
         dedyic += (dedyip - dedyie - dedyig);
         dedzic += (dedzip - dedzie - dedzig);
         dedxid += (dedxiq - dedxia - dedxib);
         dedyid += (dedyiq - dedyia - dedyib);
         dedzid += (dedziq - dedzia - dedzib);

         atomic_add(dedxia, deptx, ia);
         atomic_add(dedyia, depty, ia);
         atomic_add(dedzia, deptz, ia);
         atomic_add(dedxib, deptx, ib);
         atomic_add(dedyib, depty, ib);
         atomic_add(dedzib, deptz, ib);
         atomic_add(dedxic, deptx, ic);
         atomic_add(dedyic, depty, ic);
         atomic_add(dedzic, deptz, ic);
         atomic_add(dedxid, deptx, id);
         atomic_add(dedyid, depty, id);
         atomic_add(dedzid, deptz, id);
         atomic_add(dedxie, deptx, ie);
         atomic_add(dedyie, depty, ie);
         atomic_add(dedzie, deptz, ie);
         atomic_add(dedxig, deptx, ig);
         atomic_add(dedyig, depty, ig);
         atomic_add(dedzig, deptz, ig);

         if CONSTEXPR (do_v) {
            real vxterm = dedxid + dedxia + dedxib;
            real vyterm = dedyid + dedyia + dedyib;
            real vzterm = dedzid + dedzia + dedzib;
            vxx = xdc * vxterm + xcp * dedxip - xqd * dedxiq;
            vyx = ydc * vxterm + ycp * dedxip - yqd * dedxiq;
            vzx = zdc * vxterm + zcp * dedxip - zqd * dedxiq;
            vyy = ydc * vyterm + ycp * dedyip - yqd * dedyiq;
            vzy = zdc * vyterm + zcp * dedyip - zqd * dedyiq;
            vzz = zdc * vzterm + zcp * dedzip - zqd * dedziq;
         }
      }
   }
}
}
