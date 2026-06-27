#pragma once
#include "ff/evalence.h"
#include "math/const.h"
#include "math/libfunc.h"
#include "seq/add.h"
#include "seq/seq.h"

namespace tinker {
/**
 * Comment
 * Zhi Wang, July 1, 2019
 *
 * The original Tinker implementation has following expressions
 * xip = xib + xt * delta
 * xap = xia - xip
 *
 * And they were reorganized to
 * xap = xia - xib - xt * delta
 * for higher accuracy in the single precision mode.
 *
 * Consider an example where
 * xia = 33.553368, xib = 34.768604
 * xt = 0.33142909, delta = 0.0044494048,
 * the later expression gives a better numerical result.
 */
#pragma acc routine seq
template <class Ver>
SEQ_CUDA
void dk_angle(real& restrict e,
              real& restrict vxx,
              real& restrict vyx,
              real& restrict vzx,
              real& restrict vyy,
              real& restrict vzy,
              real& restrict vzz,

              grad_prec* restrict deax,
              grad_prec* restrict deay,
              grad_prec* restrict deaz,

              const Angle* restrict angtyp,
              real angunit,
              int i,
              const int (*restrict iang)[4],
              const real* restrict anat,
              const real* restrict ak,
              const real* restrict afld,

              real cang,
              real qang,
              real pang,
              real sang,

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

   int ia = iang[i][0];
   int ib = iang[i][1];
   int ic = iang[i][2];
   int id = iang[i][3];
   real ideal = anat[i];
   real force = ak[i];
   Angle angtypi = angtyp[i];

   real xia = x[ia];
   real yia = y[ia];
   real zia = z[ia];
   real xib = x[ib];
   real yib = y[ib];
   real zib = z[ib];
   real xic = x[ic];
   real yic = y[ic];
   real zic = z[ic];

   if (angtypi != Angle::IN_PLANE) {
      real xab = xia - xib;
      real yab = yia - yib;
      real zab = zia - zib;
      real xcb = xic - xib;
      real ycb = yic - yib;
      real zcb = zic - zib;

      real rab2 = xab * xab + yab * yab + zab * zab;
      real rcb2 = xcb * xcb + ycb * ycb + zcb * zcb;

      if (rab2 != 0 && rcb2 != 0) {
         real xp = ycb * zab - zcb * yab;
         real yp = zcb * xab - xcb * zab;
         real zp = xcb * yab - ycb * xab;
         real rp = REAL_SQRT(xp * xp + yp * yp + zp * zp);
         rp = REAL_MAX(rp, (real)0.0001);
         real dot = xab * xcb + yab * ycb + zab * zcb;
         real cosine = dot * REAL_RSQRT(rab2 * rcb2);
         cosine = REAL_MIN((real)1, REAL_MAX((real)-1, cosine));
         real angle = radian * REAL_ACOS(cosine);

         real deddt;
         if (angtypi == Angle::HARMONIC) {
            real dt = angle - ideal;
            real dt2 = dt * dt;
            real dt3 = dt2 * dt;
            real dt4 = dt2 * dt2;
            if CONSTEXPR (do_e)
               e = angunit * force * dt2 * (1 + cang * dt + qang * dt2 + pang * dt3 + sang * dt4);
            if CONSTEXPR (do_g)
               deddt = angunit * force * dt * radian * (2 + 3 * cang * dt + 4 * qang * dt2 + 5 * pang * dt3 + 6 * sang * dt4);
         } else if (angtypi == Angle::LINEAR) {
            real factor = 2 * angunit * radian * radian;
            real sine = REAL_SQRT(1 - cosine * cosine);
            if CONSTEXPR (do_e)
               e = factor * force * (1 + cosine);
            if CONSTEXPR (do_g)
               deddt = -factor * force * sine;
         } else if (angtypi == Angle::FOURIER) {
            real fold = afld[i];
            real factor = 2 * angunit * (radian / fold) * (radian / fold);
            real dt = (fold * angle - ideal) * _1radian;
            if CONSTEXPR (do_e) {
               real cosine = REAL_COS(dt);
               e = factor * force * (1 + cosine);
            }
            if CONSTEXPR (do_g) {
               real sine = REAL_SIN(dt);
               deddt = -factor * force * fold * sine;
            }
         }

         if CONSTEXPR (do_g) {
            real terma = -deddt * REAL_RECIP(rab2 * rp);
            real termc = deddt * REAL_RECIP(rcb2 * rp);
            real dedxia = terma * (yab * zp - zab * yp);
            real dedyia = terma * (zab * xp - xab * zp);
            real dedzia = terma * (xab * yp - yab * xp);
            real dedxic = termc * (ycb * zp - zcb * yp);
            real dedyic = termc * (zcb * xp - xcb * zp);
            real dedzic = termc * (xcb * yp - ycb * xp);
            real dedxib = -dedxia - dedxic;
            real dedyib = -dedyia - dedyic;
            real dedzib = -dedzia - dedzic;

            atomic_add(dedxia, deax, ia);
            atomic_add(dedyia, deay, ia);
            atomic_add(dedzia, deaz, ia);
            atomic_add(dedxib, deax, ib);
            atomic_add(dedyib, deay, ib);
            atomic_add(dedzib, deaz, ib);
            atomic_add(dedxic, deax, ic);
            atomic_add(dedyic, deay, ic);
            atomic_add(dedzic, deaz, ic);

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
   } else {
      real xid = x[id];
      real yid = y[id];
      real zid = z[id];
      real xad = xia - xid;
      real yad = yia - yid;
      real zad = zia - zid;
      real xbd = xib - xid;
      real ybd = yib - yid;
      real zbd = zib - zid;
      real xcd = xic - xid;
      real ycd = yic - yid;
      real zcd = zic - zid;
      real xt = yad * zcd - zad * ycd;
      real yt = zad * xcd - xad * zcd;
      real zt = xad * ycd - yad * xcd;
      real rt2 = xt * xt + yt * yt + zt * zt;
      real delta = -(xt * xbd + yt * ybd + zt * zbd) * REAL_RECIP(rt2);
      real xap = xia - xib - xt * delta;
      real yap = yia - yib - yt * delta;
      real zap = zia - zib - zt * delta;
      real xcp = xic - xib - xt * delta;
      real ycp = yic - yib - yt * delta;
      real zcp = zic - zib - zt * delta;
      real rap2 = xap * xap + yap * yap + zap * zap;
      real rcp2 = xcp * xcp + ycp * ycp + zcp * zcp;
      if (rap2 != 0 && rcp2 != 0) {
         real dot = xap * xcp + yap * ycp + zap * zcp;
         real cosine = dot * REAL_RSQRT(rap2 * rcp2);
         cosine = REAL_MIN((real)1, REAL_MAX((real)-1, cosine));
         real angle = radian * REAL_ACOS(cosine);
         real dt = angle - ideal;
         real dt2 = dt * dt;
         real dt3 = dt2 * dt;
         real dt4 = dt2 * dt2;

         if CONSTEXPR (do_e) {
            e = angunit * force * dt2 * (1 + cang * dt + qang * dt2 + pang * dt3 + sang * dt4);
         }

         if CONSTEXPR (do_g) {
            real deddt = angunit * force * dt * radian * (2 + 3 * cang * dt + 4 * qang * dt2 + 5 * pang * dt3 + 6 * sang * dt4);
            real xm = ycp * zap - zcp * yap;
            real ym = zcp * xap - xcp * zap;
            real zm = xcp * yap - ycp * xap;
            real rm = REAL_SQRT(xm * xm + ym * ym + zm * zm);
            rm = REAL_MAX(rm, (real)0.0001);

            // chain rule terms for first derivative components

            real terma = -deddt * REAL_RECIP(rap2 * rm);
            real termc = deddt * REAL_RECIP(rcp2 * rm);
            real dedxia = terma * (yap * zm - zap * ym);
            real dedyia = terma * (zap * xm - xap * zm);
            real dedzia = terma * (xap * ym - yap * xm);
            real dedxic = termc * (ycp * zm - zcp * ym);
            real dedyic = termc * (zcp * xm - xcp * zm);
            real dedzic = termc * (xcp * ym - ycp * xm);
            real dedxip = -dedxia - dedxic;
            real dedyip = -dedyia - dedyic;
            real dedzip = -dedzia - dedzic;

            // chain rule components for the projection of the central atom

            real delta2, term;
            delta2 = 2 * delta;
            real ptrt2 = (dedxip * xt + dedyip * yt + dedzip * zt) * REAL_RECIP(rt2);
            term = (zcd * ybd - ycd * zbd) + delta2 * (yt * zcd - zt * ycd);
            real dpdxia = delta * (ycd * dedzip - zcd * dedyip) + term * ptrt2;
            term = (xcd * zbd - zcd * xbd) + delta2 * (zt * xcd - xt * zcd);
            real dpdyia = delta * (zcd * dedxip - xcd * dedzip) + term * ptrt2;
            term = (ycd * xbd - xcd * ybd) + delta2 * (xt * ycd - yt * xcd);
            real dpdzia = delta * (xcd * dedyip - ycd * dedxip) + term * ptrt2;
            term = (yad * zbd - zad * ybd) + delta2 * (zt * yad - yt * zad);
            real dpdxic = delta * (zad * dedyip - yad * dedzip) + term * ptrt2;
            term = (zad * xbd - xad * zbd) + delta2 * (xt * zad - zt * xad);
            real dpdyic = delta * (xad * dedzip - zad * dedxip) + term * ptrt2;
            term = (xad * ybd - yad * xbd) + delta2 * (yt * xad - xt * yad);
            real dpdzic = delta * (yad * dedxip - xad * dedyip) + term * ptrt2;

            // compute derivative components for this interaction

            dedxia = dedxia + dpdxia;
            dedyia = dedyia + dpdyia;
            dedzia = dedzia + dpdzia;
            real dedxib = dedxip;
            real dedyib = dedyip;
            real dedzib = dedzip;
            dedxic = dedxic + dpdxic;
            dedyic = dedyic + dpdyic;
            dedzic = dedzic + dpdzic;
            real dedxid = -dedxia - dedxib - dedxic;
            real dedyid = -dedyia - dedyib - dedyic;
            real dedzid = -dedzia - dedzib - dedzic;

            atomic_add(dedxia, deax, ia);
            atomic_add(dedyia, deay, ia);
            atomic_add(dedzia, deaz, ia);
            atomic_add(dedxib, deax, ib);
            atomic_add(dedyib, deay, ib);
            atomic_add(dedzib, deaz, ib);
            atomic_add(dedxic, deax, ic);
            atomic_add(dedyic, deay, ic);
            atomic_add(dedzic, deaz, ic);
            atomic_add(dedxid, deax, id);
            atomic_add(dedyid, deay, id);
            atomic_add(dedzid, deaz, id);

            if CONSTEXPR (do_v) {
               vxx = xad * dedxia + xbd * dedxib + xcd * dedxic;
               vyx = yad * dedxia + ybd * dedxib + ycd * dedxic;
               vzx = zad * dedxia + zbd * dedxib + zcd * dedxic;
               vyy = yad * dedyia + ybd * dedyib + ycd * dedyic;
               vzy = zad * dedyia + zbd * dedyib + zcd * dedyic;
               vzz = zad * dedzia + zbd * dedzib + zcd * dedzic;
            }
         }
      }
   }
}
}
