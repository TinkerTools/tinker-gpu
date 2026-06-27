#pragma once
#include "ff/elec.h"
#include "math/const.h"
#include "math/libfunc.h"
#include "math/switch.h"
#include "seq/add.h"
#include "seq/seq.h"

namespace tinker {
#pragma acc routine seq
template <class Ver, class ETYP>
SEQ_CUDA
void pair_charge(real r,
                 real xr,
                 real yr,
                 real zr,
                 real cscale,
                 real chgi,
                 real chgk,
                 real ebuffer,
                 real f,
                 real aewald,
                 real cut,
                 real off,
                 real& restrict grdx,
                 real& restrict grdy,
                 real& restrict grdz,
                 int& restrict ctl,
                 real& restrict etl,
                 real& restrict vtlxx,
                 real& restrict vtlxy,
                 real& restrict vtlxz,
                 real& restrict vtlyy,
                 real& restrict vtlyz,
                 real& restrict vtlzz)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;
   constexpr bool taper_flag = eq<ETYP, NON_EWALD_TAPER>();
   MAYBE_UNUSED real e, dedx, dedy, dedz;

   if CONSTEXPR (eq<ETYP, EWALD>()) {
      real fik = f * chgi * chgk;
      real rew = aewald * r;
      real erfterm = REAL_ERFC(rew);
      real rb = r + ebuffer;
      real invrb = REAL_RECIP(rb);

      if CONSTEXPR (do_e)
         e = fik * invrb * erfterm;
      if CONSTEXPR (do_g) {
         real invr = REAL_RECIP(r);
         real invrb2 = invrb * invrb;
         real de = erfterm * invrb2;
         de += (2 * aewald / sqrtpi) * REAL_EXP(-rew * rew) * invrb;
         de *= -fik;
         de *= invr;
         dedx = de * xr;
         dedy = de * yr;
         dedz = de * zr;
      }
   } else if CONSTEXPR (eq<ETYP, NON_EWALD>() || taper_flag) {
      real fik = cscale * f * chgi * chgk;
      real rb = r + ebuffer;
      real invrb = REAL_RECIP(rb);

      // always calculate e
      e = fik * invrb;
      MAYBE_UNUSED real de, invr;
      if CONSTEXPR (do_g) {
         invr = REAL_RECIP(r);
         real invrb2 = invrb * invrb;
         de = -fik * invrb2;
      }

      // shifted energy switching
      //
      // empirical (?) 7th degree additive switching function
      // e = taper (e - shift) + trans
      //
      // cut = A; off = B; x = (r - A) / (B - A); s = B - A
      //
      // shift = fik / ((A + B)/2)
      //
      // trans = coef poly(x)
      // coef = fik (1/A - 1/B) / 9.3
      // poly(x) = 64 x^3 - 217 x^4 + 267 x^5 - 139 x^6 + 25 x^7
      //         = x^3 (1-x)^3 (64 - 25 x)
      //
      // dtrans = coef poly'(x) / s; dtrans = d/dr trans
      // poly'(x) = x^2 (1-x)^2 (25 x - 12) (7 x - 16)

      if CONSTEXPR (taper_flag) {
         // shifted energy
         real shift = fik * 2 * REAL_RECIP(cut + off);
         e -= shift;

         // switching
         if (r > cut) {
            real taper, dtaper;
            switchTaper5<do_g>(r, cut, off, taper, dtaper);

            real trans, dtrans;
            real coef = fik * (REAL_RECIP(cut) - REAL_RECIP(off)) * REAL_RECIP((real)9.3);
            real invs = REAL_RECIP(off - cut);
            real x = (r - cut) * invs;
            real y = (1 - x) * x;
            trans = coef * y * y * y * (64 - 25 * x);
            if CONSTEXPR (do_g) {
               dtrans = y * y * (25 * x - 12) * (7 * x - 16);
               dtrans *= coef * invs;
            }

            if CONSTEXPR (do_g)
               de = e * dtaper + de * taper + dtrans;
            if CONSTEXPR (do_e)
               e = e * taper + trans;
         }
      }

      if CONSTEXPR (do_g) {
         de *= invr;
         dedx = de * xr;
         dedy = de * yr;
         dedz = de * zr;
      }
   }

   if CONSTEXPR (do_a) {
      // ci and ck may be zero
      if (e != 0) {
         // scale =  1.0 -- count +1  scale =  0.4 or 0.0 -- impossible
         // scale = -0.6 -- count +0  scale = -1.0        -- count -1
         if (cscale == 1)
            ctl += 1;
         else if (cscale == -1)
            ctl -= 1;
      }
   }
   if CONSTEXPR (do_e)
      etl += e;
   if CONSTEXPR (do_g) {
      grdx += dedx;
      grdy += dedy;
      grdz += dedz;
   }
   if CONSTEXPR (do_v) {
      real vxx = xr * dedx;
      real vyx = yr * dedx;
      real vzx = zr * dedx;
      real vyy = yr * dedy;
      real vzy = zr * dedy;
      real vzz = zr * dedz;
      vtlxx += vxx;
      vtlxy += vyx;
      vtlxz += vzx;
      vtlyy += vyy;
      vtlyz += vzy;
      vtlzz += vzz;
   }
}

#pragma acc routine seq
template <bool DO_G, class ETYP, int SCALE>
SEQ_CUDA
void pair_chg_v2(real r,
                 real invr, //
                 real cscale,
                 real chgi,
                 real chgk,
                 real f,
                 real aewald,
                 real eccut,
                 real ecoff,
                 real& restrict ec,
                 real& restrict dec)
{
   if (r > ecoff) {
      ec = 0;
      if CONSTEXPR (DO_G)
         dec = 0;
      return;
   }
   real fik = f * chgi * chgk;
   if CONSTEXPR (eq<ETYP, EWALD>()) {
      const real rew = aewald * r;
      const real expterm = REAL_EXP(-rew * rew);
      real erfterm = REAL_ERFC_V2(rew, expterm);
      if CONSTEXPR (SCALE != 1)
         erfterm += cscale - 1;
      ec = fik * invr * erfterm;
      if CONSTEXPR (DO_G) {
         constexpr real two = 2.0f / sqrtpi;
         dec = erfterm * invr + two * aewald * expterm;
         dec *= -fik * invr;
      }
   } else if CONSTEXPR (eq<ETYP, NON_EWALD_TAPER>()) {
      if CONSTEXPR (SCALE != 1)
         fik *= cscale;
      ec = fik * invr;
      if CONSTEXPR (DO_G)
         dec = -ec * invr;

      // shift energy
      real shift = fik * 2 * REAL_RECIP(eccut + ecoff);
      ec -= shift;
      if (r > eccut) {
         real taper, dtaper;
         switchTaper5<DO_G>(r, eccut, ecoff, taper, dtaper);
         real trans, dtrans;
         real coef = fik * (REAL_RECIP(eccut) - REAL_RECIP(ecoff)) * REAL_RECIP((real)9.3);
         real invs = REAL_RECIP(ecoff - eccut);
         real x = (r - eccut) * invs;
         real y = (1 - x) * x;
         trans = coef * y * y * y * (64 - 25 * x);
         if CONSTEXPR (DO_G) {
            dtrans = y * y * (25 * x - 12) * (7 * x - 16);
            dtrans *= coef * invs;
            dec = ec * dtaper + dec * taper + dtrans;
         }
         ec = ec * taper + trans;
      }
   }
}

#pragma acc routine seq
template <bool DO_G, class ETYP, int SCALE>
SEQ_CUDA
void pair_chg_v3(real r, real cscale, real chgi, real chgk, real ebuffer, real f, real aewald, real eccut, real ecoff, real& restrict ec, real& restrict dec)
{
   real fik = f * chgi * chgk;
   real rb = r + ebuffer;
   real invrb = REAL_RECIP(rb);
   if CONSTEXPR (eq<ETYP, EWALD>()) {
      const real rew = aewald * r;
      const real expterm = REAL_EXP(-rew * rew);
      real erfterm = REAL_ERFC_V2(rew, expterm);
      if CONSTEXPR (SCALE != 1)
         erfterm += cscale - 1;
      ec = fik * invrb * erfterm;
      if CONSTEXPR (DO_G) {
         constexpr real two = 2.0f / sqrtpi;
         dec = erfterm * invrb + two * aewald * expterm;
         dec *= -fik * invrb;
      }
   } else if CONSTEXPR (eq<ETYP, NON_EWALD_TAPER>()) {
      if CONSTEXPR (SCALE != 1)
         fik *= cscale;
      ec = fik * invrb;
      if CONSTEXPR (DO_G)
         dec = -ec * invrb;

      // shift energy
      real shift = fik * 2 * REAL_RECIP(eccut + ecoff);
      ec -= shift;
      if (r > eccut) {
         real taper, dtaper;
         switchTaper5<DO_G>(r, eccut, ecoff, taper, dtaper);
         real trans, dtrans;
         real coef = fik * (REAL_RECIP(eccut) - REAL_RECIP(ecoff)) * REAL_RECIP((real)9.3);
         real invs = REAL_RECIP(ecoff - eccut);
         real x = (r - eccut) * invs;
         real y = (1 - x) * x;
         trans = coef * y * y * y * (64 - 25 * x);
         if CONSTEXPR (DO_G) {
            dtrans = y * y * (25 * x - 12) * (7 * x - 16);
            dtrans *= coef * invs;
            dec = ec * dtaper + dec * taper + dtrans;
         }
         ec = ec * taper + trans;
      }
   }
}
}
