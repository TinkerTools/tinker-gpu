#pragma once
#include "math/libfunc.h"
#include "seq/pair_vlambda.h"

namespace tinker {
/**
 * \ingroup vdw
 */
#pragma acc routine seq
template <bool DO_G, bool SOFTCORE>
SEQ_CUDA
void pair_lj_v0(real r, real invr, real vlambda, real rad, real eps, real& restrict ev, real& restrict dev)
{
   if CONSTEXPR (SOFTCORE) {
      if (rad == 0) {
         ev = 0;
         if CONSTEXPR (DO_G)
            dev = 0;
         return;
      }
      real sig = invr * rad;
      real sig2 = sig * sig;
      real p6 = 1 / (sig2 * sig2 * sig2);
      p6 = p6 * 2;
      real sc = p6 + (1 - vlambda) / 2;
      real invsc = 1 / sc;
      real term = 4 * vlambda * eps * invsc * invsc;
      ev = term * (1 - sc);
      if CONSTEXPR (DO_G) {
         real dterm = -6 * p6 * term * invr;
         dev = dterm * (2 * invsc - 1);
      }
   } else {
      real sig = invr * rad;
      real sig2 = sig * sig;
      real p6 = sig2 * sig2 * sig2;
      real ep6 = eps * p6;
      ev = ep6 * (p6 - 2);
      if CONSTEXPR (DO_G)
         dev = ep6 * (p6 - 1) * (-12 * invr);
   }
}

/**
 * \ingroup vdw
 */
#pragma acc routine seq
template <bool DO_G, bool SOFTCORE>
SEQ_CUDA
void pair_lj_v1(real rik, real vlambda, real rv, real eps, real vscalek, real& restrict e, real& restrict de)
{
   eps *= vscalek;
   pair_lj_v0<DO_G, SOFTCORE>(rik, 1 / rik, vlambda, rv, eps, e, de);
}

/**
 * \ingroup vdw
 */
#pragma acc routine seq
template <bool DO_G, bool SOFTCORE, class RADRULE, class EPSRULE, int SCALE>
SEQ_CUDA
void pair_lj_v2(real r,
                real invr,
                real vlambda, //
                real vscale,
                real radi,
                real epsi,
                real radk,
                real epsk,
                real evcut,
                real evoff,
                real& restrict ev,
                real& restrict dev)
{
   if (r > evoff) {
      ev = 0;
      if CONSTEXPR (DO_G)
         dev = 0;
      return;
   }
   real rad = RADRULE::avg(radi, radk);
   real eps = EPSRULE::savg(epsi, epsk);
   if CONSTEXPR (SCALE != 1)
      eps *= vscale;
   pair_lj_v0<DO_G, SOFTCORE>(r, invr, vlambda, rad, eps, ev, dev);
   // taper
   if (r > evcut) {
      real taper, dtaper;
      switchTaper5<DO_G>(r, evcut, evoff, taper, dtaper);
      if CONSTEXPR (DO_G)
         dev = ev * dtaper + dev * taper;
      ev = ev * taper;
   }
}

/**
 * \ingroup vdw
 */
#pragma acc routine seq
template <bool DO_G, bool SOFTCORE, int SCALE>
SEQ_CUDA
void pair_lj_v3(real r,
                real invr,
                real vlambda, //
                real vscale,
                real rad,
                real eps,
                real evcut,
                real evoff,
                real& restrict ev,
                real& restrict dev)
{
   if CONSTEXPR (SCALE != 1) {
      eps *= vscale;
   }
   pair_lj_v0<DO_G, SOFTCORE>(r, invr, vlambda, rad, eps, ev, dev);
   // taper
   if (r > evcut) {
      real taper, dtaper;
      switchTaper5<DO_G>(r, evcut, evoff, taper, dtaper);
      if CONSTEXPR (DO_G)
         dev = ev * dtaper + dev * taper;
      ev = ev * taper;
   }
}
}
