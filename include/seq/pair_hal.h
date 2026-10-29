#pragma once
#include "ff/dlmda.h"
#include "math/libfunc.h"
#include "math/switch.h"
#include "seq/seq.h"

namespace tinker {
/**
 * \ingroup vdw
 */
struct PairHalLambda
{
   real dedl;
   real d2edl2;
   real dlde;
};

#pragma acc routine seq
template <bool DO_G, int SCALE, bool DO_DL1, bool DO_DL2, bool DO_DLDE>
SEQ_CUDA
void pair_hal(real r,
              real vscale,
              real rv,
              real eps,
              real evcut,
              real evoff,
              const SoftCoreLmda& lp,
              real ghal,
              real dhal,
              real& restrict e,
              real& restrict de,
              PairHalLambda* dl,
              bool dlon)
{
   if CONSTEXPR (SCALE != 1)
      eps *= vscale;
   real eps0 = eps;
   real invrv = REAL_RECIP(rv);
   real rho = r * invrv;
   real rho2 = rho * rho;
   real rho6 = rho2 * rho2 * rho2;
   real rho7 = rho6 * rho;
   real rhod = rho + dhal;
   real rhod2 = rhod * rhod;
   real rhod6 = rhod2 * rhod2 * rhod2;
   real dhal7 = REAL_POW(1 + dhal, 7); // folded when dhal is a constant
   eps *= lp.p0;
   real s1 = REAL_RECIP(lp.sc + rhod6 * rhod);
   real s2 = REAL_RECIP(lp.sc + rho7 + ghal);
   real t1 = dhal7 * s1;
   real t2 = (1 + ghal) * s2;
   e = eps * t1 * (t2 - 2);
   if CONSTEXPR (DO_G) {
      real dt1drho = -7 * rhod6 * t1 * s1;
      real dt2drho = -7 * rho6 * t2 * s2;
      de = eps * (dt1drho * (t2 - 2) + t1 * dt2drho) * invrv;
   }

   // The lambda derivatives are only wanted for pairs that couple to lambda.
   // The pair energy is eps v^n T(s) with T = t1 (t2 - 2), and t1 and t2 change
   // with the softening s as -t1 s1 and -t2 s2, so each main lambda derivative
   // combines T and its derivatives with respect to s through the coefficients
   // of the soft core (ehalsc.f:ehalsc4).
   constexpr bool DO_DL = DO_DL1 or DO_DL2 or DO_DLDE;
   if (DO_DL and dlon) {
      real tt = t1 * (t2 - 2);
      real ts = -t1 * s1 * (t2 - 2) - t1 * t2 * s2;
      dl->dedl = eps0 * (lp.a1 * tt + lp.b1 * ts);

      if CONSTEXPR (DO_DL2) {
         real tss = 2 * t1 * s1 * s1 * (t2 - 2) + 2 * t1 * t2 * s1 * s2 + 2 * t1 * t2 * s2 * s2;
         dl->d2edl2 = eps0 * (lp.a2 * tt + lp.b2 * ts + lp.c2 * tss);
      }

      if CONSTEXPR (DO_DLDE) {
         real dt1drho = -7 * rhod6 * t1 * s1;
         real dt2drho = -7 * rho6 * t2 * s2;
         real tr = dt1drho * (t2 - 2) + t1 * dt2drho;
         real trs = -dt1drho * (2 * s1 * (t2 - 2) + t2 * s2) - t1 * dt2drho * (s1 + 2 * s2);
         dl->dlde = eps0 * (lp.a1 * tr + lp.b1 * trs) * invrv;
      }
   }

   if (r > evcut) {
      real taper, dtaper;
      switchTaper5<DO_G>(r, evcut, evoff, taper, dtaper);
      if CONSTEXPR (DO_G) {
         de = e * dtaper + de * taper;
         if CONSTEXPR (DO_DLDE)
            dl->dlde = dl->dedl * dtaper + dl->dlde * taper;
      }
      e *= taper;
      if CONSTEXPR (DO_DL)
         dl->dedl *= taper;
      if CONSTEXPR (DO_DL2)
         dl->d2edl2 *= taper;
   }
}

/**
 * \ingroup vdw
 */
#pragma acc routine seq
template <bool DO_G, int SCALE>
SEQ_CUDA
void pair_hal_v2(real r,
                 real vscale,
                 real rv,
                 real eps,
                 real evcut,
                 real evoff,
                 const SoftCoreLmda& lp,
                 real ghal,
                 real dhal,
                 real& restrict e,
                 real& restrict de)
{
   pair_hal<DO_G, SCALE, false, false, false>(r, vscale, rv, eps, evcut, evoff, lp, ghal, dhal, e, de, nullptr, false);
}
}
