#pragma once
#include "ff/dlmda.h"
#include "math/libfunc.h"
#include "math/switch.h"
#include "seq/seq.h"

namespace tinker {
/**
 * \ingroup vdw
 */
#pragma acc routine seq
template <bool DO_G>
SEQ_CUDA
void pair_hal(real rik,
              real rv,
              real eps,
              real vscalek,
              real vlambda, //
              real ghal,
              real dhal,
              real scexp,
              real scalphav, //
              real& restrict e,
              real& restrict de)
{
   eps *= vscalek;
   real invrv = REAL_RECIP(rv);
   real rho = rik * invrv;
   real rho2 = rho * rho;
   real rho6 = rho2 * rho2 * rho2;
   real rho7 = rho6 * rho;
   real rhod = rho + dhal;
   real rhod2 = rhod * rhod;
   real rhod6 = rhod2 * rhod2 * rhod2;
   real dhal7 = REAL_POW(1 + dhal, 7); // folded when dhal is a constant
   eps *= REAL_POW(vlambda, scexp);
   real one_minus_lambda = 1 - vlambda;
   real scal = scalphav * one_minus_lambda * one_minus_lambda;
   real s1 = REAL_RECIP(scal + rhod6 * rhod);
   real s2 = REAL_RECIP(scal + rho7 + ghal);
   real t1 = dhal7 * s1;
   real t2 = (1 + ghal) * s2;
   e = eps * t1 * (t2 - 2);
   if CONSTEXPR (DO_G) {
      real dt1drho = -7 * rhod6 * t1 * s1;
      real dt2drho = -7 * rho6 * t2 * s2;
      de = eps * (dt1drho * (t2 - 2) + t1 * dt2drho) * invrv;
   }
}

/**
 * \ingroup vdw
 */
struct PairHalLambda
{
   real dedl;
   real d2edl2;
   real dlde;
};

/**
 * \ingroup vdw
 * \brief Powers of the lambda a pair couples at: \f$ v^s \f$, \f$ v^{s-1} \f$ and
 * \f$ v^{s-2} \f$ for the soft core exponent \f$ s \f$. The host takes them once
 * per call, so the pair loop needs no pow.
 */
struct PairHalLambdaPow
{
   real p0;
   real p1;
   real p2;
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
              real vlambda,
              const PairHalLambdaPow& lp,
              real ghal,
              real dhal,
              real scexp,
              real scalphav,
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
   real lambdaexp = lp.p0;
   eps *= lambdaexp;
   real one_minus_lambda = 1 - vlambda;
   real scal = scalphav * one_minus_lambda * one_minus_lambda;
   real s1 = REAL_RECIP(scal + rhod6 * rhod);
   real s2 = REAL_RECIP(scal + rho7 + ghal);
   real t1 = dhal7 * s1;
   real t2 = (1 + ghal) * s2;
   e = eps * t1 * (t2 - 2);
   if CONSTEXPR (DO_G) {
      real dt1drho = -7 * rhod6 * t1 * s1;
      real dt2drho = -7 * rho6 * t2 * s2;
      de = eps * (dt1drho * (t2 - 2) + t1 * dt2drho) * invrv;
   }

   // The lambda derivatives are only wanted for pairs that couple to lambda.
   constexpr bool DO_DL = DO_DL1 or DO_DL2 or DO_DLDE;
   if (DO_DL and dlon) {
      real dt0dl = eps0 * scexp * lp.p1;
      real dscaldl = 2 * scalphav * (1 - vlambda);
      real ds1dl = dscaldl * s1 * s1;
      real ds2dl = dscaldl * s2 * s2;
      real dt1dl = dhal7 * ds1dl;
      real dt2dl = (1 + ghal) * ds2dl;
      dl->dedl = dt0dl * t1 * (t2 - 2) + eps * dt1dl * (t2 - 2) + eps * t1 * dt2dl;

      if CONSTEXPR (DO_DL2) {
         real d2t0dl2 = 0;
         if (scexp >= 2)
            d2t0dl2 = eps0 * scexp * (scexp - 1) * lp.p2;
         real d2t1dl2 = dhal7 * (-2 * scalphav * s1 * s1 + 2 * dscaldl * s1 * ds1dl);
         real d2t2dl2 = (1 + ghal) * (-2 * scalphav * s2 * s2 + 2 * dscaldl * s2 * ds2dl);
         dl->d2edl2 = d2t0dl2 * t1 * (t2 - 2) + eps * d2t1dl2 * (t2 - 2) + eps * t1 * d2t2dl2
            + 2 * dt0dl * dt1dl * (t2 - 2) + 2 * dt0dl * t1 * dt2dl + 2 * eps * dt1dl * dt2dl;
      }

      if CONSTEXPR (DO_DLDE) {
         real dt1drho = -7 * rhod6 * t1 * s1;
         real dt2drho = -7 * rho6 * t2 * s2;
         real d2t1dldrho = -14 * dhal7 * s1 * ds1dl * rhod6;
         real d2t2dldrho = -14 * (1 + ghal) * s2 * ds2dl * rho6;
         dl->dlde = eps0 * invrv
            * (scexp * lp.p1 * (dt1drho * (t2 - 2) + t1 * dt2drho)
               + lambdaexp * (d2t1dldrho * (t2 - 2) + t1 * d2t2dldrho + dt1dl * dt2drho + dt1drho * dt2dl));
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
                 real vlambda,
                 const PairHalLambdaPow& lp,
                 real ghal,
                 real dhal,
                 real scexp,
                 real scalphav,
                 real& restrict e,
                 real& restrict de)
{
   pair_hal<DO_G, SCALE, false, false, false>(r, vscale, rv, eps, evcut, evoff, vlambda, lp, ghal, dhal, scexp,
      scalphav, e, de, nullptr, false);
}

/**
 * \ingroup vdw
 */
#pragma acc routine seq
template <bool DO_G, int SCALE, bool DO_DL1, bool DO_DL2, bool DO_DLDE>
SEQ_CUDA
void pair_hal_v3(real r,
                 real vscale,
                 real rv,
                 real eps,
                 real evcut,
                 real evoff,
                 real vlambda,
                 const PairHalLambdaPow& lp,
                 real ghal,
                 real dhal,
                 real scexp,
                 real scalphav,
                 real& restrict e,
                 real& restrict de,
                 real& restrict dedl,
                 real& restrict d2edl2,
                 real& restrict dlde,
                 bool dlon)
{
   PairHalLambda dl = {0, 0, 0};
   pair_hal<DO_G, SCALE, DO_DL1, DO_DL2, DO_DLDE>(r, vscale, rv, eps, evcut, evoff, vlambda, lp, ghal, dhal, scexp,
      scalphav, e, de, &dl, dlon);
   dedl = dl.dedl;
   d2edl2 = dl.d2edl2;
   dlde = dl.dlde;
}
}
