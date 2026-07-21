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
              real scalpha, //
              real& restrict e,
              real& restrict de)
{
   eps *= vscalek;
   real rho = rik * REAL_RECIP(rv);
   real rho6 = REAL_POW(rho, 6);
   real rho7 = rho6 * rho;
   eps *= REAL_POW(vlambda, scexp);
   real one_minus_lambda = 1 - vlambda;
   real scal = scalpha * one_minus_lambda * one_minus_lambda;
   real s1 = REAL_RECIP(scal + REAL_POW(rho + dhal, 7));
   real s2 = REAL_RECIP(scal + rho7 + ghal);
   real t1 = REAL_POW(1 + dhal, 7) * s1;
   real t2 = (1 + ghal) * s2;
   e = eps * t1 * (t2 - 2);
   if CONSTEXPR (DO_G) {
      real dt1drho = -7 * REAL_POW(rho + dhal, 6) * t1 * s1;
      real dt2drho = -7 * rho6 * t2 * s2;
      de = eps * (dt1drho * (t2 - 2) + t1 * dt2drho) * REAL_RECIP(rv);
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

#pragma acc routine seq
template <bool DO_G, int SCALE, class LTYP>
SEQ_CUDA
void pair_hal(real r,
              real vscale,
              real rv,
              real eps,
              real evcut,
              real evoff,
              real vlambda,
              real ghal,
              real dhal,
              real scexp,
              real scalpha,
              real& restrict e,
              real& restrict de,
              PairHalLambda* dl)
{
   if CONSTEXPR (SCALE != 1)
      eps *= vscale;
   real eps0 = eps;
   real rho = r * REAL_RECIP(rv);
   real rho6 = REAL_POW(rho, 6);
   real rho7 = rho6 * rho;
   real lambdaexp = REAL_POW(vlambda, scexp);
   eps *= lambdaexp;
   real one_minus_lambda = 1 - vlambda;
   real scal = scalpha * one_minus_lambda * one_minus_lambda;
   real s1 = REAL_RECIP(scal + REAL_POW(rho + dhal, 7));
   real s2 = REAL_RECIP(scal + rho7 + ghal);
   real t1 = REAL_POW(1 + dhal, 7) * s1;
   real t2 = (1 + ghal) * s2;
   e = eps * t1 * (t2 - 2);
   if CONSTEXPR (DO_G) {
      real dt1drho = -7 * REAL_POW(rho + dhal, 6) * t1 * s1;
      real dt2drho = -7 * rho6 * t2 * s2;
      de = eps * (dt1drho * (t2 - 2) + t1 * dt2drho) * REAL_RECIP(rv);
   }

   if CONSTEXPR (eq<LTYP, DLMDA>()) {
      real dt0dl = eps0 * scexp * REAL_POW(vlambda, scexp - 1);
      real dscaldl = 2 * scalpha * (1 - vlambda);
      real ds1dl = dscaldl * s1 * s1;
      real ds2dl = dscaldl * s2 * s2;
      real dt1dl = REAL_POW(1 + dhal, 7) * ds1dl;
      real dt2dl = (1 + ghal) * ds2dl;
      dl->dedl = dt0dl * t1 * (t2 - 2) + eps * dt1dl * (t2 - 2) + eps * t1 * dt2dl;

      real d2t0dl2 = 0;
      if (scexp >= 2)
         d2t0dl2 = eps0 * scexp * (scexp - 1) * REAL_POW(vlambda, scexp - 2);
      real d2t1dl2 = REAL_POW(1 + dhal, 7) * (-2 * scalpha * s1 * s1 + 2 * dscaldl * s1 * ds1dl);
      real d2t2dl2 = (1 + ghal) * (-2 * scalpha * s2 * s2 + 2 * dscaldl * s2 * ds2dl);
      dl->d2edl2 = d2t0dl2 * t1 * (t2 - 2) + eps * d2t1dl2 * (t2 - 2) + eps * t1 * d2t2dl2 + 2 * dt0dl * dt1dl * (t2 - 2) + 2 * dt0dl * t1 * dt2dl
         + 2 * eps * dt1dl * dt2dl;

      if CONSTEXPR (DO_G) {
         real rhopdhal6 = REAL_POW(rho + dhal, 6);
         real dt1drho = -7 * rhopdhal6 * t1 * s1;
         real dt2drho = -7 * rho6 * t2 * s2;
         real d2t1dldrho = -14 * REAL_POW(1 + dhal, 7) * s1 * ds1dl * rhopdhal6;
         real d2t2dldrho = -14 * (1 + ghal) * s2 * ds2dl * rho6;
         dl->dlde = eps0 * REAL_RECIP(rv)
            * (scexp * REAL_POW(vlambda, scexp - 1) * (dt1drho * (t2 - 2) + t1 * dt2drho)
               + lambdaexp * (d2t1dldrho * (t2 - 2) + t1 * d2t2dldrho + dt1dl * dt2drho + dt1drho * dt2dl));
      }
   }

   if (r > evcut) {
      real taper, dtaper;
      switchTaper5<DO_G>(r, evcut, evoff, taper, dtaper);
      if CONSTEXPR (DO_G) {
         de = e * dtaper + de * taper;
         if CONSTEXPR (eq<LTYP, DLMDA>())
            dl->dlde = dl->dedl * dtaper + dl->dlde * taper;
      }
      e *= taper;
      if CONSTEXPR (eq<LTYP, DLMDA>()) {
         dl->dedl *= taper;
         dl->d2edl2 *= taper;
      }
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
                 real ghal,
                 real dhal,
                 real scexp,
                 real scalpha,
                 real& restrict e,
                 real& restrict de)
{
   pair_hal<DO_G, SCALE, NON_DLMDA>(r, vscale, rv, eps, evcut, evoff, vlambda, ghal, dhal, scexp, scalpha, e, de, nullptr);
}

/**
 * \ingroup vdw
 */
#pragma acc routine seq
template <bool DO_G, int SCALE>
SEQ_CUDA
void pair_hal_v3(real r,
                 real vscale,
                 real rv,
                 real eps,
                 real evcut,
                 real evoff,
                 real vlambda,
                 real ghal,
                 real dhal,
                 real scexp,
                 real scalpha,
                 real& restrict e,
                 real& restrict de,
                 real& restrict dedl,
                 real& restrict d2edl2,
                 real& restrict dlde)
{
   PairHalLambda dl;
   pair_hal<DO_G, SCALE, DLMDA>(r, vscale, rv, eps, evcut, evoff, vlambda, ghal, dhal, scexp, scalpha, e, de, &dl);
   dedl = dl.dedl;
   d2edl2 = dl.d2edl2;
   dlde = dl.dlde;
}
}
