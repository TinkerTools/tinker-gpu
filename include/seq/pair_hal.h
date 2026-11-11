#pragma once
#include "ff/dlmda.h"
#include "math/libfunc.h"
#include "math/switch.h"
#include "seq/pair_vlambda.h"
#include "seq/seq.h"

namespace tinker {
/**
 * \ingroup vdw
 * \brief A tapered buffered 14-7 pair at the soft core \c lp (ehalsc.f),
 * \f$ E = \epsilon L t_1 (t_2 - 2) \f$ with
 * \f$ t_1 = (1+\delta)^7 / (h + (\rho+\delta)^7) \f$ and
 * \f$ t_2 = (1+\gamma) / (h + \rho^7 + \gamma) \f$, the scale \f$ L \f$ being
 * \c lp.p0 and the softening \f$ h \f$ \c lp.sc; with DL1, DL2 and DLDE, also
 * its main lambda derivatives in \c dl for a pair that follows lambda
 * (\c dlon), the caller having zeroed \c dl.
 */
#pragma acc routine seq
template <bool DO_G, int SCALE, bool DL1, bool DL2, bool DLDE>
SEQ_CUDA
inline void pair_hal(real r,
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
                     PairVdwLambda* dl,
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
   constexpr bool DL = DL1 or DL2 or DLDE;
   if (DL and dlon) {
      real tt = t1 * (t2 - 2);
      real ts = -t1 * s1 * (t2 - 2) - t1 * t2 * s2;
      real tss = 0, tr = 0, trs = 0;
      if CONSTEXPR (DL2)
         tss = 2 * t1 * s1 * s1 * (t2 - 2) + 2 * t1 * t2 * s1 * s2 + 2 * t1 * t2 * s2 * s2;
      if CONSTEXPR (DLDE) {
         real dt1drho = -7 * rhod6 * t1 * s1;
         real dt2drho = -7 * rho6 * t2 * s2;
         tr = dt1drho * (t2 - 2) + t1 * dt2drho;
         trs = -dt1drho * (2 * s1 * (t2 - 2) + t2 * s2) - t1 * dt2drho * (s1 + 2 * s2);
      }
      pair_softcore_dl<DL1, DL2, DLDE>(lp, eps0, tt, ts, tss, tr, trs, invrv, dl);
   }
   pair_vdw_taper<DO_G, DL1, DL2, DLDE>(r, evcut, evoff, e, de, dl);
}

/**
 * \ingroup vdw
 */
#pragma acc routine seq
template <bool DO_G, int SCALE>
SEQ_CUDA
inline void pair_hal_v2(real r,
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
