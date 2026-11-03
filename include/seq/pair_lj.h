#pragma once
#include "math/libfunc.h"
#include "seq/pair_vlambda.h"

namespace tinker {
/// \ingroup vdw
/// The inverse distance capped at a finite value, so that two sites at the
/// same point, whose soft core pair has a zero distance derivative, take no
/// force (elj1.f) at the cost of one instruction.
SEQ_ROUTINE
inline real pair_lj_rinv(real invr)
{
   return REAL_MIN(invr, real(1e30));
}

/**
 * \ingroup vdw
 * \brief A Lennard-Jones pair. SOFTCORE: at the soft core \c lp (eljsc.f),
 * \f$ E = \epsilon L T(s) \f$ with \f$ s = 2 (r/r_v)^6 + h \f$ and
 * \f$ T(s) = 4(1-s)/s^2 \f$, the scale \f$ L \f$ being \c lp.p0 and the shift
 * \f$ h \f$ \c lp.sc; a pair that does not follow lambda takes 1 and 0, the
 * plain potential. With DL1, DL2 and DLDE, also its main lambda derivatives
 * in \c dl from \f$ T_s = 4(s-2)/s^3 \f$ and \f$ T_{ss} = 8(3-s)/s^4 \f$
 * (eljsc.f:eljsc4), for a pair that follows lambda (\c dlon); the caller
 * zeroes \c dl, which keeps it for the others. Two sites at the same point
 * take no force.
 */
#pragma acc routine seq
template <bool DO_G, bool SOFTCORE, bool DL1 = false, bool DL2 = false, bool DLDE = false>
SEQ_CUDA
void pair_lj_v0(real r, real invr, const SoftCoreLmda& lp, real rad, real eps, real& restrict ev,
   real& restrict dev, PairVdwLambda* restrict dl = nullptr, bool dlon = true)
{
   static_assert(SOFTCORE or not(DL1 or DL2 or DLDE), "");
   if CONSTEXPR (SOFTCORE) {
      // ahead of the branch, so that it is the same value as the caller's
      real rinv = pair_lj_rinv(invr);
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
      real sc = p6 + lp.sc;
      real invsc = 1 / sc;
      real term = 4 * lp.p0 * eps * invsc * invsc;
      ev = term * (1 - sc);
      // ds/dr, zero for two sites at the same point
      real dsdr = 6 * p6 * rinv;
      if CONSTEXPR (DO_G) {
         real dterm = -term * dsdr;
         dev = dterm * (2 * invsc - 1);
      }
      if CONSTEXPR (DL1 or DL2 or DLDE) {
         if (not dlon)
            return;
         // An unshifted pair at lambda zero, LJ-SOFTCORE m > 0, takes a zero
         // scale and zero coefficients, but at r below about 2e-3 A 1/s^4
         // overflows in single precision to give 0 * inf; s past the floor
         // only changes the derivatives where they vanish.
         real sd = REAL_MAX(sc, real(1e-8));
         real invsd = sc >= real(1e-8) ? invsc : real(1e8);
         real tt = 4 * (1 - sd) * invsd * invsd;
         real ts = 4 * (sd - 2) * invsd * invsd * invsd;
         real tss = 8 * (3 - sd) * invsd * invsd * invsd * invsd;
         // s is the only distance dependence of T, so its distance
         // derivatives are those in s times ds/dr
         pair_softcore_dl<DL1, DL2, DLDE>(lp, eps, tt, ts, tss, ts, tss, dsdr, dl);
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
 * \brief A tapered pair of the radius and well depth that RADRULE and
 * EPSRULE mix from those of the two sites, with, as the flags ask, its main
 * lambda derivatives as pair_lj_v0() gives them; past the cutoff \c dl keeps
 * what the caller zeroed it to.
 */
#pragma acc routine seq
template <bool DO_G, bool SOFTCORE, class RADRULE, class EPSRULE, int SCALE, bool DL1 = false, bool DL2 = false,
   bool DLDE = false>
SEQ_CUDA
void pair_lj_v2(real r,
                real invr,
                const SoftCoreLmda& lp, //
                real vscale,
                real radi,
                real epsi,
                real radk,
                real epsk,
                real evcut,
                real evoff,
                real& restrict ev,
                real& restrict dev,
                PairVdwLambda* restrict dl = nullptr,
                bool dlon = true)
{
   if (r > evoff) {
      ev = 0;
      if CONSTEXPR (DO_G)
         dev = 0;
      return;
   }
   real rad = RADRULE::savg(radi, radk);
   real eps = EPSRULE::savg(epsi, epsk);
   if CONSTEXPR (SCALE != 1)
      eps *= vscale;
   pair_lj_v0<DO_G, SOFTCORE, DL1, DL2, DLDE>(r, invr, lp, rad, eps, ev, dev, dl, dlon);
   pair_vdw_taper<DO_G, DL1, DL2, DLDE>(r, evcut, evoff, ev, dev, dl);
}

/**
 * \ingroup vdw
 * \brief A tapered pair of premixed radius and well depth, at the soft core
 * \c lp if SOFTCORE, with its main lambda derivatives as pair_lj_v0() gives
 * them.
 */
#pragma acc routine seq
template <bool DO_G, bool SOFTCORE, int SCALE, bool DL1 = false, bool DL2 = false, bool DLDE = false>
SEQ_CUDA
void pair_lj_v3(real r,
                real invr,
                const SoftCoreLmda& lp, //
                real vscale,
                real rad,
                real eps,
                real evcut,
                real evoff,
                real& restrict ev,
                real& restrict dev,
                PairVdwLambda* restrict dl = nullptr,
                bool dlon = true)
{
   if CONSTEXPR (SCALE != 1) {
      eps *= vscale;
   }
   pair_lj_v0<DO_G, SOFTCORE, DL1, DL2, DLDE>(r, invr, lp, rad, eps, ev, dev, dl, dlon);
   pair_vdw_taper<DO_G, DL1, DL2, DLDE>(r, evcut, evoff, ev, dev, dl);
}
}
