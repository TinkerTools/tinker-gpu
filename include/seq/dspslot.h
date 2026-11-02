#pragma once
#include "ff/box.h"
#include "ff/hippo/edisp.h"
#include "math/const.h"
#include "math/libfunc.h"
#include "seq/seq.h"
#include <cmath>

namespace tinker {
/// \ingroup disp
/// The pair weight \c w[s][t] of a slot \c s known only as the kernel runs, for
/// a constant \c t, chosen by value so that the weights stay in registers.
SEQ_ROUTINE
inline real dspSlotWeight(const real (&w)[3][3], int s, int t)
{
   return s == 0 ? w[0][t] : (s == 1 ? w[1][t] : w[2][t]);
}

/// The structure factor of the pairs of \c nslot slots at one wavevector,
/// \f$ \sum_{st} w_{st} \mathrm{Re}(S_s S_t^*) \f$, from their transforms \c q
/// (edisp.f:dspstruc).
SEQ_ROUTINE
inline real dspSlotStruc(int nslot, const real (&q)[3][2], const real (&w)[3][3])
{
   real st = w[0][0] * (q[0][0] * q[0][0] + q[0][1] * q[0][1]);
   if (nslot > 1) {
      st += w[1][1] * (q[1][0] * q[1][0] + q[1][1] * q[1][1]);
      st += 2 * w[0][1] * (q[0][0] * q[1][0] + q[0][1] * q[1][1]);
   }
   if (nslot > 2) {
      st += w[2][2] * (q[2][0] * q[2][0] + q[2][1] * q[2][1]);
      st += 2 * w[0][2] * (q[0][0] * q[2][0] + q[0][1] * q[2][1]);
      st += 2 * w[1][2] * (q[1][0] * q[2][0] + q[1][1] * q[2][1]);
   }
   return st;
}

/// \ingroup disp
/// The constants of the Ewald dispersion influence function (dspInfluence):
/// \f$ \pi/a \f$, \f$ 2 \pi^{3.5} \f$, \f$ a^3 \f$, \f$ -2 a \pi^2 \f$ and
/// \f$ 6 V/\pi^{1.5} \f$ for Ewald coefficient \f$ a \f$ in a box of volume
/// \f$ V \f$.
struct DspFac
{
   real bfac, fac1, fac2, fac3, denom0;
};

/// \ingroup disp
/// DspFac for Ewald coefficient \c aewald in a box of volume \c vbox,
/// evaluated on the host in double precision.
inline DspFac dspFac(double aewald, double vbox)
{
   return {real(M_PI / aewald), real(2 * std::pow(M_PI, 3.5)), real(aewald * aewald * aewald),
      real(-2 * aewald * M_PI * M_PI), real(6 * vbox / std::pow(M_PI, 1.5))};
}

/// \ingroup disp
/// The influence function of Ewald dispersion at the wavevector of the grid
/// indices \c k1, \c k2, \c k3 taken as \c r1, \c r2, \c r3, its \c h
/// vector, and the factors \c vx and \c vr of its virial, \f$ 3 v_x v_r \f$
/// per unit structure factor; all zero past the cutoff of the exponential
/// (edisp1.f:edrecip1).
SEQ_ROUTINE
inline real dspInfluence(int r1, int r2, int r3, int k1, int k2, int k3, real3 ra, real3 rb, real3 rc,
   BoxShape shape, real lvec1x, const real* restrict bsmod1, const real* restrict bsmod2,
   const real* restrict bsmod3, const DspFac& fac, real& h1, real& h2, real& h3, real& vx, real& vr)
{
   h1 = ra.x * r1 + rb.x * r2 + rc.x * r3;
   h2 = ra.y * r1 + rb.y * r2 + rc.y * r3;
   h3 = ra.z * r1 + rb.z * r2 + rc.z * r3;
   real hsq = h1 * h1 + h2 * h2 + h3 * h3;
   real h = REAL_SQRT(hsq);
   real b = h * fac.bfac;
   real hhh = h * hsq;
   real term = -hsq * fac.bfac * fac.bfac;
   real eterm = 0;
   vx = 0;
   vr = 0;
   if (term > -50) {
      real expterm = REAL_EXP(term);
      // the accurate erfc, since the two terms of eterm cancel at large h,
      // where the error of REAL_ERFC, relative to the small erfc, biased the
      // mixed precision energy and virial
      real erfcterm = REAL_ERFC_ACCURATE(b);
      if (shape == BoxShape::UNBOUND) {
         real coef = (1 - REAL_COS(pi * lvec1x * REAL_SQRT(hsq)));
         expterm *= coef;
         erfcterm *= coef;
      } else if (shape == BoxShape::OCT) {
         if ((k1 + k2 + k3) & 1) {
            expterm = 0;
            erfcterm = 0;
         } // end if ((k1 + k2 + k3) % 2 != 0)
      }
      real denom = fac.denom0 * bsmod1[k1] * bsmod2[k2] * bsmod3[k3];
      eterm = (-fac.fac1 * erfcterm * hhh - expterm * (fac.fac2 + fac.fac3 * hsq)) * REAL_RECIP(denom);
      vx = fac.fac1 * erfcterm * h + fac.fac3 * expterm;
      vr = REAL_RECIP(denom);
   }
   return eterm;
}

/// \ingroup disp
/// One wavevector of the convolution: its influence function, \c h vector and
/// virial factors (dspInfluence), and the number of times \c n it counts.
struct DspWave
{
   real eterm, h1, h2, h3, vx, vr, n;
};

/// \ingroup disp
/// The wavevectors that the grid point \c k1, \c k2, \c k3 stands for in the
/// convolution, one or two, and the influence function \c epot its potential
/// takes; returns their number.
///
/// On the full spectrum of a complex grid (not \c half) a point is its own
/// wavevector. On the half spectrum, \f$ k_1 \le n_1/2 \f$, of a real grid, a
/// point with \f$ 0 < k_1 < n_1/2 \f$ also stands for its partner under
/// conjugation, at the same structure factors: at \f$ -h \f$ it counts twice,
/// but on a plane of Nyquist indices the partner is not at \f$ -h \f$ and takes
/// its own \c h, and the potential, which must stay Hermitian as a full
/// transform's real part is, takes the mean influence function of the two.
/// Only a monoclinic or triclinic box gives the partner another \f$ |h| \f$;
/// otherwise it shares the influence function, the B-spline moduli being
/// symmetric, and differs only in the signs of its \c h, which a third of the
/// points of a warp would otherwise wait on.
SEQ_ROUTINE
inline int dspWaves(bool half, int k1, int k2, int k3, int nfft1, int nfft2, int nfft3, real3 ra, real3 rb,
   real3 rc, BoxShape shape, real lvec1x, const real* restrict bsmod1, const real* restrict bsmod2,
   const real* restrict bsmod3, const DspFac& fac, DspWave (&w)[2], real& epot)
{
   int r1 = (k1 < (nfft1 + 1) / 2) ? k1 : (k1 - nfft1);
   int r2 = (k2 < (nfft2 + 1) / 2) ? k2 : (k2 - nfft2);
   int r3 = (k3 < (nfft3 + 1) / 2) ? k3 : (k3 - nfft3);
   w[0].eterm = dspInfluence(r1, r2, r3, k1, k2, k3, ra, rb, rc, shape, lvec1x, bsmod1, bsmod2, bsmod3, fac,
      w[0].h1, w[0].h2, w[0].h3, w[0].vx, w[0].vr);
   w[0].n = 1;
   epot = w[0].eterm;
   if (not half)
      return 1;

   const bool doubled = k1 != 0 and 2 * k1 != nfft1;
   const bool nyquist = 2 * k1 == nfft1 or 2 * k2 == nfft2 or 2 * k3 == nfft3;
   if (nyquist) {
      int p1 = k1 ? nfft1 - k1 : 0;
      int p2 = k2 ? nfft2 - k2 : 0;
      int p3 = k3 ? nfft3 - k3 : 0;
      int s1 = (p1 < (nfft1 + 1) / 2) ? p1 : (p1 - nfft1);
      int s2 = (p2 < (nfft2 + 1) / 2) ? p2 : (p2 - nfft2);
      int s3 = (p3 < (nfft3 + 1) / 2) ? p3 : (p3 - nfft3);
      if (shape == BoxShape::MONO or shape == BoxShape::TRI) {
         w[1].eterm = dspInfluence(s1, s2, s3, p1, p2, p3, ra, rb, rc, shape, lvec1x, bsmod1, bsmod2, bsmod3, fac,
            w[1].h1, w[1].h2, w[1].h3, w[1].vx, w[1].vr);
         epot = (w[0].eterm + w[1].eterm) / 2;
      } else {
         w[1] = w[0];
         w[1].h1 = ra.x * s1 + rb.x * s2 + rc.x * s3;
         w[1].h2 = ra.y * s1 + rb.y * s2 + rc.y * s3;
         w[1].h3 = ra.z * s1 + rb.z * s2 + rc.z * s3;
      }
      w[1].n = 1;
      return doubled ? 2 : 1;
   }
   if (doubled)
      w[0].n = 2;
   return 1;
}
}
