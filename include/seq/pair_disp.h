#pragma once
#include "ff/elec.h"
#include "ff/hippo/edisp.h"
#include "math/libfunc.h"
#include "math/switch.h"
#include "seq/add.h"
#include "seq/damp_hippodisp.h"
#include "seq/pair_vlambda.h"
#include "seq/seq.h"

namespace tinker {
/// \ingroup disp
/// The soft core scaling \c soft of a mutated damped dispersion pair and its
/// distance derivative \c dsoft, along with, as the flags ask, its lambda
/// derivatives \c dsdl and \c d2sdl2 and the lambda derivative \c d2sdldr of
/// \c dsoft (edspsc.f:edspsc4). Beyond the switch radius every mutated pair
/// carries the plain lambda power.
#pragma acc routine seq
template <bool DL1, bool DL2, bool GDL>
SEQ_CUDA
void pair_dspsc(real r, real r2, const DispSoft& ds, real& restrict soft, real& restrict dsoft, real& restrict dsdl,
   real& restrict d2sdl2, real& restrict d2sdldr)
{
   soft = ds.dlsc;
   dsoft = 0;
   if CONSTEXPR (DL1)
      dsdl = ds.dlsc1;
   if CONSTEXPR (DL2)
      d2sdl2 = ds.dlsc2;
   if CONSTEXPR (GDL)
      d2sdldr = 0;
   if (r2 < ds.rs2) {
      real t = 1 - r2 * ds.rrs2;
      real t2 = t * t;
      real sw = t2 * t;
      real dsw = 6 * r * t2 * ds.rrs2;
      soft = ds.dlsc - (ds.dlsc - ds.dlsq) * sw;
      dsoft = (ds.dlsc - ds.dlsq) * dsw;
      if CONSTEXPR (DL1)
         dsdl = ds.dlsc1 - (ds.dlsc1 - ds.dlsq1) * sw;
      if CONSTEXPR (DL2)
         d2sdl2 = ds.dlsc2 - (ds.dlsc2 - ds.dlsq2) * sw;
      if CONSTEXPR (GDL)
         d2sdldr = (ds.dlsc1 - ds.dlsq1) * dsw;
   }
}

/// \ingroup disp
/// A damped dispersion pair, with the soft core of a \c mutik pair and, as
/// the flags ask, its lambda derivatives \c dle and \c dle2 and the lambda
/// derivative \c dlde of \c de (edisp1.f). Under Ewald summation the real
/// space of a mutated pair removes the share that the reciprocal space adds
/// with the plain lambda power.
#pragma acc routine seq
template <bool DO_G, class DTYP, int SCALE, bool DL1, bool DL2, bool GDL>
SEQ_CUDA
void pair_disp(real r,
               real r2,
               real rr1,
               real dspscale,
               real aewald,
               real ci,
               real ai,
               real ck,
               real ak,
               bool mutik,
               const DispSoft& ds,
               real edcut,
               real edoff,
               real& restrict e,
               real& restrict de,
               real& restrict dle,
               real& restrict dle2,
               real& restrict dlde)
{
   if CONSTEXPR (DL1)
      dle = 0;
   if CONSTEXPR (DL2)
      dle2 = 0;
   if CONSTEXPR (GDL)
      dlde = 0;
   if (r > edoff) {
      e = 0;
      if CONSTEXPR (DO_G)
         de = 0;
      return;
   }

   real rr2 = rr1 * rr1;
   real rr6 = rr2 * rr2 * rr2;
   real dmpik[2], damp, ddamp;
   damp_hippodisp<DO_G or GDL>(dmpik, r, rr1, ai, ak);
   damp = dmpik[0];
   if CONSTEXPR (DO_G or GDL)
      ddamp = dmpik[1];

   if CONSTEXPR (SCALE == 1)
      dspscale = 1;

   real soft = 1, dsoft = 0, dsdl = 0, d2sdl2 = 0, d2sdldr = 0;
   if (mutik)
      pair_dspsc<DL1, DL2, GDL>(r, r2, ds, soft, dsoft, dsdl, d2sdl2, d2sdldr);

   if CONSTEXPR (eq<DTYP, DEWALD>()) {
      real ralpha2 = r2 * aewald * aewald;
      real term = 1 + ralpha2 + 0.5f * ralpha2 * ralpha2;
      real expterm = REAL_EXP(-ralpha2);
      real expa = expterm * term;
      real rterm = -ralpha2 * ralpha2 * ralpha2 * rr1 * expterm;
      // the plain lambda power weights the share of the reciprocal space
      real wsc = 1, scs = dspscale;
      if (mutik) {
         wsc = ds.dlsc;
         scs = dspscale * soft;
      }
      e = -ci * ck * rr6 * (scs * damp * damp + wsc * expa - wsc);
      if CONSTEXPR (DO_G) {
         de = -6 * e * rr1 - ci * ck * rr6 * (wsc * rterm + 2 * scs * damp * ddamp);
         if (mutik)
            de -= ci * ck * rr6 * dspscale * damp * damp * dsoft;
      }
      if (mutik) {
         real e0 = -ci * ck * rr6;
         real scale = dspscale * damp * damp;
         if CONSTEXPR (DL1)
            dle = e0 * (dsdl * scale + ds.dlsc1 * (expa - 1));
         if CONSTEXPR (DL2)
            dle2 = e0 * (d2sdl2 * scale + ds.dlsc2 * (expa - 1));
         if CONSTEXPR (GDL)
            dlde = -6 * dle * rr1
               - ci * ck * rr6 * (ds.dlsc1 * rterm + 2 * dspscale * damp * ddamp * dsdl + scale * d2sdldr);
      }
   } else if CONSTEXPR (eq<DTYP, NON_EWALD_TAPER>()) {
      e = -ci * ck * rr6;
      if CONSTEXPR (DO_G or GDL) {
         de = -6 * e * rr1;
         de = de * damp * damp + 2 * e * damp * ddamp;
      }
      e = e * damp * damp;
      if (r > edcut) {
         real taper, dtaper;
         switchTaper5<DO_G or GDL>(r, edcut, edoff, taper, dtaper);
         if CONSTEXPR (DO_G or GDL)
            de = e * dtaper + de * taper;
         e = e * taper;
      }
      e *= dspscale;
      if CONSTEXPR (DO_G or GDL)
         de *= dspscale;
      // the soft core multiplies the tapered pair (edisp1.f)
      if (mutik) {
         if CONSTEXPR (DL1)
            dle = dsdl * e;
         if CONSTEXPR (DL2)
            dle2 = d2sdl2 * e;
         if CONSTEXPR (GDL)
            dlde = dsdl * de + d2sdldr * e;
         if CONSTEXPR (DO_G)
            de = soft * de + dsoft * e;
         e *= soft;
      }
   }
}

/// \ingroup disp
/// A damped dispersion pair without lambda derivatives.
#pragma acc routine seq
template <bool DO_G, class DTYP, int SCALE>
SEQ_CUDA
void pair_disp(real r,
               real r2,
               real rr1,
               real dspscale,
               real aewald,
               real ci,
               real ai,
               real ck,
               real ak,
               bool mutik,
               const DispSoft& ds,
               real edcut,
               real edoff,
               real& restrict e,
               real& restrict de)
{
   real dle, dle2, dlde;
   pair_disp<DO_G, DTYP, SCALE, false, false, false>(
      r, r2, rr1, dspscale, aewald, ci, ai, ck, ak, mutik, ds, edcut, edoff, e, de, dle, dle2, dlde);
}
}
