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
/// The dispersion soft core of ligand group 2 if \c two, else of group 1,
/// chosen field by field as pair_softcore() does.
SEQ_ROUTINE
inline DispSoft pair_dispsoft(bool two, const DispSoft& s1, const DispSoft& s2)
{
   return {two ? s2.dlsc : s1.dlsc, two ? s2.dlsq : s1.dlsq, two ? s2.dlsc1 : s1.dlsc1, two ? s2.dlsq1 : s1.dlsq1,
      two ? s2.dlsc2 : s1.dlsc2, two ? s2.dlsq2 : s1.dlsq2, s1.rs2, s1.rrs2};
}

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
/// A damped dispersion pair of class \c mc (pair_mutcls), with the soft core
/// of ligand group \c mc, \c ds1 or \c ds2, if it is mutated, and, as the
/// flags ask, its lambda derivatives \c dle and \c dle2 and the lambda
/// derivative \c dlde of \c de (edisp1.f). Under Ewald summation the real
/// space of a mutated pair removes the share that the reciprocal space adds
/// with the plain lambda power. The soft core is chosen only on the branches
/// of a mutated pair, which a pair at full weight leaves alone. DHAL: the
/// Ewald tail of a buffered 14-7 vdW, its pair weighted across the vdW switch
/// of \c dh, undamped unless \c dh asks (edisp1.f).
#pragma acc routine seq
template <bool DO_G, class DTYP, int SCALE, bool DL1, bool DL2, bool GDL, bool DHAL = false>
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
               int mc,
               const DispSoft& ds1,
               const DispSoft& ds2,
               real edcut,
               real edoff,
               const DispHal& dh,
               real& restrict e,
               real& restrict de,
               real& restrict dle,
               real& restrict dle2,
               real& restrict dlde)
{
   static_assert(not DHAL or eq<DTYP, DEWALD>(), "");
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
   real damp = 1, ddamp = 0;
   if (not DHAL or dh.damp) {
      real dmpik[2];
      damp_hippodisp<DO_G or GDL>(dmpik, r, rr1, ai, ak);
      damp = dmpik[0];
      if CONSTEXPR (DO_G or GDL)
         ddamp = dmpik[1];
   }
   // the damping squared and half its derivative, which the tail of a hal vdW
   // weights by w, its derivative going to the second (edspsc.f:dhalwt)
   real dmp2 = damp * damp, dmpd = damp * ddamp;
   if CONSTEXPR (DHAL) {
      // whole beyond the switch, which an empty window ends at once
      real w = 0, dw = 0;
      if (r >= dh.off)
         w = 1;
      else if (r > dh.cut)
         switchTaper5<DO_G or GDL>(r, dh.off, dh.cut, w, dw);
      dmpd = w * dmpd + 0.5f * dw * dmp2;
      dmp2 = w * dmp2;
   }

   if CONSTEXPR (SCALE == 1)
      dspscale = 1;

   const bool mutik = mc > 0;
   const bool two = mc == 2;
   real soft = 1, dsoft = 0, dsdl = 0, d2sdl2 = 0, d2sdldr = 0;
   if (mutik)
      pair_dspsc<DL1, DL2, GDL>(r, r2, pair_dispsoft(two, ds1, ds2), soft, dsoft, dsdl, d2sdl2, d2sdldr);

   if CONSTEXPR (eq<DTYP, DEWALD>()) {
      real ralpha2 = r2 * aewald * aewald;
      real term = 1 + ralpha2 + 0.5f * ralpha2 * ralpha2;
      real expterm = REAL_EXP(-ralpha2);
      real expa = expterm * term;
      real rterm = -ralpha2 * ralpha2 * ralpha2 * rr1 * expterm;
      // the plain lambda power weights the share of the reciprocal space
      real wsc = 1, scs = dspscale;
      if (mutik) {
         wsc = two ? ds2.dlsc : ds1.dlsc;
         scs = dspscale * soft;
      }
      e = -ci * ck * rr6 * (scs * dmp2 + wsc * expa - wsc);
      if CONSTEXPR (DO_G) {
         de = -6 * e * rr1 - ci * ck * rr6 * (wsc * rterm + 2 * scs * dmpd);
         if (mutik)
            de -= ci * ck * rr6 * dspscale * dmp2 * dsoft;
      }
      if (mutik) {
         real e0 = -ci * ck * rr6;
         real scale = dspscale * dmp2;
         const real dlsc1 = two ? ds2.dlsc1 : ds1.dlsc1;
         if CONSTEXPR (DL1)
            dle = e0 * (dsdl * scale + dlsc1 * (expa - 1));
         if CONSTEXPR (DL2)
            dle2 = e0 * (d2sdl2 * scale + (two ? ds2.dlsc2 : ds1.dlsc2) * (expa - 1));
         if CONSTEXPR (GDL)
            dlde = -6 * dle * rr1
               - ci * ck * rr6 * (dlsc1 * rterm + 2 * dspscale * dmpd * dsdl + scale * d2sdldr);
      }
   } else if CONSTEXPR (eq<DTYP, NON_EWALD_TAPER>()) {
      e = -ci * ck * rr6;
      if CONSTEXPR (DO_G or GDL) {
         de = -6 * e * rr1;
         de = de * dmp2 + 2 * e * dmpd;
      }
      e = e * dmp2;
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
template <bool DO_G, class DTYP, int SCALE, bool DHAL = false>
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
               const DispHal& dh,
               real& restrict e,
               real& restrict de)
{
   real dle, dle2, dlde;
   pair_disp<DO_G, DTYP, SCALE, false, false, false, DHAL>(
      r, r2, rr1, dspscale, aewald, ci, ai, ck, ak, mutik ? 1 : 0, ds, ds, edcut, edoff, dh, e, de, dle, dle2, dlde);
}
}
