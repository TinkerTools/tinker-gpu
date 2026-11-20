#pragma once
#include "ff/watermccheap.h"
#include "seq/pair_chgtrn.h"
#include "seq/pair_disp.h"
#include "seq/pair_hal.h"
#include "seq/pair_lj.h"
#include "seq/pair_mpole.h"
#include "seq/pair_mpole_chgpen.h"
#include "seq/pair_repel.h"
#include "seq/pair_vlambda.h"
#include "seq/pairchgtrnaplus.h"
#include "seq/pairmpoleaplus.h"

namespace tinker {
/// \ingroup mdintg
/// The cheap energy of water atom \c i, turned and put as \c ws (WMC_NSITE
/// values), with atom \c j of pair class \c mc (pair_mutcls), whose values
/// relative to the point of the oxygen are \c nb (WMC_NNB). Raises \c hit to
/// WMC_HIT if the two come closer than the neighbor allows by more than
/// \c t.edge, and to WMC_EDGE if they come within \c t.edge of it either way,
/// which the host settles.
#pragma acc routine seq
SEQ_CUDA
inline real waterMcCheapPair(const WaterMcCheapTerms& t, const WaterMcCheapDev& p, const real* restrict ws, int i,
   int j, int mc, const real* restrict nb, int& restrict hit)
{
   // from the water atom to the other, as the energy kernels take i to k
   const real xr = nb[WMC_ND + 0] - ws[WMC_X + 0];
   const real yr = nb[WMC_ND + 1] - ws[WMC_X + 1];
   const real zr = nb[WMC_ND + 2] - ws[WMC_X + 2];
   const real r2 = xr * xr + yr * yr + zr * zr;
   const real ov = nb[WMC_NOV];
   if (ov > 0) {
      const real hi = ov + t.edge, lo = ov - t.edge;
      if (r2 < hi * hi) {
         const int h = (lo > 0 and r2 < lo * lo) ? WMC_HIT : WMC_EDGE;
         hit = hit > h ? hit : h;
      }
   }
   if (r2 == 0)
      return 0;
   const real r = REAL_SQRT(r2);
   const real invr = REAL_RECIP(r);
   const real big = t.big;
   real e = 0;

   if (t.elec == 1) {
      e += t.f * p.pchg[i] * p.pchg[j] * invr;
   } else if (t.elec >= 2) {
      const real* mi = ws + WMC_M;
      const real* mk = p.rpole[j];
      real em = 0;
      if (t.elec == 2) {
         PairMPoleGrad pg;
         pair_mpole<true, false, NON_EWALD>(r2, xr, yr, zr, 1,                               //
            mi[MPL_PME_0], mi[MPL_PME_X], mi[MPL_PME_Y], mi[MPL_PME_Z],                      //
            mi[MPL_PME_XX], mi[MPL_PME_XY], mi[MPL_PME_XZ], mi[MPL_PME_YY], mi[MPL_PME_YZ], //
            mi[MPL_PME_ZZ],                                                                  //
            mk[MPL_PME_0], mk[MPL_PME_X], mk[MPL_PME_Y], mk[MPL_PME_Z],                      //
            mk[MPL_PME_XX], mk[MPL_PME_XY], mk[MPL_PME_XZ], mk[MPL_PME_YY], mk[MPL_PME_YZ], //
            mk[MPL_PME_ZZ],                                                                  //
            t.f, 0, em, pg);
      } else {
         PairMPoleGrad pg;
         real poti, potk;
         if (t.elec == 3)
            pair_mpole_chgpen<true, false, NON_EWALD, 0>(r2, xr, yr, zr, 1,                           //
               mi[MPL_PME_0], mi[MPL_PME_X], mi[MPL_PME_Y], mi[MPL_PME_Z], p.pcore[i], p.pval[i],     //
               p.palpha[i], mi[MPL_PME_XX], mi[MPL_PME_XY], mi[MPL_PME_XZ], mi[MPL_PME_YY],           //
               mi[MPL_PME_YZ], mi[MPL_PME_ZZ],                                                        //
               mk[MPL_PME_0], mk[MPL_PME_X], mk[MPL_PME_Y], mk[MPL_PME_Z], p.pcore[j], p.pval[j],     //
               p.palpha[j], mk[MPL_PME_XX], mk[MPL_PME_XY], mk[MPL_PME_XZ], mk[MPL_PME_YY],           //
               mk[MPL_PME_YZ], mk[MPL_PME_ZZ],                                                        //
               t.f, 0, em, poti, potk, pg);
         else
            pair_mpole_chgpen_aplus<true, false, NON_EWALD, 0>(r2, xr, yr, zr, 1,                     //
               mi[MPL_PME_0], mi[MPL_PME_X], mi[MPL_PME_Y], mi[MPL_PME_Z], p.pcore[i], p.pval[i],     //
               p.palpha[i], mi[MPL_PME_XX], mi[MPL_PME_XY], mi[MPL_PME_XZ], mi[MPL_PME_YY],           //
               mi[MPL_PME_YZ], mi[MPL_PME_ZZ],                                                        //
               mk[MPL_PME_0], mk[MPL_PME_X], mk[MPL_PME_Y], mk[MPL_PME_Z], p.pcore[j], p.pval[j],     //
               p.palpha[j], mk[MPL_PME_XX], mk[MPL_PME_XY], mk[MPL_PME_XZ], mk[MPL_PME_YY],           //
               mk[MPL_PME_YZ], mk[MPL_PME_ZZ],                                                        //
               t.f, 0, em, poti, potk, pg);
      }
      e += em;
   }

   // A pair the full energy leaves out (mc < 0) is left out here too.
   if (mc < 0)
      return e;

   if (t.vdw != 0) {
      // between the van der Waals sites, which a reduction factor moves toward
      // the neighbor of an atom
      const real vx = nb[WMC_NV + 0] - ws[WMC_V + 0];
      const real vy = nb[WMC_NV + 1] - ws[WMC_V + 1];
      const real vz = nb[WMC_NV + 2] - ws[WMC_V + 2];
      const real rv = REAL_SQRT(vx * vx + vy * vy + vz * vz);
      const int row = p.jvdw[i] * p.njvdw + p.jvdw[j];
      const real rad = p.radmin[row], eps = p.epsilon[row];
      const SoftCoreLmda lp = pair_vdw_softcore<true>(mc, t.vs1, t.vs2);
      real ev = 0, dev;
      if (rv > 0) {
         if (t.vdw == 1)
            pair_hal<false, 1, false, false, false>(rv, 1, rad, eps, big, big, lp, t.ghal, t.dhal, ev, dev, nullptr,
               false);
         else
            pair_lj_v0<false, true>(rv, REAL_RECIP(rv), lp, rad, eps, ev, dev);
      }
      e += REAL_MIN(ev, t.cap);
   }

   if (t.repel) {
      const real* ri = ws + WMC_R;
      const real* rk = p.rrepole[j];
      real er;
      PairRepelGrad pg;
      pair_repel<false>(r2, 1, mc > 0, pair_softcore(mc == 2, t.rs1, t.rs2), big, big, xr, yr, zr, //
         p.sizpr[i], p.dmppr[i], p.elepr[i],                                                       //
         ri[MPL_PME_0], ri[MPL_PME_X], ri[MPL_PME_Y], ri[MPL_PME_Z],                               //
         ri[MPL_PME_XX], ri[MPL_PME_XY], ri[MPL_PME_XZ], ri[MPL_PME_YY], ri[MPL_PME_YZ],          //
         ri[MPL_PME_ZZ],                                                                           //
         p.sizpr[j], p.dmppr[j], p.elepr[j],                                                       //
         rk[MPL_PME_0], rk[MPL_PME_X], rk[MPL_PME_Y], rk[MPL_PME_Z],                               //
         rk[MPL_PME_XX], rk[MPL_PME_XY], rk[MPL_PME_XZ], rk[MPL_PME_YY], rk[MPL_PME_YZ],          //
         rk[MPL_PME_ZZ], er, pg);
      e += REAL_MIN(er, t.cap);
   }

   if (t.disp) {
      real ed, ded, dle, dle2, dlde;
      const DispHal dh = {big, big, true};
      pair_disp<false, NON_EWALD_TAPER, 1, false, false, false, false>(r, r2, invr, 1, 0, p.csix[i], p.adisp[i],
         p.csix[j], p.adisp[j], mc, t.ds1, t.ds2, big, big, dh, ed, ded, dle, dle2, dlde);
      e += ed;
   }

   const real ctw = nb[WMC_NCT];
   if (t.ct != 0 and ctw != 0) {
      e_prec ec, dec;
      if (t.ct == 1)
         pair_chgtrn<false>(r, big, big, 1, t.f, p.dmpct[i], p.chgct[i], p.dmpct[j], p.chgct[j], 1, ec, dec);
      else
         pair_chgtrn_aplus<false>(r, big, big, 1, t.f, p.dmpct[i], p.chgct[i], p.dmpct[j], p.chgct[j], 1, ec, dec);
      e += ctw * ec;
   }
   return e;
}
}
