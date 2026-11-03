#include "ff/hippo/edisp.h"
#include "ff/dlmda.h"
#include "ff/energy.h"
#include "ff/evdw.h"
#include "ff/pme.h"
#include "ff/potent.h"
#include "math/zero.h"
#include "tool/externfunc.h"
#include <cassert>
#include <tinker/detail/couple.hh>
#include <tinker/detail/disp.hh>
#include <tinker/detail/dsppot.hh>
#include <tinker/detail/limits.hh>
#include <tinker/detail/mutant.hh>
#include <tinker/routines.h>

#include <algorithm>
#include <vector>

namespace tinker {
// The C6 sums of the environment and of the ligand groups, their sums of
// squared C6 for the self term, and whether each group has a site
// (edisp.f:dspslot).
static double dsp_csum[3], dsp_csq[3];
static bool dsp_have[3];
// With mutated ligands, the real grids of the slots, dsp_stride reals apart,
// and where the grid of each site starts, for the slots of dsp_offset_slot.
static real* dsp_slots;
static int dsp_stride;
static int* dsp_offset;
static int dsp_offset_slot[3];
// The class sums of the long-range dispersion correction.
static LrcSums dsp_lrc;

bool useDEwald()
{
   return use(Potent::DISP) and static_cast<bool>(limits::use_dewald);
}

bool useDHal()
{
   return useDEwald() and static_cast<bool>(dsppot::use_dhal);
}

DispHal dispHal()
{
   if (not useDHal())
      return {0, 0, true};
   return {(real)dsppot::dhcut, (real)dsppot::dhoff, static_cast<bool>(dsppot::dhdamp)};
}

void edispData(RcOp op)
{
   if (not use(Potent::DISP))
      return;

   auto rc_a = rc_flag & calc::analyz;

   if (op & RcOp::DEALLOC) {
      darray::deallocate(csix, adisp);
      darray::deallocate(dspexclude, dspexclude_scale);
      darray::deallocate(dsp_slots, dsp_offset);
      dsp_slots = nullptr;
      dsp_offset = nullptr;

      if (rc_a) {
         bufferDeallocate(rc_flag, ndisp);
         bufferDeallocate(rc_flag, edsp, vir_edsp, dedspx, dedspy, dedspz);
      }
      ndisp = nullptr;
      edsp = nullptr;
      vir_edsp = nullptr;
      dedspx = nullptr;
      dedspy = nullptr;
      dedspz = nullptr;

      dsp_lrc = {};
   }

   if (op & RcOp::ALLOC) {
      darray::allocate(n, &csix, &adisp);
      // the environment and each ligand may need a grid of their own
      // (dispSlots); a stride of whole 128-byte lines keeps every grid aligned
      // as the FFT plan of the unit expects
      dsp_slots = nullptr;
      dsp_offset = nullptr;
      dsp_stride = 0;
      if (useDEwald() and mutant::nmut > 0) {
         const auto& st = *dpme_unit;
         const int nslot = use_rel ? 3 : 2;
         const int line = 128 / sizeof(real);
         dsp_stride = (2 * (st.nfft1 / 2 + 1) * st.nfft2 * st.nfft3 + line - 1) / line * line;
         darray::allocate(nslot * dsp_stride, &dsp_slots);
         darray::allocate(n, &dsp_offset);
      }
      dsp_offset_slot[0] = dsp_offset_slot[1] = dsp_offset_slot[2] = -2;

      ndisp = nullptr;
      edsp = eng_buf_vdw;
      vir_edsp = vir_buf_vdw;
      dedspx = gx_vdw;
      dedspy = gy_vdw;
      dedspz = gz_vdw;
      if (rc_a) {
         bufferAllocate(rc_flag, &ndisp);
         bufferAllocate(rc_flag, &edsp, &vir_edsp, &dedspx, &dedspy, &dedspz);
      }

      dsp2scale = dsppot::dsp2scale;
      dsp3scale = dsppot::dsp3scale;
      dsp4scale = dsppot::dsp4scale;
      dsp5scale = dsppot::dsp5scale;
      std::vector<int> exclik;
      std::vector<real> excls;
      // see also attach.f
      const int maxn13 = 3 * sizes::maxval;
      const int maxn14 = 9 * sizes::maxval;
      const int maxn15 = 27 * sizes::maxval;
      for (int i = 0; i < n; ++i) {
         int nn;
         int bask;

         if (dsp2scale != 1) {
            nn = couple::n12[i];
            for (int j = 0; j < nn; ++j) {
               int k = couple::i12[i][j];
               k -= 1;
               if (k > i) {
                  exclik.push_back(i);
                  exclik.push_back(k);
                  excls.push_back(dsp2scale);
               }
            }
         }

         if (dsp3scale != 1) {
            nn = couple::n13[i];
            bask = i * maxn13;
            for (int j = 0; j < nn; ++j) {
               int k = couple::i13[bask + j];
               k -= 1;
               if (k > i) {
                  exclik.push_back(i);
                  exclik.push_back(k);
                  excls.push_back(dsp3scale);
               }
            }
         }

         if (dsp4scale != 1) {
            nn = couple::n14[i];
            bask = i * maxn14;
            for (int j = 0; j < nn; ++j) {
               int k = couple::i14[bask + j];
               k -= 1;
               if (k > i) {
                  exclik.push_back(i);
                  exclik.push_back(k);
                  excls.push_back(dsp4scale);
               }
            }
         }

         if (dsp5scale != 1) {
            nn = couple::n15[i];
            bask = i * maxn15;
            for (int j = 0; j < nn; ++j) {
               int k = couple::i15[bask + j];
               k -= 1;
               if (k > i) {
                  exclik.push_back(i);
                  exclik.push_back(k);
                  excls.push_back(dsp5scale);
               }
            }
         }
      }
      ndspexclude = excls.size();
      darray::allocate(ndspexclude, &dspexclude, &dspexclude_scale);
      darray::copyin(g::q0, ndspexclude, dspexclude, exclik.data());
      darray::copyin(g::q0, ndspexclude, dspexclude_scale, excls.data());
      waitFor(g::q0);
   }

   if (op & RcOp::INIT) {
      scrdsp = mutant::scrdsp;
      scexpd = mutant::scexpd;
      scexpdq = mutant::scexpdq;
      csixpr = disp::csixpr;
      darray::copyin(g::q0, n, csix, disp::csix);
      darray::copyin(g::q0, n, adisp, disp::adisp);
      waitFor(g::q0);

      for (int g = 0; g < 3; ++g) {
         dsp_csum[g] = 0;
         dsp_csq[g] = 0;
         dsp_have[g] = false;
      }
      for (int i = 0; i < n; ++i) {
         double c = disp::csix[i];
         if (c == 0)
            continue;
         int g = hostGroup(i);
         dsp_csum[g] += c;
         dsp_csq[g] += c * c;
         dsp_have[g] = true;
      }

      dsp_lrc = {};
      if (dsppot::use_dcorr && not useDEwald())
         lrcSetup(dsp_lrc, true);
   }
}
}

namespace tinker {
// The soft core of the real space pairs from its powers lambda^n and
// lambda^(n+q) of each ligand group (vdwGroupPower, edspsc.f:dspsc4). The soft
// core is linear in its two powers, so the van der Waals lambda chain rule goes
// on the powers themselves.
static DispSoft dispSoft(const LmdaFn& c, const LmdaFn& q)
{
   DispSoft ds;
   ds.dlsc = c.f;
   ds.dlsq = q.f;
   ds.dlsc1 = c.d1;
   ds.dlsq1 = q.d1;
   ds.dlsc2 = c.d2;
   ds.dlsq2 = q.d2;
   ds.rs2 = scrdsp * scrdsp;
   ds.rrs2 = (scrdsp > 0 ? 1 / (scrdsp * scrdsp) : 0);
   return ds;
}

TINKER_FVOID2(acc1, cu1, edispNonEwald, int, DispSoft, DispSoft);
static void edispNonEwald(int vers, const DispSoft& sc1, const DispSoft& sc2)
{
   TINKER_FCALL2(acc1, cu1, edispNonEwald, vers, sc1, sc2);
}

// The terms of the reciprocal and self parts summed on the host: whether the
// plain grid is used, the weighted C6 pair sum of the zero wavevector with its
// lambda derivatives, and the lambda derivatives of the self terms weighted by
// the squared C6 sums.
struct DispHost
{
   bool plain;
   double cpr, dcpr, d2cpr;
   double dself, d2self;
};

// The reciprocal and self terms weight a pair of groups as the real space
// pairs beyond the soft core switch radius are weighted, by the lambda power
// of the class of the pair (dlambda.f:grpvwt), which no single real grid can
// carry once a ligand is decoupled. Every group with sites and weight left
// goes to a slot, shared by the groups whose weights match (edisp.f:dspslot);
// a single slot of full weight and no lambda derivatives is the plain grid,
// while with no slot at all, reciprocal space has nothing to carry.
static DispSlots dispSlots(bool dodl, DispHost& h)
{
   DispSlots ds = {};
   h = {true, csixpr, 0, 0, 0, 0};
   if (mutant::nmut == 0)
      return ds;
   h.plain = false;

   LmdaFn w[3][3];
   vdwClassWeights(scexpd, dodl, w);

   // drop a group with no weight to any group on the grid, in order
   bool have[3] = {dsp_have[0], dsp_have[1], dsp_have[2]};
   bool full = true;
   for (int g = 0; g < 3; ++g) {
      if (not have[g])
         continue;
      bool live = false;
      for (int k = 0; k < 3; ++k)
         if (have[k] and (w[g][k].f != 0 or w[g][k].d1 != 0 or w[g][k].d2 != 0))
            live = true;
      have[g] = live;
      full = full and live;
   }

   // merge a group into an earlier one with the same weights
   int grp[3];
   for (int g = 0; g < 3; ++g) {
      ds.slot[g] = -1;
      if (not have[g])
         continue;
      for (int k = 0; k < g and ds.slot[g] < 0; ++k) {
         if (ds.slot[k] < 0)
            continue;
         bool same = true;
         for (int f = 0; f < 3; ++f)
            if (have[f] and (w[g][f].f != w[k][f].f or w[g][f].d1 != w[k][f].d1 or w[g][f].d2 != w[k][f].d2))
               same = false;
         if (same)
            ds.slot[g] = ds.slot[k];
      }
      if (ds.slot[g] < 0) {
         grp[ds.nslot] = g;
         ds.slot[g] = ds.nslot++;
      }
   }
   if (ds.nslot == 1 and full and not dodl and w[grp[0]][grp[0]].f == 1) {
      ds.nslot = 0;
      h.plain = true;
      return ds;
   }

   for (int s = 0; s < ds.nslot; ++s) {
      for (int t = 0; t < ds.nslot; ++t) {
         const LmdaFn& f = w[grp[s]][grp[t]];
         ds.w[s][t] = f.f;
         ds.dw[s][t] = f.d1;
         ds.d2w[s][t] = f.d2;
      }
   }
   double cpr = 0;
   for (int g = 0; g < 3; ++g) {
      if (not have[g])
         continue;
      for (int k = 0; k < 3; ++k) {
         if (not have[k])
            continue;
         double cc = dsp_csum[g] * dsp_csum[k];
         cpr += w[g][k].f * cc;
         h.dcpr += w[g][k].d1 * cc;
         h.d2cpr += w[g][k].d2 * cc;
      }
      h.dself += w[g][g].d1 * dsp_csq[g];
      h.d2self += w[g][g].d2 * dsp_csq[g];
   }
   h.cpr = cpr;
   return ds;
}

// Where the grid of each site starts, or -1 for a site of a group left out,
// set again only when the slots of the groups change.
static void dispSlotOffsets(const DispSlots& ds)
{
   if (std::equal(ds.slot, ds.slot + 3, dsp_offset_slot))
      return;
   std::vector<int> off(n);
   for (int i = 0; i < n; ++i) {
      int s = ds.slot[hostGroup(i)];
      off[i] = (s < 0 ? -1 : s * dsp_stride);
   }
   darray::copyin(g::q0, n, dsp_offset, off.data());
   waitFor(g::q0);
   std::copy(ds.slot, ds.slot + 3, dsp_offset_slot);
}

TINKER_FVOID2(acc1, cu1, edispEwaldReal, int, DispSoft, DispSoft);
TINKER_FVOID2(acc1, cu1, pmeConvDisp, int);
TINKER_FVOID2(acc1, cu1, edispEwaldRecipSelf, int);
TINKER_FVOID2(acc1, cu1, pmeConvDispSlots, int, DispSlots, real*, int);
TINKER_FVOID2(acc1, cu1, edispEwaldRecipSelfSlots, int, DispSlots, const real*, int, const int*);
static void edispEwald(int vers, const DispSoft& sc1, const DispSoft& sc2)
{
   TINKER_FCALL2(acc1, cu1, edispEwaldReal, vers, sc1, sc2);

   // recip and self, whose lambda derivatives vanish on the plain grid
   DispHost h;
   const DispSlots ds = dispSlots(vers & calc::energy_dlmda1, h);
   if (h.plain)
      vers &= ~calc::dlmda;
   auto do_e = vers & calc::energy;
   auto do_v = vers & calc::virial;
   auto do_g = vers & calc::grad;
   auto do_dl1 = vers & calc::energy_dlmda1;
   auto do_dl2 = vers & calc::energy_dlmda2;
   auto do_vdl = vers & calc::virial_dlmda;
   PMEUnit u = dpme_unit;

   if (h.plain) {
      gridDisp(u, csix);
      fftfront(u);
      TINKER_FCALL2(acc1, cu1, pmeConvDisp, vers);
      if (do_g)
         fftback(u);
      TINKER_FCALL2(acc1, cu1, edispEwaldRecipSelf, vers);
   } else if (ds.nslot > 0) {
      // the lambda derivatives of the force come with the force itself
      dispSlotOffsets(ds);
      gridDispSlots(u, csix, dsp_offset, ds.nslot * dsp_stride, dsp_slots);
      fftfrontOneSet(u, dsp_slots, ds.nslot, dsp_stride);
      TINKER_FCALL2(acc1, cu1, pmeConvDispSlots, vers, ds, dsp_slots, dsp_stride);
      if (do_g)
         fftbackOneSet(u, dsp_slots, ds.nslot, dsp_stride);
      TINKER_FCALL2(acc1, cu1, edispEwaldRecipSelfSlots, vers, ds, dsp_slots, dsp_stride, dsp_offset);
   } else if (vers & calc::analyz) {
      // every group is left out, but each site's self term is still counted
      // (edisp.f:dspself), which the self kernel does for a site without a slot
      dispSlotOffsets(ds);
      TINKER_FCALL2(acc1, cu1, edispEwaldRecipSelfSlots, vers, ds, dsp_slots, dsp_stride, dsp_offset);
   }

   // account for the total energy and virial correction term, the weighted C6
   // pair sum of the zero wavevector times fcpr
   const double aewald = u->aewald;
   const double a3 = aewald * aewald * aewald;
   const double fcpr = a3 * std::pow(M_PI, 1.5) / (6 * boxVolume());
   if (do_e || do_v) {
      energy_prec term = h.cpr * fcpr;
      if (do_e) {
         energy_edsp -= term;
         energy_vdw -= term;
      }
      if (do_v) {
         virial_edsp[0] += term; // xx
         virial_edsp[4] += term; // yy
         virial_edsp[8] += term; // zz
         virial_vdw[0] += term;  // xx
         virial_vdw[4] += term;  // yy
         virial_vdw[8] += term;  // zz
      }
   }

   // the lambda derivatives of the correction term and of the self terms
   // (edisp.f:dspself)
   if (do_dl1) {
      const double fself = a3 * a3 / 12;
      energy_prec term = h.dcpr * fcpr;
      energy_prec dl1 = h.dself * fself - term;
      devdl += dl1;
      dedl += dl1;
      if (do_dl2) {
         energy_prec dl2 = h.d2self * fself - h.d2cpr * fcpr;
         d2evdl2 += dl2;
         d2edl2 += dl2;
      }
      if (do_vdl) {
         dvirdl[0] += term;
         dvirdl[4] += term;
         dvirdl[8] += term;
      }
   }
}

void edisp(int vers)
{
   auto rc_a = rc_flag & calc::analyz;
   auto do_a = vers & calc::analyz;
   auto do_e = vers & calc::energy;
   auto do_v = vers & calc::virial;
   auto do_g = vers & calc::grad;

   zeroOnHost(energy_edsp, virial_edsp);
   size_t bsize = bufferSize();
   if (rc_a) {
      if (do_a)
         darray::zero(g::q0, bsize, ndisp);
      if (do_e)
         darray::zero(g::q0, bsize, edsp);
      if (do_v)
         darray::zero(g::q0, bsize, vir_edsp);
      if (do_g)
         darray::zero(g::q0, n, dedspx, dedspy, dedspz);
   }

   // The lambda derivatives follow vlambda and go to the van der Waals
   // channels, while its map has slope or curvature (edisp1.f).
   // The soft core powers of the real part are taken once here; the slot
   // weights of the reciprocal part take their own (dispSlots).
   const int dlvers = lmdaDerivVers(vers, vdlmdaActive());
   const bool dodl = lmdaDodl(dlvers);
   // Ligand 2 of a relative calculation takes its own soft core.
   const DispSoft sc1 = dispSoft(vdwGroupPower(scexpd, 1, dodl), vdwGroupPower(scexpd + scexpdq, 1, dodl));
   const DispSoft sc2 =
      use_rel ? dispSoft(vdwGroupPower(scexpd, 2, dodl), vdwGroupPower(scexpd + scexpdq, 2, dodl)) : sc1;
   lmdaChannel(LmdaTerm::VDW).zero(vers);
   if (useDEwald())
      edispEwald(dlvers, sc1, sc2);
   else
      edispNonEwald(dlvers, sc1, sc2);
   lmdaChannel(LmdaTerm::VDW).flush(vers);
   lrcApply(vers, dlvers, dsp_lrc, energy_edsp, virial_edsp);
   if (rc_a) {
      if (do_e) {
         EnergyBuffer u = edsp;
         energy_prec e = energyReduce(u);
         energy_edsp += e;
         energy_vdw += e;
      }
      if (do_v) {
         VirialBuffer u = vir_edsp;
         virial_prec v[9];
         virialReduce(v, u);
         for (int iv = 0; iv < 9; ++iv) {
            virial_edsp[iv] += v[iv];
            virial_vdw[iv] += v[iv];
         }
      }
      if (do_g)
         sumGradient(gx_vdw, gy_vdw, gz_vdw, dedspx, dedspy, dedspz);
   }
}
}
