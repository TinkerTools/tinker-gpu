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

namespace tinker {
// The C6 sums of the environment and of the mutated ligand, the sum of the
// squared C6 of the ligand for its self term, and the copy of the forward
// transform that separates their two transforms (edisp1.f:edrecip1).
static double dsp_sume, dsp_suml, dsp_sqll;
static real* dsp_qfwd;
// The lambda derivative of the potential, which gives the force one.
static real* dsp_qdl;
// The class sums of the long-range dispersion correction.
static LrcSums dsp_lrc;

bool useDEwald()
{
   return use(Potent::DISP) and static_cast<bool>(limits::use_dewald);
}

void edispData(RcOp op)
{
   if (not use(Potent::DISP))
      return;

   auto rc_a = rc_flag & calc::analyz;

   if (op & RcOp::DEALLOC) {
      darray::deallocate(csix, adisp);
      darray::deallocate(dspexclude, dspexclude_scale);
      darray::deallocate(dsp_qfwd, dsp_qdl);
      dsp_qfwd = nullptr;
      dsp_qdl = nullptr;

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
      // a mutated ligand may be weighted apart from the environment (dispSplit)
      dsp_qfwd = nullptr;
      dsp_qdl = nullptr;
      if (useDEwald() and mutant::nmut > 0) {
         const auto& st = *dpme_unit;
         darray::allocate(2 * st.nfft1 * st.nfft2 * st.nfft3, &dsp_qfwd);
         if (use_vdlmda and use_d2lmda)
            darray::allocate(2 * st.nfft1 * st.nfft2 * st.nfft3, &dsp_qdl);
      }

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

      dsp_sume = 0;
      dsp_suml = 0;
      dsp_sqll = 0;
      if (mutant::nmut > 0) {
         for (int i = 0; i < n; ++i) {
            double c = disp::csix[i];
            if (mutant::mutg[i] == 0) {
               dsp_sume += c;
            } else {
               dsp_suml += c;
               dsp_sqll += c * c;
            }
         }
      }

      dsp_lrc = {};
      if (dsppot::use_dcorr && not useDEwald())
         lrcSetup(dsp_lrc, true);
   }
}
}

namespace tinker {
// A lambda power vlam^e of the soft core and its lambda derivatives
// (edspsc.f:dspsc4). The soft core is linear in its two powers, so the van der
// Waals lambda chain rule goes on the powers themselves.
static LmdaFn dispPower(double e, bool dodl)
{
   return lmdaPwrChain(vlam, e, dvldlmda, d2vldlmda2, dodl);
}

// The soft core of the real space pairs from its powers lambda^n and
// lambda^(n+q) (edspsc.f:dspsc4).
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

TINKER_FVOID2(acc1, cu1, edispNonEwald, int, DispSoft);
static void edispNonEwald(int vers, const DispSoft& sc)
{
   TINKER_FCALL2(acc1, cu1, edispNonEwald, vers, sc);
}

// The terms of the reciprocal and self parts summed on the host: the weighted
// C6 pair sum of the zero wavevector with its lambda derivatives, and the
// lambda derivatives of the ligand-ligand weight, which scales the self term
// of each ligand site.
struct DispHost
{
   real cpr;
   double dcpr, d2cpr;
   double dwll, d2wll;
};

// The reciprocal and self terms weight the ligand by the lambda power v of the
// soft core, which the real space pairs carry beyond its switch radius: v
// between the ligand and the environment, and within the ligand v when it is
// annihilated or 1 when it is decoupled. The weighted C6 sum product replaces
// csixpr, and the split stays on at full coupling while the weights have
// lambda derivatives (edisp.f:dspmutwt4).
static DispSplit dispSplit(const LmdaFn& v, bool dodl, DispHost& h)
{
   DispSplit ds = {false, 1, 1, 0, 0, 0, 0, nullptr, nullptr, nullptr};
   h = {csixpr, 0, 0, 0, 0};
   if (mutant::nmut == 0)
      return ds;
   if (not dodl and v.f == 1)
      return ds;

   double wel = v.f, dwel = v.d1, d2wel = v.d2;
   double wll = 1, dwll = 0, d2wll = 0;
   if (vcouple == Vdw::ANNIHILATE) {
      wll = v.f;
      dwll = v.d1;
      d2wll = v.d2;
   }
   assert(dsp_qfwd);
   ds.split = true;
   ds.wel = wel;
   ds.wll = wll;
   ds.dwel = dwel;
   ds.dwll = dwll;
   ds.d2wel = d2wel;
   ds.d2wll = d2wll;
   ds.part = mut;
   ds.qfwd = reinterpret_cast<const real(*)[2]>(dsp_qfwd);
   double se = dsp_sume, sl = dsp_suml;
   h.cpr = se * se + 2 * wel * se * sl + wll * sl * sl;
   h.dcpr = 2 * dwel * se * sl + dwll * sl * sl;
   h.d2cpr = 2 * d2wel * se * sl + d2wll * sl * sl;
   h.dwll = dwll;
   h.d2wll = d2wll;
   return ds;
}

TINKER_FVOID2(acc1, cu1, edispEwaldReal, int, DispSoft);
TINKER_FVOID2(acc1, cu1, edispEwaldRecipSelf, int, DispSplit);
TINKER_FVOID2(acc1, cu1, pmeConvDisp, int, DispSplit);
TINKER_FVOID2(acc0, cu1, edispEwaldRecipDlmda, DispSplit);
static void edispEwald(int vers, const DispSoft& sc, const LmdaFn& v)
{
   TINKER_FCALL2(acc1, cu1, edispEwaldReal, vers, sc);

   // recip and self, whose lambda derivatives vanish without a mutated ligand
   DispHost h;
   DispSplit ds = dispSplit(v, vers & calc::energy_dlmda1, h);
   if (not ds.split)
      vers &= ~calc::dlmda;
   auto do_e = vers & calc::energy;
   auto do_v = vers & calc::virial;
   auto do_g = vers & calc::grad;
   auto do_dl1 = vers & calc::energy_dlmda1;
   auto do_dl2 = vers & calc::energy_dlmda2;
   auto do_gdl = vers & calc::grad_dlmda;
   auto do_vdl = vers & calc::virial_dlmda;
   PMEUnit u = dpme_unit;
   const int ntot = u->nfft1 * u->nfft2 * u->nfft3;

   if (do_gdl) {
      assert(dsp_qdl);
      ds.qdl = reinterpret_cast<real(*)[2]>(dsp_qdl);
   }
   gridDisp(u, csix, ds.part);
   fftfront(u);
   // each point of the convolution reads the opposite wavevector as well
   if (ds.split)
      darray::copy(g::q0, 2 * ntot, dsp_qfwd, u->qgrid);
   TINKER_FCALL2(acc1, cu1, pmeConvDisp, vers, ds);
   if (do_g) {
      fftback(u);
   }
   // the lambda derivatives of the self term are added below
   TINKER_FCALL2(acc1, cu1, edispEwaldRecipSelf, vers & ~calc::dlmda, ds);

   // the lambda derivative of the potential goes through the same grid
   if (do_gdl) {
      darray::copy(g::q0, 2 * ntot, u->qgrid, dsp_qdl);
      fftback(u);
      TINKER_FCALL2(acc0, cu1, edispEwaldRecipDlmda, ds);
   }

   // account for the total energy and virial correction term
   if CONSTEXPR (do_e || do_v) {
      const real aewald = u->aewald;
      const real denom0 = 6 * boxVolume() / std::pow(M_PI, 1.5);
      energy_prec term = h.cpr * aewald * aewald * aewald / denom0;
      if CONSTEXPR (do_e) {
         energy_edsp -= term;
         energy_vdw -= term;
      }
      if CONSTEXPR (do_v) {
         virial_edsp[0] += term; // xx
         virial_edsp[4] += term; // yy
         virial_edsp[8] += term; // zz
         virial_vdw[0] += term;  // xx
         virial_vdw[4] += term;  // yy
         virial_vdw[8] += term;  // zz
      }
   }

   // the lambda derivatives of the correction term and of the self term of
   // the ligand sites (edisp.f:dspself)
   if (do_dl1) {
      const double aewald = u->aewald;
      const double a3 = aewald * aewald * aewald;
      const double fcpr = a3 * std::pow(M_PI, 1.5) / (6 * boxVolume());
      const double fself = a3 * a3 / 12 * dsp_sqll;
      energy_prec term = h.dcpr * fcpr;
      energy_prec dl1 = h.dwll * fself - term;
      devdl += dl1;
      dedl += dl1;
      if (do_dl2) {
         energy_prec dl2 = h.d2wll * fself - h.d2cpr * fcpr;
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
   // The soft core powers are taken once for the real and reciprocal parts.
   const int dlvers = lmdaDerivVers(vers, vdlmdaActive());
   const bool dodl = dlvers & calc::energy_dlmda1;
   const LmdaFn vn = dispPower(scexpd, dodl);
   const DispSoft sc = dispSoft(vn, dispPower(scexpd + scexpdq, dodl));
   vdwLmdaZero(vers);
   if (useDEwald())
      edispEwald(dlvers, sc, vn);
   else
      edispNonEwald(dlvers, sc);
   vdwLmdaFlush(vers);
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
