#include "ff/echarge.h"
#include "ff/dlmda.h"
#include "ff/elec.h"
#include "ff/energy.h"
#include "ff/nblist.h"
#include "ff/pme.h"
#include "ff/pmestream.h"
#include "ff/potent.h"
#include "math/zero.h"
#include "tool/externfunc.h"
#include <tinker/detail/chgpot.hh>
#include <tinker/detail/couple.hh>
#include <tinker/detail/sizes.hh>

namespace tinker {
// The real grid of the lambda derivatives of the charges, in the padded layout
// of the charge grid, for the second lambda derivatives under Ewald
// (echarge1.f:ecrecip1).
static real* dlqgrid;

void echargeData(RcOp op)
{
   if (not use(Potent::CHARGE))
      return;

   auto rc_a = rc_flag & calc::analyz;

   if (op & RcOp::DEALLOC) {
      ncexclude = 0;
      darray::deallocate(cexclude, cexclude_scale, dlqgrid);
      dlqgrid = nullptr;

      if (rc_a) {
         bufferDeallocate(rc_flag, nec);
         bufferDeallocate(rc_flag, ec, vir_ec, decx, decy, decz);
      }
      nec = nullptr;
      ec = nullptr;
      vir_ec = nullptr;
      decx = nullptr;
      decy = nullptr;
      decz = nullptr;
   }

   if (op & RcOp::ALLOC) {
      ebuffer = chgpot::ebuffer;

      c2scale = chgpot::c2scale;
      c3scale = chgpot::c3scale;
      c4scale = chgpot::c4scale;
      c5scale = chgpot::c5scale;

      std::vector<int> exclik;
      std::vector<real> excl;
      // see attach.f
      const int maxn13 = 3 * sizes::maxval;
      const int maxn14 = 9 * sizes::maxval;
      const int maxn15 = 27 * sizes::maxval;
      for (int i = 0; i < n; ++i) {
         int nn;
         int bask;
         if (c2scale != 1) {
            nn = couple::n12[i];
            for (int j = 0; j < nn; ++j) {
               int k = couple::i12[i][j];
               k -= 1;
               if (k > i) {
                  exclik.push_back(i);
                  exclik.push_back(k);
                  excl.push_back(c2scale);
               }
            }
         }
         if (c3scale != 1) {
            nn = couple::n13[i];
            bask = i * maxn13;
            for (int j = 0; j < nn; ++j) {
               int k = couple::i13[bask + j];
               k -= 1;
               if (k > i) {
                  exclik.push_back(i);
                  exclik.push_back(k);
                  excl.push_back(c3scale);
               }
            }
         }
         if (c4scale != 1) {
            nn = couple::n14[i];
            bask = i * maxn14;
            for (int j = 0; j < nn; ++j) {
               int k = couple::i14[bask + j];
               k -= 1;
               if (k > i) {
                  exclik.push_back(i);
                  exclik.push_back(k);
                  excl.push_back(c4scale);
               }
            }
         }
         if (c5scale != 1) {
            nn = couple::n15[i];
            bask = i * maxn15;
            for (int j = 0; j < nn; ++j) {
               int k = couple::i15[bask + j];
               k -= 1;
               if (k > i) {
                  exclik.push_back(i);
                  exclik.push_back(k);
                  excl.push_back(c5scale);
               }
            }
         }
      }
      ncexclude = excl.size();
      darray::allocate(ncexclude, &cexclude, &cexclude_scale);
      darray::copyin(g::q0, ncexclude, cexclude, exclik.data());
      darray::copyin(g::q0, ncexclude, cexclude_scale, excl.data());
      waitFor(g::q0);

      dlqgrid = nullptr;
      if (use_edlmda and use_d2lmda and useEwald() and (pltfm_config & Platform::CUDA)) {
         const auto& st = *epme_unit;
         darray::allocate(2 * (st.nfft1 / 2 + 1) * st.nfft2 * st.nfft3, &dlqgrid);
      }

      nec = nullptr;
      ec = eng_buf_elec;
      vir_ec = vir_buf_elec;
      decx = gx_elec;
      decy = gy_elec;
      decz = gz_elec;
      if (rc_a) {
         bufferAllocate(rc_flag, &nec);
         bufferAllocate(rc_flag, &ec, &vir_ec, &decx, &decy, &decz);
      }
   }

   if (op & RcOp::INIT) {}
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, echargeNonEwald, int);
static void echargeNonEwald(int vers)
{
   TINKER_FCALL2(acc1, cu1, echargeNonEwald, vers);
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, echargeEwaldFphiSelf, int, const real*);
void echargeEwaldRecipSelf(int vers)
{
   pmeStreamStartWait(use_pme_stream);

   // ewald recip space, self term
   // ewald real space

   const PMEUnit pu = epme_unit;
   EnergyBuffer e = (vers & calc::energy) ? ec : nullptr;
   VirialBuffer v = (vers & calc::virial) ? vir_ec : nullptr;
   // The second lambda derivatives, and those of the force and the virial,
   // take the potential of the charge derivatives as well, from a grid of
   // their own; the first only needs that of the charges (echarge1.f:ecrecip1).
   real* dlgrid = (vers & calc::energy_dlmda2) ? dlqgrid : nullptr;
   if (pltfm_config & Platform::CUDA) {
      // the charge grid is real: in-place R2C and C2R FFTs on the half spectrum
      gridPchgOneSet(pu, pchg);
      fftfrontOneSet(pu);
      if (dlgrid) {
         gridPchgOneSet(pu, pchg_dq, dlgrid);
         fftfrontOneSet(pu, dlgrid);
         VirialBuffer dv = (vers & calc::virial_dlmda) ? dvirdl_buf : nullptr;
         pmeConvOneSetDlmda(pu, dlgrid, e, v, dv, elecChain().d1);
      } else {
         pmeConvOneSet(pu, e, v);
      }
      if (vers & calc::grad) {
         fftbackOneSet(pu);
         if (dlgrid)
            fftbackOneSet(pu, dlgrid);
      }
   } else {
      gridPchg(pu, pchg);
      fftfront(pu);
      pmeConv(pu, e, v);
      if (vers & calc::grad)
         fftback(pu);
   }

   // fphi_pchg, recip, self

   TINKER_FCALL2(acc1, cu1, echargeEwaldFphiSelf, vers, dlgrid);
   echargeEwaldBackground(vers);

   pmeStreamFinishRecord(use_pme_stream);
}

TINKER_FVOID2(acc1, cu1, echargeEwaldReal, int);
void echarge(int vers)
{
   auto rc_a = rc_flag & calc::analyz;
   auto do_a = vers & calc::analyz;
   auto do_e = vers & calc::energy;
   auto do_v = vers & calc::virial;
   auto do_g = vers & calc::grad;

   zeroOnHost(energy_ec, virial_ec);
   size_t bsize = bufferSize();
   if (rc_a) {
      if (do_a)
         darray::zero(g::q0, bsize, nec);
      if (do_e)
         darray::zero(g::q0, bsize, ec);
      if (do_v)
         darray::zero(g::q0, bsize, vir_ec);
      if (do_g)
         darray::zero(g::q0, n, decx, decy, decz);
   }

   lmdaChannel(LmdaTerm::ELEC).zero(vers);

   // The charges take their lambda derivatives as lmdaDrv decides
   // (dlambda.f:chgdlflags).
   const int dlvers = lmdaDerivVers(vers, edlmdaActive());
   // ahead of the reciprocal space, which may read the coordinates on the PME
   // stream while bounds() would move them on s0
   exfield(dlvers, 0);
   if (useEwald()) {
      // The analysis buffers are zeroed and reduced inside this function, on
      // the main stream. The PME stream must not start before the zeroing and
      // must have finished before the reduction, whatever "vers" is.
      const bool pme_stream_a = use_pme_stream and rc_a;
      pmeStreamStartRecord(pme_stream_a);
      echargeEwaldRecipSelf(dlvers);
      TINKER_FCALL2(acc1, cu1, echargeEwaldReal, dlvers);
      pmeStreamFinishWait(pme_stream_a);
   } else {
      echargeNonEwald(dlvers);
   }

   if (rc_a) {
      if (do_e) {
         EnergyBuffer u = ec;
         energy_prec e = energyReduce(u);
         energy_ec += e;
         energy_elec += e;
      }
      if (do_v) {
         VirialBuffer u = vir_ec;
         virial_prec v[9];
         virialReduce(v, u);
         for (int iv = 0; iv < 9; ++iv) {
            virial_ec[iv] += v[iv];
            virial_elec[iv] += v[iv];
         }
      }
      if (do_g)
         sumGradient(gx_elec, gy_elec, gz_elec, decx, decy, decz);
   }
   lmdaChannel(LmdaTerm::ELEC).flush(vers);
}
}
