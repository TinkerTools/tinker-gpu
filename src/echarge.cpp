#include "ff/echarge.h"
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
void echargeData(RcOp op)
{
   if (not use(Potent::CHARGE))
      return;

   auto rc_a = rc_flag & calc::analyz;

   if (op & RcOp::DEALLOC) {
      ncexclude = 0;
      darray::deallocate(cexclude, cexclude_scale);

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
TINKER_FVOID2(acc1, cu1, echargeEwaldFphiSelf, int);
void echargeEwaldRecipSelf(int vers)
{
   pmeStreamStartWait(use_pme_stream);

   // ewald recip space, self term
   // ewald real space

   const PMEUnit pu = epme_unit;
   gridPchg(pu, pchg);
   fftfront(pu);
   if (vers & calc::virial) {
      if (vers & calc::energy) {
         pmeConv(pu, ec, vir_ec);
      } else {
         pmeConv(pu, vir_ec);
      }
   } else {
      if (vers & calc::energy) {
         pmeConv(pu, ec);
      } else {
         pmeConv(pu);
      }
   }
   if (vers & calc::grad) {
      fftback(pu);
   }

   // fphi_pchg, recip, self

   TINKER_FCALL2(acc1, cu1, echargeEwaldFphiSelf, vers);

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

   if (useEwald()) {
      echargeEwaldRecipSelf(vers);
      TINKER_FCALL2(acc1, cu1, echargeEwaldReal, vers);
      pmeStreamFinishWait(use_pme_stream and static_cast<bool>(vers & calc::analyz));
   } else {
      echargeNonEwald(vers);
   }
   exfield(vers, 0);

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
}
}
