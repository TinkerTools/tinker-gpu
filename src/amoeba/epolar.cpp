#include "ff/amoeba/epolar.h"
#include "ff/amoeba/empole.h"
#include "ff/amoeba/induce.h"
#include "ff/modamoeba.h"
#include "ff/hippo/epolar.h"
#include "ff/elec.h"
#include "ff/energy.h"
#include "ff/hippo/cflux.h"
#include "ff/nblist.h"
#include "ff/potent.h"
#include "math/zero.h"
#include "tool/externfunc.h"
#include "tool/iofortstr.h"
#include "tool/ioprint.h"
#include <tinker/detail/couple.hh>
#include <tinker/detail/extfld.hh>
#include <tinker/detail/mplpot.hh>
#include <tinker/detail/polar.hh>
#include <tinker/detail/polgrp.hh>
#include <tinker/detail/polpot.hh>
#include <tinker/detail/uprior.hh>

namespace tinker {
TINKER_FVOID2(cpp0, cu1, epolarDataBinding, RcOp);
void epolarData(RcOp op)
{
   if (not use(Potent::POLAR)) return;
   if (mplpot::use_chgpen and not polpot::use_tholed) // HIPPO Polarization
      return;

   auto rc_a = rc_flag & calc::analyz;

   if (op & RcOp::DEALLOC) {
      njpolar = 0;
      darray::deallocate(jpolar, thlval);

      nuexclude = 0;
      darray::deallocate(uexclude, uexclude_scale);
      ndpexclude = 0;
      darray::deallocate(dpexclude, dpexclude_scale);
      ndpuexclude = 0;
      darray::deallocate(dpuexclude, dpuexclude_scale);

      darray::deallocate(polarity, thole, pdamp, polarity_inv);
      if (polpot::use_tholed) darray::deallocate(dirdamp);

      if (rc_a) {
         bufferDeallocate(rc_flag, nep);
         bufferDeallocate(rc_flag, ep, vir_ep, depx, depy, depz);
      }
      nep = nullptr;
      ep = nullptr;
      vir_ep = nullptr;
      depx = nullptr;
      depy = nullptr;
      depz = nullptr;

      darray::deallocate(ufld, dufld);
      darray::deallocate(work01_, work02_, work03_, work04_, work05_);
      if (not polpot::use_tholed) // AMOEBA
         darray::deallocate(work06_, work07_, work08_, work09_, work10_);

      if (polpred == UPred::ASPC) {
         darray::deallocate(udalt_00, udalt_01, udalt_02, udalt_03, udalt_04,
            udalt_05, udalt_06, udalt_07, udalt_08, udalt_09, udalt_10,
            udalt_11, udalt_12, udalt_13, udalt_14, udalt_15);
         if (not polpot::use_tholed) // AMOEBA
            darray::deallocate(upalt_00, upalt_01, upalt_02, upalt_03, upalt_04,
               upalt_05, upalt_06, upalt_07, upalt_08, upalt_09, upalt_10,
               upalt_11, upalt_12, upalt_13, upalt_14, upalt_15);
      } else if (polpred == UPred::GEAR) {
         darray::deallocate(udalt_00, udalt_01, udalt_02, udalt_03, udalt_04,
            udalt_05);
         if (not polpot::use_tholed) // AMOEBA
            darray::deallocate(upalt_00, upalt_01, upalt_02, upalt_03, upalt_04,
               upalt_05);
      } else if (polpred == UPred::LSQR) {
         darray::deallocate(udalt_00, udalt_01, udalt_02, udalt_03, udalt_04,
            udalt_05, udalt_06);
         darray::deallocate(udalt_lsqr_a, udalt_lsqr_b);
         if (not polpot::use_tholed) { // AMOEBA
            darray::deallocate(upalt_00, upalt_01, upalt_02, upalt_03, upalt_04,
               upalt_05, upalt_06);
            darray::deallocate(upalt_lsqr_a, upalt_lsqr_b);
         }
      }
      polpred = UPred::NONE;
      maxualt = 0;
      nualt = 0;
   }

   if (op & RcOp::ALLOC) {
      // see also attach.f
      const int maxn13 = 3 * sizes::maxval;
      const int maxn14 = 9 * sizes::maxval;
      const int maxn15 = 27 * sizes::maxval;
      const int maxp11 = polgrp::maxp11;
      const int maxp12 = polgrp::maxp12;
      const int maxp13 = polgrp::maxp13;
      const int maxp14 = polgrp::maxp14;

      struct dpu_scale
      {
         real d, p, u;
      };
      auto insert_dpu = [](std::map<std::pair<int, int>, dpu_scale>& m, int i,
                           int k, real val, char ch) {
         std::pair<int, int> key;
         key.first = i;
         key.second = k;
         auto it = m.find(key);
         if (it == m.end()) {
            dpu_scale dpu;
            dpu.d = 1;
            dpu.p = 1;
            dpu.u = 1;
            if (ch == 'd')
               dpu.d = val;
            else if (ch == 'p')
               dpu.p = val;
            else if (ch == 'u')
               dpu.u = val;
            m[key] = dpu;
         } else {
            if (ch == 'd')
               it->second.d = val;
            else if (ch == 'p')
               it->second.p = val;
            else if (ch == 'u')
               it->second.u = val;
         }
      };
      std::map<std::pair<int, int>, dpu_scale> ik_dpu;

      std::vector<int> exclik;
      std::vector<real> excls;

      u1scale = polpot::u1scale;
      u2scale = polpot::u2scale;
      u3scale = polpot::u3scale;
      u4scale = polpot::u4scale;
      exclik.clear();
      excls.clear();
      for (int i = 0; i < n; ++i) {
         int nn, bask;

         if (u1scale != 1) {
            nn = polgrp::np11[i];
            bask = i * maxp11;
            for (int j = 0; j < nn; ++j) {
               int k = polgrp::ip11[bask + j] - 1;
               if (k > i) {
                  insert_dpu(ik_dpu, i, k, u1scale, 'u');
                  exclik.push_back(i);
                  exclik.push_back(k);
                  exclik.push_back(u1scale);
               }
            }
         }

         if (u2scale != 1) {
            nn = polgrp::np12[i];
            bask = i * maxp12;
            for (int j = 0; j < nn; ++j) {
               int k = polgrp::ip12[bask + j] - 1;
               if (k > i) {
                  insert_dpu(ik_dpu, i, k, u2scale, 'u');
                  exclik.push_back(i);
                  exclik.push_back(k);
                  exclik.push_back(u2scale);
               }
            }
         }

         if (u3scale != 1) {
            nn = polgrp::np13[i];
            bask = i * maxp13;
            for (int j = 0; j < nn; ++j) {
               int k = polgrp::ip13[bask + j] - 1;
               if (k > i) {
                  insert_dpu(ik_dpu, i, k, u3scale, 'u');
                  exclik.push_back(i);
                  exclik.push_back(k);
                  exclik.push_back(u3scale);
               }
            }
         }

         if (u4scale != 1) {
            nn = polgrp::np14[i];
            bask = i * maxp14;
            for (int j = 0; j < nn; ++j) {
               int k = polgrp::ip14[bask + j] - 1;
               if (k > i) {
                  insert_dpu(ik_dpu, i, k, u4scale, 'u');
                  exclik.push_back(i);
                  exclik.push_back(k);
                  exclik.push_back(u4scale);
               }
            }
         }
      }
      nuexclude = excls.size();
      darray::allocate(nuexclude, &uexclude, &uexclude_scale);
      darray::copyin(g::q0, nuexclude, uexclude, exclik.data());
      darray::copyin(g::q0, nuexclude, uexclude_scale, excls.data());
      waitFor(g::q0);

      d1scale = polpot::d1scale;
      d2scale = polpot::d2scale;
      d3scale = polpot::d3scale;
      d4scale = polpot::d4scale;

      p2scale = polpot::p2scale;
      p3scale = polpot::p3scale;
      p4scale = polpot::p4scale;
      p5scale = polpot::p5scale;

      p2iscale = polpot::p2iscale;
      p3iscale = polpot::p3iscale;
      p4iscale = polpot::p4iscale;
      p5iscale = polpot::p5iscale;
      exclik.clear();
      excls.clear();
      struct dp_scale
      {
         real d, p;
      };
      auto insert_dp = [](std::map<int, dp_scale>& m, int k, real val,
                          char dpchar) {
         auto it = m.find(k);
         if (it == m.end()) {
            dp_scale dp;
            dp.d = 1;
            dp.p = 1;
            if (dpchar == 'd')
               dp.d = val;
            else if (dpchar == 'p')
               dp.p = val;
            m[k] = dp;
         } else {
            if (dpchar == 'd')
               it->second.d = val;
            else if (dpchar == 'p')
               it->second.p = val;
         }
      };
      for (int i = 0; i < n; ++i) {
         std::map<int, dp_scale> k_dpscale;
         int nn, bask;

         if (d1scale != 1) {
            nn = polgrp::np11[i];
            bask = i * maxp11;
            for (int j = 0; j < nn; ++j) {
               int k = polgrp::ip11[bask + j] - 1;
               if (k > i) {
                  insert_dpu(ik_dpu, i, k, d1scale, 'd');
                  insert_dp(k_dpscale, k, d1scale, 'd');
               }
            }
         }

         if (d2scale != 1) {
            nn = polgrp::np12[i];
            bask = i * maxp12;
            for (int j = 0; j < nn; ++j) {
               int k = polgrp::ip12[bask + j] - 1;
               if (k > i) {
                  insert_dpu(ik_dpu, i, k, d2scale, 'd');
                  insert_dp(k_dpscale, k, d2scale, 'd');
               }
            }
         }

         if (d3scale != 1) {
            nn = polgrp::np13[i];
            bask = i * maxp13;
            for (int j = 0; j < nn; ++j) {
               int k = polgrp::ip13[bask + j] - 1;
               if (k > i) {
                  insert_dpu(ik_dpu, i, k, d3scale, 'd');
                  insert_dp(k_dpscale, k, d3scale, 'd');
               }
            }
         }

         if (d4scale != 1) {
            nn = polgrp::np14[i];
            bask = i * maxp14;
            for (int j = 0; j < nn; ++j) {
               int k = polgrp::ip14[bask + j] - 1;
               if (k > i) {
                  insert_dpu(ik_dpu, i, k, d4scale, 'd');
                  insert_dp(k_dpscale, k, d4scale, 'd');
               }
            }
         }

         if (p2scale != 1 or p2iscale != 1) {
            nn = couple::n12[i];
            for (int j = 0; j < nn; ++j) {
               int k = couple::i12[i][j];
               real val = p2scale;
               for (int jj = 0; jj < polgrp::np11[i]; ++jj) {
                  if (k == polgrp::ip11[i * maxp11 + jj]) val = p2iscale;
               }
               k -= 1;
               if (k > i) {
                  insert_dpu(ik_dpu, i, k, val, 'p');
                  insert_dp(k_dpscale, k, val, 'p');
               }
            }
         }

         if (p3scale != 1 or p3iscale != 1) {
            nn = couple::n13[i];
            bask = i * maxn13;
            for (int j = 0; j < nn; ++j) {
               int k = couple::i13[bask + j];
               real val = p3scale;
               for (int jj = 0; jj < polgrp::np11[i]; ++jj) {
                  if (k == polgrp::ip11[i * maxp11 + jj]) val = p3iscale;
               }
               k -= 1;
               if (k > i) {
                  insert_dpu(ik_dpu, i, k, val, 'p');
                  insert_dp(k_dpscale, k, val, 'p');
               }
            }
         }

         if (p4scale != 1 or p4iscale != 1) {
            nn = couple::n14[i];
            bask = i * maxn14;
            for (int j = 0; j < nn; ++j) {
               int k = couple::i14[bask + j];
               real val = p4scale;
               for (int jj = 0; jj < polgrp::np11[i]; ++jj) {
                  if (k == polgrp::ip11[i * maxp11 + jj]) val = p4iscale;
               }
               k -= 1;
               if (k > i) {
                  insert_dpu(ik_dpu, i, k, val, 'p');
                  insert_dp(k_dpscale, k, val, 'p');
               }
            }
         }

         if (p5scale != 1 or p5iscale != 1) {
            nn = couple::n15[i];
            bask = i * maxn15;
            for (int j = 0; j < nn; ++j) {
               int k = couple::i15[bask + j];
               real val = p5scale;
               for (int jj = 0; jj < polgrp::np11[i]; ++jj) {
                  if (k == polgrp::ip11[i * maxp11 + jj]) val = p5iscale;
               }
               k -= 1;
               if (k > i) {
                  insert_dpu(ik_dpu, i, k, val, 'p');
                  insert_dp(k_dpscale, k, val, 'p');
               }
            }
         }

         for (auto& it : k_dpscale) {
            exclik.push_back(i);
            exclik.push_back(it.first);
            excls.push_back(it.second.d);
            excls.push_back(it.second.p);
         }
      }
      std::vector<int> dpu_ik_vec;
      std::vector<real> dpu_sc_vec;
      for (auto& it : ik_dpu) {
         dpu_ik_vec.push_back(it.first.first);
         dpu_ik_vec.push_back(it.first.second);
         dpu_sc_vec.push_back(it.second.d);
         dpu_sc_vec.push_back(it.second.p);
         dpu_sc_vec.push_back(it.second.u);
      }
      ndpuexclude = ik_dpu.size();
      darray::allocate(ndpuexclude, &dpuexclude, &dpuexclude_scale);
      darray::copyin(g::q0, ndpuexclude, dpuexclude, dpu_ik_vec.data());
      darray::copyin(g::q0, ndpuexclude, dpuexclude_scale, dpu_sc_vec.data());
      waitFor(g::q0);

      ndpexclude = excls.size() / 2;
      darray::allocate(ndpexclude, &dpexclude, &dpexclude_scale);
      darray::copyin(g::q0, ndpexclude, dpexclude, exclik.data());
      darray::copyin(g::q0, ndpexclude, dpexclude_scale, excls.data());
      waitFor(g::q0);

      std::map<int, int> jpolarmap;
      for (int i = 0; i < n; ++i)
         jpolarmap[polar::jpolar[i]] = 1;
      njpolar = jpolarmap.size();
      darray::allocate(n, &jpolar);
      darray::allocate(njpolar * njpolar, &thlval);

      darray::allocate(n, &polarity, &thole, &pdamp, &polarity_inv);
      if (polpot::use_tholed) darray::allocate(n, &dirdamp);

      nep = nullptr;
      ep = eng_buf_elec;
      vir_ep = vir_buf_elec;
      depx = gx_elec;
      depy = gy_elec;
      depz = gz_elec;
      if (rc_a) {
         bufferAllocate(rc_flag, &nep);
         bufferAllocate(rc_flag, &ep, &vir_ep, &depx, &depy, &depz);
      }

      if (rc_flag & calc::grad) {
         darray::allocate(n, &ufld, &dufld);
      } else {
         ufld = nullptr;
         dufld = nullptr;
      }

      darray::allocate(n, &work01_, &work02_, &work03_, &work04_, &work05_);
      if (not polpot::use_tholed) // AMOEBA
         darray::allocate(n, &work06_, &work07_, &work08_, &work09_, &work10_);

      if (uprior::use_pred) {
         FstrView predstr = uprior::polpred;
         if (predstr == "ASPC") {
            polpred = UPred::ASPC;
         } else if (predstr == "GEAR") {
            polpred = UPred::GEAR;
         } else {
            polpred = UPred::LSQR;
#if TINKER_REAL_SIZE == 4
            print(stdout,
               "\n"
               " Warning -- 32-bit floating-point induced dipoles.\n"
               "            LSQR Predictor is numerically unstable.\n"
               "            Use at your own risk.\n"
               "\n");
#endif
         }
      } else {
         polpred = UPred::NONE;
      }
      maxualt = 0;
      nualt = 0;

      if (polpred == UPred::ASPC) {
         maxualt = 16;
         darray::allocate(n, &udalt_00, &udalt_01, &udalt_02, &udalt_03,
            &udalt_04, &udalt_05, &udalt_06, &udalt_07, &udalt_08, &udalt_09,
            &udalt_10, &udalt_11, &udalt_12, &udalt_13, &udalt_14, &udalt_15);
         darray::zero(g::q0, n, udalt_00, udalt_01, udalt_02, udalt_03,
            udalt_04, udalt_05, udalt_06, udalt_07, udalt_08, udalt_09,
            udalt_10, udalt_11, udalt_12, udalt_13, udalt_14, udalt_15);
         if (not polpot::use_tholed) { // AMOEBA
            darray::allocate(n, &upalt_00, &upalt_01, &upalt_02, &upalt_03,
               &upalt_04, &upalt_05, &upalt_06, &upalt_07, &upalt_08, &upalt_09,
               &upalt_10, &upalt_11, &upalt_12, &upalt_13, &upalt_14,
               &upalt_15);
            darray::zero(g::q0, n, upalt_00, upalt_01, upalt_02, upalt_03,
               upalt_04, upalt_05, upalt_06, upalt_07, upalt_08, upalt_09,
               upalt_10, upalt_11, upalt_12, upalt_13, upalt_14, upalt_15);
         }
      } else if (polpred == UPred::GEAR) {
         maxualt = 6;
         darray::allocate(n, &udalt_00, &udalt_01, &udalt_02, &udalt_03,
            &udalt_04, &udalt_05);
         darray::zero(g::q0, n, udalt_00, udalt_01, udalt_02, udalt_03,
            udalt_04, udalt_05);
         if (not polpot::use_tholed) { // AMOEBA
            darray::allocate(n, &upalt_00, &upalt_01, &upalt_02, &upalt_03,
               &upalt_04, &upalt_05);
            darray::zero(g::q0, n, upalt_00, upalt_01, upalt_02, upalt_03,
               upalt_04, upalt_05);
         }
      } else if (polpred == UPred::LSQR) {
         maxualt = 7;
         int lenb = maxualt - 1;
         int lena = lenb * lenb; // lenb*(lenb+1)/2 should be plenty.
         darray::allocate(n, &udalt_00, &udalt_01, &udalt_02, &udalt_03,
            &udalt_04, &udalt_05, &udalt_06);
         darray::allocate(lena, &udalt_lsqr_a);
         darray::allocate(lenb, &udalt_lsqr_b);
         darray::zero(g::q0, n, udalt_00, udalt_01, udalt_02, udalt_03,
            udalt_04, udalt_05, udalt_06);
         if (not polpot::use_tholed) { // AMOEBA
            darray::allocate(n, &upalt_00, &upalt_01, &upalt_02, &upalt_03,
               &upalt_04, &upalt_05, &upalt_06);
            darray::allocate(lena, &upalt_lsqr_a);
            darray::allocate(lenb, &upalt_lsqr_b);
            darray::zero(g::q0, n, upalt_00, upalt_01, upalt_02, upalt_03,
               upalt_04, upalt_05, upalt_06);
         }
      }
   }

   if (op & RcOp::INIT) {
      std::vector<int> jpolarvec(n);
      for (int i = 0; i < n; ++i)
         jpolarvec[i] = polar::jpolar[i] - 1;
      darray::copyin(g::q0, n, jpolar, jpolarvec.data());
      darray::copyin(g::q0, njpolar * njpolar, thlval, polar::thlval);

      // TODO: rename udiag to uaccel
      udiag = polpot::uaccel;

      const double polmin = 1.0e-16;
      std::vector<double> pinvbuf(n);
      for (int i = 0; i < n; ++i) {
         pinvbuf[i] = 1.0 / std::max(polar::polarity[i], polmin);
      }
      darray::copyin(g::q0, n, polarity, polar::polarity);
      darray::copyin(g::q0, n, thole, polar::thole);
      darray::copyin(g::q0, n, pdamp, polar::pdamp);
      darray::copyin(g::q0, n, polarity_inv, pinvbuf.data());
      if (polpot::use_tholed) darray::copyin(g::q0, n, dirdamp, polar::tholed);
      waitFor(g::q0);
   }

   TINKER_FCALL2(cpp0, cu1, epolarDataBinding, op);
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, epolarNonEwald, int, const real (*)[3],
   const real (*)[3]);
static void epolarNonEwald(int vers)
{
   // v0: E_dot
   // v1: EGV = E_dot + GV
   // v3: EA = E_pair + A
   // v4: EG = E_dot + G
   // v5: G
   // v6: GV
   auto edot = vers & calc::energy; // if not do_e, edot = false
   if (vers & calc::energy and vers & calc::analyz)
      edot = 0; // if do_e and do_a, edot = false
   int ver2 = vers;
   if (edot) ver2 &= ~calc::energy; // toggle off the calc::energy flag

   induce(uind, uinp);
   if (edot) epolar0DotProd(uind, udirp);
   if (vers != calc::v0)
      TINKER_FCALL2(acc1, cu1, epolarNonEwald, ver2, uind, uinp);
}

TINKER_FVOID2(acc1, cu1, epolarEwaldRecipSelf, int, const real (*)[3],
   const real (*)[3]);
void epolarEwaldRecipSelf(int vers)
{
   TINKER_FCALL2(acc1, cu1, epolarEwaldRecipSelf, vers, uind, uinp);
}

TINKER_FVOID2(acc1, cu1, epolarEwaldReal, int, const real (*)[3],
   const real (*)[3]);
static void epolarEwaldReal(int vers)
{
   TINKER_FCALL2(acc1, cu1, epolarEwaldReal, vers, uind, uinp);
}

static void epolarEwald(int vers)
{
   // v0: E_dot
   // v1: EGV = E_dot + GV
   // v3: EA = E_pair + A
   // v4: EG = E_dot + G
   // v5: G
   // v6: GV
   auto edot = vers & calc::energy; // if not do_e, edot = false
   if (vers & calc::energy and vers & calc::analyz)
      edot = 0; // if do_e and do_a, edot = false
   int ver2 = vers;
   if (edot) ver2 &= ~calc::energy; // toggle off the calc::energy flag

   induce(uind, uinp);
   if (edot) epolar0DotProd(uind, udirp);
   if (vers != calc::v0) {
      epolarEwaldReal(ver2);
      epolarEwaldRecipSelf(ver2);
   }
}

void epolar(int vers)
{
   auto rc_a = rc_flag & calc::analyz;
   auto do_a = vers & calc::analyz;
   auto do_e = vers & calc::energy;
   auto do_v = vers & calc::virial;
   auto do_g = vers & calc::grad;
   bool use_cf = use(Potent::CHGFLX);
   bool use_cfgrad = use_cf and static_cast<bool>(do_g);

   zeroOnHost(energy_ep, virial_ep);
   size_t bsize = bufferSize();
   if (rc_a) {
      if (do_a) darray::zero(g::q0, bsize, nep);
      if (do_e) darray::zero(g::q0, bsize, ep);
      if (do_v) { darray::zero(g::q0, bsize, vir_ep); }
      if (do_g) { darray::zero(g::q0, n, depx, depy, depz); }
   }

   if (use_cf) alterchg();
   mpoleInit(vers);
   if (use_cfgrad) cfluxZeroPot();

   if (useEwald()) {
      if (polpot::use_tholed)
         epolarAplusEwald(vers, use_cfgrad);
      else
         epolarEwald(vers);
   } else {
      if (polpot::use_tholed)
         epolarAplusNonEwald(vers, use_cfgrad);
      else
         epolarNonEwald(vers);
   }
   epolarPairwiseExtfield(vers, uind);
   torque(vers, depx, depy, depz);
   if (use_cfgrad) dcflux(vers, depx, depy, depz, vir_ep);
   if (do_v) {
      VirialBuffer u2 = vir_trq;
      virial_prec v2[9];
      virialReduce(v2, u2);
      for (int iv = 0; iv < 9; ++iv) {
         virial_ep[iv] += v2[iv];
         virial_elec[iv] += v2[iv];
      }
   }

   if (rc_a) {
      if (do_e) {
         EnergyBuffer u = ep;
         energy_prec e = energyReduce(u);
         energy_ep += e;
         energy_elec += e;
      }
      if (do_v) {
         VirialBuffer u1 = vir_ep;
         virial_prec v1[9];
         virialReduce(v1, u1);
         for (int iv = 0; iv < 9; ++iv) {
            virial_ep[iv] = v1[iv];
            virial_elec[iv] += v1[iv];
         }
      }
      if (do_g) sumGradient(gx_elec, gy_elec, gz_elec, depx, depy, depz);
   }
}

TINKER_FVOID2(acc1, cu1, epolar0DotProd, const real (*)[3], const real (*)[3]);
void epolar0DotProd(const real (*uind)[3], const real (*udirp)[3])
{
   TINKER_FCALL2(acc1, cu1, epolar0DotProd, uind, udirp);
}

TINKER_FVOID2(acc1, cu1, epolarPairwiseExtfield, const real (*)[3]);
void epolarPairwiseExtfield(int vers, const real (*uind)[3]) {
   if (extfld::use_exfld and (vers & calc::analyz)) {
      TINKER_FCALL2(acc1, cu1, epolarPairwiseExtfield, uind);
   }
}
}
