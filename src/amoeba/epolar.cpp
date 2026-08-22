#include "ff/amoeba/epolar.h"
#include "ff/amoeba/empole.h"
#include "ff/amoeba/induce.h"
#include "ff/dlmda.h"
#include "ff/elec.h"
#include "ff/energy.h"
#include "ff/evdw.h"
#include "ff/hippo/cflux.h"
#include "ff/hippo/epolar.h"
#include "ff/modamoeba.h"
#include "ff/nblist.h"
#include "ff/ost.h"
#include "ff/potent.h"
#include "ff/termbuf.h"
#include "math/zero.h"
#include "tool/error.h"
#include "tool/externfunc.h"
#include "tool/iofortstr.h"
#include "tool/ioprint.h"
#include "tool/platform.h"
#include <tinker/detail/couple.hh>
#include <tinker/detail/dlmda.hh>
#include <tinker/detail/extfld.hh>
#include <tinker/detail/mplpot.hh>
#include <tinker/detail/mutant.hh>
#include <tinker/detail/polar.hh>
#include <tinker/detail/polgrp.hh>
#include <tinker/detail/polpot.hh>
#include <tinker/detail/uprior.hh>

#include <cassert>

namespace tinker {
static unsigned polar_active_mask;
static LmdaBuffer ep_dl;

TINKER_FVOID2(cpp0, cu1, epolarDataBinding, RcOp);
void epolarData(RcOp op)
{
   if (not use(Potent::POLAR))
      return;
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

      darray::deallocate(polarity, thole, pdamp, polarity_inv, polarityorig);
      if (polpot::use_tholed)
         darray::deallocate(dirdamp);

      if (rc_a)
         bufferDeallocate(rc_flag, nep);
      ep_buf.manage(op, rc_flag, {}, {}, false);
      ep_dl.manage(op, rc_flag, use_pdlmda, &depdl_buf, &d2epdl2_buf, &depdl, &d2epdl2);
      nep = nullptr;
      polar_active_mask = 0;

      darray::deallocate(ufld, dufld);
      darray::deallocate(work01_, work02_, work03_, work04_, work05_);
      if (not polpot::use_tholed) // AMOEBA
         darray::deallocate(work06_, work07_, work08_, work09_, work10_);

      if (polpred == UPred::ASPC) {
         darray::deallocate(udalt_00, udalt_01, udalt_02, udalt_03, udalt_04, udalt_05, udalt_06, udalt_07, udalt_08,
            udalt_09, udalt_10, udalt_11, udalt_12, udalt_13, udalt_14, udalt_15);
         if (not polpot::use_tholed) // AMOEBA
            darray::deallocate(upalt_00, upalt_01, upalt_02, upalt_03, upalt_04, upalt_05, upalt_06, upalt_07, upalt_08,
               upalt_09, upalt_10, upalt_11, upalt_12, upalt_13, upalt_14, upalt_15);
      } else if (polpred == UPred::GEAR) {
         darray::deallocate(udalt_00, udalt_01, udalt_02, udalt_03, udalt_04, udalt_05);
         if (not polpot::use_tholed) // AMOEBA
            darray::deallocate(upalt_00, upalt_01, upalt_02, upalt_03, upalt_04, upalt_05);
      } else if (polpred == UPred::LSQR) {
         darray::deallocate(udalt_00, udalt_01, udalt_02, udalt_03, udalt_04, udalt_05, udalt_06);
         darray::deallocate(udalt_lsqr_a, udalt_lsqr_b);
         if (not polpot::use_tholed) { // AMOEBA
            darray::deallocate(upalt_00, upalt_01, upalt_02, upalt_03, upalt_04, upalt_05, upalt_06);
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
      auto insert_dpu = [](std::map<std::pair<int, int>, dpu_scale>& m, int i, int k, real val, char ch) {
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
      auto insert_dp = [](std::map<int, dp_scale>& m, int k, real val, char dpchar) {
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
                  if (k == polgrp::ip11[i * maxp11 + jj])
                     val = p2iscale;
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
                  if (k == polgrp::ip11[i * maxp11 + jj])
                     val = p3iscale;
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
                  if (k == polgrp::ip11[i * maxp11 + jj])
                     val = p4iscale;
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
                  if (k == polgrp::ip11[i * maxp11 + jj])
                     val = p5iscale;
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
      if (use_epdt || use_plmda)
         darray::allocate(n, &polarityorig);
      else
         polarityorig = nullptr;
      if (polpot::use_tholed)
         darray::allocate(n, &dirdamp);

      nep = nullptr;
      ep_buf.manage(op, rc_flag, {&ep, &vir_ep, &depx, &depy, &depz},
         {eng_buf_elec, vir_buf_elec, gx_elec, gy_elec, gz_elec}, rc_a, //
         {&energy_ep, &virial_ep}, {&energy_elec, &virial_elec});
      ep_dl.manage(op, rc_flag, use_pdlmda, &depdl_buf, &d2epdl2_buf, &depdl, &d2epdl2);
      if (rc_a)
         bufferAllocate(rc_flag, &nep);

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
         darray::allocate(n, &udalt_00, &udalt_01, &udalt_02, &udalt_03, &udalt_04, &udalt_05, &udalt_06, &udalt_07,
            &udalt_08, &udalt_09, &udalt_10, &udalt_11, &udalt_12, &udalt_13, &udalt_14, &udalt_15);
         darray::zero(g::q0, n, udalt_00, udalt_01, udalt_02, udalt_03, udalt_04, udalt_05, udalt_06, udalt_07,
            udalt_08, udalt_09, udalt_10, udalt_11, udalt_12, udalt_13, udalt_14, udalt_15);
         if (not polpot::use_tholed) { // AMOEBA
            darray::allocate(n, &upalt_00, &upalt_01, &upalt_02, &upalt_03, &upalt_04, &upalt_05, &upalt_06, &upalt_07,
               &upalt_08, &upalt_09, &upalt_10, &upalt_11, &upalt_12, &upalt_13, &upalt_14, &upalt_15);
            darray::zero(g::q0, n, upalt_00, upalt_01, upalt_02, upalt_03, upalt_04, upalt_05, upalt_06, upalt_07,
               upalt_08, upalt_09, upalt_10, upalt_11, upalt_12, upalt_13, upalt_14, upalt_15);
         }
      } else if (polpred == UPred::GEAR) {
         maxualt = 6;
         darray::allocate(n, &udalt_00, &udalt_01, &udalt_02, &udalt_03, &udalt_04, &udalt_05);
         darray::zero(g::q0, n, udalt_00, udalt_01, udalt_02, udalt_03, udalt_04, udalt_05);
         if (not polpot::use_tholed) { // AMOEBA
            darray::allocate(n, &upalt_00, &upalt_01, &upalt_02, &upalt_03, &upalt_04, &upalt_05);
            darray::zero(g::q0, n, upalt_00, upalt_01, upalt_02, upalt_03, upalt_04, upalt_05);
         }
      } else if (polpred == UPred::LSQR) {
         maxualt = 7;
         int lenb = maxualt - 1;
         int lena = lenb * lenb; // lenb*(lenb+1)/2 should be plenty.
         darray::allocate(n, &udalt_00, &udalt_01, &udalt_02, &udalt_03, &udalt_04, &udalt_05, &udalt_06);
         darray::allocate(lena, &udalt_lsqr_a);
         darray::allocate(lenb, &udalt_lsqr_b);
         darray::zero(g::q0, n, udalt_00, udalt_01, udalt_02, udalt_03, udalt_04, udalt_05, udalt_06);
         if (not polpot::use_tholed) { // AMOEBA
            darray::allocate(n, &upalt_00, &upalt_01, &upalt_02, &upalt_03, &upalt_04, &upalt_05, &upalt_06);
            darray::allocate(lena, &upalt_lsqr_a);
            darray::allocate(lenb, &upalt_lsqr_b);
            darray::zero(g::q0, n, upalt_00, upalt_01, upalt_02, upalt_03, upalt_04, upalt_05, upalt_06);
         }
      }
   }

   if (op & RcOp::INIT) {
      if (use_epdt) {
         if (mplpot::use_chgpen)
            TINKER_THROW("Polarization dual topology does not support charge penetration.");
         if (polpot::use_tholed)
            TINKER_THROW("Polarization dual topology does not support the AMOEBA+ (tholed) model.");
         if (use(Potent::CHGFLX))
            TINKER_THROW("Polarization dual topology does not support charge flux.");
      }

      if (use_plmda and not polTracksEle()) {
         if (mplpot::use_chgpen)
            TINKER_THROW("A decoupled polarization lambda does not support charge penetration.");
         if (polpot::use_tholed)
            TINKER_THROW("A decoupled polarization lambda does not support the AMOEBA+ (tholed) model.");
         if (use(Potent::CHGFLX))
            TINKER_THROW("A decoupled polarization lambda does not support charge flux.");
      }

      std::vector<int> jpolarvec(n);
      for (int i = 0; i < n; ++i)
         jpolarvec[i] = polar::jpolar[i] - 1;
      darray::copyin(g::q0, n, jpolar, jpolarvec.data());
      darray::copyin(g::q0, njpolar * njpolar, thlval, polar::thlval);

      // TODO: rename udiag to uaccel
      udiag = polpot::uaccel;

      const double polmin = 1.0e-16;
      std::vector<double> polbuf(n);
      for (int i = 0; i < n; ++i) {
         if (use_plmda and mutant::mut[i])
            polbuf[i] = plam * dlmda::polarityorig[i];
         else
            polbuf[i] = polar::polarity[i];
      }
      std::vector<double> pinvbuf(n);
      for (int i = 0; i < n; ++i) {
         pinvbuf[i] = 1.0 / std::max(polbuf[i], polmin);
      }
      darray::copyin(g::q0, n, polarity, polbuf.data());
      darray::copyin(g::q0, n, thole, polar::thole);
      darray::copyin(g::q0, n, pdamp, polar::pdamp);
      darray::copyin(g::q0, n, polarity_inv, pinvbuf.data());
      if (use_epdt || use_plmda)
         darray::copyin(g::q0, n, polarityorig, dlmda::polarityorig);
      if (polpot::use_tholed)
         darray::copyin(g::q0, n, dirdamp, polar::tholed);
      waitFor(g::q0);

      polar_active_mask = 0;
      if (use_epdt) {
         for (int i = 0; i < n; ++i) {
            if (dlmda::polarityorig[i] == 0)
               continue;
            auto atom_mask = RdtMask::ENV;
            if (use_rel) {
               if (mutant::mutg[i] == 1)
                  atom_mask = RdtMask::LIGA;
               else if (mutant::mutg[i] == 2)
                  atom_mask = RdtMask::LIGB;
            } else if (mutant::mut[i]) {
               atom_mask = RdtMask::LIGA;
            }
            polar_active_mask |= static_cast<unsigned>(atom_mask);
         }
      }
   }

   TINKER_FCALL2(cpp0, cu1, epolarDataBinding, op);
}

TINKER_FVOID2(acc0, cu1, polarState, RdtMask, const int*);
void polarState(RdtMask mask, const int* group)
{
   TINKER_FCALL2(acc0, cu1, polarState, mask, group);
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, epolarNonEwald, int, const real (*)[3], const real (*)[3]);
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
   if (edot)
      ver2 &= ~calc::energy; // toggle off the calc::energy flag

   induce(uind, uinp);
   if (edot)
      epolar0DotProd(uind, udirp, ep);
   if (vers != calc::v0)
      TINKER_FCALL2(acc1, cu1, epolarNonEwald, ver2, uind, uinp);
}

TINKER_FVOID2(acc1, cu1, epolarEwaldRecipSelf, int, const real (*)[3], const real (*)[3],
   EnergyBuffer, VirialBuffer, grad_prec*, grad_prec*, grad_prec*);
void epolarEwaldRecipSelf(int vers, EnergyBuffer out_e, VirialBuffer out_v,
   grad_prec* out_gx, grad_prec* out_gy, grad_prec* out_gz)
{
   TINKER_FCALL2(acc1, cu1, epolarEwaldRecipSelf, vers, uind, uinp,
      out_e, out_v, out_gx, out_gy, out_gz);
}

TINKER_FVOID2(acc1, cu1, epolarEwaldReal, int, const real (*)[3], const real (*)[3]);
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
   if (edot)
      ver2 &= ~calc::energy; // toggle off the calc::energy flag

   induce(uind, uinp);
   if (edot)
      epolar0DotProd(uind, udirp, ep);
   if (vers != calc::v0) {
      epolarEwaldReal(ver2);
      AccumRef out = ep_buf.ref();
      epolarEwaldRecipSelf(ver2, out.e, out.v, out.gx, out.gy, out.gz);
   }
}

static void epolarBegin(int vers);
static void epolarKernel(int vers, bool use_cfgrad);
static void epolarFinish(int vers);

void epolar(int vers)
{
   assert(!use_epdt);

   auto do_v = vers & calc::virial;
   auto do_g = vers & calc::grad;
   bool use_cf = use(Potent::CHGFLX);
   bool use_cfgrad = use_cf and static_cast<bool>(do_g);

   epolarBegin(vers);

   if (use_plmda)
      mpoleScale(plam);

   if (use_cf)
      alterchg();
   mpoleInit(vers, use_emast and not use_plmda);
   if (use_cfgrad)
      cfluxZeroPot();

   epolarKernel(vers, use_cfgrad);
   epolarPairwiseExtfield(vers, uind);
   torque(vers, depx, depy, depz);
   if (use_cfgrad)
      dcflux(vers, depx, depy, depz, vir_ep);
   if (do_v) {
      VirialBuffer u2 = vir_trq;
      virial_prec v2[9];
      virialReduce(v2, u2);
      for (int iv = 0; iv < 9; ++iv) {
         virial_ep[iv] += v2[iv];
         virial_elec[iv] += v2[iv];
      }
   }

   if (use_plmda)
      mpoleScale(elam);

   epolarFinish(vers);
}

static void epolarZeroWork(int vers)
{
   auto rc_a = rc_flag & calc::analyz;
   auto do_a = vers & calc::analyz;
   if (rc_a and do_a)
      darray::zero(g::q0, bufferSize(), nep);
   ep_buf.zero(vers);
}

static void epolarBegin(int vers)
{
   zeroOnHost(energy_ep, virial_ep);
   epolarZeroWork(vers);
   ep_dl.zero(vers);
}

static void epolarFinish(int vers)
{
   ep_buf.flush(vers);
   ep_dl.flush(vers);
}

static bool epolarStateHasActiveSite(RdtMask mask)
{
   return static_cast<unsigned>(mask) & polar_active_mask;
}

static void epolarKernel(int vers, bool use_cfgrad)
{
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
}

TINKER_FVOID2(acc0, cu1, epolarNonEwaldDt, int, const real (*)[3], const real (*)[3], real, real);
TINKER_FVOID2(acc0, cu1, epolarEwaldRealDt, int, const real (*)[3], const real (*)[3], real, real);
TINKER_FVOID2(acc0, cu1, epolar0DotProdDt, int, const real (*)[3], const real (*)[3], EnergyBuffer, real, real, real);
TINKER_FVOID2(acc0, cu1, epolarPairwiseExtfieldDt, const real (*)[3], EnergyBuffer, real);
TINKER_FVOID2(acc0, cu1, epolarEwaldRecipSelfDt, int, const real (*)[3], const real (*)[3], EnergyBuffer,
   VirialBuffer, grad_prec*, grad_prec*, grad_prec*, const RecipDt&);

void epolarEwaldRecipSelfDt(int vers, EnergyBuffer out_e, VirialBuffer out_v, grad_prec* out_gx,
   grad_prec* out_gy, grad_prec* out_gz, const RecipDt& dt)
{
   TINKER_FCALL2(acc0, cu1, epolarEwaldRecipSelfDt, vers, uind, uinp, out_e, out_v, out_gx, out_gy, out_gz, dt);
}

void dtRestoreFullState(const int* group)
{
   mpoleRestoreFullState(group);
   polarState(RdtMask::ALL, group);
}

RecipDt dtRecipSinks(int vers, real wa, real wb)
{
   RecipDt dt;
   dt.wa = wa;
   dt.wb = wb;
   if (vers & calc::virial_dlmda)
      dt.vdl = dvirdl_buf;
   if (vers & calc::grad_dlmda) {
      dt.dgx = dfdlx;
      dt.dgy = dfdly;
      dt.dgz = dfdlz;
   }
   // The lambda torque feeds both the lambda gradient and the torque part of
   // the lambda virial, so either one is reason enough to accumulate it.
   if (vers & (calc::grad_dlmda | calc::virial_dlmda)) {
      dt.dltrqx = dltrqx;
      dt.dltrqy = dltrqy;
      dt.dltrqz = dltrqz;
   }
   return dt;
}

// One dual topology subsystem, weighted straight into the global accumulators.
// The pass masks rpole and polarity first, so every interaction it evaluates
// carries the same three weights and the kernels need no per-atom bookkeeping.
static void epolarState(int vers, RdtMask mask, const int* group, bool first_state, //
   real wa, real wb, real wc)
{
   mpoleInitStateDt(vers, mask, group, first_state);
   polarState(mask, group);

   if (not epolarStateHasActiveSite(mask)) {
      darray::zero(g::q0, n, uind, uinp);
      return;
   }

   // The dot product owns the energy and both of its lambda derivatives; the
   // kernels below switch their own energy channel off for the same versions.
   const bool edot = epolarEnergyFromDotProd(vers);

   induce(uind, uinp);
   if (edot)
      TINKER_FCALL2(acc0, cu1, epolar0DotProdDt, vers, uind, udirp, ep, wa, wb, wc);
   if (vers != calc::v0) {
      if (useEwald()) {
         TINKER_FCALL2(acc0, cu1, epolarEwaldRealDt, vers, uind, uinp, wa, wb);
         epolarEwaldRecipSelfDt(vers, ep, vir_ep, depx, depy, depz, dtRecipSinks(vers, wa, wb));
      } else {
         TINKER_FCALL2(acc0, cu1, epolarNonEwaldDt, vers, uind, uinp, wa, wb);
      }
   }
   if (extfld::use_exfld and (vers & calc::analyz))
      TINKER_FCALL2(acc0, cu1, epolarPairwiseExtfieldDt, uind, ep, wa);
}

// Whether one subsystem's interactions are the ones analysis reports.
// Polarization reports whichever endpoint ran rather than always the coupled
// one (epolar3.f:2559), so unlike the multipole and van der Waals terms there is
// never a count-only pass -- but only that endpoint's subsystems may count, or
// the two endpoints would be summed together.
static bool epolarCounts(const DtPass& p, bool relative, bool coupled, bool need1)
{
   if (not(need1 ? p.in1 : p.in0))
      return false;
   if (not relative)
      return true;
   // Within the reported endpoint, one ligand-plus-environment subsystem carries
   // the whole count if there is one; a decoupled endpoint has none, so its
   // subsystems sum instead.
   return not coupled or relSlotIsCoupled(p.slot);
}

void epolar_dt(int vers)
{
   const int dvers = lmdaDerivVers(vers, use_pdlmda);
   // use_eprdt is use_epdt and use_rel, and this is only reached under use_epdt.
   const bool relative = use_eprdt;
   // The relative schedule labels atoms by ligand; the absolute one only knows
   // mutated from not.
   const int* group = relative ? rdt_group : mut;
   auto do_g = vers & calc::grad;
   auto do_a = vers & calc::analyz;

   double w, dw, d2w;
   bool need0, need1;
   dtWeightNeed(plam, epdtexp, dpldlmda, d2pldlmda2, w, dw, d2w, need0, need1);

   epolarBegin(vers);

   DtCoef c;
   dtWeightsToCoef(c, w, dw, d2w, dpldlmda, d2pldlmda2, use_pdlmda);

   DtPass pass[nRelSlot];
   const int npass = dtPassList(relative, eprelst0, eprelst1, pass);

   // Whether the reported endpoint has a ligand-plus-environment subsystem to
   // carry its whole interaction count. Only the relative schedule builds an
   // endpoint out of several subsystems.
   bool coupled = false;
   if (relative) {
      const RelState reported = need1 ? eprelst1 : eprelst0;
      for (int k = 0; k < nRelSlot and not coupled; ++k) {
         RdtMask mask;
         bool in0, in1;
         relSlot(k, reported, reported, mask, in0, in1);
         coupled = in0 and relSlotIsCoupled(k);
      }
   }

   bool first = true;
   for (int k = 0; k < npass; ++k) {
      real wa, wb, wc;
      dtPassWeights(c, pass[k], wa, wb, wc);
      const bool counts = do_a and epolarCounts(pass[k], relative, coupled, need1);
      if (dtPassIsIdle(dvers, wa, wb, wc, counts))
         continue;
      epolarState(counts ? dvers : dvers & ~calc::analyz, pass[k].mask, group, first, wa, wb, wc);
      first = false;
   }

   // Every pass added its torque unconverted, so one conversion covers them all.
   if (do_g) {
      torque(vers, depx, depy, depz, trqx, trqy, trqz, vir_ep);
      if (dvers & (calc::grad_dlmda | calc::virial_dlmda))
         torque(vers, dfdlx, dfdly, dfdlz, dltrqx, dltrqy, dltrqz, dvirdl_buf);
   }

   epolarFinish(vers);

   // The last pass leaves rpole and polarity masked down to a subsystem.
   dtRestoreFullState(group);
}

TINKER_FVOID2(acc1, cu1, epolar0DotProd, const real (*)[3], const real (*)[3], EnergyBuffer);
void epolar0DotProd(const real (*uind)[3], const real (*udirp)[3], EnergyBuffer eout)
{
   TINKER_FCALL2(acc1, cu1, epolar0DotProd, uind, udirp, eout);
}

TINKER_FVOID2(acc1, cu1, epolarPairwiseExtfield, const real (*)[3]);
void epolarPairwiseExtfield(int vers, const real (*uind)[3])
{
   if (extfld::use_exfld and (vers & calc::analyz)) {
      TINKER_FCALL2(acc1, cu1, epolarPairwiseExtfield, uind);
   }
}
}
