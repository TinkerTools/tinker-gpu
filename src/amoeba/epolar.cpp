#include "ff/amoeba/epolar.h"
#include "ff/amoeba/empole.h"
#include "ff/amoeba/induce.h"
#include "ff/dlmda.h"
#include "ff/elec.h"
#include "ff/energy.h"
#include "ff/evdw.h"
#include "ff/hippo/cflux.h"
#include "ff/hippo/epolar.h"
#include "ff/hippo/expol.h"
#include "ff/hippo/induce.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
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
#include <vector>

namespace tinker {
static unsigned polar_active_mask;

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

      darray::deallocate(polarity, pdamp, polarity_inv, polarityorig);
      if (polpot::use_tholed)
         darray::deallocate(thdval);

      if (rc_a)
         bufferDeallocate(rc_flag, nep);
      ep_buf.manage(op, rc_flag, {}, {}, false);
      nep = nullptr;
      polar_active_mask = 0;

      darray::deallocate(ufld, dufld);
      darray::deallocate(work01_, work02_, work03_, work04_, work05_);
      if (not polpot::use_tholed) // AMOEBA
         darray::deallocate(work06_, work07_, work08_, work09_, work10_);

      induceHistoryData(RcOp::DEALLOC, not polpot::use_tholed);
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

      darray::allocate(n, &polarity, &pdamp, &polarity_inv);
      if (use_epdt || use_prst)
         darray::allocate(n, &polarityorig);
      else
         polarityorig = nullptr;
      if (polpot::use_tholed)
         darray::allocate(njpolar * njpolar, &thdval);
      else
         thdval = nullptr;

      nep = nullptr;
      ep_buf.manage(op, rc_flag, {&ep, &vir_ep, &depx, &depy, &depz},
         {eng_buf_elec, vir_buf_elec, gx_elec, gy_elec, gz_elec}, rc_a, //
         {&energy_ep, &virial_ep}, {&energy_elec, &virial_elec});
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

      polpred = polarPredictor();
#if TINKER_REAL_SIZE == 4
      if (polpred == UPred::LSQR)
         print(stdout,
            "\n"
            " Warning -- 32-bit floating-point induced dipoles.\n"
            "            LSQR Predictor is numerically unstable.\n"
            "            Use at your own risk.\n"
            "\n");
#endif
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
      induceHistoryData(RcOp::ALLOC, not polpot::use_tholed);
   }

   if (op & RcOp::INIT) {
      // The same ground mutate_check covers: the derivative differentiates the
      // stationary mutual functional with Thole or charge penetration damping,
      // so anything that is not that model, or that adds a lambda dependence
      // altpolr does not scale linearly, is out.
      if (use_epast) {
         FstrView poltypstr = polpot::poltyp;
         if (not(poltypstr == "MUTUAL"))
            TINKER_THROW("Single topology polarization dU/dLambda supports "
                         "MUTUAL polarization only.");
         if (not(polpot::use_thole or mplpot::use_chgpen))
            TINKER_THROW("Single topology polarization dU/dLambda supports "
                         "Thole or charge penetration damping only.");
         if (polpot::use_expol)
            TINKER_THROW("Single topology polarization dU/dLambda does not "
                         "support exchange polarization.");
         if (use(Potent::SOLV))
            TINKER_THROW("Single topology polarization dU/dLambda does not "
                         "support implicit solvent.");
      }

      std::vector<int> jpolarvec(n);
      for (int i = 0; i < n; ++i)
         jpolarvec[i] = polar::jpolar[i] - 1;
      darray::copyin(g::q0, n, jpolar, jpolarvec.data());
      darray::copyin(g::q0, njpolar * njpolar, thlval, polar::thlval);
      if (polpot::use_tholed)
         darray::copyin(g::q0, njpolar * njpolar, thdval, polar::thdval);

      // TODO: rename udiag to uaccel
      udiag = polpot::uaccel;

      const double polmin = 1.0e-16;
      std::vector<double> polbuf(n);
      const GrpScale psc = grpScale(plam);
      for (int i = 0; i < n; ++i) {
         if (use_prst and mutant::mutg[i] != 0)
            polbuf[i] = psc.s[mutant::mutg[i]] * dlmda::polarityorig[i];
         else
            polbuf[i] = polar::polarity[i];
      }
      std::vector<double> pinvbuf(n);
      for (int i = 0; i < n; ++i) {
         pinvbuf[i] = 1.0 / std::max(polbuf[i], polmin);
      }
      darray::copyin(g::q0, n, polarity, polbuf.data());
      darray::copyin(g::q0, n, pdamp, polar::pdamp);
      darray::copyin(g::q0, n, polarity_inv, pinvbuf.data());
      if (use_epdt || use_prst)
         darray::copyin(g::q0, n, polarityorig, dlmda::polarityorig);
      waitFor(g::q0);

      epolarSetActiveMask();
   }

   TINKER_FCALL2(cpp0, cu1, epolarDataBinding, op);
}

void epolarSetActiveMask()
{
   polar_active_mask = 0;
   if (not use_epdt)
      return;
   for (int i = 0; i < n; ++i) {
      if (dlmda::polarityorig[i] == 0)
         continue;
      auto atom_mask = RdtMask::ENV;
      if (use_rel) {
         if (mutant::mutg[i] == 1)
            atom_mask = RdtMask::LIGA;
         else if (mutant::mutg[i] == 2)
            atom_mask = RdtMask::LIGB;
      } else if (mutant::mutg[i] != 0) {
         atom_mask = RdtMask::LIGA;
      }
      polar_active_mask |= static_cast<unsigned>(atom_mask);
   }
}

TINKER_FVOID2(acc1, cu1, polarState, RdtMask, const int*, real, bool, real*, const real*);
void polarState(RdtMask mask, const int* group, real factor)
{
   // HIPPO exchange polarization scales its sizes with the polarizability
   // (mutate.f:altpolr); every caller keeps them unscaled to scale from.
   const bool expol = polpot::use_expol;
   assert(not expol or prepeporig);
   TINKER_FCALL2(acc1, cu1, polarState, mask, group, factor, expol, prepep, prepeporig);
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

TINKER_FVOID2(acc0, cu1, epolarAstDeriv, EnergyBuffer, const real (*)[3], const real (*)[3], const real (*)[3],
   const real (*)[3], const real (*)[3], const real (*)[3], const real (*)[3], const real (*)[3], bool,
   const real (*)[3][3], const real (*)[3][3]);

// The permanent field of rpole and the field of uind, for the models with one
// set of induced dipoles.
static void epolarField1(real (*field)[3])
{
   if (polpot::use_tholed)
      dfieldAplus(field);
   else
      dfieldChgpen(field);
}

static void epolarUfield1(const real (*u)[3], real (*field)[3])
{
   if (polpot::use_tholed)
      ufieldAplus(u, field);
   else
      ufieldChgpen(u, field);
}

void epolarAstDeriv()
{
   // HIPPO and AMOEBA+ solve for one set of induced dipoles in one field, which
   // stands in for both of AMOEBA's (epolar4.f).
   const bool one = mplpot::use_chgpen;
   const real(*up)[3] = one ? uind : uinp;

   // Scratch comes from the solver work arrays, idle now that induce() is done.
   real(*dfd)[3] = work01_;
   real(*dfp)[3] = one ? work01_ : work02_;
   real(*f0d)[3] = nullptr;
   real(*f0p)[3] = nullptr;
   real(*ufd)[3] = nullptr;
   real(*ufp)[3] = nullptr;

   if (plam == 0) {
      f0d = work03_, ufd = work05_;
      f0p = one ? work03_ : work04_;
      ufp = one ? work05_ : work06_;
      mpoleUsePole();
      if (one) {
         epolarField1(f0d);
         epolarUfield1(uind, ufd);
      } else {
         dfield(f0d, f0p);
         ufield(uind, uinp, ufd, ufp);
      }
   }

   mpoleUseState(chargeMask(), emGroup());
   if (one)
      epolarField1(dfd);
   else if (useEwald())
      dfieldEwald(dfd, dfp);
   else
      dfieldNonEwald(dfd, dfp);

   // Exchange polarization passes the field through polinv, and its overlap
   // with the charging ligand adds a term of its own (epolar4.f:epolar4s).
   const bool expol = polpot::use_expol;
   assert(not expol or dpscdl);
   TINKER_FCALL2(acc0, cu1, epolarAstDeriv, depdl_buf, uind, up, dfd, dfp, f0d, f0p, ufd, ufp, expol, polinv, dpscdl);
}

void epolarSingleTopology(int vers, void (*kernel)(int vers, bool use_cfgrad))
{
   auto do_v = vers & calc::virial;
   auto do_g = vers & calc::grad;
   const bool use_cf = use(Potent::CHGFLX);
   const bool use_cfgrad = use_cf and do_g;

   // Every parameter is scaled to the polarization lambda, which the multipoles
   // share only when the two lambdas agree (mutate.f:altepset).
   if (use_prst) {
      mpoleScale(plam, use_cf);
      polarState(coupledMask(), emGroup(), plam);
   }
   if (use_cf)
      alterchg();
   mpoleInit(vers, false);
   if (use_cfgrad)
      cfluxZeroPot();

   // the explicit plambda derivative of the stationary functional
   // (epolar4.f:epolar4s), from the dipoles about to be solved; with exchange
   // polarization their solve also gathers the overlap with the charging ligand
   const bool do_astdl = use_epast and (lmdaDerivVers(vers, pdlmdaActive()) & calc::energy_dlmda1);
   expol_mutpsc = do_astdl and polpot::use_expol;
   kernel(vers, use_cfgrad);
   expol_mutpsc = false;

   epolarPairwiseExtfield(vers, uind);
   torque(vers, depx, depy, depz);
   if (use_cfgrad)
      dcflux(vers, depx, depy, depz, vir_ep);
   if (polpot::use_expol and do_g)
      dexpol(vers);
   if (do_v) {
      VirialBuffer u2 = vir_trq;
      virial_prec v2[9];
      virialReduce(v2, u2);
      for (int iv = 0; iv < 9; ++iv) {
         virial_ep[iv] += v2[iv];
         virial_elec[iv] += v2[iv];
      }
   }
   if (do_astdl)
      epolarAstDeriv();
}

void epolar(int vers)
{
   assert(!use_epdt);

   epolarBegin(vers);
   epolarSingleTopology(vers, epolarKernel);
   // pole and rpole are left in the polarization state; every reader of them
   // asks for the state it needs.
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
   polarLmdaZero(vers);
}

static void epolarFinish(int vers)
{
   ep_buf.flush(vers);
   polarLmdaFlush(vers);
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

TINKER_FVOID2(acc0, cu1, epolar1NonEwaldDt, int, int, const real (*)[3], real, real);
TINKER_FVOID2(acc0, cu1, epolar1EwaldRealDt, int, int, const real (*)[3], real, real);
TINKER_FVOID2(acc0, cu1, epolarChgpenEwaldRecipSelfDt, int, int, const real (*)[3], EnergyBuffer, VirialBuffer,
   grad_prec*, grad_prec*, grad_prec*, const RecipDt&);

void epolarEwaldRecipSelfDt(int vers, EnergyBuffer out_e, VirialBuffer out_v, grad_prec* out_gx,
   grad_prec* out_gy, grad_prec* out_gz, const RecipDt& dt)
{
   TINKER_FCALL2(acc0, cu1, epolarEwaldRecipSelfDt, vers, uind, uinp, out_e, out_v, out_gx, out_gy, out_gz, dt);
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

// One pass of HIPPO or AMOEBA+, the models with one set of induced dipoles. An
// active site carries its unscaled multipoles, charge penetration charges and
// flux change, and an inactive one none, so the flux potential of the pass
// weighs each active group's atoms by wa, and by wb for the lambda derivative.
static void epolarState1(int vers, RdtMask mask, const int* group, bool edot, real wa, real wb, real wc)
{
   const int use_cfgrad = use(Potent::CHGFLX) and (vers & calc::grad);
   induce2(uind);
   if (edot)
      TINKER_FCALL2(acc0, cu1, epolar0DotProdDt, vers, uind, udir, ep, wa, wb, wc);
   if (use_cfgrad)
      cfluxZeroPot();
   if (vers != calc::v0) {
      if (useEwald()) {
         TINKER_FCALL2(acc0, cu1, epolar1EwaldRealDt, vers, use_cfgrad, uind, wa, wb);
         TINKER_FCALL2(acc0, cu1, epolarChgpenEwaldRecipSelfDt, vers, use_cfgrad, uind, ep, vir_ep, depx, depy,
            depz, dtRecipSinks(vers, wa, wb));
      } else {
         TINKER_FCALL2(acc0, cu1, epolar1NonEwaldDt, vers, use_cfgrad, uind, wa, wb);
      }
   }
   if (use_cfgrad) {
      const unsigned m = static_cast<unsigned>(mask);
      const unsigned gmask[3] = {static_cast<unsigned>(RdtMask::ENV), static_cast<unsigned>(RdtMask::LIGA),
         static_cast<unsigned>(RdtMask::LIGB)};
      GrpScale w{};
      for (int g = 0; g < 3; ++g)
         w.s[g] = (m & gmask[g]) ? wa : 0;
      dcfluxGroup(group, w, depx, depy, depz, (vers & calc::virial) ? vir_ep : nullptr);
      if (vers & calc::grad_dlmda) {
         for (int g = 0; g < 3; ++g)
            w.s[g] = (m & gmask[g]) ? wb : 0;
         dcfluxGroup(group, w, dfdlx, dfdly, dfdlz, (vers & calc::virial_dlmda) ? dvirdl_buf : nullptr);
      }
   }
   // the exchange polarization gradient of the pass, from the sizes polarState()
   // masked with the polarizability (epolar1.f:epolar1calc)
   if (polpot::use_expol and (vers & calc::grad))
      dexpol(vers, uind, wa, wb);
}

// One dual topology subsystem, weighted straight into the global accumulators.
// The pass masks rpole and polarity first, so every interaction it evaluates
// carries the same three weights and the kernels need no per-atom bookkeeping.
static void epolarState(int vers, RdtMask mask, const int* group, bool first_state, //
   real wa, real wb, real wc)
{
   // A dual topology driver accumulates torque over every subsystem and converts
   // it once at the end, so the torque buffers are cleared on the first pass only.
   if (first_state)
      mpoleBegin(vers, false);
   mpoleUseState(mask, group);
   polarState(mask, group);

   if (not epolarStateHasActiveSite(mask)) {
      darray::zero(g::q0, n, uind, uinp);
      return;
   }

   // The dot product owns the energy and both of its lambda derivatives; the
   // kernels below switch their own energy channel off for the same versions.
   const bool edot = epolarEnergyFromDotProd(vers);

   if (mplpot::use_chgpen) {
      epolarState1(vers, mask, group, edot, wa, wb, wc);
   } else {
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
   }
   if (extfld::use_exfld and (vers & calc::analyz))
      TINKER_FCALL2(acc0, cu1, epolarPairwiseExtfieldDt, uind, ep, wa);
}

// Whether one subsystem's interactions are the ones analysis reports.
// Polarization reports an endpoint that carries weight rather than always the
// coupled one (epolar3.f:2450), so there is never a count-only pass -- but only
// that endpoint may count, or the two endpoints would be summed together. When
// both carry weight, epolar3.f reports the larger count and this the lambda = 1
// one. Endpoint 1 carries the weight w of the coupled state.
static bool epolarCounts(int k, const DtPass pass[2])
{
   return k == (pass[1].wa != 0 ? 1 : 0);
}

void epolarDtPasses(int vers)
{
   const int dvers = lmdaDerivVers(vers, pdlmdaActive());
   const int* group = emGroup();
   auto do_g = vers & calc::grad;
   auto do_a = vers & calc::analyz;

   // the masked endpoints carry the flux change of these coordinates
   if (use(Potent::CHGFLX))
      alterchg();

   DtPass pass[2];
   const int npass = dtPasses(pass);

   bool first = true;
   for (int k = 0; k < npass; ++k) {
      const DtPass& p = pass[k];
      const bool counts = do_a and epolarCounts(k, pass);
      if (dtPassIsIdle(dvers, p, counts)) {
         induceForgetHistory(k);
         continue;
      }
      induceUseHistory(k, p.mask);
      epolarState(counts ? dvers : dvers & ~calc::analyz, p.mask, group, first, p.wa, p.wb, p.wc);
      first = false;
   }

   // Every pass added its torque unconverted, so one conversion covers them all.
   if (do_g) {
      torque(vers, depx, depy, depz, trqx, trqy, trqz, vir_ep);
      if (dvers & (calc::grad_dlmda | calc::virial_dlmda))
         torque(vers, dfdlx, dfdly, dfdlz, dltrqx, dltrqy, dltrqz, dvirdl_buf);
   }

   // The last pass leaves polarity masked down to a subsystem. It leaves rpole
   // masked too, but every reader of rpole asks for the state it needs.
   polarState(RdtMask::ALL, group);
}

void epolar_dt(int vers)
{
   epolarBegin(vers);
   epolarDtPasses(vers);
   epolarFinish(vers);
}

// The dipoles a reported moment is made of are the ones the polarization energy
// is made of: -dE/dF for a uniform field. Single topology solves once at the
// polarization lambda, as epolar() does. Dual topology leaves only its last
// endpoint pass in uind and udir, so it has no such dipoles to report.
void epolarPhysicalInduced()
{
   if (use_epdt)
      TINKER_THROW("Induced dipoles cannot be reported with dual topology polarization");

   if (use_prst) {
      mpoleScale(plam, use(Potent::CHGFLX));
      polarState(coupledMask(), emGroup(), plam);
      if (use(Potent::CHGFLX))
         alterchg();
      mpoleInit(calc::energy, false);
      if (mplpot::use_chgpen)
         induce2(uind);
      else
         induce(uind, uinp);
   } else {
      mpoleEnsurePhysical();
      mpoleInit(calc::energy, false);
      if (mplpot::use_chgpen)
         induce2(uind);
      else
         induce(uind, uinp);
   }
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
