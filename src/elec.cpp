#include "ff/elec.h"
#include "ff/echarge.h"
#include "ff/energy.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
#include "ff/potent.h"
#include "tool/iofortstr.h"
#include <tinker/detail/atoms.hh>
#include <tinker/detail/chgpen.hh>
#include <tinker/detail/chgpot.hh>
#include <tinker/detail/couple.hh>
#include <tinker/detail/extfld.hh>
#include <tinker/detail/kchrge.hh>
#include <tinker/detail/limits.hh>
#include <tinker/detail/mplpot.hh>
#include <tinker/detail/mpole.hh>
#include <tinker/detail/mutant.hh>
#include <tinker/detail/polgrp.hh>
#include <tinker/detail/polpot.hh>

namespace tinker {
static void pchgData(RcOp op)
{
   if (not use(Potent::CHARGE))
      return;

   if (op & RcOp::DEALLOC) {
      darray::deallocate(pchg);
   }

   if (op & RcOp::ALLOC) {
      darray::allocate(n, &pchg);
   }

   if (op & RcOp::INIT) {
      std::vector<real> pchgbuf(n);
      for (int i = 0; i < n; ++i) {
         int itype = atoms::type[i] - 1;
         pchgbuf[i] = kchrge::chg[itype];
         double el;
         if (mutant::mut[i])
            el = mutant::elambda;
         else
            el = 1;
         pchgbuf[i] *= el;
      }
      darray::copyin(g::q0, n, pchg, pchgbuf.data());
      waitFor(g::q0);
   }
}

TINKER_FVOID2(cpp0, cu1, mpoleDataBinding, RcOp);
static void mpoleData(RcOp op)
{
   if (not use(Potent::MPOLE) and not use(Potent::POLAR) and not use(Potent::REPULS))
      return;

   if (op & RcOp::DEALLOC) {
      darray::deallocate(zaxis, pole, rpole, udir, udirp, uind, uinp);
      darray::deallocate(trqx, trqy, trqz, vir_trq);
   }

   if (op & RcOp::ALLOC) {
      darray::allocate(n, &zaxis, &pole, &rpole);

      if (use(Potent::POLAR)) {
         darray::allocate(n, &uind, &uinp, &udir, &udirp);
      } else {
         uind = nullptr;
         uinp = nullptr;
         udir = nullptr;
         udirp = nullptr;
      }

      if (rc_flag & calc::grad) {
         darray::allocate(n, &trqx, &trqy, &trqz);
      } else {
         trqx = nullptr;
         trqy = nullptr;
         trqz = nullptr;
      }
      if (rc_flag & calc::virial) {
         darray::allocate(bufferSize(), &vir_trq);
      } else {
         vir_trq = nullptr;
      }

      TINKER_FCALL2(cpp0, cu1, mpoleDataBinding, op);
   }

   if (op & RcOp::INIT) {
      // Regarding chkpole routine:
      // 1. The chiralities of the atoms are unlikely to change in MD; but
      // still possible in Monte Carlo;
      // 2. yaxis values are directly copied from Tinker, and are NOT
      // subtracted by 1 becasue of the checks in chkpole.
      static_assert(sizeof(LocalFrame) == 4 * sizeof(int), "");
      std::vector<LocalFrame> zaxisbuf(n);
      for (int i = 0; i < n; ++i) {
         zaxisbuf[i].zaxis = mpole::zaxis[i] - 1;
         zaxisbuf[i].xaxis = mpole::xaxis[i] - 1;
         zaxisbuf[i].yaxis = mpole::yaxis[i];
         FstrView str = mpole::polaxe[i];
         int val;
         if (str == "Z-Only")
            val = LFRM_Z_ONLY;
         else if (str == "Z-then-X")
            val = LFRM_Z_THEN_X;
         else if (str == "Bisector")
            val = LFRM_BISECTOR;
         else if (str == "Z-Bisect")
            val = LFRM_Z_BISECT;
         else if (str == "3-Fold")
            val = LFRM_3_FOLD;
         else
            val = LFRM_NONE;
         zaxisbuf[i].polaxe = val;
      }
      darray::copyin(g::q0, n, zaxis, zaxisbuf.data());
      waitFor(g::q0);

      std::vector<double> polebuf(MPL_TOTAL * n);
      for (int i = 0; i < n; ++i) {
         int b1 = MPL_TOTAL * i;
         int b2 = mpole::maxpole * i;
         // Tinker c = 0, dx = 1, dy = 2, dz = 3
         // Tinker qxx = 4, qxy = 5, qxz = 6
         //        qyx    , qyy = 8, qyz = 9
         //        qzx    , qzy    , qzz = 12
         polebuf[b1 + MPL_PME_0] = mpole::pole[b2 + 0];
         polebuf[b1 + MPL_PME_X] = mpole::pole[b2 + 1];
         polebuf[b1 + MPL_PME_Y] = mpole::pole[b2 + 2];
         polebuf[b1 + MPL_PME_Z] = mpole::pole[b2 + 3];
         polebuf[b1 + MPL_PME_XX] = mpole::pole[b2 + 4];
         polebuf[b1 + MPL_PME_XY] = mpole::pole[b2 + 5];
         polebuf[b1 + MPL_PME_XZ] = mpole::pole[b2 + 6];
         polebuf[b1 + MPL_PME_YY] = mpole::pole[b2 + 8];
         polebuf[b1 + MPL_PME_YZ] = mpole::pole[b2 + 9];
         polebuf[b1 + MPL_PME_ZZ] = mpole::pole[b2 + 12];
      }
      darray::copyin(g::q0, n, pole, polebuf.data());
      waitFor(g::q0);
   }
}

static void mdpuscaleData(RcOp op)
{
   if (not use(Potent::MPOLE) and not use(Potent::POLAR))
      return;

   if (op & RcOp::DEALLOC) {
      nmexclude = 0;
      darray::deallocate(mexclude, mexclude_scale);
      nmdpuexclude = 0;
      darray::deallocate(mdpuexclude, mdpuexclude_scale);
   }

   if (op & RcOp::ALLOC) {
      using key_t = std::pair<int, int>;
      struct m
      {
         real m;
      };
      std::map<key_t, m> ikm;
      auto insert_m = [](std::map<key_t, m>& a, int i, int k, real val, char ch) {
         key_t key;
         key.first = i;
         key.second = k;
         auto it = a.find(key);
         if (it == a.end()) {
            m x;
            x.m = 1;
            if (ch == 'm')
               x.m = val;
            a[key] = x;
         } else {
            if (ch == 'm')
               it->second.m = val;
         }
      };
      struct mdpu
      {
         real m, d, p, u;
      };
      std::map<key_t, mdpu> ik_scale;
      auto insert_mdpu = [](std::map<key_t, mdpu>& a, int i, int k, real val, char ch) {
         key_t key;
         key.first = i;
         key.second = k;
         auto it = a.find(key);
         if (it == a.end()) {
            mdpu x;
            x.m = 1;
            x.d = 1;
            x.p = 1;
            x.u = 1;
            if (ch == 'm')
               x.m = val;
            else if (ch == 'd')
               x.d = val;
            else if (ch == 'p')
               x.p = val;
            else if (ch == 'u')
               x.u = val;
            a[key] = x;
         } else {
            if (ch == 'm')
               it->second.m = val;
            else if (ch == 'd')
               it->second.d = val;
            else if (ch == 'p')
               it->second.p = val;
            else if (ch == 'u')
               it->second.u = val;
         }
      };

      // see also attach.f
      const int maxn12 = sizes::maxval;
      const int maxn13 = 3 * sizes::maxval;
      const int maxn14 = 9 * sizes::maxval;
      const int maxn15 = 27 * sizes::maxval;
      const int* couple_i12 = &couple::i12[0][0];
      const int* couple_i13 = couple::i13;
      const int* couple_i14 = couple::i14;
      const int* couple_i15 = couple::i15;
      const real m2scale = mplpot::m2scale;
      const real m3scale = mplpot::m3scale;
      const real m4scale = mplpot::m4scale;
      const real m5scale = mplpot::m5scale;

      const int maxp11 = polgrp::maxp11;
      const int maxp12 = polgrp::maxp12;
      const int maxp13 = polgrp::maxp13;
      const int maxp14 = polgrp::maxp14;
      const real d1scale = polpot::d1scale;
      const real d2scale = polpot::d2scale;
      const real d3scale = polpot::d3scale;
      const real d4scale = polpot::d4scale;
      const real p2scale = polpot::p2scale;
      const real p3scale = polpot::p3scale;
      const real p4scale = polpot::p4scale;
      const real p5scale = polpot::p5scale;
      const real p2iscale = polpot::p2iscale;
      const real p3iscale = polpot::p3iscale;
      const real p4iscale = polpot::p4iscale;
      const real p5iscale = polpot::p5iscale;
      const real u1scale = polpot::u1scale;
      const real u2scale = polpot::u2scale;
      const real u3scale = polpot::u3scale;
      const real u4scale = polpot::u4scale;

      int nn, bask;
      for (int i = 0; i < n; ++i) {
         // m
         if (m2scale != 1) {
            nn = couple::n12[i];
            bask = i * maxn12;
            for (int j = 0; j < nn; ++j) {
               int k = couple_i12[bask + j];
               k -= 1;
               if (k > i) {
                  insert_m(ikm, i, k, m2scale, 'm');
                  insert_mdpu(ik_scale, i, k, m2scale, 'm');
               }
            }
         }

         if (m3scale != 1) {
            nn = couple::n13[i];
            bask = i * maxn13;
            for (int j = 0; j < nn; ++j) {
               int k = couple_i13[bask + j];
               k -= 1;
               if (k > i) {
                  insert_m(ikm, i, k, m3scale, 'm');
                  insert_mdpu(ik_scale, i, k, m3scale, 'm');
               }
            }
         }

         if (m4scale != 1) {
            nn = couple::n14[i];
            bask = i * maxn14;
            for (int j = 0; j < nn; ++j) {
               int k = couple_i14[bask + j];
               k -= 1;
               if (k > i) {
                  insert_m(ikm, i, k, m4scale, 'm');
                  insert_mdpu(ik_scale, i, k, m4scale, 'm');
               }
            }
         }

         if (m5scale != 1) {
            nn = couple::n15[i];
            bask = i * maxn15;
            for (int j = 0; j < nn; ++j) {
               int k = couple_i15[bask + j];
               k -= 1;
               if (k > i) {
                  insert_m(ikm, i, k, m5scale, 'm');
                  insert_mdpu(ik_scale, i, k, m5scale, 'm');
               }
            }
         }
      }

      const bool usepolar = use(Potent::POLAR);
      for (int i = 0; usepolar and i < n; ++i) {
         // p
         if (p2scale != 1 || p2iscale != 1) {
            nn = couple::n12[i];
            bask = i * maxn12;
            for (int j = 0; j < nn; ++j) {
               int k = couple_i12[bask + j];
               real val = p2scale;
               for (int jj = 0; jj < polgrp::np11[i]; ++jj) {
                  if (k == polgrp::ip11[i * maxp11 + jj])
                     val = p2iscale;
               }
               k -= 1;
               if (k > i) {
                  insert_mdpu(ik_scale, i, k, val, 'p');
               }
            }
         }

         if (p3scale != 1 || p3iscale != 1) {
            nn = couple::n13[i];
            bask = i * maxn13;
            for (int j = 0; j < nn; ++j) {
               int k = couple_i13[bask + j];
               real val = p3scale;
               for (int jj = 0; jj < polgrp::np11[i]; ++jj) {
                  if (k == polgrp::ip11[i * maxp11 + jj])
                     val = p3iscale;
               }
               k -= 1;
               if (k > i) {
                  insert_mdpu(ik_scale, i, k, val, 'p');
               }
            }
         }

         if (p4scale != 1 || p4iscale != 1) {
            nn = couple::n14[i];
            bask = i * maxn14;
            for (int j = 0; j < nn; ++j) {
               int k = couple_i14[bask + j];
               real val = p4scale;
               for (int jj = 0; jj < polgrp::np11[i]; ++jj) {
                  if (k == polgrp::ip11[i * maxp11 + jj])
                     val = p4iscale;
               }
               k -= 1;
               if (k > i) {
                  insert_mdpu(ik_scale, i, k, val, 'p');
               }
            }
         }

         if (p5scale != 1 || p5iscale != 1) {
            nn = couple::n15[i];
            bask = i * maxn15;
            for (int j = 0; j < nn; ++j) {
               int k = couple_i15[bask + j];
               real val = p5scale;
               for (int jj = 0; jj < polgrp::np11[i]; ++jj) {
                  if (k == polgrp::ip11[i * maxp11 + jj])
                     val = p5iscale;
               }
               k -= 1;
               if (k > i) {
                  insert_mdpu(ik_scale, i, k, val, 'p');
               }
            }
         }

         // d
         if (d1scale != 1) {
            nn = polgrp::np11[i];
            bask = i * maxp11;
            for (int j = 0; j < nn; ++j) {
               int k = polgrp::ip11[bask + j] - 1;
               if (k > i) {
                  insert_mdpu(ik_scale, i, k, d1scale, 'd');
               }
            }
         }

         if (d2scale != 1) {
            nn = polgrp::np12[i];
            bask = i * maxp12;
            for (int j = 0; j < nn; ++j) {
               int k = polgrp::ip12[bask + j] - 1;
               if (k > i) {
                  insert_mdpu(ik_scale, i, k, d2scale, 'd');
               }
            }
         }

         if (d3scale != 1) {
            nn = polgrp::np13[i];
            bask = i * maxp13;
            for (int j = 0; j < nn; ++j) {
               int k = polgrp::ip13[bask + j] - 1;
               if (k > i) {
                  insert_mdpu(ik_scale, i, k, d3scale, 'd');
               }
            }
         }

         if (d4scale != 1) {
            nn = polgrp::np14[i];
            bask = i * maxp14;
            for (int j = 0; j < nn; ++j) {
               int k = polgrp::ip14[bask + j] - 1;
               if (k > i) {
                  insert_mdpu(ik_scale, i, k, d4scale, 'd');
               }
            }
         }

         // u
         if (u1scale != 1) {
            nn = polgrp::np11[i];
            bask = i * maxp11;
            for (int j = 0; j < nn; ++j) {
               int k = polgrp::ip11[bask + j] - 1;
               if (k > i) {
                  insert_mdpu(ik_scale, i, k, u1scale, 'u');
               }
            }
         }

         if (u2scale != 1) {
            nn = polgrp::np12[i];
            bask = i * maxp12;
            for (int j = 0; j < nn; ++j) {
               int k = polgrp::ip12[bask + j] - 1;
               if (k > i) {
                  insert_mdpu(ik_scale, i, k, u2scale, 'u');
               }
            }
         }

         if (u3scale != 1) {
            nn = polgrp::np13[i];
            bask = i * maxp13;
            for (int j = 0; j < nn; ++j) {
               int k = polgrp::ip13[bask + j] - 1;
               if (k > i) {
                  insert_mdpu(ik_scale, i, k, u3scale, 'u');
               }
            }
         }

         if (u4scale != 1) {
            nn = polgrp::np14[i];
            bask = i * maxp14;
            for (int j = 0; j < nn; ++j) {
               int k = polgrp::ip14[bask + j] - 1;
               if (k > i) {
                  insert_mdpu(ik_scale, i, k, u4scale, 'u');
               }
            }
         }
      }

      std::vector<int> exclik;
      std::vector<real> excls;
      for (auto& it : ikm) {
         exclik.push_back(it.first.first);
         exclik.push_back(it.first.second);
         excls.push_back(it.second.m);
      }
      nmexclude = excls.size();
      darray::allocate(nmexclude, &mexclude, &mexclude_scale);
      darray::copyin(g::q0, nmexclude, mexclude, exclik.data());
      darray::copyin(g::q0, nmexclude, mexclude_scale, excls.data());
      waitFor(g::q0);

      std::vector<int> ik_vec;
      std::vector<real> scal_vec;
      for (auto& it : ik_scale) {
         ik_vec.push_back(it.first.first);
         ik_vec.push_back(it.first.second);
         scal_vec.push_back(it.second.m);
         scal_vec.push_back(it.second.d);
         scal_vec.push_back(it.second.p);
         scal_vec.push_back(it.second.u);
      }
      nmdpuexclude = ik_scale.size();
      darray::allocate(nmdpuexclude, &mdpuexclude, &mdpuexclude_scale);
      darray::copyin(g::q0, nmdpuexclude, mdpuexclude, ik_vec.data());
      darray::copyin(g::q0, nmdpuexclude, mdpuexclude_scale, scal_vec.data());
      waitFor(g::q0);
   }

   if (op & RcOp::INIT) {}
}

static void chgpenData(RcOp op)
{
   if (op & RcOp::DEALLOC) {
      pentyp = Chgpen::NONE;
      nmdwexclude = 0;
      darray::deallocate(mdwexclude, mdwexclude_scale);
      nwexclude = 0;
      darray::deallocate(wexclude, wexclude_scale);
      darray::deallocate(pval0, pval, palpha, pcore);
   }

   if (op & RcOp::ALLOC) {
      // see also attach.h
      const int maxn12 = sizes::maxval;
      const int maxn13 = 3 * sizes::maxval;
      const int maxn14 = 9 * sizes::maxval;
      const int maxn15 = 27 * sizes::maxval;
      const int maxp11 = polgrp::maxp11;

      const int* couple_i12 = &couple::i12[0][0];
      const int* couple_i13 = couple::i13;
      const int* couple_i14 = couple::i14;
      const int* couple_i15 = couple::i15;

      struct mdw
      {
         real m, d, w;
      };

      // mdw excl list
      auto insert_mdw = [](std::map<std::pair<int, int>, mdw>& a, int i, int k, real val, char ch) {
         std::pair<int, int> key;
         key.first = i;
         key.second = k;
         auto it = a.find(key);
         if (it == a.end()) {
            mdw x;
            x.m = 1;
            x.d = 1;
            x.w = 1;
            if (ch == 'm')
               x.m = val;
            else if (ch == 'd')
               x.d = val;
            else if (ch == 'w')
               x.w = val;
            a[key] = x;
         } else {
            if (ch == 'm')
               it->second.m = val;
            else if (ch == 'd')
               it->second.d = val;
            else if (ch == 'w')
               it->second.w = val;
         }
      };

      std::map<std::pair<int, int>, mdw> ik_mdw;

      m2scale = mplpot::m2scale;
      m3scale = mplpot::m3scale;
      m4scale = mplpot::m4scale;
      m5scale = mplpot::m5scale;

      int nn, bask;

      const bool usempole = use(Potent::MPOLE) or use(Potent::CHGTRN);
      for (int i = 0; usempole and i < n; ++i) {
         if (m2scale != 1) {
            nn = couple::n12[i];
            for (int j = 0; j < nn; ++j) {
               int k = couple::i12[i][j] - 1;
               if (k > i)
                  insert_mdw(ik_mdw, i, k, m2scale, 'm');
            }
         }

         if (m3scale != 1) {
            nn = couple::n13[i];
            bask = i * maxn13;
            for (int j = 0; j < nn; ++j) {
               int k = couple::i13[bask + j] - 1;
               if (k > i)
                  insert_mdw(ik_mdw, i, k, m3scale, 'm');
            }
         }

         if (m4scale != 1) {
            nn = couple::n14[i];
            bask = i * maxn14;
            for (int j = 0; j < nn; ++j) {
               int k = couple::i14[bask + j] - 1;
               if (k > i)
                  insert_mdw(ik_mdw, i, k, m4scale, 'm');
            }
         }

         if (m5scale != 1) {
            nn = couple::n15[i];
            bask = i * maxn15;
            for (int j = 0; j < nn; ++j) {
               int k = couple::i15[bask + j] - 1;
               if (k > i)
                  insert_mdw(ik_mdw, i, k, m5scale, 'm');
            }
         }
      }

      const real p2scale = polpot::p2scale;
      const real p3scale = polpot::p3scale;
      const real p4scale = polpot::p4scale;
      const real p5scale = polpot::p5scale;
      const real p2iscale = polpot::p2iscale;
      const real p3iscale = polpot::p3iscale;
      const real p4iscale = polpot::p4iscale;
      const real p5iscale = polpot::p5iscale;

      // setup dscale values based on polar-scale and polar-iscale
      const bool usepolar = use(Potent::POLAR);
      for (int i = 0; usepolar and i < n; ++i) {
         if (p2scale != 1 or p2iscale != 1) {
            nn = couple::n12[i];
            bask = i * maxn12;
            for (int j = 0; j < nn; ++j) {
               int k = couple_i12[bask + j];
               real val = p2scale;
               for (int jj = 0; jj < polgrp::np11[i]; ++jj) {
                  if (k == polgrp::ip11[i * maxp11 + jj])
                     val = p2iscale;
               }
               k -= 1;
               if (k > i) {
                  insert_mdw(ik_mdw, i, k, val, 'd');
               }
            }
         }

         if (p3scale != 1 or p3iscale != 1) {
            nn = couple::n13[i];
            bask = i * maxn13;
            for (int j = 0; j < nn; ++j) {
               int k = couple_i13[bask + j];
               real val = p3scale;
               for (int jj = 0; jj < polgrp::np11[i]; ++jj) {
                  if (k == polgrp::ip11[i * maxp11 + jj])
                     val = p3iscale;
               }
               k -= 1;
               if (k > i) {
                  insert_mdw(ik_mdw, i, k, val, 'd');
               }
            }
         }

         if (p4scale != 1 or p4iscale != 1) {
            nn = couple::n14[i];
            bask = i * maxn14;
            for (int j = 0; j < nn; ++j) {
               int k = couple_i14[bask + j];
               real val = p4scale;
               for (int jj = 0; jj < polgrp::np11[i]; ++jj) {
                  if (k == polgrp::ip11[i * maxp11 + jj])
                     val = p4iscale;
               }
               k -= 1;
               if (k > i) {
                  insert_mdw(ik_mdw, i, k, val, 'd');
               }
            }
         }

         if (p5scale != 1 or p5iscale != 1) {
            nn = couple::n15[i];
            bask = i * maxn15;
            for (int j = 0; j < nn; ++j) {
               int k = couple_i15[bask + j];
               real val = p5scale;
               for (int jj = 0; jj < polgrp::np11[i]; ++jj) {
                  if (k == polgrp::ip11[i * maxp11 + jj])
                     val = p5iscale;
               }
               k -= 1;
               if (k > i) {
                  insert_mdw(ik_mdw, i, k, val, 'd');
               }
            }
         }
      }

      w2scale = polpot::w2scale;
      w3scale = polpot::w3scale;
      w4scale = polpot::w4scale;
      w5scale = polpot::w5scale;

      std::vector<int> exclik;
      std::vector<real> excls;

      for (int i = 0; usepolar and i < n; ++i) {
         if (w2scale != 1) {
            nn = couple::n12[i];
            for (int j = 0; j < nn; ++j) {
               int k = couple::i12[i][j] - 1;
               if (k > i) {
                  insert_mdw(ik_mdw, i, k, w2scale, 'w');
                  exclik.push_back(i);
                  exclik.push_back(k);
                  excls.push_back(w2scale);
               }
            }
         }

         if (w3scale != 1) {
            nn = couple::n13[i];
            bask = i * maxn13;
            for (int j = 0; j < nn; ++j) {
               int k = couple::i13[bask + j] - 1;
               if (k > i) {
                  insert_mdw(ik_mdw, i, k, w3scale, 'w');
                  exclik.push_back(i);
                  exclik.push_back(k);
                  excls.push_back(w3scale);
               }
            }
         }

         if (w4scale != 1) {
            nn = couple::n14[i];
            bask = i * maxn14;
            for (int j = 0; j < nn; ++j) {
               int k = couple::i14[bask + j] - 1;
               if (k > i) {
                  insert_mdw(ik_mdw, i, k, w4scale, 'w');
                  exclik.push_back(i);
                  exclik.push_back(k);
                  excls.push_back(w4scale);
               }
            }
         }

         if (w5scale != 1) {
            nn = couple::n15[i];
            bask = i * maxn15;
            for (int j = 0; j < nn; ++j) {
               int k = couple::i15[bask + j] - 1;
               if (k > i) {
                  insert_mdw(ik_mdw, i, k, w5scale, 'w');
                  exclik.push_back(i);
                  exclik.push_back(k);
                  excls.push_back(w5scale);
               }
            }
         }
      }

      nwexclude = excls.size();
      darray::allocate(nwexclude, &wexclude, &wexclude_scale);
      darray::copyin(g::q0, nwexclude, wexclude, exclik.data());
      darray::copyin(g::q0, nwexclude, wexclude_scale, excls.data());
      waitFor(g::q0);

      std::vector<int> ik_vec;
      std::vector<real> scal_vec;
      for (auto& it : ik_mdw) {
         ik_vec.push_back(it.first.first);
         ik_vec.push_back(it.first.second);
         scal_vec.push_back(it.second.m);
         scal_vec.push_back(it.second.d);
         scal_vec.push_back(it.second.w);
      }
      nmdwexclude = ik_mdw.size();
      darray::allocate(nmdwexclude, &mdwexclude, &mdwexclude_scale);
      darray::copyin(g::q0, nmdwexclude, mdwexclude, ik_vec.data());
      darray::copyin(g::q0, nmdwexclude, mdwexclude_scale, scal_vec.data());
      waitFor(g::q0);
      darray::allocate(n, &pcore, &pval0, &pval, &palpha);
   }

   if (op & RcOp::INIT) {
      FstrView pentstr = mplpot::pentyp;
      if (pentstr == "GORDON1")
         pentyp = Chgpen::GORDON1;
      else if (pentstr == "GORDON2")
         pentyp = Chgpen::GORDON2;
      else
         pentyp = Chgpen::NONE;
      darray::copyin(g::q0, n, pcore, chgpen::pcore);
      darray::copyin(g::q0, n, pval0, chgpen::pval0);
      darray::copyin(g::q0, n, pval, chgpen::pval);
      darray::copyin(g::q0, n, palpha, chgpen::palpha);
      waitFor(g::q0);
   }
}
}

namespace tinker {
bool useEwald()
{
   bool flag = useEnergyElec() and static_cast<bool>(limits::use_ewald);
   return flag;
}

void elecData(RcOp op)
{
   if (op & RcOp::INIT) {
      electric = chgpot::electric;
      dielec = chgpot::dielec;
      elam = mutant::elambda;
   }
   RcMan pchg42{pchgData, op};
   RcMan pole42{mpoleData, op};
   RcMan mdpuscale42{mdpuscaleData, op};
   RcMan chgpen42{chgpenData, op};
}

TINKER_FVOID2(acc1, cu1, exfieldCharge, int);
TINKER_FVOID2(acc1, cu1, exfieldDipole, int);
void exfield(int vers, int useDipole)
{
   if (not extfld::use_exfld)
      return;

   if (useDipole)
      TINKER_FCALL2(acc1, cu1, exfieldDipole, vers);
   else
      TINKER_FCALL2(acc1, cu1, exfieldCharge, vers);
}

TINKER_FVOID2(acc1, cu1, extfieldModifyDField, real (*)[3], real (*)[3]);
void extfieldModifyDField(real (*field)[3], real (*fieldp)[3])
{
   if (not extfld::use_exfld)
      return;

   TINKER_FCALL2(acc1, cu1, extfieldModifyDField, field, fieldp);
}
}
