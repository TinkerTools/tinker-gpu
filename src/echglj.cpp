#include "ff/echglj.h"
#include "ff/atom.h"
#include "ff/box.h"
#include "ff/elec.h"
#include "ff/energy.h"
#include "ff/nblist.h"
#include "ff/potent.h"
#include "math/zero.h"
#include "tool/externfunc.h"
#include "tool/iofortstr.h"
#include "tool/iotext.h"
#include <set>
#include <tinker/detail/atomid.hh>
#include <tinker/detail/atoms.hh>
#include <tinker/detail/chgpot.hh>
#include <tinker/detail/couple.hh>
#include <tinker/detail/keys.hh>
#include <tinker/detail/kvdws.hh>
#include <tinker/detail/params.hh>
#include <tinker/detail/vdwpot.hh>

namespace tinker {
TINKER_FVOID2(acc0, cu1, echgljData, RcOp);
void echgljData(RcOp op)
{
   if (not(clistVersion() & Nbl::SPATIAL))
      return;

   if (not use(Potent::CHARGE) or not use(Potent::VDW))
      return;

   if (op & RcOp::DEALLOC) {
      ncvexclude = 0;
      vdwpr_in_use = false;
      darray::deallocate(cvexclude, cvexclude_scale);
      darray::deallocate(atom_rad, atom_eps);
      darray::deallocate(mut_coalesced, chg_coalesced, radeps_coalesced);
   }

   if (op & RcOp::ALLOC) {
      // check "VDWPR" keyword
      if (not vdwpr_in_use) {
         auto parse_vdwpr = [](std::string line, int& i, int& k, double& rad, double& eps) {
            try {
               auto vs = Text::split(line);
               std::string ke = vs.at(0);
               Text::upcase(ke);
               if (ke == "VDWPR" or ke == "VDWPAIR") {
                  i = std::stoi(vs.at(1));
                  k = std::stoi(vs.at(2));
                  rad = std::stod(vs.at(3));
                  eps = std::stod(vs.at(4));
                  return true;
               }
               return false;
            } catch (...) {
               return false;
            }
         };
         const int* src = atoms::type;
         if (vdwindex == Vdw::ATOM_CLASS)
            src = atomid::class_;
         std::set<int> all_tid(src, src + n);
         auto end = all_tid.end();
         int i, k;
         double rad, eps;
         std::string record;
         // first prm
         for (int ii = 0; ii < params::nprm and !vdwpr_in_use; ++ii) {
            FstrView fsv = params::prmline[ii];
            record = fsv.trim();
            bool okay = parse_vdwpr(record, i, k, rad, eps);
            if (okay) {
               auto iit = all_tid.find(i);
               auto kit = all_tid.find(k);
               if (iit != end and kit != end) {
                  vdwpr_in_use = true;
               }
            }
         }
         // then key
         for (int ii = 0; ii < keys::nkey and not vdwpr_in_use; ++ii) {
            FstrView fsv = keys::keyline[ii];
            record = fsv.trim();
            bool okay = parse_vdwpr(record, i, k, rad, eps);
            if (okay) {
               auto iit = all_tid.find(i);
               auto kit = all_tid.find(k);
               if (iit != end and kit != end) {
                  vdwpr_in_use = true;
               }
            }
         }
      }

      struct cv
      {
         real c, v;
      };
      auto insert_cv = [](std::map<std::pair<int, int>, cv>& a, int i, int k, real val, char ch) {
         std::pair<int, int> key;
         key.first = i;
         key.second = k;
         auto it = a.find(key);
         if (it == a.end()) {
            cv x;
            x.c = 1;
            x.v = 1;
            if (ch == 'c')
               x.c = val;
            else if (ch == 'v')
               x.v = val;
            a[key] = x;
         } else {
            if (ch == 'c')
               it->second.c = val;
            else if (ch == 'v')
               it->second.v = val;
         }
      };

      // see also attach.f
      const int maxn13 = 3 * sizes::maxval;
      const int maxn14 = 9 * sizes::maxval;
      const int maxn15 = 27 * sizes::maxval;
      const real c2scale = chgpot::c2scale;
      const real c3scale = chgpot::c3scale;
      const real c4scale = chgpot::c4scale;
      const real c5scale = chgpot::c5scale;
      const real v2scale = vdwpot::v2scale;
      const real v3scale = vdwpot::v3scale;
      const real v4scale = vdwpot::v4scale;
      const real v5scale = vdwpot::v5scale;

      std::map<std::pair<int, int>, cv> ik_scale;
      for (int i = 0; i < n; ++i) {
         int nn;
         int bask;

         // c
         if (c2scale != 1) {
            nn = couple::n12[i];
            for (int j = 0; j < nn; ++j) {
               int k = couple::i12[i][j];
               k -= 1;
               if (k > i) {
                  insert_cv(ik_scale, i, k, c2scale, 'c');
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
                  insert_cv(ik_scale, i, k, c3scale, 'c');
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
                  insert_cv(ik_scale, i, k, c4scale, 'c');
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
                  insert_cv(ik_scale, i, k, c5scale, 'c');
               }
            }
         }

         // v
         if (v2scale != 1) {
            nn = couple::n12[i];
            for (int j = 0; j < nn; ++j) {
               int k = couple::i12[i][j];
               k -= 1;
               if (k > i) {
                  insert_cv(ik_scale, i, k, v2scale, 'v');
               }
            }
         }
         if (v3scale != 1) {
            nn = couple::n13[i];
            bask = i * maxn13;
            for (int j = 0; j < nn; ++j) {
               int k = couple::i13[bask + j];
               k -= 1;
               if (k > i) {
                  insert_cv(ik_scale, i, k, v3scale, 'v');
               }
            }
         }
         if (v4scale != 1) {
            nn = couple::n14[i];
            bask = i * maxn14;
            for (int j = 0; j < nn; ++j) {
               int k = couple::i14[bask + j];
               k -= 1;
               if (k > i) {
                  insert_cv(ik_scale, i, k, v4scale, 'v');
               }
            }
         }
         if (v5scale != 1) {
            nn = couple::n15[i];
            bask = i * maxn15;
            for (int j = 0; j < nn; ++j) {
               int k = couple::i15[bask + j];
               k -= 1;
               if (k > i) {
                  insert_cv(ik_scale, i, k, v5scale, 'v');
               }
            }
         }
      }
      std::vector<int> ik_vec;
      std::vector<real> scal_vec;
      for (auto& it : ik_scale) {
         ik_vec.push_back(it.first.first);
         ik_vec.push_back(it.first.second);
         scal_vec.push_back(it.second.c);
         scal_vec.push_back(it.second.v);
      }
      ncvexclude = ik_scale.size();
      darray::allocate(ncvexclude, &cvexclude, &cvexclude_scale);
      darray::copyin(g::q0, ncvexclude, cvexclude, ik_vec.data());
      darray::copyin(g::q0, ncvexclude, cvexclude_scale, scal_vec.data());
      waitFor(g::q0);
      darray::allocate(n, &atom_rad, &atom_eps);
      darray::allocate(n, &mut_coalesced);
      darray::allocate(n, &chg_coalesced);
      darray::allocate(2 * n, &radeps_coalesced);
   }

   if (op & RcOp::INIT) {
      std::vector<real> vrad(n), veps(n);
      if (vdwindex == Vdw::ATOM_TYPE) {
         for (int i = 0; i < n; ++i) {
            int jj = atoms::type[i] - 1;
            vrad[i] = kvdws::rad[jj];
            veps[i] = kvdws::eps[jj];
         }
      } else if (vdwindex == Vdw::ATOM_CLASS) {
         for (int i = 0; i < n; ++i) {
            int jj = atomid::class_[i] - 1;
            vrad[i] = kvdws::rad[jj];
            veps[i] = kvdws::eps[jj];
         }
      }
      darray::copyin(g::q0, n, atom_rad, vrad.data());
      darray::copyin(g::q0, n, atom_eps, veps.data());
      waitFor(g::q0);
   }

   TINKER_FCALL2(acc0, cu1, echgljData, op);
}

TINKER_FVOID2(acc0, cu1, echgljRadArithEpsGeomNonEwald, int);
TINKER_FVOID2(acc0, cu1, echgljRadArithEpsGeomEwaldReal, int);
void echglj(int vers)
{
   auto do_e = vers & calc::energy;
   auto do_v = vers & calc::virial;

   zeroOnHost(energy_ec, virial_ec);
   zeroOnHost(energy_ev, virial_ev);

   assert(vdwtyp == Vdw::LJ);
   assert(radrule == Vdw::ARITHMETIC);
   assert(epsrule == Vdw::GEOMETRIC);
   if (useEwald()) {
      echargeEwaldRecipSelf(vers);
      TINKER_FCALL2(acc0, cu1, echgljRadArithEpsGeomEwaldReal, vers);
   } else {
      TINKER_FCALL2(acc0, cu1, echgljRadArithEpsGeomNonEwald, vers);
   }
   exfield(vers, 0);

   if (do_e) {
      if (elrc_vol != 0) {
         energy_prec corr = elrc_vol / boxVolume();
         energy_ev += corr;
         energy_vdw += corr;
      }
   }
   if (do_v) {
      if (vlrc_vol != 0) {
         virial_prec term = vlrc_vol / boxVolume();
         virial_ev[0] += term; // xx
         virial_ev[4] += term; // yy
         virial_ev[8] += term; // zz
         virial_vdw[0] += term;
         virial_vdw[4] += term;
         virial_vdw[8] += term;
      }
   }
}
}
