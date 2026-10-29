#include "ff/evdw.h"
#include "ff/dlmda.h"
#include "ff/energy.h"
#include "ff/nblist.h"
#include "ff/ost.h"
#include "ff/potent.h"
#include "ff/termbuf.h"
#include "math/zero.h"
#include "tool/error.h"
#include "tool/externfunc.h"
#include "tool/iofortstr.h"
#include "tool/platform.h"
#include "tool/iotext.h"
#include <tinker/detail/atomid.hh>
#include <tinker/detail/atoms.hh>
#include <tinker/detail/bound.hh>
#include <tinker/detail/couple.hh>
#include <tinker/detail/kdsp.hh>
#include <tinker/detail/keys.hh>
#include <tinker/detail/mutant.hh>
#include <tinker/detail/params.hh>
#include <tinker/detail/shunt.hh>
#include <tinker/detail/sizes.hh>
#include <tinker/detail/vdw.hh>
#include <tinker/detail/vdwpot.hh>
#include <tinker/routines.h>

#include <algorithm>
#include <cassert>
#include <cmath>

namespace tinker {
inline namespace v1 {
using new_type = int; // new vdw class/type
using old_type = int; // old vdw class/type
}
static std::map<old_type, new_type> jmap;
static std::vector<new_type> jvec;
static std::vector<new_type> jvdwbuf;
static int jcount;

// The class sums of the long-range correction.
static LrcSums vdw_lrc;

void vdwSoftcoreData(RcOp op)
{
   // mpoleScale() reads the ligand groups whenever poleorig exists, and the
   // multipole and polarization pair kernels read them to skip dead groups.
   if ((not use(Potent::VDW)) and (not use(Potent::REPULS)) and (not use(Potent::DISP)) and (not use(Potent::CHGTRN))
      and (not use(Potent::MPOLE)) and (not use(Potent::POLAR)) and (not usePoleorig()))
      return;

   if (op & RcOp::DEALLOC)
      darray::deallocate(mut);

   if (op & RcOp::ALLOC)
      darray::allocate(n, &mut);

   if (op & RcOp::INIT) {
      vlam = mutant::vlambda;
      if (static_cast<int>(Vdw::DECOUPLE) == mutant::vcouple)
         vcouple = Vdw::DECOUPLE;
      else if (static_cast<int>(Vdw::ANNIHILATE) == mutant::vcouple)
         vcouple = Vdw::ANNIHILATE;
      std::vector<int> mutvec(n);
      for (int i = 0; i < n; ++i) {
         if (mutant::mutg[i] != 0) {
            mutvec[i] = 1;
         } else {
            mutvec[i] = 0;
         }
      }
      darray::copyin(g::q0, n, mut, mutvec.data());
      waitFor(g::q0);
   }
}

void evdwData(RcOp op)
{
   if (not use(Potent::VDW))
      return;

   auto rc_a = rc_flag & calc::analyz;

   if (op & RcOp::DEALLOC) {
      // local static variables
      jmap.clear();
      jvec.clear();
      jvdwbuf.clear();
      jcount = 0;

      if (vdwtyp == Vdw::HAL) {
         darray::deallocate(ired, kred, xred, yred, zred, gxred, gyred, gzred);
         darray::deallocate(gxred_dlmda, gyred_dlmda, gzred_dlmda);
      }

      darray::deallocate(jvdw, radmin, epsilon);

      nvexclude = 0;
      nvexclude_scaled = 0;
      darray::deallocate(vexclude, vexclude_scale, vexclude14);
      vexclude14 = nullptr;

      if (nvdw14 > 0) {
         darray::deallocate(radmin4, epsilon4, vdw14ik);
         nvdw14 = 0;
      }

      if (rc_a)
         bufferDeallocate(rc_flag, nev);
      ev_buf.manage(op, rc_flag, {}, {}, false);
      nev = nullptr;
      gxred_dlmda = nullptr;
      gyred_dlmda = nullptr;
      gzred_dlmda = nullptr;

      vdw_lrc = {};
   }

   if (op & RcOp::ALLOC) {
      FstrView str = vdwpot::vdwtyp;
      if (str == "LENNARD-JONES")
         vdwtyp = Vdw::LJ;
      else if (str == "BUCKINGHAM")
         vdwtyp = Vdw::BUCK;
      else if (str == "MM3-HBOND")
         vdwtyp = Vdw::MM3HB;
      else if (str == "BUFFERED-14-7")
         vdwtyp = Vdw::HAL;
      else if (str == "GAUSSIAN")
         vdwtyp = Vdw::GAUSS;
      else
         assert(false);


      FstrView str1 = vdwpot::vdwindex;
      if (str1 == "CLASS")
         vdwindex = Vdw::ATOM_CLASS;
      else if (str1 == "TYPE")
         vdwindex = Vdw::ATOM_TYPE;
      else
         assert(false);

      FstrView str2 = vdwpot::radrule;
      if (str2 == "ARITHMETIC")
         radrule = Vdw::ARITHMETIC;
      else if (str2 == "GEOMETRIC")
         radrule = Vdw::GEOMETRIC;
      else if (str2 == "CUBIC-MEAN")
         radrule = Vdw::CUBIC_MEAN;
      else
         assert(false);

      FstrView str3 = vdwpot::epsrule;
      if (str3 == "ARITHMETIC")
         epsrule = Vdw::ARITHMETIC;
      else if (str3 == "GEOMETRIC")
         epsrule = Vdw::GEOMETRIC;
      else if (str3 == "CUBIC-MEAN")
         epsrule = Vdw::CUBIC_MEAN;
      else if (str3 == "HHG")
         epsrule = Vdw::HHG;
      else if (str3 == "W-H")
         epsrule = Vdw::W_H;
      else
         assert(false);

      const int dlmask = lmdaDerivMask(rc_flag, use_vdlmda);
      if (vdwtyp == Vdw::HAL) {
         darray::allocate(n, &ired, &kred, &xred, &yred, &zred);
         if (rc_flag & calc::grad) {
            darray::allocate(n, &gxred, &gyred, &gzred);
            if (dlmask & calc::grad_dlmda)
               darray::allocate(n, &gxred_dlmda, &gyred_dlmda, &gzred_dlmda);
         } else {
            gxred = nullptr;
            gyred = nullptr;
            gzred = nullptr;
            gxred_dlmda = nullptr;
            gyred_dlmda = nullptr;
            gzred_dlmda = nullptr;
         }
      }

      darray::allocate(n, &jvdw);

      jvdwbuf.resize(n);
      assert(jmap.size() == 0);
      assert(jvec.size() == 0);
      jcount = 0;
#if 0
      for (int i = 0; i < n; ++i) {
         int jt = vdw::jvdw[i] - 1;
         auto iter = jmap.find(jt);
         if (iter == jmap.end()) {
            jvdwbuf[i] = jcount;
            jvec.push_back(jt);
            jmap[jt] = jcount;
            ++jcount;
         } else {
            jvdwbuf[i] = iter->second;
         }
      }
#else
      // vdw::jvdw now stores the shortened class/index values
      for (int i = 0; i < n; ++i) {
         jvdwbuf[i] = vdw::jvdw[i] - 1;
         int jt = (vdwindex == Vdw::ATOM_CLASS) ? atomid::class_[i] - 1 : atoms::type[i] - 1;
         auto iter = jmap.find(jt);
         if (iter == jmap.end()) {
            jvec.push_back(jt);
            jmap[jt] = jvdwbuf[i];
            ++jcount;
         }
      }
#endif

      darray::allocate(jcount * jcount, &radmin, &epsilon);

      v2scale = vdwpot::v2scale;
      v3scale = vdwpot::v3scale;
      v4scale = vdwpot::v4scale;
      v5scale = vdwpot::v5scale;

      std::vector<int> exclik;
      std::vector<real> excls;
      // see also attach.f
      const int maxn13 = 3 * sizes::maxval;
      const int maxn14 = 9 * sizes::maxval;
      const int maxn15 = 27 * sizes::maxval;
      for (int i = 0; i < n; ++i) {
         int nn;
         int bask;

         if (v2scale != 1) {
            nn = couple::n12[i];
            for (int j = 0; j < nn; ++j) {
               int k = couple::i12[i][j];
               k -= 1;
               if (k > i) {
                  exclik.push_back(i);
                  exclik.push_back(k);
                  excls.push_back(v2scale);
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
                  exclik.push_back(i);
                  exclik.push_back(k);
                  excls.push_back(v3scale);
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
                  exclik.push_back(i);
                  exclik.push_back(k);
                  excls.push_back(v4scale);
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
                  exclik.push_back(i);
                  exclik.push_back(k);
                  excls.push_back(v5scale);
               }
            }
         }
      }
      // check VDW14 interations
      std::vector<int> excl14;
      nvdw14 = 0;
      if (v4scale != 0) {
         // otherwise, there is no reason to worry about vdw14 energies

         // rad4 and eps4 (of module kvdws) have been overwritten by kvdw
         // must parse the parameter file and key file again for VDW14 keyword
         // vdw14         8               1.9000    -0.1000

         auto parse_v14 = [](std::string line, int& j, double& r4, double& e4) -> bool {
            try {
               auto vs = Text::split(line);
               std::string k = vs.at(0);
               Text::upcase(k);
               if (k == "VDW14") {
                  j = std::stoi(vs.at(1));
                  r4 = std::stod(vs.at(2));
                  e4 = std::stod(vs.at(3));
                  return true;
               }
               return false;
            } catch (...) {
               return false;
            }
         };
         std::map<int, double> kvdws__rad4, kvdws__eps4;
         // first prm
         for (int i = 0; i < params::nprm; ++i) {
            FstrView fsv = params::prmline[i];
            std::string record = fsv.trim();
            int j;
            double r4, e4;
            bool okay = parse_v14(record, j, r4, e4);
            if (okay) {
#if 0
               kvdws__rad4[j - 1] = r4;
               kvdws__eps4[j - 1] = e4;
#else
               auto iter = jmap.find(j - 1);
               if (iter != jmap.end()) {
                  int jt = iter->second;
                  kvdws__rad4[jt] = r4;
                  kvdws__eps4[jt] = e4;
               }
#endif
            }
         }
         // then key
         for (int i = 0; i < keys::nkey; ++i) {
            FstrView fsv = keys::keyline[i];
            std::string record = fsv.trim();
            int j;
            double r4, e4;
            bool okay = parse_v14(record, j, r4, e4);
            if (okay) {
#if 0
               kvdws__rad4[j - 1] = r4;
               kvdws__eps4[j - 1] = e4;
#else
               auto iter = jmap.find(j - 1);
               if (iter != jmap.end()) {
                  int jt = iter->second;
                  kvdws__rad4[jt] = r4;
                  kvdws__eps4[jt] = e4;
               }
#endif
            }
         }

         std::vector<int> v14ikbuf;
         for (int i = 0; i < n; ++i) {
            int nn = couple::n14[i];
            int bask = i * maxn14;
            int i_vclass = vdw::jvdw[i] - 1;
            bool i_has_v14prm = (kvdws__rad4.count(i_vclass) > 0) || (kvdws__eps4.count(i_vclass) > 0);
            for (int j = 0; j < nn; ++j) {
               int k = couple::i14[bask + j];
               k -= 1;
               int k_vclass = vdw::jvdw[k] - 1;
               bool k_has_v14prm = (kvdws__rad4.count(k_vclass) > 0) || (kvdws__eps4.count(k_vclass) > 0);
               if (k > i && (i_has_v14prm || k_has_v14prm)) {
                  v14ikbuf.push_back(i);
                  v14ikbuf.push_back(k);
                  ++nvdw14;
               }
            }
         }

         if (nvdw14 > 0) {
            // radmin4 and epsilon4 are similar to radmin and epsilon
            darray::allocate(jcount * jcount, &radmin4, &epsilon4);
            darray::allocate(nvdw14, &vdw14ik);
            darray::copyin(g::q0, nvdw14, vdw14ik, v14ikbuf.data());
            waitFor(g::q0);
         }

         // The buffered 14-7 kernels take the vdw14 values inside their
         // exclusion loop (ehal1.f), so every 1-4 pair that has them joins the
         // exclusion list, at a 1-4 scale of one too, flagged to use them.
         if (vdwtyp == Vdw::HAL and nvdw14 > 0) {
            std::map<std::pair<int, int>, size_t> where;
            for (size_t ie = 0; ie < excls.size(); ++ie)
               where[{exclik[2 * ie], exclik[2 * ie + 1]}] = ie;
            excl14.assign(excls.size(), 0);
            for (int ip = 0; ip < nvdw14; ++ip) {
               int i = v14ikbuf[2 * ip];
               int k = v14ikbuf[2 * ip + 1];
               auto iter = where.find({i, k});
               if (iter != where.end()) {
                  excl14[iter->second] = 1;
               } else {
                  exclik.push_back(i);
                  exclik.push_back(k);
                  excls.push_back(v4scale);
                  excl14.push_back(1);
               }
            }
         }
      }

      // Put the pairs that still interact first. The scale-0 pairs after them
      // only mask the neighbor list, so the CUDA HAL kernel skips them.
      {
         std::vector<int> ik2;
         std::vector<real> s2;
         std::vector<int> v142;
         for (int pass = 0; pass < 2; ++pass) {
            for (size_t ie = 0; ie < excls.size(); ++ie) {
               bool scaled = excls[ie] != 0 or (excl14.size() and excl14[ie]);
               if (scaled == (pass == 0)) {
                  ik2.push_back(exclik[2 * ie]);
                  ik2.push_back(exclik[2 * ie + 1]);
                  s2.push_back(excls[ie]);
                  if (excl14.size())
                     v142.push_back(excl14[ie]);
               }
            }
            if (pass == 0)
               nvexclude_scaled = s2.size();
         }
         exclik.swap(ik2);
         excls.swap(s2);
         excl14.swap(v142);
      }

      nvexclude = excls.size();
      darray::allocate(nvexclude, &vexclude, &vexclude_scale);
      darray::copyin(g::q0, nvexclude, vexclude, exclik.data());
      darray::copyin(g::q0, nvexclude, vexclude_scale, excls.data());
      if (excl14.size()) {
         darray::allocate(nvexclude, &vexclude14);
         darray::copyin(g::q0, nvexclude, vexclude14, excl14.data());
      } else {
         vexclude14 = nullptr;
      }
      waitFor(g::q0);

      nev = nullptr;
      ev_buf.manage(op, rc_flag, {&ev, &vir_ev, &devx, &devy, &devz},
         {eng_buf_vdw, vir_buf_vdw, gx_vdw, gy_vdw, gz_vdw}, rc_a, //
         {&energy_ev, &virial_ev}, {&energy_vdw, &virial_vdw});
      if (rc_a)
         bufferAllocate(rc_flag, &nev);
   }

   if (op & RcOp::INIT) {
      // Halgren
      if (vdwtyp == Vdw::HAL) {
         ghal = vdwpot::ghal;
         dhal = vdwpot::dhal;
         scexp = mutant::scexp;
         scalphav = mutant::scalphav;
         scexpm = mutant::scexpm;
         scexpp = mutant::scexpp;

         std::vector<int> iredbuf(n);
         std::vector<double> kredbuf(n);
         for (int i = 0; i < n; ++i) {
            int jt = vdw::ired[i] - 1;
            iredbuf[i] = jt;
            kredbuf[i] = vdw::kred[i];
         }
         darray::copyin(g::q0, n, ired, iredbuf.data());
         darray::copyin(g::q0, n, kred, kredbuf.data());
         waitFor(g::q0);
      }

      darray::copyin(g::q0, n, jvdw, jvdwbuf.data());
      waitFor(g::q0);
      njvdw = jcount;

      // see also kvdw.f
      std::vector<double> radvec, epsvec;
      for (int it_new = 0; it_new < jcount; ++it_new) {
         // int it_old = jvec[it_new];
         // int base = it_old * sizes::maxclass;
         int base = it_new * njvdw;
         for (int jt_new = 0; jt_new < jcount; ++jt_new) {
            // int jt_old = jvec[jt_new];
            // int offset = base + jt_old;
            int offset = base + jt_new;
            radvec.push_back(vdw::radmin[offset]);
            epsvec.push_back(vdw::epsilon[offset]);
         }
      }
      darray::copyin(g::q0, jcount * jcount, radmin, radvec.data());
      darray::copyin(g::q0, jcount * jcount, epsilon, epsvec.data());
      waitFor(g::q0);

      if (nvdw14) {
         std::vector<double> rad4buf, eps4buf;
         for (int it_new = 0; it_new < jcount; ++it_new) {
            // int it_old = jvec[it_new];
            // int base = it_old * sizes::maxclass;
            int base = it_new * njvdw;
            for (int jt_new = 0; jt_new < jcount; ++jt_new) {
               // int jt_old = jvec[jt_new];
               // int offset = base + jt_old;
               int offset = base + jt_new;
               rad4buf.push_back(vdw::radmin4[offset]);
               eps4buf.push_back(vdw::epsilon4[offset]);
            }
         }
         darray::copyin(g::q0, jcount * jcount, radmin4, rad4buf.data());
         darray::copyin(g::q0, jcount * jcount, epsilon4, eps4buf.data());
         waitFor(g::q0);
      }

      vdw_lrc = {};
      if (vdwpot::use_vcorr)
         lrcSetup(vdw_lrc, false);
   }
}

void lrcSetup(LrcSums& ls, bool dsp)
{
   // dispersion has no relative free energy treatment (mutate.f), so it never
   // has a ligand 2
   assert(not(dsp and use_rel));
   ls = {0, 0, 0, 1, 1};
   if (not bound::use_bounds)
      return;

   // The soft core buffered 14-7 potential scales a mutated pair by the soft
   // core power of its group lambda, its softening being negligible against the
   // buffered distance beyond the cutoff; beyond its switch radius the soft core
   // of the damped dispersion leaves only its lambda power on a mutated pair.
   if (dsp) {
      ls.nw = mutant::scexpd;
   } else if (vdwtyp == Vdw::HAL) {
      ls.nw = mutant::scexp;
      ls.nb = mutant::scexp;
   }

   // The coefficients of the switching function.
   if (dsp)
      tinker_f_switch({const_cast<char*>("DISP  "), 6});
   else
      tinker_f_switch({const_cast<char*>("VDW   "), 6});
   const double cut = shunt::cut, off = shunt::off;
   const double c0 = shunt::c0, c1 = shunt::c1, c2 = shunt::c2;
   const double c3 = shunt::c3, c4 = shunt::c4, c5 = shunt::c5;

   // The number of steps and range of the numerical integration.
   const int nstep = 2;
   const double range = 100.0;
   const int ndelta = (int)((double)nstep * (range - cut));
   const double rdelta = (range - cut) / (double)ndelta;
   const double offset = cut - 0.5 * rdelta;

   // Count the types, or the classes for dispersion, of the environment and
   // each ligand group.
   std::vector<int> ivt;
   std::vector<double> jvt, avt, bvt;
   for (int i = 0; i < n; ++i) {
      int it = dsp ? atomid::class_[i] : vdw::jvdw[i];
      int k = std::find(ivt.begin(), ivt.end(), it) - ivt.begin();
      if (k == (int)ivt.size()) {
         ivt.push_back(it);
         jvt.push_back(0);
         avt.push_back(0);
         bvt.push_back(0);
      }
      jvt[k] += 1;
      if (mutant::mutg[i] == 1)
         avt[k] += 1;
      if (mutant::mutg[i] == 2)
         bvt[k] += 1;
   }

   // Sum the pairs of each class, with ligand 2 coupling as the complement of
   // vlambda and no interaction between the two ligand groups.
   const int nvt = ivt.size();
   const double ghal = vdwpot::ghal, dhal = vdwpot::dhal;
   const double abuck = vdwpot::abuck, bbuck = vdwpot::bbuck, cbuck = vdwpot::cbuck;
   for (int i = 0; i < nvt; ++i) {
      int it = ivt[i];
      double fi = 4 * M_PI * jvt[i];
      double fia = 4 * M_PI * avt[i];
      double fib = 4 * M_PI * bvt[i];
      double fie = fi - fia - fib;
      for (int k = i; k < nvt; ++k) {
         int kt = ivt[k];
         double fka = avt[k];
         double fkb = bvt[k];
         double fke = jvt[k] - fka - fkb;
         double ee = fie * fke;
         double ea = fie * fka + fia * fke;
         double eb = fie * fkb + fib * fke;
         double aa = fia * fka;
         double bb = fib * fkb;
         double f0, f1, f2;
         if (mutant::vcouple == 0) {
            f0 = ee + aa + bb, f1 = ea, f2 = eb;
         } else {
            f0 = ee, f1 = ea + aa, f2 = eb + bb;
         }
         if (k == i)
            f0 *= 0.5, f1 *= 0.5, f2 *= 0.5;

         double cik = 0, rv = 0, eps = 0, rv2 = 0, rv6 = 0, rv7 = 0;
         if (dsp) {
            cik = kdsp::dspsix[it - 1] * kdsp::dspsix[kt - 1];
         } else {
            int offs = (it - 1) * njvdw + (kt - 1);
            rv = vdw::radmin[offs];
            eps = vdw::epsilon[offs];
            rv2 = rv * rv;
            rv6 = rv2 * rv2 * rv2;
            rv7 = rv6 * rv;
         }
         double etot = 0;
         for (int j = 1; j <= ndelta; ++j) {
            double r = offset + j * rdelta;
            double r2 = r * r;
            double r3 = r2 * r;
            double r6 = r3 * r3;
            double r7 = r6 * r;
            double e = 0;
            if (dsp) {
               e = -cik / r6;
            } else if (vdwtyp == Vdw::LJ) {
               double p6 = rv6 / r6;
               double p12 = p6 * p6;
               e = eps * (p12 - 2 * p6);
            } else if (vdwtyp == Vdw::HAL) {
               double rho = r7 + ghal * rv7;
               double tau = (dhal + 1) / (r + dhal * rv);
               double tau7 = std::pow(tau, 7);
               e = eps * rv7 * tau7 * ((ghal + 1) * rv7 / rho - 2);
            } else if (vdwtyp == Vdw::BUCK or vdwtyp == Vdw::MM3HB) {
               double p = std::sqrt(rv2 / r2);
               double p6 = rv6 / r6;
               double expterm = abuck * std::exp(-bbuck / p);
               e = eps * (expterm - cbuck * p6);
            }
            if (r < off) {
               double r4 = r2 * r2;
               double r5 = r2 * r3;
               double taper = c5 * r5 + c4 * r4 + c3 * r3 + c2 * r2 + c1 * r + c0;
               e = e * (1 - taper);
            }
            etot += e * rdelta * r2;
         }
         ls.s0 += f0 * etot;
         ls.s1 += f1 * etot;
         ls.s2 += f2 * etot;
      }
   }
}

LmdaFn lrcEval(const LrcSums& ls, double vl, double dl, double d2l)
{
   const bool dodl = dl != 0 or d2l != 0;
   const LmdaFn w = lmdaPwrChain(vl, ls.nw, dl, d2l, dodl);
   const LmdaFn b = lmdaPwrChain(1 - vl, ls.nb, -dl, -d2l, dodl);
   return {ls.s0 + w.f * ls.s1 + b.f * ls.s2, w.d1 * ls.s1 + b.d1 * ls.s2, w.d2 * ls.s1 + b.d2 * ls.s2};
}

void lrcApply(int vers, int dlvers, const LrcSums& ls, energy_prec& eterm, virial_prec (&vterm)[9])
{
   auto do_e = vers & calc::energy;
   auto do_v = vers & calc::virial;
   const bool dodl = dlvers & calc::energy_dlmda1;
   const LmdaFn e = lrcEval(ls, vlam, dodl ? dvldlmda : 0, dodl ? d2vldlmda2 : 0);
   const double vol = boxVolume();

   if (do_e) {
      if (e.f != 0) {
         energy_prec corr = e.f / vol;
         eterm += corr;
         energy_vdw += corr;
      }
      if ((dlvers & calc::energy_dlmda1) and e.d1 != 0) {
         energy_prec corr = e.d1 / vol;
         devdl += corr;
         dedl += corr;
      }
      if ((dlvers & calc::energy_dlmda2) and e.d2 != 0) {
         energy_prec corr = e.d2 / vol;
         d2evdl2 += corr;
         d2edl2 += corr;
      }
   }
   // the virial of a constant over the volume is minus its energy
   if (do_v) {
      if (e.f != 0) {
         virial_prec term = -e.f / vol;
         for (int k : {0, 4, 8}) {
            vterm[k] += term;
            virial_vdw[k] += term;
         }
      }
      if ((dlvers & calc::virial_dlmda) and e.d1 != 0) {
         virial_prec term = -e.d1 / vol;
         for (int k : {0, 4, 8})
            dvirdl[k] += term;
      }
   }
}

static void evdwKernel(int vers)
{
   if (vdwtyp == Vdw::LJ)
      elj(vers);
   else if (vdwtyp == Vdw::BUCK)
      ebuck(vers);
   else if (vdwtyp == Vdw::MM3HB)
      emm3hb(vers);
   else if (vdwtyp == Vdw::HAL)
      ehal(vers);
   else if (vdwtyp == Vdw::GAUSS)
      egauss(vers);
   else
      assert(false);
}

void evdw(int vers)
{
   auto rc_a = rc_flag & calc::analyz;
   auto do_a = vers & calc::analyz;

   zeroOnHost(energy_ev, virial_ev);
   if (rc_a and do_a)
      darray::zero(g::q0, bufferSize(), nev);
   ev_buf.zero(vers);
   vdwLmdaZero(vers);

   evdwKernel(vers);
   lrcApply(vers, lmdaDerivVers(vers, vdlmdaActive()), vdw_lrc, energy_ev, virial_ev);

   ev_buf.flush(vers);
   vdwLmdaFlush(vers);
}

void evdwLrcVol(energy_prec& elrc, virial_prec& vlrc)
{
   const LmdaFn e = lrcEval(vdw_lrc, vlam, 0, 0);
   elrc = e.f;
   vlrc = -e.f;
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, elj, int);
void elj(int vers)
{
   TINKER_FCALL2(acc1, cu1, elj, vers);
}

TINKER_FVOID2(acc0, cu1, elj14, int);
void elj14(int vers)
{
   TINKER_FCALL2(acc0, cu1, elj14, vers);
}

TINKER_FVOID2(acc1, cu1, ebuck, int);
void ebuck(int vers)
{
   TINKER_FCALL2(acc1, cu1, ebuck, vers);
}

TINKER_FVOID2(acc1, cu1, emm3hb, int);
void emm3hb(int vers)
{
   TINKER_FCALL2(acc1, cu1, emm3hb, vers);
}

TINKER_FVOID2(acc1, cu1, egauss, int);
void egauss(int vers)
{
   TINKER_FCALL2(acc1, cu1, egauss, vers);
}

TINKER_FVOID2(acc1, cu1, ehal, int);
void ehal(int vers)
{
   TINKER_FCALL2(acc1, cu1, ehal, lmdaDerivVers(vers, vdlmdaActive()));
}

TINKER_FVOID2(acc1, cu1, ehalReduceXyz);
void ehalReduceXyz()
{
   TINKER_FCALL2(acc1, cu1, ehalReduceXyz);
}

TINKER_FVOID2(acc1, cu1, ehalResolveGradient, const grad_prec*, const grad_prec*, const grad_prec*, grad_prec*,
   grad_prec*, grad_prec*);
void ehalResolveGradient(const grad_prec* gxred, const grad_prec* gyred, const grad_prec* gzred, grad_prec* devx,
   grad_prec* devy, grad_prec* devz)
{
   TINKER_FCALL2(acc1, cu1, ehalResolveGradient, gxred, gyred, gzred, devx, devy, devz);
}
}
