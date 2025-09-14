#include "ff/evdw.h"
#include "ff/energy.h"
#include "ff/nblist.h"
#include "ff/potent.h"
#include "math/zero.h"
#include "tool/externfunc.h"
#include "tool/iofortstr.h"
#include "tool/iotext.h"
#include <tinker/detail/atomid.hh>
#include <tinker/detail/atoms.hh>
#include <tinker/detail/couple.hh>
#include <tinker/detail/keys.hh>
#include <tinker/detail/mutant.hh>
#include <tinker/detail/params.hh>
#include <tinker/detail/sizes.hh>
#include <tinker/detail/vdw.hh>
#include <tinker/detail/vdwpot.hh>
#include <tinker/routines.h>

#include <cassert>

namespace tinker {
inline namespace v1 {
using new_type = int; // new vdw class/type
using old_type = int; // old vdw class/type
}
static std::map<old_type, new_type> jmap;
static std::vector<new_type> jvec;
static std::vector<new_type> jvdwbuf;
static int jcount;

void vdwSoftcoreData(RcOp op)
{
   if ((not use(Potent::VDW)) and (not use(Potent::REPULS)) and (not use(Potent::DISP)) and (not use(Potent::CHGTRN)))
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
         if (mutant::mut[i]) {
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

      if (vdwtyp == Vdw::HAL)
         darray::deallocate(ired, kred, xred, yred, zred, gxred, gyred, gzred);

      darray::deallocate(jvdw, radmin, epsilon);

      nvexclude = 0;
      darray::deallocate(vexclude, vexclude_scale);

      if (nvdw14 > 0) {
         darray::deallocate(radmin4, epsilon4, vdw14ik);
         nvdw14 = 0;
      }

      if (rc_a) {
         bufferDeallocate(rc_flag, nev);
         bufferDeallocate(rc_flag, ev, vir_ev, devx, devy, devz);
      }
      nev = nullptr;
      ev = nullptr;
      vir_ev = nullptr;
      devx = nullptr;
      devy = nullptr;
      devz = nullptr;

      elrc_vol = 0;
      vlrc_vol = 0;
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

      if (vdwtyp == Vdw::HAL) {
         darray::allocate(n, &ired, &kred, &xred, &yred, &zred);
         if (rc_flag & calc::grad) {
            darray::allocate(n, &gxred, &gyred, &gzred);
         } else {
            gxred = nullptr;
            gyred = nullptr;
            gzred = nullptr;
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
      nvexclude = excls.size();
      darray::allocate(nvexclude, &vexclude, &vexclude_scale);
      darray::copyin(g::q0, nvexclude, vexclude, exclik.data());
      darray::copyin(g::q0, nvexclude, vexclude_scale, excls.data());
      waitFor(g::q0);

      // check VDW14 interations
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
      }

      nev = nullptr;
      ev = eng_buf_vdw;
      vir_ev = vir_buf_vdw;
      devx = gx_vdw;
      devy = gy_vdw;
      devz = gz_vdw;
      if (rc_a) {
         bufferAllocate(rc_flag, &nev);
         bufferAllocate(rc_flag, &ev, &vir_ev, &devx, &devy, &devz);
      }
   }

   if (op & RcOp::INIT) {
      // Halgren
      if (vdwtyp == Vdw::HAL) {
         ghal = vdwpot::ghal;
         dhal = vdwpot::dhal;
         scexp = mutant::scexp;
         scalpha = mutant::scalpha;

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

      // Initialize elrc and vlrc.
      if (vdwpot::use_vcorr) {
         double elrc = 0, vlrc = 0;
         tinker_f_evcorr1({const_cast<char*>("VDW"), 3}, &elrc, &vlrc);
         elrc_vol = elrc * boxVolume();
         vlrc_vol = vlrc * boxVolume();
      } else {
         elrc_vol = 0;
         vlrc_vol = 0;
      }
   }
}

void evdw(int vers)
{
   auto rc_a = rc_flag & calc::analyz;
   auto do_a = vers & calc::analyz;
   auto do_e = vers & calc::energy;
   auto do_v = vers & calc::virial;
   auto do_g = vers & calc::grad;

   zeroOnHost(energy_ev, virial_ev);
   size_t bsize = bufferSize();
   if (rc_a) {
      if (do_a)
         darray::zero(g::q0, bsize, nev);
      if (do_e)
         darray::zero(g::q0, bsize, ev);
      if (do_v)
         darray::zero(g::q0, bsize, vir_ev);
      if (do_g)
         darray::zero(g::q0, n, devx, devy, devz);
   }

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
   if (rc_a) {
      if (do_e) {
         EnergyBuffer u = ev;
         energy_prec e = energyReduce(u);
         energy_ev += e;
         energy_vdw += e;
      }
      if (do_v) {
         VirialBuffer u = vir_ev;
         virial_prec v[9];
         virialReduce(v, u);
         for (int iv = 0; iv < 9; ++iv) {
            virial_ev[iv] += v[iv];
            virial_vdw[iv] += v[iv];
         }
      }
      if (do_g)
         sumGradient(gx_vdw, gy_vdw, gz_vdw, devx, devy, devz);
   }
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
   TINKER_FCALL2(acc1, cu1, ehal, vers);
}

TINKER_FVOID2(acc1, cu1, ehalReduceXyz);
void ehalReduceXyz()
{
   TINKER_FCALL2(acc1, cu1, ehalReduceXyz);
}

TINKER_FVOID2(acc1, cu1, ehalResolveGradient);
void ehalResolveGradient()
{
   TINKER_FCALL2(acc1, cu1, ehalResolveGradient);
}
}
