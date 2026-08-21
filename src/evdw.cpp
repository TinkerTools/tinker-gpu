#include "ff/evdw.h"
#include "ff/dlmda.h"
#include "ff/energy.h"
#include "ff/nblist.h"
#include "ff/ost.h"
#include "ff/potent.h"
#include "ff/termbuf.h"
#include "math/zero.h"
#include "seq/ost.h"
#include "tool/error.h"
#include "tool/externfunc.h"
#include "tool/iofortstr.h"
#include "tool/platform.h"
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

static energy_prec elrc0_vol;
static energy_prec elrc1_vol;
static virial_prec vlrc0_vol;
static virial_prec vlrc1_vol;
// Long-range correction of each parameter-zeroed subsystem, indexed as relSlot.
static energy_prec elrc_slot[nRelSlot];
static virial_prec vlrc_slot[nRelSlot];

void vdwSoftcoreData(RcOp op)
{
   if ((not use(Potent::VDW)) and (not use(Potent::REPULS)) and (not use(Potent::DISP)) and (not use(Potent::CHGTRN))
      and (not(use(Potent::MPOLE) and (use_emast || use_emdt)))
      and (not(use(Potent::POLAR) and (use_epdt || use_plmda))))
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

      if (vdwtyp == Vdw::HAL) {
         darray::deallocate(ired, kred, xred, yred, zred, gxred, gyred, gzred);
         darray::deallocate(gxred_dlmda, gyred_dlmda, gzred_dlmda);
      }

      darray::deallocate(jvdw, radmin, epsilon);

      nvexclude = 0;
      darray::deallocate(vexclude, vexclude_scale);

      if (nvdw14 > 0) {
         darray::deallocate(radmin4, epsilon4, vdw14ik);
         nvdw14 = 0;
      }

      if (rc_a)
         bufferDeallocate(rc_flag, nev);
      ev_buf.manage(op, rc_flag, {}, {}, false);
      if (rc_a)
         darray::deallocate(devdl_buf, d2evdl2_buf);
      devdl_buf = nullptr;
      d2evdl2_buf = nullptr;
      nev = nullptr;
      gxred_dlmda = nullptr;
      gyred_dlmda = nullptr;
      gzred_dlmda = nullptr;

      elrc_vol = 0;
      vlrc_vol = 0;
      elrc0_vol = 0;
      elrc1_vol = 0;
      vlrc0_vol = 0;
      vlrc1_vol = 0;
      for (int k = 0; k < nRelSlot; ++k) {
         elrc_slot[k] = 0;
         vlrc_slot[k] = 0;
      }
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
      ev_buf.manage(op, rc_flag, {&ev, &vir_ev, &devx, &devy, &devz},
         {eng_buf_vdw, vir_buf_vdw, gx_vdw, gy_vdw, gz_vdw}, rc_a, //
         {&energy_ev, &virial_ev}, {&energy_vdw, &virial_vdw});
      devdl_buf = nullptr;
      d2evdl2_buf = nullptr;
      if (dlmask & calc::energy_dlmda1) {
         if (rc_a)
            darray::allocate(bufferSize(), &devdl_buf);
         else
            devdl_buf = dedl_buf;
      }
      if (dlmask & calc::energy_dlmda2) {
         if (rc_a)
            darray::allocate(bufferSize(), &d2evdl2_buf);
         else
            d2evdl2_buf = d2edl2_buf;
      }
      if (rc_a)
         bufferAllocate(rc_flag, &nev);
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
         if (use_evadt) {
            double vlambda_orig = mutant::vlambda;
            double elrc0 = 0, vlrc0 = 0;
            double elrc1 = 0, vlrc1 = 0;
            mutant::vlambda = 0;
            tinker_f_evcorr1({const_cast<char*>("VDW"), 3}, &elrc0, &vlrc0);
            mutant::vlambda = 1;
            tinker_f_evcorr1({const_cast<char*>("VDW"), 3}, &elrc1, &vlrc1);
            mutant::vlambda = vlambda_orig;
            elrc0_vol = elrc0 * boxVolume();
            elrc1_vol = elrc1 * boxVolume();
            vlrc0_vol = vlrc0 * boxVolume();
            vlrc1_vol = vlrc1 * boxVolume();
         } else if (use_evrdt) {
            // One correction per parameter-zeroed subsystem, in relSlot order.
            // Tinker charges evcorr1 inside ehal1calc/ehal3calc, so each
            // subsystem carries its own; the endpoints are summed from these
            // per the coupling states, which is why every slot is needed and
            // not just the four a plain relative schedule happens to use.
            static constexpr int kLa[nRelSlot] = {1, 0, 0, 1, 0};
            static constexpr int kLb[nRelSlot] = {0, 1, 0, 0, 1};
            static constexpr int kLe[nRelSlot] = {1, 1, 1, 0, 0};
            for (int k = 0; k < nRelSlot; ++k) {
               int la = kLa[k], lb = kLb[k], le = kLe[k];
               double eslot = 0, vslot = 0;
               tinker_f_submask(&la, &lb, &le);
               tinker_f_evcorr1({const_cast<char*>("VDW"), 3}, &eslot, &vslot);
               elrc_slot[k] = eslot * boxVolume();
               vlrc_slot[k] = vslot * boxVolume();
            }
            int active = 1;
            tinker_f_submask(&active, &active, &active);
         }
         double elrc = 0, vlrc = 0;
         tinker_f_evcorr1({const_cast<char*>("VDW"), 3}, &elrc, &vlrc);
         elrc_vol = elrc * boxVolume();
         vlrc_vol = vlrc * boxVolume();
      } else {
         elrc_vol = 0;
         vlrc_vol = 0;
         elrc0_vol = 0;
         elrc1_vol = 0;
         vlrc0_vol = 0;
         vlrc1_vol = 0;
         for (int k = 0; k < nRelSlot; ++k) {
            elrc_slot[k] = 0;
            vlrc_slot[k] = 0;
         }
      }
   }
}

static void evdwZeroBuffers(int vers)
{
   auto rc_a = rc_flag & calc::analyz;
   auto do_a = vers & calc::analyz;
   if (rc_a and do_a)
      darray::zero(g::q0, bufferSize(), nev);
   ev_buf.zero(vers);
   if (rc_a) {
      const int vdl = lmdaDerivVers(vers, use_vdlmda);
      if (vdl & calc::energy_dlmda1)
         darray::zero(g::q0, bufferSize(), devdl_buf);
      if (vdl & calc::energy_dlmda2)
         darray::zero(g::q0, bufferSize(), d2evdl2_buf);
   }
}

static void evdwBegin(int vers)
{
   zeroOnHost(energy_ev, virial_ev);
   evdwZeroBuffers(vers);
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

static void evdwFinish(int vers, energy_prec elrcv, virial_prec vlrcv, energy_prec delrcv = 0, energy_prec d2elrcv = 0,
   virial_prec dvlrcv = 0)
{
   auto rc_a = rc_flag & calc::analyz;
   auto do_e = vers & calc::energy;
   auto do_v = vers & calc::virial;
   const int vdl = lmdaDerivVers(vers, use_vdlmda);

   if (do_e) {
      if (elrcv != 0) {
         energy_prec corr = elrcv / boxVolume();
         energy_ev += corr;
         energy_vdw += corr;
      }
      if ((vdl & calc::energy_dlmda1) and delrcv != 0) {
         energy_prec corr = delrcv / boxVolume();
         devdl += corr;
         dedl += corr;
      }
      if ((vdl & calc::energy_dlmda2) and d2elrcv != 0) {
         energy_prec corr = d2elrcv / boxVolume();
         d2evdl2 += corr;
         d2edl2 += corr;
      }
   }
   if (do_v) {
      if (vlrcv != 0) {
         virial_prec term = vlrcv / boxVolume();
         virial_ev[0] += term; // xx
         virial_ev[4] += term; // yy
         virial_ev[8] += term; // zz
         virial_vdw[0] += term;
         virial_vdw[4] += term;
         virial_vdw[8] += term;
      }
      if ((vdl & calc::virial_dlmda) and dvlrcv != 0) {
         virial_prec term = dvlrcv / boxVolume();
         dvirdl[0] += term;
         dvirdl[4] += term;
         dvirdl[8] += term;
      }
   }
   ev_buf.flush(vers);
   if (rc_a) {
      if (vdl & calc::energy_dlmda1) {
         energy_prec e = energyReduce(devdl_buf);
         devdl += e;
         dedl += e;
      }
      if (vdl & calc::energy_dlmda2) {
         energy_prec e = energyReduce(d2evdl2_buf);
         d2evdl2 += e;
         d2edl2 += e;
      }
   }
}

void evdw(int vers)
{
   evdwBegin(vers);
   evdwKernel(vers);
   evdwFinish(vers, elrc_vol, vlrc_vol);
}

// Absolute dual topology. Endpoint 1 is the fully coupled system; endpoint 0 is
// the same system at vlam = 0, where the softcore annihilates exactly the pairs
// it touches. Groups are the binary mut flags.
static DtCoef ehalDtCoefAdt(double w, double dw, double d2w)
{
   DtCoef c;
   dtWeightsToCoef(c, w, dw, d2w, dvldlmda, d2vldlmda2, use_vdlmda);
   c.in0bits = 0;
   c.in1bits = 0;
   for (int gi = 0; gi < 2; ++gi) {
      for (int gk = gi; gk < 2; ++gk) {
         dtCellSet(c.in1bits, gi, gk);
         bool soft = (vcouple == Vdw::DECOUPLE) ? (gi != gk) : (gi or gk);
         if (not soft)
            dtCellSet(c.in0bits, gi, gk);
      }
   }
   // Analysis reports the coupled endpoint's count even when that endpoint
   // carries no weight (ehal3.f:1243-1251), so the counted set is not a subset
   // of the live endpoints and cannot be gated on need1.
   c.cntbits = c.in1bits;
   return c;
}

// Relative dual topology. Groups are the ternary rdt_group labels.
static DtCoef ehalDtCoefRdt(double w, double dw, double d2w, bool need1)
{
   DtCoef c;
   dtWeightsToCoef(c, w, dw, d2w, dvldlmda, d2vldlmda2, use_vdlmda);
   c.in0bits = dtStateBits(evrelst0);
   c.in1bits = dtStateBits(evrelst1);
   c.cntbits = dtCountBits(need1 ? evrelst1 : evrelst0);
   return c;
}

// Interpolates the long-range correction between two endpoint values and hands
// the result, with its lambda derivatives, to evdwFinish.
static void evdwFinishMixed(int vers, double weight1, double dweight1, double d2weight1, energy_prec elrc0,
   energy_prec elrc1, virial_prec vlrc0, virial_prec vlrc1)
{
   energy_prec mix_elrc = weight1 * elrc1 + (1 - weight1) * elrc0;
   virial_prec mix_vlrc = weight1 * vlrc1 + (1 - weight1) * vlrc0;
   energy_prec mix_delrc = dweight1 * dvldlmda * (elrc1 - elrc0);
   energy_prec mix_d2elrc = (d2weight1 * dvldlmda * dvldlmda + dweight1 * d2vldlmda2) * (elrc1 - elrc0);
   virial_prec mix_dvlrc = dweight1 * dvldlmda * (vlrc1 - vlrc0);
   evdwFinish(vers, mix_elrc, mix_vlrc, mix_delrc, mix_d2elrc, mix_dvlrc);
}

void evdw_dt(int vers)
{
   assert(vdwtyp == Vdw::HAL);
   const bool relative = use_evrdt;

   double weight1, dweight1, d2weight1;
   bool need0, need1;
   dtWeightNeed(vlam, evdtexp, dvldlmda, d2vldlmda2, weight1, dweight1, d2weight1, need0, need1);

   evdwBegin(vers);

   ehalDt(vers,
      relative ? ehalDtCoefRdt(weight1, dweight1, d2weight1, need1)
               : ehalDtCoefAdt(weight1, dweight1, d2weight1));

   energy_prec elrc0, elrc1;
   virial_prec vlrc0, vlrc1;
   if (relative) {
      elrc0 = 0, elrc1 = 0, vlrc0 = 0, vlrc1 = 0;
      for (int k = 0; k < nRelSlot; ++k) {
         RdtMask mask;
         bool in0, in1;
         relSlot(k, evrelst0, evrelst1, mask, in0, in1);
         if (in0) {
            elrc0 += elrc_slot[k];
            vlrc0 += vlrc_slot[k];
         }
         if (in1) {
            elrc1 += elrc_slot[k];
            vlrc1 += vlrc_slot[k];
         }
      }
   } else {
      elrc0 = elrc0_vol, elrc1 = elrc1_vol;
      vlrc0 = vlrc0_vol, vlrc1 = vlrc1_vol;
   }

   if (not need0) {
      elrc0 = elrc1;
      vlrc0 = vlrc1;
   } else if (not need1) {
      elrc1 = elrc0;
      vlrc1 = vlrc0;
   }

   evdwFinishMixed(vers, weight1, dweight1, d2weight1, elrc0, elrc1, vlrc0, vlrc1);
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
   TINKER_FCALL2(acc1, cu1, ehal, lmdaDerivVers(vers, use_evast));
}

TINKER_FVOID2(acc0, cu1, ehalDt, int, const DtCoef&);
void ehalDt(int vers, const DtCoef& coef)
{
   TINKER_FCALL2(acc0, cu1, ehalDt, lmdaDerivVers(vers, use_vdlmda), coef);
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
