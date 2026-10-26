#include "ff/amoeba/induce.h"
#include "ff/atom.h"
#include "ff/dlmda.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
#include "ff/nblist.h"
#include "ff/potent.h"
#include "tool/externfunc.h"
#include "tool/iofortstr.h"
#include "tool/ioprint.h"
#include "tool/iotext.h"
#include <tinker/detail/inform.hh>
#include <tinker/detail/keys.hh>
#include <tinker/detail/mplpot.hh>
#include <tinker/detail/polar.hh>
#include <tinker/detail/polpot.hh>
#include <tinker/detail/units.hh>
#include <tinker/detail/uprior.hh>

#include <string>
#include <utility>
#include <vector>

namespace tinker {
TINKER_FVOID2(acc1, cu1, diagPrecond, const real (*)[3], const real (*)[3], //
   real (*)[3], real (*)[3]);
void diagPrecond(const real (*rsd)[3], const real (*rsdp)[3], real (*zrsd)[3], real (*zrsdp)[3])
{
   TINKER_FCALL2(acc1, cu1, diagPrecond, rsd, rsdp, zrsd, zrsdp);
}

void sparsePrecondBuild() {}

TINKER_FVOID2(acc1, cu1, sparsePrecondApply, const real (*)[3], const real (*)[3], real (*)[3], real (*)[3]);
void sparsePrecondApply(const real (*rsd)[3], const real (*rsdp)[3], real (*zrsd)[3], real (*zrsdp)[3])
{
   TINKER_FCALL2(acc1, cu1, sparsePrecondApply, rsd, rsdp, zrsd, zrsdp);
}

static void markHistorySaved();

TINKER_FVOID2(acc1, cu1, ulspredSaveP1, real (*)[3], real (*)[3], //
   const real (*)[3], const real (*)[3]);
void ulspredSave(const real (*uind)[3], const real (*uinp)[3])
{
   if (not induceHistoryUsable())
      return;
   // a trial that bypasses the history leaves no trace in it
   if (induceHistoryBypassed())
      return;
   markHistorySaved();

   // clang-format off
   real(*ud)[3];
   real(*up)[3];
   int pos = nualt % maxualt;
   switch (pos) {
      case  0: ud = udalt_00; if (uinp) up = upalt_00; break;
      case  1: ud = udalt_01; if (uinp) up = upalt_01; break;
      case  2: ud = udalt_02; if (uinp) up = upalt_02; break;
      case  3: ud = udalt_03; if (uinp) up = upalt_03; break;
      case  4: ud = udalt_04; if (uinp) up = upalt_04; break;
      case  5: ud = udalt_05; if (uinp) up = upalt_05; break;
      case  6: ud = udalt_06; if (uinp) up = upalt_06; break;
      case  7: ud = udalt_07; if (uinp) up = upalt_07; break;
      case  8: ud = udalt_08; if (uinp) up = upalt_08; break;
      case  9: ud = udalt_09; if (uinp) up = upalt_09; break;
      case 10: ud = udalt_10; if (uinp) up = upalt_10; break;
      case 11: ud = udalt_11; if (uinp) up = upalt_11; break;
      case 12: ud = udalt_12; if (uinp) up = upalt_12; break;
      case 13: ud = udalt_13; if (uinp) up = upalt_13; break;
      case 14: ud = udalt_14; if (uinp) up = upalt_14; break;
      case 15: ud = udalt_15; if (uinp) up = upalt_15; break;
      default: ud =  nullptr; up =  nullptr; break;
   }
   nualt = nualt + 1;
   // clang-format on
   if (nualt > 2 * maxualt)
      nualt = nualt - maxualt;

   TINKER_FCALL2(acc1, cu1, ulspredSaveP1, ud, up, uind, uinp);
}

TINKER_FVOID2(acc1, cu1, ulspredSum, real (*)[3], real (*)[3]);
void ulspredSum(real (*uind)[3], real (*uinp)[3])
{
   TINKER_FCALL2(acc1, cu1, ulspredSum, uind, uinp);
}
}

namespace tinker {
static real (**udalt[])[3] = {&udalt_00, &udalt_01, &udalt_02, &udalt_03, &udalt_04, &udalt_05, &udalt_06, &udalt_07,
   &udalt_08, &udalt_09, &udalt_10, &udalt_11, &udalt_12, &udalt_13, &udalt_14, &udalt_15};
static real (**upalt[])[3] = {&upalt_00, &upalt_01, &upalt_02, &upalt_03, &upalt_04, &upalt_05, &upalt_06, &upalt_07,
   &upalt_08, &upalt_09, &upalt_10, &upalt_11, &upalt_12, &upalt_13, &upalt_14, &upalt_15};

namespace {
constexpr UHistoryKey kNoState = {RdtMask::ALL, nullptr, 1};

// The second dual topology history, which the globals do not hold.
struct UHistory
{
   real (*ud[16])[3];
   real (*up[16])[3];
   int nualt;
   UHistoryMark mark;
};
UHistory s_other;
UHistoryMark s_mark;          // of the history the globals hold
UHistoryKey s_key = kNoState; // the state of the solves to come
unsigned long long s_serial;  // energy calls that saved to a history
unsigned long long s_epoch;   // the xyz_epoch of the last save
bool s_two = false;           // whether there is a second history
bool s_twoset = false;        // and whether it keeps uinp as well

bool sameState(const UHistoryKey& a, const UHistoryKey& b)
{
   return a.mask == b.mask and a.group == b.group and a.factor == b.factor;
}

// Whether a history may be extrapolated by a solve at state k in energy call
// `call`: it was built for that state and saved in that call or the one before.
bool extendable(const UHistoryMark& m, const UHistoryKey& k, unsigned long long call)
{
   return sameState(m.key, k) and m.saved + 1 >= call;
}

// the energy call a save would belong to now
unsigned long long currentCall()
{
   return xyz_epoch == s_epoch ? s_serial : s_serial + 1;
}

void swapHistory()
{
   for (int i = 0; i < maxualt; ++i) {
      std::swap(*udalt[i], s_other.ud[i]);
      if (s_twoset)
         std::swap(*upalt[i], s_other.up[i]);
   }
   std::swap(nualt, s_other.nualt);
   std::swap(s_mark, s_other.mark);
}
}

static void markHistorySaved()
{
   if (xyz_epoch != s_epoch) {
      ++s_serial;
      s_epoch = xyz_epoch;
   }
   s_mark.saved = s_serial;
}

void induceHistoryData(RcOp op, bool two_sets)
{
   if (op & RcOp::DEALLOC) {
      if (s_two) {
         // the epolar data routines free whichever history the globals hold
         for (int i = 0; i < maxualt; ++i) {
            darray::deallocate(s_other.ud[i]);
            if (s_twoset)
               darray::deallocate(s_other.up[i]);
         }
      }
      s_other = UHistory{};
      s_mark = UHistoryMark{kNoState, 0};
      s_key = kNoState;
      s_two = false;
      s_twoset = false;
   }

   if (op & RcOp::ALLOC) {
      s_mark = UHistoryMark{kNoState, 0};
      // LSQR keeps fits of its own beside the dipoles, so it gets no second
      // history and dual topology does not predict with it
      if (use_epdt and (polpred == UPred::ASPC or polpred == UPred::GEAR)) {
         s_two = true;
         s_twoset = two_sets;
         s_other = UHistory{};
         for (int i = 0; i < maxualt; ++i) {
            darray::allocate(n, &s_other.ud[i]);
            darray::zero(g::q0, n, s_other.ud[i]);
            if (two_sets) {
               darray::allocate(n, &s_other.up[i]);
               darray::zero(g::q0, n, s_other.up[i]);
            }
         }
         s_other.mark = UHistoryMark{kNoState, 0};
      }
   }
}

void inducePolarState(RdtMask mask, const int* group, real factor)
{
   s_key = {mask, group, factor};
}

void induceSelectHistory()
{
   if (not induceHistoryUsable() or induceHistoryBypassed())
      return;
   // the other history if it was built for this state, or else the staler one
   const bool other = sameState(s_other.mark.key, s_key) or s_other.mark.saved < s_mark.saved;
   if (s_two and not sameState(s_mark.key, s_key) and other)
      swapHistory();
   if (not extendable(s_mark, s_key, currentCall())) {
      nualt = 0;
      s_mark.key = s_key;
   }
}

bool induceHistoryUsable()
{
   return polpred != UPred::NONE and (not use_epdt or s_two);
}

int induceLoadHistory(RdtMask mask, real factor)
{
   const UHistoryKey key = {mask, emGroup(), factor};
   if (s_two and not sameState(s_mark.key, key) and sameState(s_other.mark.key, key))
      swapHistory();
   // the next energy call is another one
   return extendable(s_mark, key, s_serial + 1) ? nualt : 0;
}

InducedSnapshot::InducedSnapshot()
   : m_ud{}
   , m_up{}
   , m_nslot(0)
   , m_nualt(0)
   , m_mark{kNoState, 0}
   , m_serial(0)
   , m_epoch(0)
{
   // Every buffer is picked by the flags that allocate it, never by its pointer:
   // deallocation leaves the pointers of an earlier system dangling.
   if (not use(Potent::POLAR))
      return;

   // the dipoles, as mpoleData allocates them
   for (auto* s : {&uind, &uinp, &udir, &udirp}) {
      Copy<real[3]> c{s, nullptr};
      darray::allocate(n, &c.copy);
      m_vec.push_back(c);
   }

   // A solve writes one slot of the predictor history and advances nualt
   // (ulspredSave): slot nualt % maxualt, or the first one if it empties the
   // history first (induceSelectHistory). That slot, nualt and the marks are all
   // a trial solve leaves behind. The history is in use as epolarData allocates
   // it: AMOEBA alone keeps the second set, and neither AMOEBA+ nor HIPPO does.
   if (maxualt > 0 and not use_epdt) {
      for (int k = 0; k < 2; ++k) {
         darray::allocate(n, &m_ud[k].copy);
         if (not polpot::use_tholed and not mplpot::use_chgpen)
            darray::allocate(n, &m_up[k].copy);
      }
   }

   // exchange polarization, as expolData allocates them
   if (polpot::use_expol) {
      for (auto* s : {&polinv, &polscale}) {
         Copy<real[3][3]> c{s, nullptr};
         darray::allocate(n, &c.copy);
         m_mat.push_back(c);
      }
   }
}

InducedSnapshot::~InducedSnapshot()
{
   for (auto& c : m_vec)
      darray::deallocate(c.copy);
   for (auto& c : m_mat)
      darray::deallocate(c.copy);
   for (int k = 0; k < 2; ++k)
      darray::deallocate(m_ud[k].copy, m_up[k].copy);
}

void InducedSnapshot::save(bool history)
{
   for (auto& c : m_vec)
      darray::copy(g::q0, n, c.copy, *c.src);
   // flat, as mdsave copies them; the 3x3 element type does not flatten const
   for (auto& c : m_mat)
      darray::copy(g::q0, 9 * n, &c.copy[0][0][0], &(*c.src)[0][0][0]);
   m_nualt = nualt;
   m_mark = s_mark;
   m_serial = s_serial;
   m_epoch = s_epoch;

   // the slots the trial solve may overwrite
   m_nslot = 0;
   if (history and m_ud[0].copy) {
      const int slots[2] = {nualt % maxualt, 0};
      m_nslot = slots[0] == 0 ? 1 : 2;
      for (int k = 0; k < m_nslot; ++k) {
         m_ud[k].src = udalt[slots[k]];
         darray::copy(g::q0, n, m_ud[k].copy, *m_ud[k].src);
         if (m_up[k].copy) {
            m_up[k].src = upalt[slots[k]];
            darray::copy(g::q0, n, m_up[k].copy, *m_up[k].src);
         }
      }
   }
}

void InducedSnapshot::restore()
{
   for (auto& c : m_vec)
      darray::copy(g::q0, n, *c.src, c.copy);
   for (auto& c : m_mat)
      darray::copy(g::q0, 9 * n, &(*c.src)[0][0][0], &c.copy[0][0][0]);
   nualt = m_nualt;
   s_mark = m_mark;
   s_serial = m_serial;
   s_epoch = m_epoch;

   for (int k = 0; k < m_nslot; ++k) {
      darray::copy(g::q0, n, *m_ud[k].src, m_ud[k].copy);
      if (m_up[k].copy)
         darray::copy(g::q0, n, *m_up[k].src, m_up[k].copy);
   }
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, induceMutualPcg1, real (*)[3], real (*)[3]);
static void induceMutualPcg1(real (*uind)[3], real (*uinp)[3])
{
   TINKER_FCALL2(acc1, cu1, induceMutualPcg1, uind, uinp);
}

UPred polarPredictor()
{
   bool pred = uprior::use_pred;
   std::string name = FstrView(uprior::polpred).trim();

   // The Fortran reader turns prediction off for free energy perturbation and
   // for the adaptive lambda bias methods, so that their trial energies cannot
   // enter the history, and for dual topology polarization (predict.f). Here
   // the trial solves bypass the predictor (InducedTrial), and each history
   // is extrapolated only at the state it was built for (induceSelectHistory),
   // so the dynamics keeps it.
   const bool trials = use_fep or use_ost or use_abf or use_meta;
   if ((trials or use_epdt) and (rc_flag & calc::md)) {
      for (int i = 0; i < keys::nkey; ++i) {
         FstrView record = keys::keyline[i];
         auto vs = Text::split(record.trim());
         if (vs.empty())
            continue;
         std::string keyword = vs[0];
         Text::upcase(keyword);
         if (keyword != "POLAR-PREDICT")
            continue;
         name = vs.size() > 1 ? vs[1] : "ASPC";
         Text::upcase(name);
         pred = (name != "IEL");
      }
   }

   if (not pred)
      return UPred::NONE;
   if (name == "ASPC")
      return UPred::ASPC;
   if (name == "GEAR")
      return UPred::GEAR;
   return UPred::LSQR;
}

void inducePrint(const real (*ud)[3])
{
   if (inform::debug and use(Potent::POLAR)) {
      std::vector<double> uindbuf;
      uindbuf.resize(3 * n);
      darray::copyout(g::q0, n, uindbuf.data(), ud);
      waitFor(g::q0);
      bool header = true;
      for (int i = 0; i < n; ++i) {
         if (polar::polarity[i] != 0) {
            if (header) {
               header = false;
               print(stdout, "\n Induced Dipole Moments (Debye) :\n");
               print(stdout, "\n    Atom %1$13s X %1$10s Y %1$10s Z %1$9s Total\n\n", "");
            }
            double u1 = uindbuf[3 * i];
            double u2 = uindbuf[3 * i + 1];
            double u3 = uindbuf[3 * i + 2];
            double unorm = std::sqrt(u1 * u1 + u2 * u2 + u3 * u3);
            u1 *= units::debye;
            u2 *= units::debye;
            u3 *= units::debye;
            unorm *= units::debye;
            print(stdout, "%8d     %13.4f%13.4f%13.4f %13.4f\n", i + 1, u1, u2, u3, unorm);
         }
      }
   }
}

void induce(real (*ud)[3], real (*up)[3])
{
   induceSelectHistory();
   induceMutualPcg1(ud, up);
   if (not induceFailed()) {
      ulspredSave(ud, up);
      inducePrint(ud);
   }
}
}
