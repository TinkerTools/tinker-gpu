#include "ff/amoeba/induce.h"
#include "ff/atom.h"
#include "ff/dlmda.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
#include "ff/nblist.h"
#include "ff/potent.h"
#include "tool/externfunc.h"
#include "tool/ioprint.h"
#include <tinker/detail/inform.hh>
#include <tinker/detail/polar.hh>
#include <tinker/detail/polpot.hh>
#include <tinker/detail/units.hh>

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

TINKER_FVOID2(acc1, cu1, ulspredSaveP1, real (*)[3], real (*)[3], //
   const real (*)[3], const real (*)[3]);
void ulspredSave(const real (*uind)[3], const real (*uinp)[3])
{
   if (polpred == UPred::NONE || use_epdt)
      return;

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

InducedSnapshot::InducedSnapshot()
   : m_ud{nullptr, nullptr}
   , m_up{nullptr, nullptr}
   , m_nualt(0)
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

   // A solve writes only slot nualt % maxualt of the predictor history and
   // advances nualt (ulspredSave), so that slot and nualt are all a trial leaves
   // behind. The history is in use as epolarData allocates it.
   if (maxualt > 0 and not use_epdt) {
      darray::allocate(n, &m_ud.copy);
      if (not polpot::use_tholed)
         darray::allocate(n, &m_up.copy);
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
   darray::deallocate(m_ud.copy, m_up.copy);
}

void InducedSnapshot::save()
{
   for (auto& c : m_vec)
      darray::copy(g::q0, n, c.copy, *c.src);
   // flat, as mdsave copies them; the 3x3 element type does not flatten const
   for (auto& c : m_mat)
      darray::copy(g::q0, 9 * n, &c.copy[0][0][0], &(*c.src)[0][0][0]);
   m_nualt = nualt;

   // the slot the trial solve will overwrite
   if (m_ud.copy) {
      int pos = nualt % maxualt;
      m_ud.src = udalt[pos];
      darray::copy(g::q0, n, m_ud.copy, *m_ud.src);
      if (m_up.copy) {
         m_up.src = upalt[pos];
         darray::copy(g::q0, n, m_up.copy, *m_up.src);
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

   if (m_ud.copy) {
      darray::copy(g::q0, n, *m_ud.src, m_ud.copy);
      if (m_up.copy)
         darray::copy(g::q0, n, *m_up.src, m_up.copy);
   }
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, induceMutualPcg1, real (*)[3], real (*)[3]);
static void induceMutualPcg1(real (*uind)[3], real (*uinp)[3])
{
   TINKER_FCALL2(acc1, cu1, induceMutualPcg1, uind, uinp);
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
   induceMutualPcg1(ud, up);
   if (not induceFailed()) {
      ulspredSave(ud, up);
      inducePrint(ud);
   }
}
}
