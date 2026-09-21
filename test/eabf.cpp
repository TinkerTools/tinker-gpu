#include "ff/eabf.h"
#include "ff/energy.h"
#include "ff/eost.h"
#include "math/const.h"
#include "test.h"
#include "testrt.h"

#include <cmath>
#include <vector>

// Port of the Fortran ABF unit tests of the ABF routines in
// tinker/test/test_eabf.f. Pure host math -- no molecular system, no GPU.
//
// The expected values and tolerances are copied from test_eabf.f. A failing
// check indicates a real discrepancy in src/eabf.cpp, to be fixed there -- do
// not adjust the expected values.

using namespace tinker;

namespace {
// resetabf -- set the lambda bias state shared with OST to unit-test defaults
// and leave every OST-only array empty, so ABF must never touch them
// (test_eabf.f:resetabf).
void resetabf(int nl, int nhist)
{
   nlmda = nl;
   wlmda = 1.0 / (double)(nlmda - 1);
   wlmda2 = 0.5 * wlmda;
   nlmdahist = 0;
   sizelmdahist = nhist;
   lmdaintv = 10;
   lmdanpa = 3;
   lmdanpb = 3;
   lmdanpc = 4;
   lambda = 0.0;
   lmdaavg = 0.0;
   lmdastd = 0.0;
   dedl = 0.0;
   dedlavg = 0.0;
   dedlstd = 0.0;
   deffdl = 0.0;
   lmdaddgdl = 0.0;
   lmdadeltag = 0.0;
   lmdadfdl = 0.0;
   lmdadt = 0.0;
   use_abf = true;
   use_ost = false;
   use_meta = false;
   use_lmdacv = false;
   lmdacvstd = 0.0;
   lmdacvrat = 0.0;

   lmdaihist.assign(sizelmdahist + 1, 0);
   lmdalhist.assign(sizelmdahist + 1, 0.0);
   lmdafhist.assign(sizelmdahist + 1, 0.0);
   lmdallist.assign(lmdaintv, 0.0);
   lmdaflist.assign(lmdaintv, 0.0);
   lmdafmean.assign(nlmda + 1, 0.0);
   lmdafsum.assign(nlmda + 1, 0.0);
   lmdafwt.assign(nlmda + 1, 0.0);

   osthist.clear();
   ostnext.clear();
   osthead.clear();
   osthhist.clear();
   ostwlhist.clear();
   ostwfhist.clear();
   gkernel.clear();
   gfkernel.clear();
   glfkernel.clear();
   glkernel.clear();
   vkernelmax.clear();
}
}

TEST_CASE("EABF-bias", "[ff][eabf]")
{
   // seed a linear mean force f(lambda) = 2 + 4*lambda on the grid
   TestLmdaFlagGuard guard;
   resetabf(5, 4);
   for (int i = 1; i <= nlmda; ++i)
      lmdafmean[i] = 2.0 + 4.0 * (double)(i - 1) * wlmda;
   for (int k = 0; k < 9; ++k)
      vir[k] = 1.0;
   lambda = 0.3;
   esum = 1.0;
   eabfBias(calc::energy);

   // the free energy 2*lambda + 2*lambda**2 leaves the energy and its slope
   // 2 + 4*lambda is saved for the lambda particle
   COMPARE_REALS(esum, 1.0 - 0.78, 1.0e-12);
   COMPARE_REALS(lmdadfdl, 3.2, 1.0e-12);
   for (int k = 0; k < 9; ++k)
      REQUIRE(vir[k] == 1.0);
}

TEST_CASE("EABF-dyn", "[ff][eabf]")
{
   // a settled interval records its lambda and dU/dlambda average
   TestLmdaFlagGuard guard;
   resetabf(5, 4);
   lmdaintv = 4;
   lmdanpa = 0;
   lmdanpb = 0;
   lmdanpc = 4;
   lmdadt = 0.0;
   lmdadfdl = 0.5;
   for (int istep = 1; istep <= lmdaintv; ++istep) {
      lambda = 0.5;
      dedl = 3.0;
      elmdaDyn(istep);
      if (istep < lmdaintv)
         COMPARE_INTS(nlmdahist, 0); // waits for the interval end
   }
   COMPARE_REALS(deffdl, 2.5, 1.0e-12);
   COMPARE_INTS(nlmdahist, 1);
   COMPARE_INTS(lmdaihist[1], lmdaintv);
   REQUIRE(lmdalhist[1] == 0.5);
   COMPARE_REALS(lmdafhist[1], 3.0, 1.0e-12);
   REQUIRE(lmdafwt[3] == 1.0);
   COMPARE_REALS(lmdafmean[3], 3.0, 1.0e-12);
   COMPARE_REALS(lmdadeltag, 0.75, 1.0e-12);

   // a noisy interval is still recorded while the gate is off
   for (int istep = lmdaintv + 1; istep <= 2 * lmdaintv; ++istep) {
      lambda = 0.5;
      dedl = 1.0;
      if (istep % 2 == 0)
         dedl = 11.0;
      elmdaDyn(istep);
   }
   COMPARE_INTS(nlmdahist, 2);
   REQUIRE(lmdafwt[3] == 2.0);
   COMPARE_REALS(lmdafmean[3], 4.5, 1.0e-12);

   // with the gate on a noisy interval leaves the history unchanged
   use_lmdacv = true;
   lmdacvstd = 1.0;
   lmdacvrat = 0.0;
   double egsave = lmdadeltag;
   for (int istep = 2 * lmdaintv + 1; istep <= 3 * lmdaintv; ++istep) {
      lambda = 0.5;
      dedl = 1.0;
      if (istep % 2 == 0)
         dedl = 11.0;
      elmdaDyn(istep);
   }
   COMPARE_INTS(nlmdahist, 2);
   REQUIRE(lmdafwt[3] == 2.0);
   COMPARE_REALS(lmdafmean[3], 4.5, 1.0e-12);
   COMPARE_REALS(lmdadeltag, egsave, 1.0e-12);

   // with the gate on a settled interval is still recorded
   for (int istep = 3 * lmdaintv + 1; istep <= 4 * lmdaintv; ++istep) {
      lambda = 0.5;
      dedl = 3.0;
      elmdaDyn(istep);
   }
   COMPARE_INTS(nlmdahist, 3);
   REQUIRE(lmdafwt[3] == 3.0);
   COMPARE_REALS(lmdafmean[3], 4.0, 1.0e-12);
}

TEST_CASE("EABF-mean", "[ff][eabf]")
{
   // the bias force follows the running mean of the bin samples
   TestLmdaFlagGuard guard;
   resetabf(5, 8);
   for (int i = 1; i <= 6; ++i) {
      lmdalhist[i] = 0.25;
      lmdafhist[i] = (double)i;
      nlmdahist = i;
      addAbfHist(i);
      COMPARE_REALS(lmdafmean[2], 0.5 * (double)(i + 1), 1.0e-12);
   }

   // a rebuild from the history reproduces the accumulated bins
   buildAbfKernel();
   REQUIRE(lmdafwt[2] == 6.0);
   COMPARE_REALS(lmdafsum[2], 21.0, 1.0e-12);
   COMPARE_REALS(lmdafmean[2], 3.5, 1.0e-12);
   REQUIRE(lmdafmean[1] == 0.0);
}

TEST_CASE("EABF-gate", "[ff][eabf]")
{
   // drive an interval with a deterministic frictionless lambda particle, so
   // that any lambda motion comes from the gate alone
   TestLmdaFlagGuard guard;
   resetabf(5, 4);
   lmdaintv = 6;
   lmdanpa = 2;
   lmdanpb = 2;
   lmdanpc = 2;
   lmdadt = 0.1;
   lmdamass = 1.0;
   lmdafric = 0.0;
   lmdatheta = 0.25 * pi;
   lmdavtheta = 0.0;
   lambda = 0.5;
   lmdadfdl = 0.0;

   // lam is indexed by istep
   double lam[7] = {0};
   for (int istep = 1; istep <= lmdaintv; ++istep) {
      dedl = 1.0;
      elmdaDyn(istep);
      lam[istep] = lambda;
   }

   // the particle moves only while the interval is in phase a
   REQUIRE(lam[1] != lam[2]);
   double frozen = lam[2];
   for (int istep = 3; istep <= lmdaintv; ++istep)
      REQUIRE(lam[istep] == frozen);

   // the sample is recorded exactly on the frozen lambda
   COMPARE_INTS(nlmdahist, 1);
   REQUIRE(lmdalhist[1] == frozen);
   COMPARE_REALS(lmdafhist[1], 1.0, 1.0e-12);
}

TEST_CASE("EABF-resize", "[ff][eabf]")
{
   // three samples overflow a history sized for two
   TestLmdaFlagGuard guard;
   resetabf(5, 2);
   lmdaintv = 4;
   lmdanpa = 0;
   lmdanpb = 0;
   lmdanpc = 4;
   lmdadt = 0.0;
   lmdadfdl = 0.0;
   for (int istep = 1; istep <= 3 * lmdaintv; ++istep) {
      lambda = 0.25;
      dedl = (double)((istep - 1) / lmdaintv + 1);
      elmdaDyn(istep);
   }
   COMPARE_INTS(nlmdahist, 3);
   COMPARE_INTS(sizelmdahist, 4);
   COMPARE_INTS(lmdaihist[1], 4);
   COMPARE_INTS(lmdaihist[3], 12);
   COMPARE_REALS(lmdafhist[1], 1.0, 1.0e-12);
   COMPARE_REALS(lmdafhist[2], 2.0, 1.0e-12);
   COMPARE_REALS(lmdafhist[3], 3.0, 1.0e-12);
   REQUIRE(lmdalhist[3] == 0.25);
   COMPARE_REALS(lmdafmean[2], 2.0, 1.0e-12);
}
