#include "ff/atom.h"
#include "ff/dlmda.h"
#include "ff/energy.h"
#include "tool/xtesthelper.h"

#include "test.h"
#include "testrt.h"
#include "tinker9.h"

#include <tinker/detail/dlmda.hh>

#include <string>

using namespace tinker;

#if TINKER_GPULANG_CUDA

namespace {
// The fixtures of Tinker's test_testlmda.f, driven through the same evaluate
// step xtestlmda uses. The reference files are captured tinker9 output and hold
// the analytical derivatives only.
struct Fixture
{
   const char* key;
   const char* ref;
   bool checknumer;
};

const Fixture kFixtures[] = {
   {"01_water_adt_l05.key", "testlmda.1.txt", true},
   {"02_water_ast_l05.key", "testlmda.2.txt", true},
};

// Finite difference stepsize, in lambda. Smaller steps sharpen the first
// derivatives but wreck the second ones, which divide by eps squared.
constexpr double kEps = 1.0e-2;

void runFixture(const Fixture& fx)
{
   std::string dir = TINKER9_DIRSTR "/test/file/testlmda/";
   const char* xyzname = "water2.xyz";

   TestFile fxyz(dir + xyzname, xyzname);
   TestFile fkey(dir + fx.key, fx.key);
   TestFile fprm(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");

   const char* argv[] = {"dummy", xyzname, "-k", fx.key};
   int argc = 4;

   // These key files drive lambda through OST rather than the "lambda-deriv"
   // keyword, so use_dlmda has to be turned on for the derivative buffers to be
   // allocated (see xTestlmda).
   testBeginWithArgs(argc, argv, true);

   FdTestOptions opts;
   opts.analyt = true;
   opts.numer = true;
   opts.eps = kEps;

   rc_flag = testlmdaFlags(opts);
   initialize();

   auto r = testlmdaEvaluate(opts);
   TestReference reffile(std::string(TINKER9_DIRSTR "/test/ref/") + fx.ref);
   const TestLmdaReference& ref = reffile.getLmda();
   REQUIRE((int)ref.lgrad.size() >= n);

   const double eps_d = testGetEps(1.0e-3, 1.0e-4);
   // dV/dL is printed with 3 decimals in the references.
   const double eps_v = 1.0e-2;

   // ---- Analytical values against the reference ----------------------------
   for (int k = 0; k < 4; ++k) {
      COMPARE_REALS(r.dedl[k], ref.dedl[k], eps_d);
      COMPARE_REALS(r.d2edl2[k], ref.d2edl2[k], eps_d);
   }
   COMPARE_GRADIENT_FLAT(r.dfdl, ref.lgrad, eps_d);
   COMPARE_VIR9(r.dvirdl, ref.dvdl, eps_v);

   // ---- Numerical values against the analytical ones -----------------------
   // Central-difference truncation dominates here; the second derivatives carry
   // an extra factor of 1/eps of it.
   const double eps_n1 = 5.0e-2;
   const double eps_n2 = 2.0e-1;
   const double eps_nf = 1.0e-1;
   if (fx.checknumer) {
      for (int k = 0; k < 4; ++k) {
         COMPARE_REALS(r.ndedl[k], r.dedl[k], eps_n1);
         COMPARE_REALS(r.nd2edl2[k], r.d2edl2[k], eps_n2);
      }
      COMPARE_GRADIENT_FLAT2(r.ndfdl, r.dfdl, eps_nf);
      for (int k = 0; k < 9; ++k)
         COMPARE_REALS(r.ndvirdl[k], r.dvirdl[k], eps_nf);
   } else {
      // Only the vdw (1) and multipole (2) columns are meaningful.
      COMPARE_REALS(r.ndedl[1], r.dedl[1], eps_n1);
      COMPARE_REALS(r.ndedl[2], r.dedl[2], eps_n1);
      COMPARE_REALS(r.nd2edl2[1], r.d2edl2[1], eps_n2);
      COMPARE_REALS(r.nd2edl2[2], r.d2edl2[2], eps_n2);
   }

   finish();
   testEnd();

   // Avoid contaminating later randomized tests.
   dlmda::use_dlmda = 0;
   use_dlmda = false;
   use_ost = false;
   use_meta = false;
   use_ti = false;
}
}

TEST_CASE("TESTLMDA-01_water_adt_l05", "[ff][testlmda]") { runFixture(kFixtures[0]); }
TEST_CASE("TESTLMDA-02_water_ast_l05", "[ff][testlmda]") { runFixture(kFixtures[1]); }

#endif
