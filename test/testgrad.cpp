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

namespace {
struct Fixture
{
   const char* key;
   const char* ref;
};

const Fixture kFixtures[] = {
   {"01_water_ye_m10v10.key", "testgrad.1.txt"},
   {"02_water_ye_m05v05.key", "testgrad.2.txt"},
   {"03_water_ye_m00v00.key", "testgrad.3.txt"},
   {"04_water_ast_ye_l10.key", "testgrad.4.txt"},
   {"05_water_ast_ye_l05.key", "testgrad.5.txt"},
   {"06_water_ast_ye_l00.key", "testgrad.6.txt"},
};

// Finite difference stepsize, in Angstroms.
constexpr double kEps = 1.0e-2;

void runFixture(const Fixture& fx)
{
   std::string dir = TINKER9_DIRSTR "/test/file/testgrad/";
   const char* xyzname = "water2.xyz";

   TestFile fxyz(dir + xyzname, xyzname);
   TestFile fkey(dir + fx.key, fx.key);
   TestFile fprm(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");

   const char* argv[] = {"dummy", xyzname, "-k", fx.key};
   int argc = 4;
   testBeginWithArgs(argc, argv);

   FdTestOptions opts;
   opts.analyt = true;
   opts.numer = true;
   opts.eps = kEps;

   rc_flag = testgradFlags(opts);
   initialize();

   auto r = testgradEvaluate(opts);
   TestReference ref(std::string(TINKER9_DIRSTR "/test/ref/") + fx.ref);
   REQUIRE(ref.getGradientCount() == n);
   REQUIRE(ref.getNumerGradientCount() == n);
   auto ref_g = ref.getGradient();
   auto ref_gn = ref.getNumerGradient();

   // The references carry 4 decimals, so they cannot pin anything tighter than
   // 1e-3 no matter how the build is configured.
   const double eps_e = testGetEps(1.0e-3, 1.0e-4);
   const double eps_g = testGetEps(1.0e-3, 1.0e-4);
   // Central differences at this stepsize, on top of mixed-precision energies.
   const double eps_n = 1.0e-2;

   COMPARE_REALS(r.energy, ref.getEnergy(), eps_e);
   COMPARE_GRADIENT_FLAT(r.ganlyt, ref_g, eps_g);
   COMPARE_GRADIENT_FLAT(r.gnumer, ref_gn, eps_n);

   finish();
   testEnd();

   // The lambda-scaled fixtures leave the derivative machinery switched on.
   // Clear it so later tests in the same binary start from a clean state, the
   // same way testlmda.cpp does.
   dlmda::use_dlmda = 0;
   dlmda::use_edlmda = 0;
   dlmda::use_pdlmda = 0;
   dlmda::use_vdlmda = 0;
   use_dlmda = false;
   use_edlmda = false;
   use_pdlmda = false;
   use_vdlmda = false;
   use_ost = false;
   use_meta = false;
   use_ti = false;
   use_mainlmda = false;
}
}

TEST_CASE("TESTGRAD-01_water_ye_m10v10", "[ff][testgrad]") { runFixture(kFixtures[0]); }
TEST_CASE("TESTGRAD-02_water_ye_m05v05", "[ff][testgrad]") { runFixture(kFixtures[1]); }
TEST_CASE("TESTGRAD-03_water_ye_m00v00", "[ff][testgrad]") { runFixture(kFixtures[2]); }

#if TINKER_GPULANG_CUDA
TEST_CASE("TESTGRAD-04_water_ast_ye_l10", "[ff][testgrad][ast]") { runFixture(kFixtures[3]); }
TEST_CASE("TESTGRAD-05_water_ast_ye_l05", "[ff][testgrad][ast]") { runFixture(kFixtures[4]); }
TEST_CASE("TESTGRAD-06_water_ast_ye_l00", "[ff][testgrad][ast]") { runFixture(kFixtures[5]); }
#endif
