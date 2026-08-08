#include "ff/atom.h"
#include "ff/energy.h"
#include "tool/xtesthelper.h"

#include "test.h"
#include "testrt.h"
#include "tinker9.h"

#include <string>

using namespace tinker;

namespace {
// The three lambda values of Tinker's test_testgrad.f, driven through the same
// evaluate step xtestgrad uses. The reference files are captured tinker9 output.
struct Fixture
{
   const char* key;
   const char* ref;
};

const Fixture kFixtures[] = {
   {"01_water_ye_m10v10.key", "testgrad.1.txt"},
   {"02_water_ye_m05v05.key", "testgrad.2.txt"},
   {"03_water_ye_m00v00.key", "testgrad.3.txt"},
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
   auto ref_g = ref.getGradient();

   // The references carry 4 decimals, so they cannot pin anything tighter than
   // 1e-3 no matter how the build is configured.
   const double eps_e = testGetEps(1.0e-3, 1.0e-4);
   const double eps_g = testGetEps(1.0e-3, 1.0e-4);
   // Central differences at this stepsize, on top of mixed-precision energies.
   const double eps_n = 1.0e-2;

   COMPARE_REALS(r.energy, ref.getEnergy(), eps_e);
   COMPARE_GRADIENT_FLAT(r.ganlyt, ref_g, eps_g);
   COMPARE_GRADIENT_FLAT2(r.gnumer, r.ganlyt, eps_n);

   finish();
   testEnd();
}
}

TEST_CASE("TESTGRAD-01_water_ye_m10v10", "[ff][testgrad]") { runFixture(kFixtures[0]); }
TEST_CASE("TESTGRAD-02_water_ye_m05v05", "[ff][testgrad]") { runFixture(kFixtures[1]); }
TEST_CASE("TESTGRAD-03_water_ye_m00v00", "[ff][testgrad]") { runFixture(kFixtures[2]); }
