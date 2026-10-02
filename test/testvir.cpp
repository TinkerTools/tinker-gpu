#include "ff/atom.h"
#include "ff/dlmda.h"
#include "ff/energy.h"
#include "tool/xtesthelper.h"

#include "test.h"
#include "testrt.h"
#include "tinker9.h"

#include <string>

using namespace tinker;

namespace {
struct Fixture
{
   const char* name;
   const char* xyz; ///< Coordinates, under test/file/testvir.
   const char* prm; ///< Parameters, under test/file.
};

const Fixture kFixtures[] = {
   {"01_charge_ewald", "ions.xyz", "commit_350df099/amber99sb.prm"},
   {"02_charge_ewald_lights", "ions.xyz", "commit_350df099/amber99sb.prm"},
   {"03_charge_ewald_nlist", "ions.xyz", "commit_350df099/amber99sb.prm"},
   {"04_mpole_ewald", "ionwat.xyz", "commit_ebe3611e/amoeba09.prm"},
   {"05_mpole_ewald_nlist", "ionwat.xyz", "commit_ebe3611e/amoeba09.prm"},
   {"06_mpole_lambda_l05", "ionwat.xyz", "commit_ebe3611e/amoeba09.prm"},
   {"07_mpole_lambda_nlist_l05", "ionwat.xyz", "commit_ebe3611e/amoeba09.prm"},
};

// The fixture of a given name. Cases look their fixture up by name so that
// retiring a fixture does not shift every case after it.
const Fixture& fx(const char* name)
{
   for (const auto& f : kFixtures)
      if (std::string(f.name) == name)
         return f;
   FAIL("no testvir fixture named " << name);
   return kFixtures[0];
}

void runFixture(const Fixture& fx)
{
   // The lambda fixtures leave the derivative machinery switched on; clear it
   // once the case ends, however it ends.
   TestLmdaFlagReset lmdaReset;

   const std::string files = TINKER9_DIRSTR "/test/file/";
   const std::string dir = files + "testvir/";

   TestFile fxyz(dir + fx.xyz, fx.xyz);
   std::string keyname = std::string(fx.name) + ".key";
   TestFile fkey(dir + keyname, keyname);
   TestFile fprm(files + fx.prm);

   const char* argv[] = {"dummy", fx.xyz, "-k", keyname.c_str()};
   int argc = 4;
   TestSession session(argc, argv);

   rc_flag = testvirFlags();
   session.init();

   auto r = testvirEvaluate();
   REQUIRE(r.numer);
   TestReference ref(std::string(TINKER9_DIRSTR "/test/ref/testvir/") + fx.name + ".txt");

   // The references, from testvir.x, carry 3 decimals.
   const double eps_a = 1.0e-3;
   // Central differences over the lattice vectors. A double precision build
   // reproduces the references; in mixed precision the diagonal, where the steps
   // change the volume, lands up to 2e-2 off.
   const double eps_n = testGetEps(5.0e-2, 1.0e-2);

   COMPARE_VIR9(r.vanlyt, ref.getVirial(), eps_a);
   COMPARE_VIR9(r.vnumer, ref.getNumerVirial(), eps_n);
   // The analytical virial against its own finite difference, as test_testvir.f does.
   for (int k = 0; k < 9; ++k)
      COMPARE_REALS(r.vanlyt[k], r.vnumer[k], eps_n);

   session.end();
}
}

TEST_CASE("TESTVIR-01_charge_ewald", "[ff][testvir]") { runFixture(fx("01_charge_ewald")); }
TEST_CASE("TESTVIR-02_charge_ewald_lights", "[ff][testvir]") { runFixture(fx("02_charge_ewald_lights")); }
TEST_CASE("TESTVIR-03_charge_ewald_nlist", "[ff][testvir]") { runFixture(fx("03_charge_ewald_nlist")); }
TEST_CASE("TESTVIR-04_mpole_ewald", "[ff][testvir]") { runFixture(fx("04_mpole_ewald")); }
TEST_CASE("TESTVIR-05_mpole_ewald_nlist", "[ff][testvir]") { runFixture(fx("05_mpole_ewald_nlist")); }

#if TINKER_GPULANG_CUDA
TEST_CASE("TESTVIR-06_mpole_lambda_l05", "[ff][testvir][ast]") { runFixture(fx("06_mpole_lambda_l05")); }
TEST_CASE("TESTVIR-07_mpole_lambda_nlist_l05", "[ff][testvir][ast]") { runFixture(fx("07_mpole_lambda_nlist_l05")); }
#endif

// tinker9 has no vacuum cell dipole term, so pmeData() refuses EWALD-BOUNDARY.
// The throw comes partway through initialize(): the modules set up before
// pmeData() stay allocated, and the session's destructor still ends the run.
TEST_CASE("TESTVIR-EwaldBoundary", "[ff][testvir]")
{
   const std::string files = TINKER9_DIRSTR "/test/file/";
   const char* xyzname = "ions.xyz";
   const char* keyname = "ewald_boundary.key";
   TestFile fxyz(files + "testvir/" + xyzname, xyzname);
   TestFile fkey(files + "testvir/01_charge_ewald.key", keyname, "ewald-boundary\n");
   TestFile fprm(files + "commit_350df099/amber99sb.prm");

   const char* argv[] = {"dummy", xyzname, "-k", keyname};
   int argc = 4;
   TestSession session(argc, argv);
   rc_flag = testvirFlags();
   REQUIRE_THROWS(session.init());
}
