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
   const char* xyz = "testgrad/water2.xyz";         ///< Coordinates, under test/file.
   const char* prm = "commit_6fe8e913/water03.prm"; ///< Parameters, under test/file.
   double ntol = 1.0e-2; ///< Tolerance of the numerical gradient in mixed precision.
   double dtol = 1.0e-4; ///< Tolerance of the analytical gradient in double precision.
};

const Fixture kFixtures[] = {
   {"01_water_ye_m10v10"},
   {"02_water_ye_m05v05"},
   {"03_water_ye_m00v00"},
   {"04_water_ast_ye_l10"},
   {"05_water_ast_ye_l05"},
   {"06_water_ast_ye_l00"},
   // Guest 3 under Ewald, with and without the neighbor list. The numerical
   // gradient lands 1.3e-2 from the 1e-5 step reference in mixed precision. The
   // reference solves the induced dipoles only to POLAR-EPS 1e-5, which leaves
   // it 7e-5 from converged, and a double precision build lands up to 2e-4 from
   // it; at POLAR-EPS 1e-9 every build and list path agrees to 1e-8.
   {"09_g3", "g3/g3.xyz", "g3/g3.prm", 2.0e-2, 5.0e-4},
   {"10_g3_nlist", "g3/g3.xyz", "g3/g3.prm", 2.0e-2, 5.0e-4},
   // AMOEBA+ with a zero Thole value on one type, which the Thole pair table
   // replaces by the larger of the two, and with a POLPAIR Thole that differs
   // from the combining rule; the mutual polarization gradient must take the
   // pair value, as in epolar1.f. The analytical gradient is what these pin.
   // In mixed precision the charge penetration multipole energy is up to 2e-3
   // kcal/mol off: its core-core, core-valence and valence-valence terms are
   // each thousands of kcal/mol and cancel in float. So the numerical gradient
   // lands up to 0.33 from the reference at this stepsize, and further at
   // smaller ones; a double precision build is within 1.2e-3. Lower ntol once
   // pair_mpole_chgpen_aplus regroups those terms around the total charges.
   {"11_aplus_thole0", "aplus2022/tetramer.xyz", "aplus2022/AMOEBAplus_Org.prm", 5.0e-1},
   {"12_aplus_polpair", "aplus2022/tetramer.xyz", "aplus2022/AMOEBAplus_Org.prm", 5.0e-1},
};

// The fixture of a given name. Cases look their fixture up by name so that
// retiring a fixture does not shift every case after it.
const Fixture& fx(const char* name)
{
   for (const auto& f : kFixtures)
      if (std::string(f.name) == name)
         return f;
   FAIL("no testgrad fixture named " << name);
   return kFixtures[0];
}

// Finite difference stepsize, in Angstroms.
constexpr double kEps = 1.0e-2;

void runFixture(const Fixture& fx)
{
   // The lambda fixtures leave the derivative machinery switched on; clear it
   // once the case ends, however it ends.
   TestLmdaFlagReset lmdaReset;

   const std::string files = TINKER9_DIRSTR "/test/file/";
   const std::string xyzname = testBaseName(fx.xyz);

   TestFile fxyz(files + fx.xyz, xyzname);
   std::string keyname = std::string(fx.name) + ".key";
   TestFile fkey(files + "testgrad/" + keyname, keyname);
   TestFile fprm(files + fx.prm);

   const char* argv[] = {"dummy", xyzname.c_str(), "-k", keyname.c_str()};
   int argc = 4;
   TestSession session(argc, argv);

   FdTestOptions opts;
   opts.analyt = true;
   opts.numer = true;
   opts.eps = kEps;

   rc_flag = testgradFlags(opts);
   session.init();

   auto r = testgradEvaluate(opts);
   TestReference ref(std::string(TINKER9_DIRSTR "/test/ref/testgrad/") + fx.name + ".txt");
   REQUIRE(ref.getGradientCount() == n);
   REQUIRE(ref.getNumerGradientCount() == n);
   auto ref_g = ref.getGradient();
   auto ref_gn = ref.getNumerGradient();

   // The references carry 4 decimals, so they cannot pin anything tighter than
   // 1e-3 no matter how the build is configured.
   const double eps_e = testGetEps(1.0e-3, 1.0e-4);
   const double eps_g = testGetEps(1.0e-3, fx.dtol);
   // Central differences at this stepsize, on top of mixed-precision energies.
   const double eps_n = testGetEps(fx.ntol, 1.0e-2);

   COMPARE_REALS(r.energy, ref.getEnergy(), eps_e);
   COMPARE_GRADIENT_FLAT(r.ganlyt, ref_g, eps_g);
   COMPARE_GRADIENT_FLAT(r.gnumer, ref_gn, eps_n);

   session.end();
}
}

TEST_CASE("TESTGRAD-01_water_ye_m10v10", "[ff][testgrad]") { runFixture(fx("01_water_ye_m10v10")); }
TEST_CASE("TESTGRAD-02_water_ye_m05v05", "[ff][testgrad]") { runFixture(fx("02_water_ye_m05v05")); }
TEST_CASE("TESTGRAD-03_water_ye_m00v00", "[ff][testgrad]") { runFixture(fx("03_water_ye_m00v00")); }
TEST_CASE("TESTGRAD-09_g3", "[ff][testgrad]") { runFixture(fx("09_g3")); }
TEST_CASE("TESTGRAD-10_g3_nlist", "[ff][testgrad]") { runFixture(fx("10_g3_nlist")); }
TEST_CASE("TESTGRAD-11_aplus_thole0", "[ff][testgrad][aplus]") { runFixture(fx("11_aplus_thole0")); }
TEST_CASE("TESTGRAD-12_aplus_polpair", "[ff][testgrad][aplus]") { runFixture(fx("12_aplus_polpair")); }

#if TINKER_GPULANG_CUDA
TEST_CASE("TESTGRAD-04_water_ast_ye_l10", "[ff][testgrad][ast]") { runFixture(fx("04_water_ast_ye_l10")); }
TEST_CASE("TESTGRAD-05_water_ast_ye_l05", "[ff][testgrad][ast]") { runFixture(fx("05_water_ast_ye_l05")); }
TEST_CASE("TESTGRAD-06_water_ast_ye_l00", "[ff][testgrad][ast]") { runFixture(fx("06_water_ast_ye_l00")); }
#endif
