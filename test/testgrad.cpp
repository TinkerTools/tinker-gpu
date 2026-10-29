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
   double etol = 1.0e-3; ///< Tolerance of the energy in mixed precision.
};

const Fixture kFixtures[] = {
   {"01_water_ye_m10v10"},
   {"02_water_ye_m05v05"},
   {"03_water_ye_m00v00"},
   {"04_water_ast_ye_l10"},
   {"05_water_ast_ye_l05"},
   {"06_water_ast_ye_l00"},
   {"07_g3", "g3/g3.xyz", "g3/g3.prm", 2.0e-2, 5.0e-4},
   {"08_g3_nlist", "g3/g3.xyz", "g3/g3.prm", 2.0e-2, 5.0e-4},
   {"09_aplus_thole0", "aplus2022/tetramer.xyz", "aplus2022/AMOEBAplus_Org.prm"},
   {"10_aplus_polpair", "aplus2022/tetramer.xyz", "aplus2022/AMOEBAplus_Org.prm"},
   {"11_hippo_ast_l05", "dmso/dmso.xyz", "hippo/hippo19.prm", 2.0e-2},
   {"12_hipwat_group_inter", "testgrad/hipwat.xyz", "commit_aad9340c/water21.prm"},
   // Ewald dispersion of the DMSO box is 4e-3 off in mixed precision, with or
   // without the mutation; a double build reproduces it
   {"16_hippo_dewald_dec_l05", "dmso/dmso.xyz", "hippo/hippo19.prm", 2.0e-2, 1.0e-4, 6.0e-3},
   {"17_hipwat_group_ct_replica", "testgrad/hipwat.xyz", "commit_aad9340c/water21.prm"},
   {"18_hippo_rep_sc_real_l04", "dmso/dmso.xyz", "hippo/hippo19.prm", 2.0e-2},
   // 7e-3 off in mixed precision for the same reason, mutated or not
   {"19_hippo_disp_sc_l04", "dmso/dmso.xyz", "hippo/hippo19.prm", 2.0e-2, 1.0e-4, 1.0e-2},
   {"21_hippo_dewald_group_l04", "dmso/dmso.xyz", "hippo/hippo19.prm"},
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
   const double eps_e = testGetEps(fx.etol, 1.0e-4);
   const double eps_g = testGetEps(1.0e-3, fx.dtol);
   // Central differences at this stepsize, on top of mixed-precision energies.
   const double eps_n = testGetEps(fx.ntol, 1.0e-2);

   COMPARE_REALS(r.energy, ref.getEnergy(), eps_e);
   COMPARE_GRADIENT_FLAT(r.ganlyt, ref_g, eps_g);
   COMPARE_GRADIENT_FLAT(r.gnumer, ref_gn, eps_n);

   session.end();
}

// A fixture whose key tinker9 refuses as it sets up. The throw comes from
// elecData(), before pmeData() opens anything a later case could trip on.
void runRefusedFixture(const Fixture& fx)
{
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
   rc_flag = testgradFlags(opts);
   REQUIRE_THROWS(session.init());
}
}

TEST_CASE("TESTGRAD-01_water_ye_m10v10", "[ff][testgrad]") { runFixture(fx("01_water_ye_m10v10")); }
TEST_CASE("TESTGRAD-02_water_ye_m05v05", "[ff][testgrad]") { runFixture(fx("02_water_ye_m05v05")); }
TEST_CASE("TESTGRAD-03_water_ye_m00v00", "[ff][testgrad]") { runFixture(fx("03_water_ye_m00v00")); }
TEST_CASE("TESTGRAD-07_g3", "[ff][testgrad]") { runFixture(fx("07_g3")); }
TEST_CASE("TESTGRAD-08_g3_nlist", "[ff][testgrad]") { runFixture(fx("08_g3_nlist")); }
TEST_CASE("TESTGRAD-09_aplus_thole0", "[ff][testgrad][aplus]") { runFixture(fx("09_aplus_thole0")); }
TEST_CASE("TESTGRAD-10_aplus_polpair", "[ff][testgrad][aplus]") { runFixture(fx("10_aplus_polpair")); }
TEST_CASE("TESTGRAD-11_hippo_ast_l05", "[ff][testgrad][hippo]") { runFixture(fx("11_hippo_ast_l05")); }
// Ewald dispersion of a decoupled DMSO, whose reciprocal and self terms follow
// the van der Waals lambda (test_testgrad.f case 16).
TEST_CASE("TESTGRAD-16_hippo_dewald_dec_l05", "[ff][testgrad][hippo][disp]") { runFixture(fx("16_hippo_dewald_dec_l05")); }
// Damped dispersion of a decoupled DMSO with a DISP-SOFTCORE switch radius that
// reaches the closest mutated pairs and non-integer exponents, under Ewald
// summation by neighbor list (test_testgrad.f case 19). Case 20 needs replica
// images, which tinker9 does not have.
TEST_CASE("TESTGRAD-19_hippo_disp_sc_l04", "[ff][testgrad][hippo][disp]") { runFixture(fx("19_hippo_disp_sc_l04")); }
// Pauli repulsion of a decoupled DMSO with a non-integer REPEL-SOFTCORE
// exponent and shift shape, by neighbor list (test_testgrad.f case 18).
TEST_CASE("TESTGRAD-18_hippo_rep_sc_real_l04", "[ff][testgrad][hippo][repel]") { runFixture(fx("18_hippo_rep_sc_real_l04")); }
// tinker9 applies no atom group weights to any term, so it refuses these keys
// rather than compute the unweighted energy (test_testgrad.f cases 12 to 15,
// 17 with charge transfer alone, and 21 with Ewald dispersion alone).
TEST_CASE("TESTGRAD-12_hipwat_group_inter", "[ff][testgrad][hippo]") { runRefusedFixture(fx("12_hipwat_group_inter")); }
TEST_CASE("TESTGRAD-17_hipwat_group_ct_replica", "[ff][testgrad][hippo]") { runRefusedFixture(fx("17_hipwat_group_ct_replica")); }
TEST_CASE("TESTGRAD-21_hippo_dewald_group_l04", "[ff][testgrad][hippo][disp]") { runRefusedFixture(fx("21_hippo_dewald_group_l04")); }

#if TINKER_GPULANG_CUDA
TEST_CASE("TESTGRAD-04_water_ast_ye_l10", "[ff][testgrad][ast]") { runFixture(fx("04_water_ast_ye_l10")); }
TEST_CASE("TESTGRAD-05_water_ast_ye_l05", "[ff][testgrad][ast]") { runFixture(fx("05_water_ast_ye_l05")); }
TEST_CASE("TESTGRAD-06_water_ast_ye_l00", "[ff][testgrad][ast]") { runFixture(fx("06_water_ast_ye_l00")); }
#endif
