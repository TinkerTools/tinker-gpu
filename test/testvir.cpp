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
   const char* xyz; ///< Coordinates, under test/file.
   const char* prm; ///< Parameters, under test/file.
   double atol = 1.0e-3; ///< Tolerance of the analytical virial in mixed precision.
   double ntol = 5.0e-2; ///< Tolerance of the numerical virial in mixed precision.
   double step = 0.02;   ///< Finite difference stepsize of the lattice vectors, in Angstroms.
   double dstep = 0.02;  ///< The stepsize in double precision.
};

const Fixture kFixtures[] = {
   {"01_charge_ewald", "testvir/ions.xyz", "commit_350df099/amber99sb.prm"},
   {"02_charge_ewald_lights", "testvir/ions.xyz", "commit_350df099/amber99sb.prm"},
   {"03_charge_ewald_nlist", "testvir/ions.xyz", "commit_350df099/amber99sb.prm"},
   {"04_mpole_ewald", "testvir/ionwat.xyz", "commit_ebe3611e/amoeba09.prm"},
   {"05_mpole_ewald_nlist", "testvir/ionwat.xyz", "commit_ebe3611e/amoeba09.prm"},
   {"06_mpole_lambda_l05", "testvir/ionwat.xyz", "commit_ebe3611e/amoeba09.prm"},
   {"07_mpole_lambda_nlist_l05", "testvir/ionwat.xyz", "commit_ebe3611e/amoeba09.prm"},
   {"08_mpole_cflux_lambda_l04", "testvir/apgas.xyz", "commit_152ecec0/AMOEBAplus_Org.prm"},
   {"09_mpole_cflux_l04", "testvir/apgas.xyz", "commit_152ecec0/AMOEBAplus_Org.prm"},
   // the numerical virial of the DMSO box, a finite difference of mixed
   // precision energies, is 0.11 off in yz
   {"10_disp_dewald_ann_l05", "dmso/dmso.xyz", "hippo/hippo19.prm", 1.0e-3, 0.15},
   // the long range dispersion correction with a wide taper window, whose
   // virial is minus its energy
   {"12_disp_corr", "testlmda/hipmix.xyz", "commit_4a558d00/hippo19.prm"},
   // the box has steep Lennard-Jones contacts, its yy virial -28400: at the
   // default step their truncation takes the numerical virial 1.6 off, 7e-3
   // at 2e-3 in double precision, while the single precision pair energies
   // leave it 1.8 off at 1e-2 in mixed precision and more at any other step
   {"15_tip3_ljchg_lambda_nlist_l05", "testlmda/tip3box.xyz", "commit_350df099/amber99sb.prm", 1.0e-3, 3.0, 1.0e-2,
      2.0e-3},
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
   const std::string xyzname = testBaseName(fx.xyz);

   TestFile fxyz(files + fx.xyz, xyzname);
   std::string keyname = std::string(fx.name) + ".key";
   TestFile fkey(files + "testvir/" + keyname, keyname);
   TestFile fprm(files + fx.prm);

   const char* argv[] = {"dummy", xyzname.c_str(), "-k", keyname.c_str()};
   int argc = 4;
   TestSession session(argc, argv);

   rc_flag = testvirFlags();
   session.init();

   auto r = testvirEvaluate(testGetEps(fx.step, fx.dstep));
   REQUIRE(r.numer);
   TestReference ref(std::string(TINKER9_DIRSTR "/test/ref/testvir/") + fx.name + ".txt");

   // The references, from testvir.x, carry 3 decimals.
   const double eps_a = testGetEps(fx.atol, 1.0e-3);
   // Central differences over the lattice vectors. A double precision build
   // reproduces the references; in mixed precision the diagonal, where the steps
   // change the volume, lands up to 2e-2 off.
   const double eps_n = testGetEps(fx.ntol, 1.0e-2);

   COMPARE_VIR9(r.vanlyt, ref.getVirial(), eps_a);
   COMPARE_VIR9(r.vnumer, ref.getNumerVirial(), eps_n);
   // The analytical virial against its own finite difference, as test_testvir.f does.
   for (int k = 0; k < 9; ++k)
      COMPARE_REALS(r.vanlyt[k], r.vnumer[k], eps_n);

   session.end();
}
}

TEST_CASE("TESTVIR-01_charge_ewald", "[ff][testvir][L3]") { runFixture(fx("01_charge_ewald")); }
TEST_CASE("TESTVIR-02_charge_ewald_lights", "[ff][testvir][L3]") { runFixture(fx("02_charge_ewald_lights")); }
TEST_CASE("TESTVIR-03_charge_ewald_nlist", "[ff][testvir][L3]") { runFixture(fx("03_charge_ewald_nlist")); }
TEST_CASE("TESTVIR-04_mpole_ewald", "[ff][testvir][L3]") { runFixture(fx("04_mpole_ewald")); }
TEST_CASE("TESTVIR-05_mpole_ewald_nlist", "[ff][testvir][L3]") { runFixture(fx("05_mpole_ewald_nlist")); }
// The flux change of one AMOEBA+ water hydrogen is scaled, so the net charge
// varies with geometry and the Ewald background term adds a charge flux virial.
TEST_CASE("TESTVIR-09_mpole_cflux_l04", "[ff][testvir][cflux][L3]") { runFixture(fx("09_mpole_cflux_l04")); }
// Ewald dispersion of an annihilated DMSO, whose reciprocal and self terms
// follow the van der Waals lambda (test_testvir.f case 10).
TEST_CASE("TESTVIR-10_disp_dewald_ann_l05", "[ff][testvir][hippo][disp][L3]") { runFixture(fx("10_disp_dewald_ann_l05")); }
// Case 11 needs replica images, its 7 A charge transfer cutoff reaching past
// half of its 12 A box, which tinker9 does not have.
TEST_CASE("TESTVIR-12_disp_corr", "[ff][testvir][hippo][disp][L3]") { runFixture(fx("12_disp_corr")); }

#if TINKER_GPULANG_CUDA
TEST_CASE("TESTVIR-06_mpole_lambda_l05", "[ff][testvir][ast][L3]") { runFixture(fx("06_mpole_lambda_l05")); }
TEST_CASE("TESTVIR-07_mpole_lambda_nlist_l05", "[ff][testvir][ast][L3]") { runFixture(fx("07_mpole_lambda_nlist_l05")); }
// 09 through the multipole lambda derivative, where the background term's flux
// virial comes from the net flux change of each lambda group.
TEST_CASE("TESTVIR-08_mpole_cflux_lambda_l04", "[ff][testvir][ast][cflux][L3]") { runFixture(fx("08_mpole_cflux_lambda_l04")); }
// A TIP3P water decoupled from an AMBER box through the soft core
// Lennard-Jones and scaled partial charges, with the lambda derivatives on,
// under Ewald with a neighbor list and the long range correction
// (test_testvir.f case 15).
TEST_CASE("TESTVIR-15_tip3_ljchg_lambda_nlist_l05", "[ff][testvir][charge][lj][L3]") { runFixture(fx("15_tip3_ljchg_lambda_nlist_l05")); }
#endif

// tinker9 has no vacuum cell dipole term, so pmeData() refuses EWALD-BOUNDARY.
// The throw comes partway through initialize(): the modules set up before
// pmeData() stay allocated, and the session's destructor still ends the run.
TEST_CASE("TESTVIR-EwaldBoundary", "[ff][testvir][L3]")
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
