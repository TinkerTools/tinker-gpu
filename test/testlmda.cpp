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
struct Fixture
{
   const char* name;
   double eps;   ///< Finite difference stepsize, in lambda.
   double ntol;  ///< Scale on the tolerances of the numerical derivatives.
};

// The default stepsize is 1e-2: smaller steps sharpen the first derivatives but
// wreck the second ones, which divide by eps squared. The single topology
// fixtures have no second derivatives to protect -- use_epast zeroes them -- and
// 1e-2 leaves too much truncation in the first ones at lambda 1, so they step by
// 1e-3. Going smaller is worse, not better: the forward stencil at lambda 0 does
// not cancel its center term, so mixed-precision round-off grows as 1/eps.
// The staged morph leg squeezes van der Waals into the 0.3-0.7 window, which
// steepens every lambda derivative: 2e-3 balances truncation against round-off,
// and the numerical second derivative still scatters from run to run, so its
// numerical tolerances are doubled.
const Fixture kFixtures[] = {
   {"01_water_adt_l05", 1.0e-2, 1.0},
   {"02_water_ast_l05", 1.0e-2, 1.0},
   {"03_water_adt_l06exp", 1.0e-2, 1.0},
   {"04_water_ast_l06exp", 1.0e-2, 1.0},
   {"05_water_ast_nodl_l05", 1.0e-2, 1.0},
   {"06_water_ast_vonly_l05", 1.0e-2, 1.0},
   {"07_water_ast_ye_l10", 1.0e-3, 1.0},
   {"08_water_ast_ye_l05", 1.0e-3, 1.0},
   {"09_water_ast_ye_l00", 1.0e-3, 1.0},
   {"10_water_ast_vcorr_l05", 1.0e-2, 1.0},
   {"11_water_ast_vcorr_annih_l05", 1.0e-2, 1.0},
   {"12_water_ast_vcorr_l06exp", 1.0e-2, 1.0},
   {"13_water_rdt_vcorr_l05", 1.0e-2, 1.0},
   {"14_water_rdt_vcorr_annih_l05", 1.0e-2, 1.0},
   {"15_water_rdt_l06exp_nlist", 1.0e-2, 1.0},
   {"16_water_rdt_lights_l05", 1.0e-2, 1.0},
   {"17_water_rels_vdwm_vcorr_l050", 2.0e-3, 2.0},
};

void runFixture(const Fixture& fx)
{
   std::string dir = TINKER9_DIRSTR "/test/file/testlmda/";
   const char* xyzname = "water2.xyz";

   TestFile fxyz(dir + xyzname, xyzname);
   std::string keyname = std::string(fx.name) + ".key";
   TestFile fkey(dir + keyname, keyname);
   TestFile fprm(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");

   const char* argv[] = {"dummy", xyzname, "-k", keyname.c_str()};
   int argc = 4;

   testBeginWithArgs(argc, argv);

   FdTestOptions opts;
   opts.analyt = true;
   opts.numer = true;
   opts.eps = fx.eps;

   rc_flag = testlmdaFlags(opts);
   initialize();

   auto r = testlmdaEvaluate(opts);
   TestReference reffile(std::string(TINKER9_DIRSTR "/test/ref/testlmda/") + fx.name + ".txt");
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

   // ---- Numerical values against the same reference ------------------------
   // Central-difference truncation dominates here; the second derivatives carry
   // an extra factor of 1/eps of it.
   const double eps_n1 = 5.0e-2 * fx.ntol;
   const double eps_n2 = 2.0e-1 * fx.ntol;
   const double eps_nf = 1.0e-1 * fx.ntol;
   for (int k = 0; k < 4; ++k) {
      COMPARE_REALS(r.ndedl[k], ref.dedl[k], eps_n1);
      COMPARE_REALS(r.nd2edl2[k], ref.d2edl2[k], eps_n2);
   }
   COMPARE_GRADIENT_FLAT(r.ndfdl, ref.lgrad, eps_nf);
   COMPARE_VIR9(r.ndvirdl, ref.dvdl, eps_nf);

   finish();
   testEnd();

   // Avoid contaminating later randomized tests.
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
   dlmda::use_relstage = 0;
   use_relstage = false;
   use_rel = false;
}
}

TEST_CASE("TESTLMDA-01_water_adt_l05", "[ff][testlmda]") { runFixture(kFixtures[0]); }
TEST_CASE("TESTLMDA-02_water_ast_l05", "[ff][testlmda]") { runFixture(kFixtures[1]); }
TEST_CASE("TESTLMDA-03_water_adt_l06exp", "[ff][testlmda]") { runFixture(kFixtures[2]); }
TEST_CASE("TESTLMDA-04_water_ast_l06exp", "[ff][testlmda]") { runFixture(kFixtures[3]); }
TEST_CASE("TESTLMDA-05_water_ast_nodl_l05", "[ff][testlmda]") { runFixture(kFixtures[4]); }
TEST_CASE("TESTLMDA-06_water_ast_vonly_l05", "[ff][testlmda]") { runFixture(kFixtures[5]); }
TEST_CASE("TESTLMDA-07_water_ast_ye_l10", "[ff][testlmda][ast]") { runFixture(kFixtures[6]); }
TEST_CASE("TESTLMDA-08_water_ast_ye_l05", "[ff][testlmda][ast]") { runFixture(kFixtures[7]); }
TEST_CASE("TESTLMDA-09_water_ast_ye_l00", "[ff][testlmda][ast]") { runFixture(kFixtures[8]); }
TEST_CASE("TESTLMDA-10_water_ast_vcorr_l05", "[ff][testlmda]") { runFixture(kFixtures[9]); }
TEST_CASE("TESTLMDA-11_water_ast_vcorr_annih_l05", "[ff][testlmda]") { runFixture(kFixtures[10]); }
TEST_CASE("TESTLMDA-12_water_ast_vcorr_l06exp", "[ff][testlmda]") { runFixture(kFixtures[11]); }
TEST_CASE("TESTLMDA-13_water_rdt_vcorr_l05", "[ff][testlmda][rdt]") { runFixture(kFixtures[12]); }
TEST_CASE("TESTLMDA-14_water_rdt_vcorr_annih_l05", "[ff][testlmda][rdt]") { runFixture(kFixtures[13]); }
TEST_CASE("TESTLMDA-15_water_rdt_l06exp_nlist", "[ff][testlmda][rdt]") { runFixture(kFixtures[14]); }
TEST_CASE("TESTLMDA-16_water_rdt_lights_l05", "[ff][testlmda][rdt]") { runFixture(kFixtures[15]); }
TEST_CASE("TESTLMDA-17_water_rels_vdwm_vcorr_l050", "[ff][testlmda][rdt]") { runFixture(kFixtures[16]); }
#endif
