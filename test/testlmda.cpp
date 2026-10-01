#include "ff/atom.h"
#include "ff/dlmda.h"
#include "ff/energy.h"
#include "tool/xtesthelper.h"

#include "test.h"
#include "testrt.h"
#include "tinker9.h"

#include <string>

using namespace tinker;

#if TINKER_GPULANG_CUDA

namespace {
struct Fixture
{
   const char* name;
   double eps = 1.0e-2;  ///< Finite difference stepsize, in lambda.
   double ntol = 1.0;    ///< Scale on the tolerances of the numerical derivatives.
   double dtol = 1.0e-4; ///< Tolerance of the analytical derivatives in double precision.
   const char* xyz = "testlmda/water2.xyz";         ///< Coordinates, under test/file.
   const char* prm = "commit_6fe8e913/water03.prm"; ///< Parameters, under test/file.
};

// The default stepsize is 1e-2: smaller steps sharpen the first derivatives but
// wreck the second ones, which divide by eps squared. The staged morph leg
// squeezes van der Waals into the 0.3-0.7 window, which steepens every lambda
// derivative: 2e-3 balances truncation against round-off, and the numerical
// second derivative still scatters from run to run, so its numerical tolerances
// are doubled. The staged charging legs squeeze the
// electrostatic map into a 0.3 wide window the same way, but their numerical
// second derivatives scatter more: at 2e-3 round-off swamps them, and at 1e-2
// truncation throws the first ones off by a few tenths, so they step by 4e-3
// and triple the numerical tolerances.
const Fixture kFixtures[] = {
   {"01_water_adt_l05"},
   {"02_water_ast_l05"},
   {"03_water_adt_l06exp"},
   {"04_water_ast_l06exp"},
   {"05_water_ast_nodl_l05"},
   {"06_water_ast_vonly_l05"},
   {"10_water_ast_vcorr_l05"},
   {"11_water_ast_vcorr_annih_l05"},
   {"12_water_ast_vcorr_l06exp"},
   {"14_water_rels_vdwm_vcorr_annih_l05"},
   {"17_water_rels_vdwm_vcorr_l050", 2.0e-3, 2.0},
   {"18_water_rels_lig1_l085", 4.0e-3, 3.0},
   // The reference solves the induced dipoles only to POLAR-EPS 1e-5, so even a
   // double precision build lands about 1.3e-4 from its dF/dL (mutate.cpp: 140).
   {"19_water_rels_lig2_l015", 4.0e-3, 3.0, 5.0e-4},
   {"20_water_rels_lig1_ne_l085", 4.0e-3, 3.0},
   {"21_water_rels_lig1_nlist_exf_l085", 4.0e-3, 3.0},
   {"22_water_rels_lig1_st_l085", 4.0e-3, 3.0},
   {"23_water_rels_lig2_st_l015", 4.0e-3, 3.0},
   {"24_water_rels_lig1_st_ne_l085", 4.0e-3, 3.0},
   {"25_water_rels_lig1_st_polonly_l085", 4.0e-3, 3.0},
   {"26_water_rels_lig1_dt_polonly_l078", 4.0e-3, 3.0},
   {"27_water_rels_lig1_st_prng_l088", 4.0e-3, 3.0},
   {"28_water_rels_lig1_dt_prng_l088", 4.0e-3, 3.0},
   {"29_water_rels_lig2_st_pmap_l015", 4.0e-3, 3.0},
   // A +1 ion cluster with a water under Ewald; decoupling the chloride carries
   // a lambda-scaled net charge into the uniform background term's dV/dL.
   {"30_ionwat_ewald_l05", 1.0e-2, 1.0, 1.0e-4, "testlmda/ionwat.xyz", "commit_ebe3611e/amoeba09.prm"},
   {"31_ionwat_ewald_nlist_l05", 1.0e-2, 1.0, 1.0e-4, "testlmda/ionwat.xyz", "commit_ebe3611e/amoeba09.prm"},
};

// The fixture of a given name. Cases look their fixture up by name so that
// retiring a fixture does not shift every case after it.
const Fixture& fx(const char* name)
{
   for (const auto& f : kFixtures)
      if (std::string(f.name) == name)
         return f;
   FAIL("no testlmda fixture named " << name);
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
   TestFile fkey(files + "testlmda/" + keyname, keyname);
   TestFile fprm(files + fx.prm);

   const char* argv[] = {"dummy", xyzname.c_str(), "-k", keyname.c_str()};
   int argc = 4;

   TestSession session(argc, argv);

   FdTestOptions opts;
   opts.analyt = true;
   opts.numer = true;
   opts.eps = fx.eps;

   rc_flag = testlmdaFlags(opts);
   session.init();

   auto r = testlmdaEvaluate(opts);
   TestReference reffile(std::string(TINKER9_DIRSTR "/test/ref/testlmda/") + fx.name + ".txt");
   const TestLmdaReference& ref = reffile.getLmda();
   REQUIRE((int)ref.lgrad.size() >= n);

   const double eps_d = testGetEps(1.0e-3, fx.dtol);
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

   session.end();
}
}

TEST_CASE("TESTLMDA-01_water_adt_l05", "[ff][testlmda]") { runFixture(fx("01_water_adt_l05")); }
TEST_CASE("TESTLMDA-02_water_ast_l05", "[ff][testlmda]") { runFixture(fx("02_water_ast_l05")); }
TEST_CASE("TESTLMDA-03_water_adt_l06exp", "[ff][testlmda]") { runFixture(fx("03_water_adt_l06exp")); }
TEST_CASE("TESTLMDA-04_water_ast_l06exp", "[ff][testlmda]") { runFixture(fx("04_water_ast_l06exp")); }
TEST_CASE("TESTLMDA-05_water_ast_nodl_l05", "[ff][testlmda]") { runFixture(fx("05_water_ast_nodl_l05")); }
TEST_CASE("TESTLMDA-06_water_ast_vonly_l05", "[ff][testlmda]") { runFixture(fx("06_water_ast_vonly_l05")); }
TEST_CASE("TESTLMDA-10_water_ast_vcorr_l05", "[ff][testlmda]") { runFixture(fx("10_water_ast_vcorr_l05")); }
TEST_CASE("TESTLMDA-11_water_ast_vcorr_annih_l05", "[ff][testlmda]") { runFixture(fx("11_water_ast_vcorr_annih_l05")); }
TEST_CASE("TESTLMDA-12_water_ast_vcorr_l06exp", "[ff][testlmda]") { runFixture(fx("12_water_ast_vcorr_l06exp")); }
TEST_CASE("TESTLMDA-14_water_rels_vdwm_vcorr_annih_l05", "[ff][testlmda][rdt]") { runFixture(fx("14_water_rels_vdwm_vcorr_annih_l05")); }
TEST_CASE("TESTLMDA-17_water_rels_vdwm_vcorr_l050", "[ff][testlmda][rdt]") { runFixture(fx("17_water_rels_vdwm_vcorr_l050")); }
TEST_CASE("TESTLMDA-18_water_rels_lig1_l085", "[ff][testlmda][rdt]") { runFixture(fx("18_water_rels_lig1_l085")); }
TEST_CASE("TESTLMDA-19_water_rels_lig2_l015", "[ff][testlmda][rdt]") { runFixture(fx("19_water_rels_lig2_l015")); }
TEST_CASE("TESTLMDA-20_water_rels_lig1_ne_l085", "[ff][testlmda][rdt]") { runFixture(fx("20_water_rels_lig1_ne_l085")); }
TEST_CASE("TESTLMDA-21_water_rels_lig1_nlist_exf_l085", "[ff][testlmda][rdt]") { runFixture(fx("21_water_rels_lig1_nlist_exf_l085")); }
TEST_CASE("TESTLMDA-22_water_rels_lig1_st_l085", "[ff][testlmda][rdt]") { runFixture(fx("22_water_rels_lig1_st_l085")); }
TEST_CASE("TESTLMDA-23_water_rels_lig2_st_l015", "[ff][testlmda][rdt]") { runFixture(fx("23_water_rels_lig2_st_l015")); }
TEST_CASE("TESTLMDA-24_water_rels_lig1_st_ne_l085", "[ff][testlmda][rdt]") { runFixture(fx("24_water_rels_lig1_st_ne_l085")); }
TEST_CASE("TESTLMDA-25_water_rels_lig1_st_polonly_l085", "[ff][testlmda][rdt]") { runFixture(fx("25_water_rels_lig1_st_polonly_l085")); }
TEST_CASE("TESTLMDA-26_water_rels_lig1_dt_polonly_l078", "[ff][testlmda][rdt]") { runFixture(fx("26_water_rels_lig1_dt_polonly_l078")); }
TEST_CASE("TESTLMDA-27_water_rels_lig1_st_prng_l088", "[ff][testlmda][rdt]") { runFixture(fx("27_water_rels_lig1_st_prng_l088")); }
TEST_CASE("TESTLMDA-28_water_rels_lig1_dt_prng_l088", "[ff][testlmda][rdt]") { runFixture(fx("28_water_rels_lig1_dt_prng_l088")); }
TEST_CASE("TESTLMDA-29_water_rels_lig2_st_pmap_l015", "[ff][testlmda][rdt]") { runFixture(fx("29_water_rels_lig2_st_pmap_l015")); }
TEST_CASE("TESTLMDA-30_ionwat_ewald_l05", "[ff][testlmda]") { runFixture(fx("30_ionwat_ewald_l05")); }
TEST_CASE("TESTLMDA-31_ionwat_ewald_nlist_l05", "[ff][testlmda]") { runFixture(fx("31_ionwat_ewald_nlist_l05")); }
#endif
