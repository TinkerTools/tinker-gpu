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

const Fixture kFixtures[] = {
   {"01_water_adt_l05"},
   {"02_water_ast_l05"},
   {"03_water_adt_l06exp"},
   {"04_water_ast_l06exp"},
   {"05_water_ast_nodl_l05"},
   {"06_water_ast_vonly_l05"},
   {"07_water_ast_vcorr_l05"},
   {"08_water_ast_vcorr_annih_l05"},
   {"09_water_ast_vcorr_l06exp"},
   {"10_water_rels_vdwm_vcorr_annih_l05"},
   {"11_water_rels_vdwm_vcorr_l050", 2.0e-3, 2.0},
   {"12_water_rels_lig1_l085", 4.0e-3, 3.0},
   {"13_water_rels_lig2_l015", 4.0e-3, 3.0, 5.0e-4},
   {"14_water_rels_lig1_ne_l085", 4.0e-3, 3.0},
   {"15_water_rels_lig1_nlist_exf_l085", 4.0e-3, 3.0},
   {"16_water_rels_lig1_st_l085", 4.0e-3, 3.0},
   {"17_water_rels_lig2_st_l015", 4.0e-3, 3.0},
   {"18_water_rels_lig1_st_ne_l085", 4.0e-3, 3.0},
   {"19_water_rels_lig1_st_polonly_l085", 4.0e-3, 3.0},
   {"20_water_rels_lig1_dt_polonly_l078", 4.0e-3, 3.0},
   {"21_water_rels_lig1_st_prng_l088", 4.0e-3, 3.0},
   {"22_water_rels_lig1_dt_prng_l088", 4.0e-3, 3.0},
   {"23_water_rels_lig2_st_pmap_l015", 4.0e-3, 3.0},
   {"24_ionwat_ewald_l05", 1.0e-2, 1.0, 1.0e-4, "testlmda/ionwat.xyz", "commit_ebe3611e/amoeba09.prm"},
   {"25_ionwat_ewald_nlist_l05", 1.0e-2, 1.0, 1.0e-4, "testlmda/ionwat.xyz", "commit_ebe3611e/amoeba09.prm"},
   {"26_ionwat_pol_ewald_l05", 1.0e-2, 1.0, 1.0e-4, "testlmda/ionwat.xyz", "commit_ebe3611e/amoeba09.prm"},
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

// With analyz off the fixture runs as dynamics does: the terms keep no lambda
// derivative buffers of their own and add into the shared ones, so only the
// totals are reported and compared.
void runFixture(const Fixture& fx, bool analyz = true)
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
   if (not analyz)
      rc_flag &= ~calc::analyz;
   session.init();
   if (not analyz) {
      auto shared = [](EnergyBuffer term, EnergyBuffer total) { return term == nullptr or term == total; };
      if (use_dlmda)
         REQUIRE(dedl_buf != nullptr);
      REQUIRE(shared(demdl_buf, dedl_buf));
      REQUIRE(shared(depdl_buf, dedl_buf));
      REQUIRE(shared(devdl_buf, dedl_buf));
      REQUIRE(shared(d2emdl2_buf, d2edl2_buf));
      REQUIRE(shared(d2epdl2_buf, d2edl2_buf));
      REQUIRE(shared(d2evdl2_buf, d2edl2_buf));
   }
   // The total and, under analysis, the van der Waals, multipole and
   // polarization breakdown.
   const int nterm = analyz ? 4 : 1;

   auto r = testlmdaEvaluate(opts);
   TestReference reffile(std::string(TINKER9_DIRSTR "/test/ref/testlmda/") + fx.name + ".txt");
   const TestLmdaReference& ref = reffile.getLmda();
   REQUIRE((int)ref.lgrad.size() >= n);

   const double eps_d = testGetEps(1.0e-3, fx.dtol);
   // dV/dL is printed with 3 decimals in the references.
   const double eps_v = 1.0e-2;

   // ---- Analytical values against the reference ----------------------------
   for (int k = 0; k < nterm; ++k) {
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
   for (int k = 0; k < nterm; ++k) {
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
TEST_CASE("TESTLMDA-07_water_ast_vcorr_l05", "[ff][testlmda]") { runFixture(fx("07_water_ast_vcorr_l05")); }
TEST_CASE("TESTLMDA-08_water_ast_vcorr_annih_l05", "[ff][testlmda]") { runFixture(fx("08_water_ast_vcorr_annih_l05")); }
TEST_CASE("TESTLMDA-09_water_ast_vcorr_l06exp", "[ff][testlmda]") { runFixture(fx("09_water_ast_vcorr_l06exp")); }
TEST_CASE("TESTLMDA-10_water_rels_vdwm_vcorr_annih_l05", "[ff][testlmda][rdt]") { runFixture(fx("10_water_rels_vdwm_vcorr_annih_l05")); }
TEST_CASE("TESTLMDA-11_water_rels_vdwm_vcorr_l050", "[ff][testlmda][rdt]") { runFixture(fx("11_water_rels_vdwm_vcorr_l050")); }
TEST_CASE("TESTLMDA-12_water_rels_lig1_l085", "[ff][testlmda][rdt]") { runFixture(fx("12_water_rels_lig1_l085")); }
TEST_CASE("TESTLMDA-13_water_rels_lig2_l015", "[ff][testlmda][rdt]") { runFixture(fx("13_water_rels_lig2_l015")); }
TEST_CASE("TESTLMDA-14_water_rels_lig1_ne_l085", "[ff][testlmda][rdt]") { runFixture(fx("14_water_rels_lig1_ne_l085")); }
TEST_CASE("TESTLMDA-15_water_rels_lig1_nlist_exf_l085", "[ff][testlmda][rdt]") { runFixture(fx("15_water_rels_lig1_nlist_exf_l085")); }
TEST_CASE("TESTLMDA-16_water_rels_lig1_st_l085", "[ff][testlmda][rdt]") { runFixture(fx("16_water_rels_lig1_st_l085")); }
TEST_CASE("TESTLMDA-17_water_rels_lig2_st_l015", "[ff][testlmda][rdt]") { runFixture(fx("17_water_rels_lig2_st_l015")); }
TEST_CASE("TESTLMDA-18_water_rels_lig1_st_ne_l085", "[ff][testlmda][rdt]") { runFixture(fx("18_water_rels_lig1_st_ne_l085")); }
TEST_CASE("TESTLMDA-19_water_rels_lig1_st_polonly_l085", "[ff][testlmda][rdt]") { runFixture(fx("19_water_rels_lig1_st_polonly_l085")); }
TEST_CASE("TESTLMDA-20_water_rels_lig1_dt_polonly_l078", "[ff][testlmda][rdt]") { runFixture(fx("20_water_rels_lig1_dt_polonly_l078")); }
TEST_CASE("TESTLMDA-21_water_rels_lig1_st_prng_l088", "[ff][testlmda][rdt]") { runFixture(fx("21_water_rels_lig1_st_prng_l088")); }
TEST_CASE("TESTLMDA-22_water_rels_lig1_dt_prng_l088", "[ff][testlmda][rdt]") { runFixture(fx("22_water_rels_lig1_dt_prng_l088")); }
TEST_CASE("TESTLMDA-23_water_rels_lig2_st_pmap_l015", "[ff][testlmda][rdt]") { runFixture(fx("23_water_rels_lig2_st_pmap_l015")); }
TEST_CASE("TESTLMDA-24_ionwat_ewald_l05", "[ff][testlmda]") { runFixture(fx("24_ionwat_ewald_l05")); }
TEST_CASE("TESTLMDA-25_ionwat_ewald_nlist_l05", "[ff][testlmda]") { runFixture(fx("25_ionwat_ewald_nlist_l05")); }
TEST_CASE("TESTLMDA-26_ionwat_pol_ewald_l05", "[ff][testlmda]") { runFixture(fx("26_ionwat_pol_ewald_l05")); }

// Every fixture once more without analysis, as dynamics runs it.
TEST_CASE("TESTLMDA-DYN-01_water_adt_l05", "[ff][testlmda][dyn]") { runFixture(fx("01_water_adt_l05"), false); }
TEST_CASE("TESTLMDA-DYN-02_water_ast_l05", "[ff][testlmda][dyn]") { runFixture(fx("02_water_ast_l05"), false); }
TEST_CASE("TESTLMDA-DYN-03_water_adt_l06exp", "[ff][testlmda][dyn]") { runFixture(fx("03_water_adt_l06exp"), false); }
TEST_CASE("TESTLMDA-DYN-04_water_ast_l06exp", "[ff][testlmda][dyn]") { runFixture(fx("04_water_ast_l06exp"), false); }
TEST_CASE("TESTLMDA-DYN-05_water_ast_nodl_l05", "[ff][testlmda][dyn]") { runFixture(fx("05_water_ast_nodl_l05"), false); }
TEST_CASE("TESTLMDA-DYN-06_water_ast_vonly_l05", "[ff][testlmda][dyn]") { runFixture(fx("06_water_ast_vonly_l05"), false); }
TEST_CASE("TESTLMDA-DYN-07_water_ast_vcorr_l05", "[ff][testlmda][dyn]") { runFixture(fx("07_water_ast_vcorr_l05"), false); }
TEST_CASE("TESTLMDA-DYN-08_water_ast_vcorr_annih_l05", "[ff][testlmda][dyn]") { runFixture(fx("08_water_ast_vcorr_annih_l05"), false); }
TEST_CASE("TESTLMDA-DYN-09_water_ast_vcorr_l06exp", "[ff][testlmda][dyn]") { runFixture(fx("09_water_ast_vcorr_l06exp"), false); }
TEST_CASE("TESTLMDA-DYN-10_water_rels_vdwm_vcorr_annih_l05", "[ff][testlmda][dyn][rdt]") { runFixture(fx("10_water_rels_vdwm_vcorr_annih_l05"), false); }
TEST_CASE("TESTLMDA-DYN-11_water_rels_vdwm_vcorr_l050", "[ff][testlmda][dyn][rdt]") { runFixture(fx("11_water_rels_vdwm_vcorr_l050"), false); }
TEST_CASE("TESTLMDA-DYN-12_water_rels_lig1_l085", "[ff][testlmda][dyn][rdt]") { runFixture(fx("12_water_rels_lig1_l085"), false); }
TEST_CASE("TESTLMDA-DYN-13_water_rels_lig2_l015", "[ff][testlmda][dyn][rdt]") { runFixture(fx("13_water_rels_lig2_l015"), false); }
TEST_CASE("TESTLMDA-DYN-14_water_rels_lig1_ne_l085", "[ff][testlmda][dyn][rdt]") { runFixture(fx("14_water_rels_lig1_ne_l085"), false); }
TEST_CASE("TESTLMDA-DYN-15_water_rels_lig1_nlist_exf_l085", "[ff][testlmda][dyn][rdt]") { runFixture(fx("15_water_rels_lig1_nlist_exf_l085"), false); }
TEST_CASE("TESTLMDA-DYN-16_water_rels_lig1_st_l085", "[ff][testlmda][dyn][rdt]") { runFixture(fx("16_water_rels_lig1_st_l085"), false); }
TEST_CASE("TESTLMDA-DYN-17_water_rels_lig2_st_l015", "[ff][testlmda][dyn][rdt]") { runFixture(fx("17_water_rels_lig2_st_l015"), false); }
TEST_CASE("TESTLMDA-DYN-18_water_rels_lig1_st_ne_l085", "[ff][testlmda][dyn][rdt]") { runFixture(fx("18_water_rels_lig1_st_ne_l085"), false); }
TEST_CASE("TESTLMDA-DYN-19_water_rels_lig1_st_polonly_l085", "[ff][testlmda][dyn][rdt]") { runFixture(fx("19_water_rels_lig1_st_polonly_l085"), false); }
TEST_CASE("TESTLMDA-DYN-20_water_rels_lig1_dt_polonly_l078", "[ff][testlmda][dyn][rdt]") { runFixture(fx("20_water_rels_lig1_dt_polonly_l078"), false); }
TEST_CASE("TESTLMDA-DYN-21_water_rels_lig1_st_prng_l088", "[ff][testlmda][dyn][rdt]") { runFixture(fx("21_water_rels_lig1_st_prng_l088"), false); }
TEST_CASE("TESTLMDA-DYN-22_water_rels_lig1_dt_prng_l088", "[ff][testlmda][dyn][rdt]") { runFixture(fx("22_water_rels_lig1_dt_prng_l088"), false); }
TEST_CASE("TESTLMDA-DYN-23_water_rels_lig2_st_pmap_l015", "[ff][testlmda][dyn][rdt]") { runFixture(fx("23_water_rels_lig2_st_pmap_l015"), false); }
TEST_CASE("TESTLMDA-DYN-24_ionwat_ewald_l05", "[ff][testlmda][dyn]") { runFixture(fx("24_ionwat_ewald_l05"), false); }
TEST_CASE("TESTLMDA-DYN-25_ionwat_ewald_nlist_l05", "[ff][testlmda][dyn]") { runFixture(fx("25_ionwat_ewald_nlist_l05"), false); }
TEST_CASE("TESTLMDA-DYN-26_ionwat_pol_ewald_l05", "[ff][testlmda][dyn]") { runFixture(fx("26_ionwat_pol_ewald_l05"), false); }
#endif
