#include "ff/atom.h"
#include "ff/dlmda.h"
#include "ff/egvop.h"
#include "ff/energy.h"
#include "ff/evdw.h"
#include "ff/modamoeba.h"

#include "test.h"
#include "testrt.h"
#include "tinker9.h"

#include <array>
#include <string>
#include <vector>

using namespace tinker;

#if TINKER_GPULANG_CUDA

namespace {
// One water mutation fixture ported from Tinker's test_mutate.f. "base" selects
// the shared coordinate file (water.xyz or water2.xyz); checkm/checkp/checkv
// select which level-3 named components are verified (Atomic Multipoles,
// Polarization, Van der Waals); dolmda enables the level-4 lambda-derivative
// checks (fixtures 030 and later). The neighbor-list duplication from the
// Fortran test is intentionally omitted.
struct Fixture
{
   const char* name;
   const char* base;
   bool checkm, checkp, checkv, dolmda;
   const char* cat;
};

const Fixture kFixtures[] = {
   {"001_water_ye_m10", "water", true, true, false, false, "mv"},
   {"002_water_ne_m10", "water", true, true, false, false, "mv"},
   {"003_water_ye_m05", "water", true, true, false, false, "mv"},
   {"004_water_ne_m05", "water", true, true, false, false, "mv"},
   {"005_water_ye_m00", "water", true, true, false, false, "mv"},
   {"006_water_ne_m00", "water", true, true, false, false, "mv"},
   {"007_water_v10", "water", false, false, true, false, "mv"},
   {"008_water_v05", "water", false, false, true, false, "mv"},
   {"009_water_v00", "water", false, false, true, false, "mv"},
   {"010_water_ye_m10", "water", true, false, false, false, "mp"},
   {"011_water_ne_m10", "water", true, false, false, false, "mp"},
   {"012_water_ye_m05", "water", true, false, false, false, "mp"},
   {"013_water_ne_m05", "water", true, false, false, false, "mp"},
   {"014_water_ye_m00", "water", true, false, false, false, "mp"},
   {"015_water_ne_m00", "water", true, false, false, false, "mp"},
   {"016_water_ye_p10", "water", false, true, false, false, "mp"},
   {"017_water_ne_p10", "water", false, true, false, false, "mp"},
   {"018_water_ye_p05", "water", false, true, false, false, "mp"},
   {"019_water_ne_p05", "water", false, true, false, false, "mp"},
   {"020_water_ye_p00", "water", false, true, false, false, "mp"},
   {"021_water_ne_p00", "water", false, true, false, false, "mp"},
   {"022_water_ye_m10p05", "water", true, true, false, false, "mp"},
   {"023_water_ne_m10p05", "water", true, true, false, false, "mp"},
   {"024_water_ye_m05p10", "water", true, true, false, false, "mp"},
   {"025_water_ne_m05p10", "water", true, true, false, false, "mp"},
   {"026_water_ye_m05p00", "water", true, true, false, false, "mp"},
   {"027_water_ne_m05p00", "water", true, true, false, false, "mp"},
   {"028_water_ye_m00p05", "water", true, true, false, false, "mp"},
   {"029_water_ne_m00p05", "water", true, true, false, false, "mp"},
   {"030_water_ast_ye_m10", "water2", true, false, false, true, "ast"},
   {"031_water_ast_ne_m10", "water2", true, false, false, true, "ast"},
   {"032_water_ast_ye_m05", "water2", true, false, false, true, "ast"},
   {"033_water_ast_ne_m05", "water2", true, false, false, true, "ast"},
   {"034_water_ast_ye_m00", "water2", true, false, false, true, "ast"},
   {"035_water_ast_ne_m00", "water2", true, false, false, true, "ast"},
   {"036_water_ast_v10", "water2", false, false, true, true, "ast"},
   {"037_water_ast_v05", "water2", false, false, true, true, "ast"},
   {"038_water_ast_v00", "water2", false, false, true, true, "ast"},
   {"039_water_ast_ye_m05p10", "water2", true, true, false, true, "ast"},
   {"040_water_ast_ne_m05p10", "water2", true, true, false, true, "ast"},
   {"041_water_adt_ye_m10", "water2", true, false, false, true, "adt"},
   {"042_water_adt_ne_m10", "water2", true, false, false, true, "adt"},
   {"043_water_adt_ye_m05", "water2", true, false, false, true, "adt"},
   {"044_water_adt_ne_m05", "water2", true, false, false, true, "adt"},
   {"045_water_adt_ye_m00", "water2", true, false, false, true, "adt"},
   {"046_water_adt_ne_m00", "water2", true, false, false, true, "adt"},
   {"047_water_adt_ye_p10", "water2", false, true, false, true, "adt"},
   {"048_water_adt_ne_p10", "water2", false, true, false, true, "adt"},
   {"049_water_adt_ye_p05", "water2", false, true, false, true, "adt"},
   {"050_water_adt_ne_p05", "water2", false, true, false, true, "adt"},
   {"051_water_adt_ye_p00", "water2", false, true, false, true, "adt"},
   {"052_water_adt_ne_p00", "water2", false, true, false, true, "adt"},
   {"053_water_adt_v10", "water2", false, false, true, true, "adt"},
   {"054_water_adt_v05", "water2", false, false, true, true, "adt"},
   {"055_water_adt_v00", "water2", false, false, true, true, "adt"},
   {"056_water_adt_ye_m05p10", "water2", true, true, false, true, "adt"},
   {"057_water_adt_ne_m05p10", "water2", true, true, false, true, "adt"},
   {"058_water_rdt_ye_m10", "water2", true, false, false, true, "rdt"},
   {"059_water_rdt_ne_m10", "water2", true, false, false, true, "rdt"},
   {"060_water_rdt_ye_m05", "water2", true, false, false, true, "rdt"},
   {"061_water_rdt_ne_m05", "water2", true, false, false, true, "rdt"},
   {"062_water_rdt_ye_m00", "water2", true, false, false, true, "rdt"},
   {"063_water_rdt_ne_m00", "water2", true, false, false, true, "rdt"},
   {"064_water_rdt_ye_p10", "water2", false, true, false, true, "rdt"},
   {"065_water_rdt_ne_p10", "water2", false, true, false, true, "rdt"},
   {"066_water_rdt_ye_p05", "water2", false, true, false, true, "rdt"},
   {"067_water_rdt_ne_p05", "water2", false, true, false, true, "rdt"},
   {"068_water_rdt_ye_p00", "water2", false, true, false, true, "rdt"},
   {"069_water_rdt_ne_p00", "water2", false, true, false, true, "rdt"},
   {"070_water_rdt_v10", "water2", false, false, true, true, "rdt"},
   {"071_water_rdt_v05", "water2", false, false, true, true, "rdt"},
   {"072_water_rdt_v00", "water2", false, false, true, true, "rdt"},
   {"073_water_rdt_ye_m05p10", "water2", true, true, false, true, "rdt"},
   {"074_water_rdt_ne_m05p10", "water2", true, true, false, true, "rdt"},
   {"075_water_qnt_ast_l10", "water2", true, true, true, true, "qnt"},
   {"076_water_qnt_ast_l05", "water2", true, true, true, true, "qnt"},
   {"077_water_qnt_ast_l00", "water2", true, true, true, true, "qnt"},
   {"078_water_qnt_adt_l10", "water2", true, true, true, true, "qnt"},
   {"079_water_qnt_adt_l05", "water2", true, true, true, true, "qnt"},
   {"080_water_qnt_adt_l00", "water2", true, true, true, true, "qnt"},
   {"081_water_qnt_rdt_l10", "water2", true, true, true, true, "qnt"},
   {"082_water_qnt_rdt_l05", "water2", true, true, true, true, "qnt"},
   {"083_water_qnt_rdt_l00", "water2", true, true, true, true, "qnt"},
   {"084_water_exp_ast_l10", "water2", true, true, true, true, "exp"},
   {"085_water_exp_ast_l05", "water2", true, true, true, true, "exp"},
   {"086_water_exp_ast_l00", "water2", true, true, true, true, "exp"},
   {"087_water_exp_adt_l10", "water2", true, true, true, true, "exp"},
   {"088_water_exp_adt_l05", "water2", true, true, true, true, "exp"},
   {"089_water_exp_adt_l00", "water2", true, true, true, true, "exp"},
   {"090_water_exp_rdt_l10", "water2", true, true, true, true, "exp"},
   {"091_water_exp_rdt_l05", "water2", true, true, true, true, "exp"},
   {"092_water_exp_rdt_l00", "water2", true, true, true, true, "exp"},
   {"093_water_inv_ast_l10", "water2", true, true, true, true, "inv"},
   {"094_water_inv_ast_l05", "water2", true, true, true, true, "inv"},
   {"095_water_inv_ast_l00", "water2", true, true, true, true, "inv"},
   {"096_water_inv_adt_l10", "water2", true, true, true, true, "inv"},
   {"097_water_inv_adt_l05", "water2", true, true, true, true, "inv"},
   {"098_water_inv_adt_l00", "water2", true, true, true, true, "inv"},
   {"099_water_inv_rdt_l10", "water2", true, true, true, true, "inv"},
   {"100_water_inv_rdt_l05", "water2", true, true, true, true, "inv"},
   {"101_water_inv_rdt_l00", "water2", true, true, true, true, "inv"},
   {"102_water_exf_ast_m10", "water2", true, false, false, true, "exf"},
   {"103_water_exf_ast_m05", "water2", true, false, false, true, "exf"},
   {"104_water_exf_ast_m00", "water2", true, false, false, true, "exf"},
   {"105_water_exf_adt_m10", "water2", true, false, false, true, "exf"},
   {"106_water_exf_adt_m05", "water2", true, false, false, true, "exf"},
   {"107_water_exf_adt_m00", "water2", true, false, false, true, "exf"},
   {"108_water_exf_rdt_m10", "water2", true, false, false, true, "exf"},
   {"109_water_exf_rdt_m05", "water2", true, false, false, true, "exf"},
   {"110_water_exf_rdt_m00", "water2", true, false, false, true, "exf"},
   {"111_water_exf_adt_p10", "water2", false, true, false, true, "exf"},
   {"112_water_exf_adt_p05", "water2", false, true, false, true, "exf"},
   {"113_water_exf_adt_p00", "water2", false, true, false, true, "exf"},
   {"114_water_exf_rdt_p10", "water2", false, true, false, true, "exf"},
   {"115_water_exf_rdt_p05", "water2", false, true, false, true, "exf"},
   {"116_water_exf_rdt_p00", "water2", false, true, false, true, "exf"},
   {"117_water_exf_adt_m05p10", "water2", true, true, false, true, "exf"},
   {"118_water_exf_rdt_m05p05", "water2", true, true, false, true, "exf"},
   {"119_water_adt_ye_m10p10v05", "water2", true, true, false, true, "emplar"},
   {"120_water_adt_ne_m10p10v05", "water2", true, true, false, true, "emplar"},
   {"121_water_adt_ye_m05p05v00", "water2", true, true, false, true, "emplar"},
   {"122_water_adt_ne_m05p05v00", "water2", true, true, false, true, "emplar"},
   {"123_water_adt_ye_m00p00v10", "water2", true, true, false, true, "emplar"},
   {"124_water_adt_ne_m00p00v10", "water2", true, true, false, true, "emplar"},
   {"125_water_rdt_ye_m10p10v05", "water2", true, true, false, true, "emplar"},
   {"126_water_rdt_ne_m10p10v05", "water2", true, true, false, true, "emplar"},
   {"127_water_rdt_ye_m05p05v00", "water2", true, true, false, true, "emplar"},
   {"128_water_rdt_ne_m05p05v00", "water2", true, true, false, true, "emplar"},
   {"129_water_rdt_ye_m00p00v10", "water2", true, true, false, true, "emplar"},
   {"130_water_rdt_ne_m00p00v10", "water2", true, true, false, true, "emplar"},
   {"131_water_qnt_adt_l10", "water2", true, true, true, true, "legskip"},
   {"132_water_qnt_adt_l00", "water2", true, true, true, true, "legskip"},
   {"133_water_qnt_rdt_l10", "water2", true, true, true, true, "legskip"},
   {"134_water_qnt_rdt_l00", "water2", true, true, true, true, "legskip"},
   {"135_water_rels_ye_l100", "water2", true, true, true, true, "rels"},
   {"136_water_rels_ye_l085", "water2", true, true, true, true, "rels"},
   {"137_water_rels_ye_l070", "water2", true, true, true, true, "rels"},
   {"138_water_rels_ye_l050", "water2", true, true, true, true, "rels"},
   {"139_water_rels_ye_l030", "water2", true, true, true, true, "rels"},
   {"140_water_rels_ye_l015", "water2", true, true, true, true, "rels"},
   {"141_water_rels_ye_l000", "water2", true, true, true, true, "rels"},
};

void runFixture(const Fixture& fx)
{
   std::string dir = TINKER9_DIRSTR "/test/file/mutate/";
   std::string xyzdst = std::string(fx.base) + ".xyz";
   std::string keyname = std::string(fx.name) + ".key";
   std::string refpath = std::string(TINKER9_DIRSTR "/test/ref/mutate/") + fx.name + ".txt";

   TestFile fxyz(dir + xyzdst, xyzdst);
   TestFile fkey(dir + keyname, keyname);
   TestFile fprm(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");

   const char* argv[] = {"dummy", xyzdst.c_str(), "-k", keyname.c_str()};
   int argc = 4;

   const double eps_e = testGetEps(1.0e-3, 1.0e-4);
   const double eps_g = testGetEps(1.0e-3, 1.0e-4);
   const double eps_v = testGetEps(2.0e-3, 1.0e-3);
   const double eps_l = testGetEps(1.0e-3, 1.0e-4);

   rc_flag = calc::xyz | calc::mass | calc::vmask;

   // These key files enable the lambda-derivative machinery through the
   // "lambda-deriv" keyword, so the Fortran-side use_dlmda needs no nudging here.
   testBeginWithArgs(argc, argv);
   initialize();

   TestReference ref(refpath);
   double ref_e = ref.getEnergy();
   auto ref_v = ref.getVirial();
   auto ref_g = ref.getGradient();

   // The lambda-derivative sections are absent from the non-dolmda references,
   // in which case every field reads back as zero and goes unused.
   const TestLmdaReference& lr = ref.getLmda();
   if (fx.dolmda)
      REQUIRE((int)lr.lgrad.size() >= n);

   // Per-atom lambda gradient (dfsumdl*) vs reference. Valid whenever calc::grad
   // is requested, since lmdachain builds dfsumdl* under its do_g branch.
   auto checkLmdaGrad = [&]() {
      std::vector<double> lx(n), ly(n), lz(n);
      copyGradient(calc::grad, lx.data(), ly.data(), lz.data(), dfsumdlx, dfsumdly, dfsumdlz);
      for (int i = 0; i < n; ++i) {
         COMPARE_REALS(lx[i], lr.lgrad[i][0], eps_g);
         COMPARE_REALS(ly[i], lr.lgrad[i][1], eps_g);
         COMPARE_REALS(lz[i], lr.lgrad[i][2], eps_g);
      }
   };

   // Repeat the full check battery twice against the built system.
   for (int irun = 0; irun < 1; ++irun) {
      // v0
      energy(calc::v0);
      COMPARE_REALS(esum, ref_e, eps_e);

      // v1
      energy(calc::v1);
      COMPARE_REALS(esum, ref_e, eps_e);
      COMPARE_GRADIENT(ref_g, eps_g);
      for (int i = 0; i < 3; ++i)
         for (int j = 0; j < 3; ++j)
            COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_v);

      if (fx.dolmda) {
         COMPARE_REALS(dedl, lr.dedl[0], eps_l);
         COMPARE_REALS(devdl, lr.dedl[1], eps_l);
         COMPARE_REALS(demdl, lr.dedl[2], eps_l);
         COMPARE_REALS(depdl, lr.dedl[3], eps_l);
         COMPARE_REALS(d2edl2, lr.d2edl2[0], eps_l);
         COMPARE_REALS(d2evdl2, lr.d2edl2[1], eps_l);
         COMPARE_REALS(d2emdl2, lr.d2edl2[2], eps_l);
         COMPARE_REALS(d2epdl2, lr.d2edl2[3], eps_l);
         checkLmdaGrad();
         for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
               COMPARE_REALS(dvirdl[i * 3 + j], lr.dvdl[i][j], eps_v);
      }

      // v3
      energy(calc::v3);
      COMPARE_REALS(esum, ref_e, eps_e);
      double eng;
      int cnt;
      if (fx.checkm) {
         ref.getEnergyCountByName("Atomic Multipoles", eng, cnt);
         COMPARE_COUNT(nem, cnt);
         COMPARE_ENERGY(em, eng, eps_e);
      }
      if (fx.checkp) {
         ref.getEnergyCountByName("Polarization", eng, cnt);
         COMPARE_COUNT(nep, cnt);
         COMPARE_ENERGY(ep, eng, eps_e);
      }
      if (fx.checkv) {
         ref.getEnergyCountByName("Van der Waals", eng, cnt);
         COMPARE_COUNT(nev, cnt);
         COMPARE_ENERGY(ev, eng, eps_e);
      }

      // v4
      energy(calc::v4);
      COMPARE_REALS(esum, ref_e, eps_e);
      COMPARE_GRADIENT(ref_g, eps_g);
      if (fx.dolmda) {
         COMPARE_REALS(dedl, lr.dedl[0], eps_l);
         COMPARE_REALS(devdl, lr.dedl[1], eps_l);
         COMPARE_REALS(demdl, lr.dedl[2], eps_l);
         COMPARE_REALS(depdl, lr.dedl[3], eps_l);
         COMPARE_REALS(d2edl2, lr.d2edl2[0], eps_l);
         COMPARE_REALS(d2evdl2, lr.d2edl2[1], eps_l);
         COMPARE_REALS(d2emdl2, lr.d2edl2[2], eps_l);
         COMPARE_REALS(d2epdl2, lr.d2edl2[3], eps_l);
         checkLmdaGrad();
      }

      // level 5 -- gradient only (no energy, no virial)
      energy(calc::v5);
      COMPARE_GRADIENT(ref_g, eps_g);
      if (fx.dolmda)
         checkLmdaGrad();

      // level 6 -- gradient + virial (no energy)
      energy(calc::v6);
      COMPARE_GRADIENT(ref_g, eps_g);
      for (int i = 0; i < 3; ++i)
         for (int j = 0; j < 3; ++j)
            COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_v);
      if (fx.dolmda) {
         checkLmdaGrad();
         for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
               COMPARE_REALS(dvirdl[i * 3 + j], lr.dvdl[i][j], eps_v);
      }
   }

   finish();
   testEnd();
}

void runLegSkipFixture(const Fixture& fx, bool expect4i, bool expect4f)
{
   runFixture(fx);

   REQUIRE(use_ele4i == expect4i);
   REQUIRE(use_pol4i == expect4i);
   REQUIRE(use_vdw4i == expect4i);
   REQUIRE(use_ele4f == expect4f);
   REQUIRE(use_pol4f == expect4f);
   REQUIRE(use_vdw4f == expect4f);
}
} // namespace

TEST_CASE("MUTATE-001_water_ye_m10", "[ff][mutate][mv]") { runFixture(kFixtures[0]); }
TEST_CASE("MUTATE-002_water_ne_m10", "[ff][mutate][mv]") { runFixture(kFixtures[1]); }
TEST_CASE("MUTATE-003_water_ye_m05", "[ff][mutate][mv]") { runFixture(kFixtures[2]); }
TEST_CASE("MUTATE-004_water_ne_m05", "[ff][mutate][mv]") { runFixture(kFixtures[3]); }
TEST_CASE("MUTATE-005_water_ye_m00", "[ff][mutate][mv]") { runFixture(kFixtures[4]); }
TEST_CASE("MUTATE-006_water_ne_m00", "[ff][mutate][mv]") { runFixture(kFixtures[5]); }
TEST_CASE("MUTATE-007_water_v10", "[ff][mutate][mv]") { runFixture(kFixtures[6]); }
TEST_CASE("MUTATE-008_water_v05", "[ff][mutate][mv]") { runFixture(kFixtures[7]); }
TEST_CASE("MUTATE-009_water_v00", "[ff][mutate][mv]") { runFixture(kFixtures[8]); }
TEST_CASE("MUTATE-010_water_ye_m10", "[ff][mutate][mp]") { runFixture(kFixtures[9]); }
TEST_CASE("MUTATE-011_water_ne_m10", "[ff][mutate][mp]") { runFixture(kFixtures[10]); }
TEST_CASE("MUTATE-012_water_ye_m05", "[ff][mutate][mp]") { runFixture(kFixtures[11]); }
TEST_CASE("MUTATE-013_water_ne_m05", "[ff][mutate][mp]") { runFixture(kFixtures[12]); }
TEST_CASE("MUTATE-014_water_ye_m00", "[ff][mutate][mp]") { runFixture(kFixtures[13]); }
TEST_CASE("MUTATE-015_water_ne_m00", "[ff][mutate][mp]") { runFixture(kFixtures[14]); }
TEST_CASE("MUTATE-016_water_ye_p10", "[ff][mutate][mp]") { runFixture(kFixtures[15]); }
TEST_CASE("MUTATE-017_water_ne_p10", "[ff][mutate][mp]") { runFixture(kFixtures[16]); }
TEST_CASE("MUTATE-018_water_ye_p05", "[ff][mutate][mp]") { runFixture(kFixtures[17]); }
TEST_CASE("MUTATE-019_water_ne_p05", "[ff][mutate][mp]") { runFixture(kFixtures[18]); }
TEST_CASE("MUTATE-020_water_ye_p00", "[ff][mutate][mp]") { runFixture(kFixtures[19]); }
TEST_CASE("MUTATE-021_water_ne_p00", "[ff][mutate][mp]") { runFixture(kFixtures[20]); }
TEST_CASE("MUTATE-022_water_ye_m10p05", "[ff][mutate][mp]") { runFixture(kFixtures[21]); }
TEST_CASE("MUTATE-023_water_ne_m10p05", "[ff][mutate][mp]") { runFixture(kFixtures[22]); }
TEST_CASE("MUTATE-024_water_ye_m05p10", "[ff][mutate][mp]") { runFixture(kFixtures[23]); }
TEST_CASE("MUTATE-025_water_ne_m05p10", "[ff][mutate][mp]") { runFixture(kFixtures[24]); }
TEST_CASE("MUTATE-026_water_ye_m05p00", "[ff][mutate][mp]") { runFixture(kFixtures[25]); }
TEST_CASE("MUTATE-027_water_ne_m05p00", "[ff][mutate][mp]") { runFixture(kFixtures[26]); }
TEST_CASE("MUTATE-028_water_ye_m00p05", "[ff][mutate][mp]") { runFixture(kFixtures[27]); }
TEST_CASE("MUTATE-029_water_ne_m00p05", "[ff][mutate][mp]") { runFixture(kFixtures[28]); }
TEST_CASE("MUTATE-030_water_ast_ye_m10", "[ff][mutate][ast]") { runFixture(kFixtures[29]); }
TEST_CASE("MUTATE-031_water_ast_ne_m10", "[ff][mutate][ast]") { runFixture(kFixtures[30]); }
TEST_CASE("MUTATE-032_water_ast_ye_m05", "[ff][mutate][ast]") { runFixture(kFixtures[31]); }
TEST_CASE("MUTATE-033_water_ast_ne_m05", "[ff][mutate][ast]") { runFixture(kFixtures[32]); }
TEST_CASE("MUTATE-034_water_ast_ye_m00", "[ff][mutate][ast]") { runFixture(kFixtures[33]); }
TEST_CASE("MUTATE-035_water_ast_ne_m00", "[ff][mutate][ast]") { runFixture(kFixtures[34]); }
TEST_CASE("MUTATE-036_water_ast_v10", "[ff][mutate][ast]") { runFixture(kFixtures[35]); }
TEST_CASE("MUTATE-037_water_ast_v05", "[ff][mutate][ast]") { runFixture(kFixtures[36]); }
TEST_CASE("MUTATE-038_water_ast_v00", "[ff][mutate][ast]") { runFixture(kFixtures[37]); }
TEST_CASE("MUTATE-039_water_ast_ye_m05p10", "[ff][mutate][ast]") { runFixture(kFixtures[38]); }
TEST_CASE("MUTATE-040_water_ast_ne_m05p10", "[ff][mutate][ast]") { runFixture(kFixtures[39]); }
TEST_CASE("MUTATE-041_water_adt_ye_m10", "[ff][mutate][adt]") { runFixture(kFixtures[40]); }
TEST_CASE("MUTATE-042_water_adt_ne_m10", "[ff][mutate][adt]") { runFixture(kFixtures[41]); }
TEST_CASE("MUTATE-043_water_adt_ye_m05", "[ff][mutate][adt]") { runFixture(kFixtures[42]); }
TEST_CASE("MUTATE-044_water_adt_ne_m05", "[ff][mutate][adt]") { runFixture(kFixtures[43]); }
TEST_CASE("MUTATE-045_water_adt_ye_m00", "[ff][mutate][adt]") { runFixture(kFixtures[44]); }
TEST_CASE("MUTATE-046_water_adt_ne_m00", "[ff][mutate][adt]") { runFixture(kFixtures[45]); }
TEST_CASE("MUTATE-047_water_adt_ye_p10", "[ff][mutate][adt]") { runFixture(kFixtures[46]); }
TEST_CASE("MUTATE-048_water_adt_ne_p10", "[ff][mutate][adt]") { runFixture(kFixtures[47]); }
TEST_CASE("MUTATE-049_water_adt_ye_p05", "[ff][mutate][adt]") { runFixture(kFixtures[48]); }
TEST_CASE("MUTATE-050_water_adt_ne_p05", "[ff][mutate][adt]") { runFixture(kFixtures[49]); }
TEST_CASE("MUTATE-051_water_adt_ye_p00", "[ff][mutate][adt]") { runFixture(kFixtures[50]); }
TEST_CASE("MUTATE-052_water_adt_ne_p00", "[ff][mutate][adt]") { runFixture(kFixtures[51]); }
TEST_CASE("MUTATE-053_water_adt_v10", "[ff][mutate][adt]") { runFixture(kFixtures[52]); }
TEST_CASE("MUTATE-054_water_adt_v05", "[ff][mutate][adt]") { runFixture(kFixtures[53]); }
TEST_CASE("MUTATE-055_water_adt_v00", "[ff][mutate][adt]") { runFixture(kFixtures[54]); }
TEST_CASE("MUTATE-056_water_adt_ye_m05p10", "[ff][mutate][adt]") { runFixture(kFixtures[55]); }
TEST_CASE("MUTATE-057_water_adt_ne_m05p10", "[ff][mutate][adt]") { runFixture(kFixtures[56]); }
TEST_CASE("MUTATE-058_water_rdt_ye_m10", "[ff][mutate][rdt]") { runFixture(kFixtures[57]); }
TEST_CASE("MUTATE-059_water_rdt_ne_m10", "[ff][mutate][rdt]") { runFixture(kFixtures[58]); }
TEST_CASE("MUTATE-060_water_rdt_ye_m05", "[ff][mutate][rdt]") { runFixture(kFixtures[59]); }
TEST_CASE("MUTATE-061_water_rdt_ne_m05", "[ff][mutate][rdt]") { runFixture(kFixtures[60]); }
TEST_CASE("MUTATE-062_water_rdt_ye_m00", "[ff][mutate][rdt]") { runFixture(kFixtures[61]); }
TEST_CASE("MUTATE-063_water_rdt_ne_m00", "[ff][mutate][rdt]") { runFixture(kFixtures[62]); }
TEST_CASE("MUTATE-064_water_rdt_ye_p10", "[ff][mutate][rdt]") { runFixture(kFixtures[63]); }
TEST_CASE("MUTATE-065_water_rdt_ne_p10", "[ff][mutate][rdt]") { runFixture(kFixtures[64]); }
TEST_CASE("MUTATE-066_water_rdt_ye_p05", "[ff][mutate][rdt]") { runFixture(kFixtures[65]); }
TEST_CASE("MUTATE-067_water_rdt_ne_p05", "[ff][mutate][rdt]") { runFixture(kFixtures[66]); }
TEST_CASE("MUTATE-068_water_rdt_ye_p00", "[ff][mutate][rdt]") { runFixture(kFixtures[67]); }
TEST_CASE("MUTATE-069_water_rdt_ne_p00", "[ff][mutate][rdt]") { runFixture(kFixtures[68]); }
TEST_CASE("MUTATE-070_water_rdt_v10", "[ff][mutate][rdt]") { runFixture(kFixtures[69]); }
TEST_CASE("MUTATE-071_water_rdt_v05", "[ff][mutate][rdt]") { runFixture(kFixtures[70]); }
TEST_CASE("MUTATE-072_water_rdt_v00", "[ff][mutate][rdt]") { runFixture(kFixtures[71]); }
TEST_CASE("MUTATE-073_water_rdt_ye_m05p10", "[ff][mutate][rdt]") { runFixture(kFixtures[72]); }
TEST_CASE("MUTATE-074_water_rdt_ne_m05p10", "[ff][mutate][rdt]") { runFixture(kFixtures[73]); }
TEST_CASE("MUTATE-075_water_qnt_ast_l10", "[ff][mutate][qnt]") { runFixture(kFixtures[74]); }
TEST_CASE("MUTATE-076_water_qnt_ast_l05", "[ff][mutate][qnt]") { runFixture(kFixtures[75]); }
TEST_CASE("MUTATE-077_water_qnt_ast_l00", "[ff][mutate][qnt]") { runFixture(kFixtures[76]); }
TEST_CASE("MUTATE-078_water_qnt_adt_l10", "[ff][mutate][qnt]") { runFixture(kFixtures[77]); }
TEST_CASE("MUTATE-079_water_qnt_adt_l05", "[ff][mutate][qnt]") { runFixture(kFixtures[78]); }
TEST_CASE("MUTATE-080_water_qnt_adt_l00", "[ff][mutate][qnt]") { runFixture(kFixtures[79]); }
TEST_CASE("MUTATE-081_water_qnt_rdt_l10", "[ff][mutate][qnt]") { runFixture(kFixtures[80]); }
TEST_CASE("MUTATE-082_water_qnt_rdt_l05", "[ff][mutate][qnt]") { runFixture(kFixtures[81]); }
TEST_CASE("MUTATE-083_water_qnt_rdt_l00", "[ff][mutate][qnt]") { runFixture(kFixtures[82]); }
TEST_CASE("MUTATE-084_water_exp_ast_l10", "[ff][mutate][exp]") { runFixture(kFixtures[83]); }
TEST_CASE("MUTATE-085_water_exp_ast_l05", "[ff][mutate][exp]") { runFixture(kFixtures[84]); }
TEST_CASE("MUTATE-086_water_exp_ast_l00", "[ff][mutate][exp]") { runFixture(kFixtures[85]); }
TEST_CASE("MUTATE-087_water_exp_adt_l10", "[ff][mutate][exp]") { runFixture(kFixtures[86]); }
TEST_CASE("MUTATE-088_water_exp_adt_l05", "[ff][mutate][exp]") { runFixture(kFixtures[87]); }
TEST_CASE("MUTATE-089_water_exp_adt_l00", "[ff][mutate][exp]") { runFixture(kFixtures[88]); }
TEST_CASE("MUTATE-090_water_exp_rdt_l10", "[ff][mutate][exp]") { runFixture(kFixtures[89]); }
TEST_CASE("MUTATE-091_water_exp_rdt_l05", "[ff][mutate][exp]") { runFixture(kFixtures[90]); }
TEST_CASE("MUTATE-092_water_exp_rdt_l00", "[ff][mutate][exp]") { runFixture(kFixtures[91]); }
TEST_CASE("MUTATE-093_water_inv_ast_l10", "[ff][mutate][inv]") { runFixture(kFixtures[92]); }
TEST_CASE("MUTATE-094_water_inv_ast_l05", "[ff][mutate][inv]") { runFixture(kFixtures[93]); }
TEST_CASE("MUTATE-095_water_inv_ast_l00", "[ff][mutate][inv]") { runFixture(kFixtures[94]); }
TEST_CASE("MUTATE-096_water_inv_adt_l10", "[ff][mutate][inv]") { runFixture(kFixtures[95]); }
TEST_CASE("MUTATE-097_water_inv_adt_l05", "[ff][mutate][inv]") { runFixture(kFixtures[96]); }
TEST_CASE("MUTATE-098_water_inv_adt_l00", "[ff][mutate][inv]") { runFixture(kFixtures[97]); }
TEST_CASE("MUTATE-099_water_inv_rdt_l10", "[ff][mutate][inv]") { runFixture(kFixtures[98]); }
TEST_CASE("MUTATE-100_water_inv_rdt_l05", "[ff][mutate][inv]") { runFixture(kFixtures[99]); }
TEST_CASE("MUTATE-101_water_inv_rdt_l00", "[ff][mutate][inv]") { runFixture(kFixtures[100]); }
TEST_CASE("MUTATE-102_water_exf_ast_m10", "[ff][mutate][exf]") { runFixture(kFixtures[101]); }
TEST_CASE("MUTATE-103_water_exf_ast_m05", "[ff][mutate][exf]") { runFixture(kFixtures[102]); }
TEST_CASE("MUTATE-104_water_exf_ast_m00", "[ff][mutate][exf]") { runFixture(kFixtures[103]); }
TEST_CASE("MUTATE-105_water_exf_adt_m10", "[ff][mutate][exf]") { runFixture(kFixtures[104]); }
TEST_CASE("MUTATE-106_water_exf_adt_m05", "[ff][mutate][exf]") { runFixture(kFixtures[105]); }
TEST_CASE("MUTATE-107_water_exf_adt_m00", "[ff][mutate][exf]") { runFixture(kFixtures[106]); }
TEST_CASE("MUTATE-108_water_exf_rdt_m10", "[ff][mutate][exf]") { runFixture(kFixtures[107]); }
TEST_CASE("MUTATE-109_water_exf_rdt_m05", "[ff][mutate][exf]") { runFixture(kFixtures[108]); }
TEST_CASE("MUTATE-110_water_exf_rdt_m00", "[ff][mutate][exf]") { runFixture(kFixtures[109]); }
TEST_CASE("MUTATE-111_water_exf_adt_p10", "[ff][mutate][exf]") { runFixture(kFixtures[110]); }
TEST_CASE("MUTATE-112_water_exf_adt_p05", "[ff][mutate][exf]") { runFixture(kFixtures[111]); }
TEST_CASE("MUTATE-113_water_exf_adt_p00", "[ff][mutate][exf]") { runFixture(kFixtures[112]); }
TEST_CASE("MUTATE-114_water_exf_rdt_p10", "[ff][mutate][exf]") { runFixture(kFixtures[113]); }
TEST_CASE("MUTATE-115_water_exf_rdt_p05", "[ff][mutate][exf]") { runFixture(kFixtures[114]); }
TEST_CASE("MUTATE-116_water_exf_rdt_p00", "[ff][mutate][exf]") { runFixture(kFixtures[115]); }
TEST_CASE("MUTATE-117_water_exf_adt_m05p10", "[ff][mutate][exf]") { runFixture(kFixtures[116]); }
TEST_CASE("MUTATE-118_water_exf_rdt_m05p05", "[ff][mutate][exf]") { runFixture(kFixtures[117]); }
TEST_CASE("MUTATE-119_water_adt_ye_m10p10v05", "[ff][mutate][emplar]") { runFixture(kFixtures[118]); }
TEST_CASE("MUTATE-120_water_adt_ne_m10p10v05", "[ff][mutate][emplar]") { runFixture(kFixtures[119]); }
TEST_CASE("MUTATE-121_water_adt_ye_m05p05v00", "[ff][mutate][emplar]") { runFixture(kFixtures[120]); }
TEST_CASE("MUTATE-122_water_adt_ne_m05p05v00", "[ff][mutate][emplar]") { runFixture(kFixtures[121]); }
TEST_CASE("MUTATE-123_water_adt_ye_m00p00v10", "[ff][mutate][emplar]") { runFixture(kFixtures[122]); }
TEST_CASE("MUTATE-124_water_adt_ne_m00p00v10", "[ff][mutate][emplar]") { runFixture(kFixtures[123]); }
TEST_CASE("MUTATE-125_water_rdt_ye_m10p10v05", "[ff][mutate][emplar]") { runFixture(kFixtures[124]); }
TEST_CASE("MUTATE-126_water_rdt_ne_m10p10v05", "[ff][mutate][emplar]") { runFixture(kFixtures[125]); }
TEST_CASE("MUTATE-127_water_rdt_ye_m05p05v00", "[ff][mutate][emplar]") { runFixture(kFixtures[126]); }
TEST_CASE("MUTATE-128_water_rdt_ne_m05p05v00", "[ff][mutate][emplar]") { runFixture(kFixtures[127]); }
TEST_CASE("MUTATE-129_water_rdt_ye_m00p00v10", "[ff][mutate][emplar]") { runFixture(kFixtures[128]); }
TEST_CASE("MUTATE-130_water_rdt_ne_m00p00v10", "[ff][mutate][emplar]") { runFixture(kFixtures[129]); }
TEST_CASE("MUTATE-131_water_qnt_adt_l10", "[ff][mutate][legskip]") { runLegSkipFixture(kFixtures[130], false, true); }
TEST_CASE("MUTATE-132_water_qnt_adt_l00", "[ff][mutate][legskip]") { runLegSkipFixture(kFixtures[131], true, false); }
TEST_CASE("MUTATE-133_water_qnt_rdt_l10", "[ff][mutate][legskip]") { runLegSkipFixture(kFixtures[132], false, true); }
TEST_CASE("MUTATE-134_water_qnt_rdt_l00", "[ff][mutate][legskip]") { runLegSkipFixture(kFixtures[133], true, false); }
TEST_CASE("MUTATE-135_water_rels_ye_l100", "[ff][mutate][rels]") { runFixture(kFixtures[134]); }
TEST_CASE("MUTATE-136_water_rels_ye_l085", "[ff][mutate][rels]") { runFixture(kFixtures[135]); }
TEST_CASE("MUTATE-137_water_rels_ye_l070", "[ff][mutate][rels]") { runFixture(kFixtures[136]); }
TEST_CASE("MUTATE-138_water_rels_ye_l050", "[ff][mutate][rels]") { runFixture(kFixtures[137]); }
TEST_CASE("MUTATE-139_water_rels_ye_l030", "[ff][mutate][rels]") { runFixture(kFixtures[138]); }
TEST_CASE("MUTATE-140_water_rels_ye_l015", "[ff][mutate][rels]") { runFixture(kFixtures[139]); }
TEST_CASE("MUTATE-141_water_rels_ye_l000", "[ff][mutate][rels]") { runFixture(kFixtures[140]); }

#endif
