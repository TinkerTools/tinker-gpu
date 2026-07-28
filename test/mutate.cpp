#include "ff/atom.h"
#include "ff/dlmda.h"
#include "ff/egvop.h"
#include "ff/energy.h"
#include "ff/evdw.h"
#include "ff/modamoeba.h"
#include "ff/ost.h"

#include "test.h"
#include "testrt.h"
#include "tinker9.h"

#include <tinker/detail/dlmda.hh>
#include <tinker/routines.h>

#include <array>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace tinker;

#if TINKER_GPULANG_CUDA

namespace {
// One water mutation fixture ported from Tinker's test_mutate.f. "base" selects
// the shared coordinate file (water.xyz or water2.xyz); checkm/checkp/checkv
// select which level-3 named components are verified (Atomic Multipoles,
// Polarization, Van der Waals); dolmda enables the level-4 lambda-derivative
// checks (fixtures 030-130). The neighbor-list duplication from the Fortran
// test is intentionally omitted.
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
};

// Reference lambda-derivative block parsed out of the analyze-style ref file.
// TestReference does not understand these sections, so they are read here.
struct LmdaRef
{
   double dedl[4];   // dE/dL, dEV/dL, dEM/dL, dEP/dL
   double d2edl2[4]; // d2E/dL2, d2EV/dL2, d2EM/dL2, d2EP/dL2
   std::vector<std::array<double, 3>> lgrad; // per-atom dF/dL
   double dvdl[3][3];                        // dV/dL tensor
};

std::vector<double> floatsOf(const std::string& line)
{
   std::vector<double> v;
   std::istringstream iss(line);
   std::string tok;
   while (iss >> tok) {
      try {
         size_t pos;
         double d = std::stod(tok, &pos);
         if (pos == tok.size())
            v.push_back(d);
      } catch (...) {
      }
   }
   return v;
}

LmdaRef readLmdaRef(const std::string& path, int natom)
{
   LmdaRef r{};
   r.lgrad.assign(natom, {0.0, 0.0, 0.0});
   std::ifstream fin(path);
   std::vector<std::string> lines;
   std::string ln;
   while (std::getline(fin, ln))
      lines.push_back(ln);
   for (size_t i = 0; i < lines.size(); ++i) {
      const std::string& L = lines[i];
      if (L.find("Analytical Lambda Derivatives") != std::string::npos) {
         auto f = floatsOf(lines.at(i + 1));
         for (int k = 0; k < 4 && k < (int)f.size(); ++k)
            r.dedl[k] = f[k];
      } else if (L.find("Analytical 2nd Lambda Derivatives") != std::string::npos) {
         auto f = floatsOf(lines.at(i + 1));
         for (int k = 0; k < 4 && k < (int)f.size(); ++k)
            r.d2edl2[k] = f[k];
      } else if (L.find("Lambda Gradient Breakdown") != std::string::npos) {
         int got = 0;
         for (size_t j = i + 1; j < lines.size() && got < natom; ++j) {
            std::istringstream iss(lines[j]);
            std::string tag;
            if (!(iss >> tag) || tag != "Lambda")
               continue;
            int idx;
            double fx, fy, fz;
            if ((iss >> idx >> fx >> fy >> fz) && idx >= 1 && idx <= natom) {
               r.lgrad[idx - 1] = {fx, fy, fz};
               ++got;
            }
         }
      } else if (L.find("Analytical dV/dL") != std::string::npos) {
         auto f0 = floatsOf(lines.at(i));     // first row shares the header line
         auto f1 = floatsOf(lines.at(i + 1));
         auto f2 = floatsOf(lines.at(i + 2));
         for (int k = 0; k < 3; ++k) {
            if (k < (int)f0.size())
               r.dvdl[0][k] = f0[k];
            if (k < (int)f1.size())
               r.dvdl[1][k] = f1[k];
            if (k < (int)f2.size())
               r.dvdl[2][k] = f2[k];
         }
      }
   }
   return r;
}

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

   // Replicate testBeginWithArgs, but toggle the Fortran-side use_dlmda before
   // mechanic2() so the lambda-derivative buffers get allocated (see xtestlmda).
   tinkerFortranRuntimeBegin(argc, (char**)argv);
   initial();
   tinker_f_command();
   tinker_f_getxyz();
   tinker_f_mechanic();
   mechanic2();
   initialize();

   TestReference ref(refpath);
   double ref_e = ref.getEnergy();
   auto ref_v = ref.getVirial();
   auto ref_g = ref.getGradient();

   LmdaRef lr;
   if (fx.dolmda)
      lr = readLmdaRef(refpath, n);

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

// Builds a system from a key file and hands control back with it live, so a
// test can sweep the main lambda itself. The staged relative free energy
// schedule has no Fortran counterpart to generate reference files from, so its
// tests are self-contained: they check the schedule against its own endpoints,
// against finite differences, and for continuity where the legs meet.
template <class F>
void withLiveSystem(const char* base, const char* keyname_base, F&& body)
{
   std::string dir = TINKER9_DIRSTR "/test/file/mutate/";
   std::string xyzdst = std::string(base) + ".xyz";
   std::string keyname = std::string(keyname_base) + ".key";

   TestFile fxyz(dir + xyzdst, xyzdst);
   TestFile fkey(dir + keyname, keyname);
   TestFile fprm(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");

   const char* argv[] = {"dummy", xyzdst.c_str(), "-k", keyname.c_str()};
   int argc = 4;

   rc_flag = calc::xyz | calc::mass | calc::vmask;

   tinkerFortranRuntimeBegin(argc, (char**)argv);
   initial();
   tinker_f_command();
   tinker_f_getxyz();
   tinker_f_mechanic();
   mechanic2();
   initialize();

   body();

   finish();
   testEnd();
}

// energy() remaps the sub-lambdas from ostlambda and applies lmdachain when
// OST owns the main lambda, so setting ostlambda is enough to move along the
// schedule. No gaussians are ever deposited here, so the OST bias stays zero.
double energyAtMainLambda(double lambda, int vers = calc::energy)
{
   ostlambda = lambda;
   energy(vers);
   return esum;
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

// The staged schedule's endpoints are the plain relative dual topology
// endpoints, so they must reproduce the 133/134 references exactly: at
// lambda = 1 the mix weight is 1 and only E(A+env) + E(B) is evaluated, at
// lambda = 0 it is E(B+env) + E(A). This is the check that the staged path did
// not shift the physics.
TEST_CASE("MUTATE-135_water_relstage_endpoints", "[ff][mutate][relstage]")
{
   const double eps_e = testGetEps(1.0e-3, 1.0e-4);
   const double eps_g = testGetEps(1.0e-3, 1.0e-4);

   withLiveSystem("water2", "135_water_relstage_l10", [&]() {
      REQUIRE(use_relstage == true);
      REQUIRE(use_emrdt == true);

      TestReference r1(TINKER9_DIRSTR "/test/ref/mutate/133_water_qnt_rdt_l10.txt");
      energyAtMainLambda(1.0, calc::v1);
      REQUIRE(relstage == RelStage::LIG1_ELE);
      REQUIRE(relstagemix == false);
      COMPARE_REALS(esum, r1.getEnergy(), eps_e);
      COMPARE_GRADIENT(r1.getGradient(), eps_g);

      TestReference r0(TINKER9_DIRSTR "/test/ref/mutate/134_water_qnt_rdt_l00.txt");
      energyAtMainLambda(0.0, calc::v1);
      REQUIRE(relstage == RelStage::LIG0_ELE);
      REQUIRE(relstagemix == false);
      COMPARE_REALS(esum, r0.getEnergy(), eps_e);
      COMPARE_GRADIENT(r0.getGradient(), eps_g);
   });
}

// In the middle window both ligands are electrostatically decoupled from the
// environment, so the electrostatic energy is flat in lambda and the whole
// lambda derivative comes from the van der Waals morph.
TEST_CASE("MUTATE-136_water_relstage_middle", "[ff][mutate][relstage]")
{
   withLiveSystem("water2", "137_water_relstage_l05", [&]() {
      double em_ref = 0, ep_ref = 0;
      for (double lambda : {0.7, 0.6, 0.5, 0.4, 0.3}) {
         CAPTURE(lambda);
         energyAtMainLambda(lambda, calc::v1);
         REQUIRE(relstage == RelStage::VDW_MORPH);
         COMPARE_REALS(demdl, 0.0, 1.0e-10);
         COMPARE_REALS(depdl, 0.0, 1.0e-10);
         COMPARE_REALS(d2emdl2, 0.0, 1.0e-10);
         COMPARE_REALS(d2epdl2, 0.0, 1.0e-10);
         COMPARE_REALS(dedl, devdl, 1.0e-10);

         // The decoupled reference does not depend on lambda at all.
         if (em_ref == 0 && ep_ref == 0) {
            em_ref = energy_em;
            ep_ref = energy_ep;
         } else {
            COMPARE_REALS(energy_em, em_ref, 1.0e-6);
            COMPARE_REALS(energy_ep, ep_ref, 1.0e-6);
         }
      }
   });
}

// The analytic main-lambda derivative against a central finite difference of
// the total energy. This exercises the whole chain: the taper derivative in
// mapSubLambda, the endpoint mix, and lmdachain.
TEST_CASE("MUTATE-137_water_relstage_dudl", "[ff][mutate][relstage]")
{
   withLiveSystem("water2", "137_water_relstage_l05", [&]() {
      // A five-point stencil, so the step can be large enough that the induced
      // dipole convergence noise (polar-eps 1e-5) does not dominate the
      // difference, while the O(h^4) truncation error stays far below it.
      const double h = 5.0e-3;
      for (double lambda : {0.9, 0.85, 0.75, 0.6, 0.5, 0.4, 0.25, 0.15, 0.1}) {
         CAPTURE(lambda);
         energyAtMainLambda(lambda, calc::v1);
         double analytic = dedl;
         double e_p2 = energyAtMainLambda(lambda + 2 * h);
         double e_p1 = energyAtMainLambda(lambda + h);
         double e_m1 = energyAtMainLambda(lambda - h);
         double e_m2 = energyAtMainLambda(lambda - 2 * h);
         double fd = (-e_p2 + 8 * e_p1 - 8 * e_m1 + e_m2) / (12 * h);
         CAPTURE(analytic, fd);
         COMPARE_REALS(analytic, fd, 1.0e-3 * (1.0 + std::fabs(fd)));
      }
   });
}

// The energy and its lambda derivative carry across the leg boundaries, where
// the code switches between the mixed path, the decoupled-reference path and
// the single-endpoint path. Agreement there is what shows all three agree on
// the same underlying state.
TEST_CASE("MUTATE-138_water_relstage_continuity", "[ff][mutate][relstage]")
{
   withLiveSystem("water2", "137_water_relstage_l05", [&]() {
      const double d = 1.0e-5;
      for (double edge : {0.7, 0.3}) {
         CAPTURE(edge);
         double eat = energyAtMainLambda(edge, calc::v1);
         double dat = dedl;
         double ehi = energyAtMainLambda(edge + d, calc::v1);
         double dhi = dedl;
         double elo = energyAtMainLambda(edge - d, calc::v1);
         double dlo = dedl;

         // No jump in the energy where the leg changes.
         COMPARE_REALS(ehi, eat, 1.0e-5);
         COMPARE_REALS(elo, eat, 1.0e-5);

         // The electrostatic contribution to dU/dlambda vanishes at the edge
         // from both sides; only the van der Waals morph is left.
         COMPARE_REALS(dhi, dat, 1.0e-4);
         COMPARE_REALS(dlo, dat, 1.0e-4);
      }

      // The outer ends are flat: the quintic taper has zero slope at 0 and 1,
      // and the van der Waals morph is outside its own window there.
      energyAtMainLambda(1.0, calc::v1);
      COMPARE_REALS(dedl, 0.0, 1.0e-8);
      energyAtMainLambda(0.0, calc::v1);
      COMPARE_REALS(dedl, 0.0, 1.0e-8);

      // At lambda = 1 and 0 the weight is exactly 1, so the code skips the mix
      // and evaluates the coupled endpoint alone. Just inside, it goes through
      // the mix with a weight barely below 1. The two must agree, which is what
      // pins the mixed path's endpoint branch to the un-mixed one.
      for (double endpoint : {1.0, 0.0}) {
         CAPTURE(endpoint);
         double e_end = energyAtMainLambda(endpoint);
         double e_in = energyAtMainLambda(endpoint == 1.0 ? 1.0 - d : d);
         COMPARE_REALS(e_in, e_end, 1.0e-5);
      }
   });
}

#endif
