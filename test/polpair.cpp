#include "ff/modamoeba.h"
#include "tool/platform.h"

#include "test.h"
#include "testrt.h"

#include <memory>
#include <string>

using namespace tinker;

static const char* xn = "test_polpair.xyz";
static const char* kn = "test_polpair.key";

TEST_CASE("PolPair-Ewald", "[ff][amoeba][polpair]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/polpair/nacl.xyz", xn);
   TestFile fk1(TINKER9_DIRSTR "/test/file/polpair/nacl.key", kn,
      "\n"
      "ewald"
      "\n");
   TestFile fp1(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");

   const char* argv[] = {"dummy", xn};
   int argc = 2;

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(0.0001, 0.0001);
   const double eps_v = testGetEps(0.001, 0.001);

   TestReference r(TINKER9_DIRSTR "/test/ref/polpair.1.txt");
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_g = r.getGradient();

   rc_flag = calc::xyz | calc::energy | calc::grad | calc::virial;
   TestSession session(argc, argv);
   session.init();

   energy(calc::v0);
   COMPARE_REALS(esum, ref_e, eps_e);

   energy(calc::v1);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_GRADIENT(ref_g, eps_g);
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
         COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_v);

   energy(calc::v4);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_GRADIENT(ref_g, eps_g);

   energy(calc::v5);
   COMPARE_GRADIENT(ref_g, eps_g);

   energy(calc::v6);
   COMPARE_GRADIENT(ref_g, eps_g);
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
         COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_v);

   session.end();
}

TEST_CASE("PolPair-NonEwald", "[ff][amoeba][polpair]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/polpair/nacl.xyz", xn);
   TestFile fk1(TINKER9_DIRSTR "/test/file/polpair/nacl.key", kn);
   TestFile fp1(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");

   const char* argv[] = {"dummy", xn};
   int argc = 2;

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(0.0001, 0.0001);
   const double eps_v = testGetEps(0.001, 0.001);

   TestReference r(TINKER9_DIRSTR "/test/ref/polpair.2.txt");
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_g = r.getGradient();

   rc_flag = calc::xyz | calc::energy | calc::grad | calc::virial;
   TestSession session(argc, argv);
   session.init();

   energy(calc::v0);
   COMPARE_REALS(esum, ref_e, eps_e);

   energy(calc::v1);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_GRADIENT(ref_g, eps_g);
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
         COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_v);

   energy(calc::v4);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_GRADIENT(ref_g, eps_g);

   energy(calc::v5);
   COMPARE_GRADIENT(ref_g, eps_g);

   energy(calc::v6);
   COMPARE_GRADIENT(ref_g, eps_g);
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
         COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_v);

   session.end();
}

TEST_CASE("PolPair-Ewald-A", "[ff][amoeba][polpair]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/polpair/nacl.xyz", xn);
   TestFile fk1(TINKER9_DIRSTR "/test/file/polpair/nacl.key", kn,
      "\n"
      "ewald"
      "\n");
   TestFile fp1(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");

   const char* argv[] = {"dummy", xn};
   int argc = 2;

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(0.0001, 0.0001);
   const double eps_v = testGetEps(0.001, 0.001);

   TestReference r(TINKER9_DIRSTR "/test/ref/polpair.1.txt");
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_g = r.getGradient();

   rc_flag = calc::xyz | calc::vmask;
   TestSession session(argc, argv);
   session.init();

   energy(calc::v0);
   COMPARE_REALS(esum, ref_e, eps_e);

   energy(calc::v1);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_GRADIENT(ref_g, eps_g);
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
         COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_v);

   energy(calc::v3);
   COMPARE_REALS(esum, ref_e, eps_e);

   energy(calc::v4);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_GRADIENT(ref_g, eps_g);

   energy(calc::v5);
   COMPARE_GRADIENT(ref_g, eps_g);

   energy(calc::v6);
   COMPARE_GRADIENT(ref_g, eps_g);
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
         COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_v);

   session.end();
}

TEST_CASE("PolPair-NonEwald-A", "[ff][amoeba][polpair]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/polpair/nacl.xyz", xn);
   TestFile fk1(TINKER9_DIRSTR "/test/file/polpair/nacl.key", kn);
   TestFile fp1(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");

   const char* argv[] = {"dummy", xn};
   int argc = 2;

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(0.0001, 0.0001);
   const double eps_v = testGetEps(0.001, 0.001);

   TestReference r(TINKER9_DIRSTR "/test/ref/polpair.2.txt");
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_g = r.getGradient();

   rc_flag = calc::xyz | calc::vmask;
   TestSession session(argc, argv);
   session.init();

   energy(calc::v0);
   COMPARE_REALS(esum, ref_e, eps_e);

   energy(calc::v1);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_GRADIENT(ref_g, eps_g);
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
         COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_v);

   energy(calc::v3);
   COMPARE_REALS(esum, ref_e, eps_e);

   energy(calc::v4);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_GRADIENT(ref_g, eps_g);

   energy(calc::v5);
   COMPARE_GRADIENT(ref_g, eps_g);

   energy(calc::v6);
   COMPARE_GRADIENT(ref_g, eps_g);
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
         COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_v);

   session.end();
}

namespace {
// A system whose Thole pair values come out 0 or vanishingly small. Tinker's
// pair tables (kpolar.f) treat 0 as "no damping specified", so such a pair is
// undamped (damping.f), while 1e-30 damps it fully; neither one is a physical
// limit of the other.
struct TholeCase
{
   const char* xyz;
   const char* key;
   const char* prm; ///< nullptr when the key carries its own parameters.
   const char* keyextra;
   const char* ref;
   double epsGradMixed = 0.0001; ///< Gradient tolerance in mixed precision.
};

const TholeCase kWaterZeroPair = {"/test/file/mutate/water.xyz", "/test/file/polpair/water.key",
   "/test/file/commit_6fe8e913/water03.prm", "polpair 2 2 0.0 0.0\n", "/test/ref/polpair.3.txt"};
// The H Thole of 0 falls back to the O value in the O-H pairs, and leaves the
// H-H pair value 0, so the pair tables match kWaterZeroPair.
const TholeCase kWaterZeroThole = {"/test/file/mutate/water.xyz", "/test/file/polpair/water.key",
   "/test/file/commit_6fe8e913/water03.prm", "polarize 2 0.496 0.0 1\n", "/test/ref/polpair.3.txt"};
// The fully damped H-H pairs have no interaction, which their nonzero Ewald
// real-space energy must not hide from the count (epolar3.f).
const TholeCase kWaterTinyPair = {"/test/file/mutate/water.xyz", "/test/file/polpair/water.key",
   "/test/file/commit_6fe8e913/water03.prm", "polpair 2 2 1e-30 0.0\n", "/test/ref/polpair.4.txt"};
// The shipped Dang-Chang model: undamped point dipoles, written as Thole 0. Its
// M site sits 0.215 Ang from O, so the field sums cancel heavily and mixed
// precision loses about 2e-4 in the gradient.
const TholeCase kDang = {"/test/file/polpair/dang.xyz", "/test/file/polpair/dang.key", nullptr, "",
   "/test/ref/polpair.5.txt", 0.0005};

// Without analyz, a CUDA build evaluates multipoles and polarization in the
// fused emplar kernel; with it, in the separate kernels that count.
void runTholeCase(const TholeCase& tc, bool analyz)
{
   std::string ke = std::string("\n") + tc.keyextra;
   // The OpenACC build runs this on its own kernels.
#if TINKER_GPULANG_OPENACC
   ke += "gpu-package openacc\n";
#endif
   TestFile fx1(std::string(TINKER9_DIRSTR) + tc.xyz, xn);
   TestFile fk1(std::string(TINKER9_DIRSTR) + tc.key, kn, ke);
   std::unique_ptr<TestFile> fp1;
   if (tc.prm)
      fp1 = std::make_unique<TestFile>(std::string(TINKER9_DIRSTR) + tc.prm);

   const char* argv[] = {"dummy", xn};
   int argc = 2;

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(tc.epsGradMixed, 0.0001);
   const double eps_v = testGetEps(0.001, 0.001);

   TestReference r(std::string(TINKER9_DIRSTR) + tc.ref);
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_g = r.getGradient();

   rc_flag = calc::xyz | (analyz ? calc::vmask : calc::energy | calc::grad | calc::virial);
   TestSession session(argc, argv);
   session.init();
#if TINKER_GPULANG_OPENACC
   REQUIRE(pltfm_config & Platform::ACC);
#endif

   energy(calc::v0);
   COMPARE_REALS(esum, ref_e, eps_e);

   energy(calc::v1);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_GRADIENT(ref_g, eps_g);
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
         COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_v);

   if (analyz) {
      energy(calc::v3);
      COMPARE_REALS(esum, ref_e, eps_e);
      double eng;
      int cnt;
      r.getEnergyCountByName("Polarization", eng, cnt);
      COMPARE_COUNT(nep, cnt);
      COMPARE_ENERGY(ep, eng, eps_e);
   }

   energy(calc::v4);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_GRADIENT(ref_g, eps_g);

   energy(calc::v5);
   COMPARE_GRADIENT(ref_g, eps_g);

   energy(calc::v6);
   COMPARE_GRADIENT(ref_g, eps_g);
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
         COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_v);

   session.end();
}
}

TEST_CASE("PolPair-ZeroPair", "[ff][amoeba][polpair]")
{
   runTholeCase(kWaterZeroPair, false);
}
TEST_CASE("PolPair-ZeroPair-A", "[ff][amoeba][polpair]")
{
   runTholeCase(kWaterZeroPair, true);
}
TEST_CASE("PolPair-ZeroThole", "[ff][amoeba][polpair]")
{
   runTholeCase(kWaterZeroThole, false);
}
TEST_CASE("PolPair-ZeroThole-A", "[ff][amoeba][polpair]")
{
   runTholeCase(kWaterZeroThole, true);
}
TEST_CASE("PolPair-TinyPair", "[ff][amoeba][polpair]")
{
   runTholeCase(kWaterTinyPair, false);
}
TEST_CASE("PolPair-TinyPair-A", "[ff][amoeba][polpair]")
{
   runTholeCase(kWaterTinyPair, true);
}
TEST_CASE("PolPair-Dang", "[ff][amoeba][polpair]")
{
   runTholeCase(kDang, false);
}
TEST_CASE("PolPair-Dang-A", "[ff][amoeba][polpair]")
{
   runTholeCase(kDang, true);
}
