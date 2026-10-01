#include "ff/evdw.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
#include "tool/platform.h"

#include "test.h"
#include "testrt.h"

#include <string>

using namespace tinker;

TEST_CASE("APlus-Liquid-Alyz", "[ff][aplus]")
{
   rc_flag = calc::xyz | calc::energy | calc::grad | calc::virial;
   rc_flag |= calc::analyz;

   const char* xn = "test_aplus.xyz";
   const char* kn = "test_aplus.key";
   const char* ke = "\n"
                    "ewald"
                    "\n";

   TestFile fx1(TINKER9_DIRSTR "/test/file/aplus2022/tetramer.xyz", xn);
   TestFile fk1(TINKER9_DIRSTR "/test/file/aplus2022/liquid.key", kn, ke);
   TestFile fp1(TINKER9_DIRSTR "/test/file/aplus2022/AMOEBAplus_Org.prm");

   TestReference r(TINKER9_DIRSTR "/test/ref/aplusliquid.1.txt");
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_g = r.getGradient();

   const double eps_e = testGetEps(0.0009, 0.0001);
   const double eps_g = testGetEps(0.0003, 0.0001);
   const double eps_v = testGetEps(0.001, 0.001);

   const char* argv[] = {"dummy", xn};
   int argc = 2;

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
   double eng;
   int cnt;
   r.getEnergyCountByName("Van der Waals", eng, cnt);
   COMPARE_COUNT(nev, cnt);
   COMPARE_ENERGY(ev, eng, eps_e);
   r.getEnergyCountByName("Atomic Multipoles", eng, cnt);
   COMPARE_COUNT(nem, cnt);
   COMPARE_ENERGY(em, eng, eps_e);
   r.getEnergyCountByName("Polarization", eng, cnt);
   COMPARE_COUNT(nep, cnt);
   COMPARE_ENERGY(ep, eng, eps_e);
   r.getEnergyCountByName("Charge Transfer", eng, cnt);
   COMPARE_COUNT(nct, cnt);
   COMPARE_ENERGY(ect, eng, eps_e);

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

TEST_CASE("APlus-Liquid", "[ff][aplus]")
{
   rc_flag = calc::xyz | calc::energy | calc::grad | calc::virial;

   const char* xn = "test_aplus.xyz";
   const char* kn = "test_aplus.key";
   const char* ke = "\n"
                    "ewald"
                    "\n";

   TestFile fx1(TINKER9_DIRSTR "/test/file/aplus2022/tetramer.xyz", xn);
   TestFile fk1(TINKER9_DIRSTR "/test/file/aplus2022/liquid.key", kn, ke);
   TestFile fp1(TINKER9_DIRSTR "/test/file/aplus2022/AMOEBAplus_Org.prm");

   TestReference r(TINKER9_DIRSTR "/test/ref/aplusliquid.1.txt");
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_g = r.getGradient();

   const double eps_e = testGetEps(0.0009, 0.0001);
   const double eps_g = testGetEps(0.0003, 0.0001);
   const double eps_v = testGetEps(0.001, 0.001);

   const char* argv[] = {"dummy", xn};
   int argc = 2;

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

TEST_CASE("APlus-Liquid-NonEwald", "[ff][aplus]")
{
   rc_flag = calc::xyz | calc::energy | calc::grad | calc::virial;
   rc_flag |= calc::analyz;

   const char* xn = "test_aplus.xyz";
   const char* kn = "test_aplus.key";

   TestFile fx1(TINKER9_DIRSTR "/test/file/aplus2022/tetramer.xyz", xn);
   TestFile fk1(TINKER9_DIRSTR "/test/file/aplus2022/liquid.key", kn);
   TestFile fp1(TINKER9_DIRSTR "/test/file/aplus2022/AMOEBAplus_Org.prm");

   TestReference r(TINKER9_DIRSTR "/test/ref/aplusliquid.2.txt");
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_g = r.getGradient();

   const double eps_e = testGetEps(0.0060, 0.0001);
   const double eps_g = testGetEps(0.0008, 0.0001);
   const double eps_v = testGetEps(0.002, 0.001);

   const char* argv[] = {"dummy", xn};
   int argc = 2;

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
   double eng;
   int cnt;
   r.getEnergyCountByName("Van der Waals", eng, cnt);
   COMPARE_COUNT(nev, cnt);
   COMPARE_ENERGY(ev, eng, eps_e);
   r.getEnergyCountByName("Atomic Multipoles", eng, cnt);
   COMPARE_COUNT(nem, cnt);
   COMPARE_ENERGY(em, eng, eps_e);
   r.getEnergyCountByName("Polarization", eng, cnt);
   COMPARE_COUNT(nep, cnt);
   COMPARE_ENERGY(ep, eng, eps_e);
   r.getEnergyCountByName("Charge Transfer", eng, cnt);
   COMPARE_COUNT(nct, cnt);
   COMPARE_ENERGY(ect, eng, eps_e);

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
// Runs the tetramer under liquid.key with Ewald and \c keyextra appended, and
// compares every term, count, gradient and virial to \c ref.
void runAplusLiquidCase(const char* keyextra, const char* ref)
{
   rc_flag = calc::xyz | calc::energy | calc::grad | calc::virial;
   rc_flag |= calc::analyz;

   const char* xn = "test_aplus.xyz";
   const char* kn = "test_aplus.key";
   std::string ke = std::string("\n"
                                "ewald\n")
      + keyextra;
   // The OpenACC build runs this on its own kernels, whose interaction count
   // these cases check.
#if TINKER_GPULANG_OPENACC
   ke += "gpu-package openacc\n";
#endif

   TestFile fx1(TINKER9_DIRSTR "/test/file/aplus2022/tetramer.xyz", xn);
   TestFile fk1(TINKER9_DIRSTR "/test/file/aplus2022/liquid.key", kn, ke);
   TestFile fp1(TINKER9_DIRSTR "/test/file/aplus2022/AMOEBAplus_Org.prm");

   TestReference r(std::string(TINKER9_DIRSTR) + ref);
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_g = r.getGradient();

   const double eps_e = testGetEps(0.0009, 0.0001);
   const double eps_g = testGetEps(0.0003, 0.0001);
   const double eps_v = testGetEps(0.001, 0.001);

   const char* argv[] = {"dummy", xn};
   int argc = 2;

   TestSession session(argc, argv);
   session.init();
#if TINKER_GPULANG_OPENACC
   REQUIRE(pltfm_config & Platform::ACC);
#endif

   energy(calc::v1);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_GRADIENT(ref_g, eps_g);
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
         COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_v);

   energy(calc::v3);
   COMPARE_REALS(esum, ref_e, eps_e);
   double eng;
   int cnt;
   r.getEnergyCountByName("Van der Waals", eng, cnt);
   COMPARE_COUNT(nev, cnt);
   COMPARE_ENERGY(ev, eng, eps_e);
   r.getEnergyCountByName("Atomic Multipoles", eng, cnt);
   COMPARE_COUNT(nem, cnt);
   COMPARE_ENERGY(em, eng, eps_e);
   r.getEnergyCountByName("Polarization", eng, cnt);
   COMPARE_COUNT(nep, cnt);
   COMPARE_ENERGY(ep, eng, eps_e);
   r.getEnergyCountByName("Charge Transfer", eng, cnt);
   COMPARE_COUNT(nct, cnt);
   COMPARE_ENERGY(ect, eng, eps_e);

   session.end();
}
}

// No polarizability on the propanol H (type 17). Its pairs with one another
// have no polarization energy and must not count as interactions (epolar3.f),
// and the direct field its zero damping width leaves undamped (damptholed).
TEST_CASE("APlus-Liquid-ZeroPol", "[ff][aplus]")
{
   runAplusLiquidCase("polarize  17  0.0  0.3900  0.7000  14\n", "/test/ref/aplusliquid.3.txt");
}

// The damping pair values come from Tinker's tables (kpolar.f), not from the
// per-atom values: a zero falls back to the partner's value, and polpair
// overrides both. A pair value of 0 means "no damping" (damping.f).

// Thole 0 on type 17: its pairs take the partner's Thole.
TEST_CASE("APlus-Liquid-ZeroThole", "[ff][aplus]")
{
   runAplusLiquidCase("polarize  17  0.4800  0.0000  0.7000  14\n", "/test/ref/aplusliquid.4.txt");
}

// Direct damping 0 on type 17: its pairs take the partner's direct damping.
TEST_CASE("APlus-Liquid-ZeroDirDamp", "[ff][aplus]")
{
   runAplusLiquidCase("polarize  17  0.4800  0.3900  0.0000  14\n", "/test/ref/aplusliquid.5.txt");
}

// Pair values that differ from the combining rule.
TEST_CASE("APlus-Liquid-PolPair", "[ff][aplus]")
{
   runAplusLiquidCase("polpair  13  17  0.1000  0.4000\n", "/test/ref/aplusliquid.6.txt");
}

// Both pair values 0: the 13-17 pairs are undamped.
TEST_CASE("APlus-Liquid-ZeroPair", "[ff][aplus]")
{
   runAplusLiquidCase("polpair  13  17  0.0  0.0\n", "/test/ref/aplusliquid.7.txt");
}

// Both pair values 1e-30: the 13-17 pairs are fully damped, and their nonzero
// Ewald real-space energy must not count them as interactions (epolar3.f).
TEST_CASE("APlus-Liquid-TinyPair", "[ff][aplus]")
{
   runAplusLiquidCase("polpair  13  17  1e-30  1e-30\n", "/test/ref/aplusliquid.8.txt");
}
