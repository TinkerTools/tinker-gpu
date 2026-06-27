#include "ff/evdw.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"

#include "test.h"
#include "testrt.h"

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

   testBeginWithArgs(argc, argv);
   initialize();

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

   finish();
   testEnd();
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

   testBeginWithArgs(argc, argv);
   initialize();

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

   finish();
   testEnd();
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

   testBeginWithArgs(argc, argv);
   initialize();

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

   finish();
   testEnd();
}
