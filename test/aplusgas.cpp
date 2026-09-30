#include "test.h"
#include "testrt.h"

using namespace tinker;

TEST_CASE("APlus-Gas-Alyz", "[ff][aplus]")
{
   rc_flag = calc::xyz | calc::energy | calc::grad | calc::virial;
   rc_flag |= calc::analyz;

   const char* xn = "test_aplus.xyz";
   const char* kn = "test_aplus.key";

   TestFile fx1(TINKER9_DIRSTR "/test/file/aplus2022/EtOH-Wat-OH_0.70.xyz", xn);
   TestFile fk1(TINKER9_DIRSTR "/test/file/aplus2022/gas.key", kn);
   TestFile fp1(TINKER9_DIRSTR "/test/file/aplus2022/AMOEBAplus_Org.prm");

   TestReference r(TINKER9_DIRSTR "/test/ref/aplusgas.1.txt");
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_g = r.getGradient();

   const double eps_e = testGetEps(0.0030, 0.0001);
   const double eps_g = testGetEps(0.0013, 0.0001);
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

TEST_CASE("APlus-Gas", "[ff][aplus]")
{
   rc_flag = calc::xyz | calc::energy | calc::grad | calc::virial;

   const char* xn = "test_aplus.xyz";
   const char* kn = "test_aplus.key";

   TestFile fx1(TINKER9_DIRSTR "/test/file/aplus2022/EtOH-Wat-OH_0.70.xyz", xn);
   TestFile fk1(TINKER9_DIRSTR "/test/file/aplus2022/gas.key", kn);
   TestFile fp1(TINKER9_DIRSTR "/test/file/aplus2022/AMOEBAplus_Org.prm");

   TestReference r(TINKER9_DIRSTR "/test/ref/aplusgas.1.txt");
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_g = r.getGradient();

   const double eps_e = testGetEps(0.0030, 0.0001);
   const double eps_g = testGetEps(0.0013, 0.0001);
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
