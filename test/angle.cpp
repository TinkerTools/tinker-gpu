#include "ff/evalence.h"

#include "test.h"
#include "testrt.h"

using namespace tinker;

TEST_CASE("Angle-Trpcage", "[ff][eangle][trpcage]")
{
   const char* k = "test_trpcage.key";
   const char* k0 = "angleterm  only\n";
   const char* x1 = "test_trpcage.xyz";

   TestFile fke(TINKER9_DIRSTR "/test/file/trpcage/trpcage.key", k, k0);
   TestFile fx1(TINKER9_DIRSTR "/test/file/trpcage/trpcage.xyz", x1);
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_291a85c1/amoebapro13.prm");

   TestReference r(TINKER9_DIRSTR "/test/ref/angle.1.txt");
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_count = r.getCount();
   auto ref_g = r.getGradient();

   const double eps_e = 0.0001;
   const double eps_g = testGetEps(0.001, 0.0001);
   const double eps_v = testGetEps(0.003, 0.001);

   const char* argv[] = {"dummy", x1};
   int argc = 2;
   TestSession session(argc, argv);
   rc_flag = calc::xyz | calc::vmask;
   session.init();

   energy(calc::v3);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_INTS(nangle, ref_count);

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

TEST_CASE("Angle-2-fourier", "[ff][eangle][fourier][anglef]")
{
   const char* k1 = "test_anglef.key";
   const char* x1 = "test_anglef.xyz";

   TestFile fke(TINKER9_DIRSTR "/test/file/anglef/anglef.key", k1);
   TestFile fx1(TINKER9_DIRSTR "/test/file/anglef/anglef.xyz", x1);
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");

   TestReference r(TINKER9_DIRSTR "/test/ref/angle.2.txt");
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_count = r.getCount();
   auto ref_g = r.getGradient();

   const double eps_e = 0.0001;
   const double eps_g = testGetEps(0.0001, 0.0001);
   const double eps_v = testGetEps(0.001, 0.001);

   const char* argv[] = {"dummy", x1};
   int argc = 2;
   TestSession session(argc, argv);
   rc_flag = calc::xyz | calc::vmask;
   session.init();

   energy(calc::v3);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_INTS(nangle, ref_count);

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
