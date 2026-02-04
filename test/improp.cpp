#include "ff/evalence.h"

#include "test.h"
#include "testrt.h"

using namespace tinker;

TEST_CASE("Improp-Trpcage-Charmm19", "[ff][eimprop][trpcage]")
{
   const char* k1 = "test_trpcage.key";
   const char* k2 = "impropterm only\n";
   const char* x1 = "test_trpcage.xyz";

   TestFile fke(TINKER9_DIRSTR "/test/file/trpcage/trp_charmm.key", k1, k2);
   TestFile fxy(TINKER9_DIRSTR "/test/file/trpcage/trp_charmm.xyz", x1);
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_4f11c0d4/charmm19.prm");

   TestReference r(TINKER9_DIRSTR "/test/ref/improp.1.txt");
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_count = r.getCount();
   auto ref_g = r.getGradient();

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(0.0003, 0.0001);
   const double eps_v = testGetEps(0.001, 0.001);

   const char* argv[] = {"dummy", x1};
   int argc = 2;
   testBeginWithArgs(argc, argv);
   rc_flag = calc::xyz | calc::vmask;
   initialize();

   energy(calc::v3);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_INTS(niprop, ref_count);

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

TEST_CASE("Improp-BDNA3-Charmm36", "[ff][eimprop][bdna3]")
{
   const char* k1 = "test_bdna3.key";
   const char* k2 = "impropterm only\n";
   const char* x1 = "test_bdna3.xyz";

   TestFile fke(TINKER9_DIRSTR "/test/file/bdna3/bdna3.key", k1, k2);
   TestFile fxy(TINKER9_DIRSTR "/test/file/bdna3/bdna3.xyz", x1);
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_4f11c0d4/charmm36m.prm");

   TestReference r(TINKER9_DIRSTR "/test/ref/improp.2.txt");
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_count = r.getCount();
   auto ref_g = r.getGradient();

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(0.001, 0.0001);
   const double eps_v = testGetEps(0.001, 0.001);

   const char* argv[] = {"dummy", x1};
   int argc = 2;
   testBeginWithArgs(argc, argv);
   rc_flag = calc::xyz | calc::vmask;
   initialize();

   energy(calc::v3);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_INTS(niprop, ref_count);

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
