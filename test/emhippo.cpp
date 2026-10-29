#include "ff/modamoeba.h"

#include "test.h"
#include "testrt.h"

using namespace tinker;

TEST_CASE("EMPOLE-1-EWALD-HIPPO", "[ff][hippo][emhippo][ewald]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/c5h12acnh2/c5h12acnh2.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/hippo/mpole/ewald.key");
   TestFile fp1(TINKER9_DIRSTR "/test/file/hippo/hippo19.prm");
   const char* xn = "c5h12acnh2.xyz";
   const char* kn = "ewald.key";
   const char* argv[] = {"dummy", xn, "-k", kn};
   int argc = 4;

   const double eps_e = testGetEps(0.0003, 0.0001);
   const double eps_g = testGetEps(0.0002, 0.0001);
   const double eps_v = testGetEps(0.001, 0.001);

   TestReference r(TINKER9_DIRSTR "/test/ref/emhippo.1.txt");
   auto ref_c = r.getCount();
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
   COMPARE_INTS(countReduce(nem), ref_c);

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

TEST_CASE("EMPOLE-2-NONEWALD-HIPPO", "[ff][hippo][emhippo][nonewald]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/c5h12acnh2/c5h12acnh2.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/hippo/mpole/newald.key");
   TestFile fp1(TINKER9_DIRSTR "/test/file/hippo/hippo19.prm");
   const char* xn = "c5h12acnh2.xyz";
   const char* kn = "newald.key";
   const char* argv[] = {"dummy", xn, "-k", kn};
   int argc = 4;

   const double eps_e = testGetEps(0.0003, 0.0001);
   const double eps_g = testGetEps(0.0002, 0.0001);
   const double eps_v = testGetEps(0.001, 0.001);

   TestReference r(TINKER9_DIRSTR "/test/ref/emhippo.2.txt");
   auto ref_c = r.getCount();
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
   COMPARE_INTS(countReduce(nem), ref_c);

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
