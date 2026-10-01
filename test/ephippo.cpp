#include "ff/modamoeba.h"
#include "tool/platform.h"

#include "test.h"
#include "testrt.h"

using namespace tinker;

TEST_CASE("EPOLAR-1-EWALD-HIPPO", "[ff][hippo][ephippo][ewald]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/dmso/dmso.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/hippo/polar/ewald.key");
   TestFile fp1(TINKER9_DIRSTR "/test/file/hippo/hippo19.prm");
   const char* xn = "dmso.xyz";
   const char* kn = "ewald.key";
   const char* argv[] = {"dummy", xn, "-k", kn};
   int argc = 4;

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(0.0001, 0.0001);
   const double eps_v = testGetEps(0.001, 0.001);

   TestReference r(TINKER9_DIRSTR "/test/ref/ephippo.1.txt");
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
   COMPARE_INTS(countReduce(nep), ref_c);

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

TEST_CASE("EPOLAR-2-NONEWALD-HIPPO", "[ff][hippo][ephippo][nonewald]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/dmso/dmso.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/hippo/polar/newald.key");
   TestFile fp1(TINKER9_DIRSTR "/test/file/hippo/hippo19.prm");
   const char* xn = "dmso.xyz";
   const char* kn = "newald.key";
   const char* argv[] = {"dummy", xn, "-k", kn};
   int argc = 4;

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(0.0001, 0.0001);
   const double eps_v = testGetEps(0.001, 0.001);

   TestReference r(TINKER9_DIRSTR "/test/ref/ephippo.2.txt");
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
   COMPARE_INTS(countReduce(nep), ref_c);

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

// No polarizability on the methyl H, so every H-H pair has no polarization
// energy and must not count as an interaction (epolar3.f).
TEST_CASE("EPOLAR-3-EWALD-HIPPO-ZEROPOL", "[ff][hippo][ephippo][ewald]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/dmso/dmso.xyz");
   const char* xn = "dmso.xyz";
   const char* kn = "ewald.key";
   std::string ke = "\npolarize         162          0.0          161\n";
   // The OpenACC build runs this on its own kernels, whose interaction count
   // this case exists to check.
#if TINKER_GPULANG_OPENACC
   ke += "gpu-package openacc\n";
#endif
   TestFile fk1(TINKER9_DIRSTR "/test/file/hippo/polar/ewald.key", kn, ke);
   TestFile fp1(TINKER9_DIRSTR "/test/file/hippo/hippo19.prm");
   const char* argv[] = {"dummy", xn, "-k", kn};
   int argc = 4;

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(0.0001, 0.0001);
   const double eps_v = testGetEps(0.001, 0.001);

   TestReference r(TINKER9_DIRSTR "/test/ref/ephippo.3.txt");
   auto ref_c = r.getCount();
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_g = r.getGradient();

   rc_flag = calc::xyz | calc::vmask;
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
   COMPARE_INTS(countReduce(nep), ref_c);

   session.end();
}
