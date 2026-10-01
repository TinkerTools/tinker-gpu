#include "ff/hippo/edisp.h"
#include "tool/platform.h"

#include "test.h"
#include "testrt.h"

using namespace tinker;

TEST_CASE("EDISP-1-NONDEWALD", "[ff][hippo][edisp][nondewald]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/c5h12acnh2/c5h12acnh2.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/disp/ndewald.key");
   TestFile fp1(TINKER9_DIRSTR "/test/file/disp/hippo19.prm");
   const char* xn = "c5h12acnh2.xyz";
   const char* kn = "ndewald.key";
   const char* argv[] = {"dummy", xn, "-k", kn};
   int argc = 4;

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(0.0001, 0.0001);
   const double eps_v = testGetEps(0.001, 0.001);

   TestReference r(TINKER9_DIRSTR "/test/ref/disp.1.txt");
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
   COMPARE_INTS(countReduce(ndisp), ref_c);

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

TEST_CASE("EDISP-2-DEWALD", "[ff][hippo][edisp][dewald]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/c5h12acnh2/c5h12acnh2.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/disp/dewald.key");
   TestFile fp1(TINKER9_DIRSTR "/test/file/disp/hippo19.prm");
   const char* xn = "c5h12acnh2.xyz";
   const char* kn = "dewald.key";
   const char* argv[] = {"dummy", xn, "-k", kn};
   int argc = 4;

   const double eps_e = testGetEps(0.0030, 0.0001);
   const double eps_g = testGetEps(0.0006, 0.0001);
   const double eps_v = testGetEps(0.0065, 0.001);

   TestReference r(TINKER9_DIRSTR "/test/ref/disp.2.txt");
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
   COMPARE_INTS(countReduce(ndisp), ref_c);

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

// No dispersion on the methyl H, so its excluded pairs have no energy. The
// neighbor loop never counts them, so the exclusion loop must not take them
// away either (edisp3.f).
TEST_CASE("EDISP-3-DEWALD-ZEROC6", "[ff][hippo][edisp][dewald]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/c5h12acnh2/c5h12acnh2.xyz");
   const char* xn = "c5h12acnh2.xyz";
   const char* kn = "dewald.key";
   std::string ke = "\ndispersion         2          0.0     4.3225\n";
   // The OpenACC build runs this on its own kernels, whose interaction count
   // this case exists to check.
#if TINKER_GPULANG_OPENACC
   ke += "gpu-package openacc\n";
#endif
   TestFile fk1(TINKER9_DIRSTR "/test/file/disp/dewald.key", kn, ke);
   TestFile fp1(TINKER9_DIRSTR "/test/file/disp/hippo19.prm");
   const char* argv[] = {"dummy", xn, "-k", kn};
   int argc = 4;

   const double eps_e = testGetEps(0.0030, 0.0001);
   // Mixed precision misses the double precision gradient by up to ~6.2e-4
   // here; the double build matches the reference.
   const double eps_g = testGetEps(0.0010, 0.0001);
   const double eps_v = testGetEps(0.0065, 0.001);

   TestReference r(TINKER9_DIRSTR "/test/ref/disp.3.txt");
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
   COMPARE_INTS(countReduce(ndisp), ref_c);

   session.end();
}
