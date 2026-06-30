#include "ff/ennintermol.h"

#include "test.h"
#include "testrt.h"

using namespace tinker;

#if TINKER_GPULANG_CUDA

TEST_CASE("NNMET-1-CU-WATER", "[ff][nn][ennmet][cu-water]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/nn/nn_6o.xyz", "nn_6o.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/nn/nn_6o.key", "nn_6o.key");
   TestFile fp1(TINKER9_DIRSTR "/test/file/nn/amoeba09-cunn.prm", "nn_amoeba09-cunn.prm");
   const char* xn = "nn_6o.xyz";
   const char* kn = "nn_6o.key";
   const char* argv[] = {"dummy", xn, "-k", kn};
   int argc = 4;

   const double eps_e = testGetEps(0.0005, 0.0001);
   const double eps_g = testGetEps(0.0005, 0.0001);
   const double eps_v = testGetEps(0.001, 0.001);

   TestReference r(TINKER9_DIRSTR "/test/ref/nn_6o.txt");
   auto ref_c = r.getCount();
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_g = r.getGradient();

   rc_flag = calc::xyz | calc::vmask;
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
   COMPARE_INTS(nennmet, ref_c);

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

#endif
