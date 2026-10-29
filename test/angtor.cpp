#include "ff/evalence.h"

#include "test.h"
#include "testrt.h"

using namespace tinker;

TEST_CASE("Angtor-A4", "[ff][eangtor][a4]")
{
   const char* k = "test_a4.key";
   const char* k0 = "angtorterm  only\n";
   const char* x1 = "test_a4.xyz";

   TestFile fke(TINKER9_DIRSTR "/test/file/nuc/a4.key", k, k0);
   TestFile fx1(TINKER9_DIRSTR "/test/file/nuc/a4.xyz", x1);
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_080e8f1d/amoebanuc17.prm");

   TestReference r(TINKER9_DIRSTR "/test/ref/angtor.txt");
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_count = r.getCount();
   auto ref_g = r.getGradient();

   const double eps_e = 0.0001;
   const double eps_g = 0.0001;
   const double eps_v = 0.001;

   const char* argv[] = {"dummy", x1};
   int argc = 2;
   TestSession session(argc, argv);
   rc_flag = calc::xyz | calc::vmask;
   session.init();

   energy(calc::v3);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_INTS(nangtor, ref_count);

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
