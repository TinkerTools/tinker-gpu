#include "ff/evdw.h"

#include "test.h"
#include "testrt.h"

using namespace tinker;

TEST_CASE("Truncated-Octahedron", "[ff][pbc][arbox]")
{
   rc_flag = calc::xyz | calc::vmask;

   const char* kname = "test_t8.key";
   const char* xname = "test_t8.xyz";

   const double eps_e = 0.0001;
   const double eps_g = 0.0002;
   const double eps_v = 0.001;
   const int eps_count = 1;
   const char* argv[] = {"dummy", xname};
   int argc = 2;

   TestFile fxy(TINKER9_DIRSTR "/test/file/trunc8/arbox.xyz", xname);
   TestFile fke(TINKER9_DIRSTR "/test/file/trunc8/arbox.key", kname);
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");

   TestReference r(TINKER9_DIRSTR "/test/ref/truncated8.1.txt");
   auto ref_e = r.getEnergy();
   auto ref_v = r.getVirial();
   auto ref_count = r.getCount();
   auto ref_g = r.getGradient();

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
   // COMPARE_INTS(countReduce(nev), ref_count);
   COMPARE_INTS_EPS(countReduce(nev), ref_count, eps_count);

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
