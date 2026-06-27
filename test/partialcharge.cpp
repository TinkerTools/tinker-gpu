#include "ff/echarge.h"

#include "test.h"
#include "testrt.h"

using namespace tinker;

TEST_CASE("PartialCharge-Trpcage", "[ff][echarge][ewald][nonewald][trpcage]")
{
   rc_flag = calc::xyz | calc::vmask;

   const char* kname = "test_pchg.key";
   std::string k0 = "CHARGETERM ONLY\n";
   const char* xname = "test_pchg.xyz";

   const double eps_e = 0.0001;
   const double eps_g = 0.0001;
   const double eps_v = 0.001;
   const char* argv[] = {"dummy", xname};
   int argc = 2;

   SECTION("  - ec -- no pbc, no cutoff, non-ewald, taper")
   {
      TestFile fxy(TINKER9_DIRSTR "/test/file/trpcage/trp_charmm.xyz", xname);
      TestFile fke(TINKER9_DIRSTR "/test/file/trpcage/trp_charmm.key", kname, k0);
      TestFile fpr(TINKER9_DIRSTR "/test/file/commit_11e84c69/charmm19.prm");

      TestReference r(TINKER9_DIRSTR "/test/ref/partialcharge.1.txt");
      auto ref_e = r.getEnergy();
      auto ref_v = r.getVirial();
      auto ref_count = r.getCount();
      auto ref_g = r.getGradient();

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
      COMPARE_INTS(countReduce(nec), ref_count);

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

   SECTION("  - ec -- pbc, cutoff, ewald")
   {
      std::string k1 = k0
         + "EWALD\n"
           "NEIGHBOR-LIST\n"
           "CUTOFF           9.0\n"
           "LIST-BUFFER      0.5\n"
           "A-AXIS            25\n"
           "B-AXIS            30\n"
           "C-AXIS            20\n";

      TestFile fxy(TINKER9_DIRSTR "/test/file/trpcage/trp_charmm.xyz", xname);
      TestFile fke(TINKER9_DIRSTR "/test/file/trpcage/trp_charmm.key", kname, k1);
      TestFile fpr(TINKER9_DIRSTR "/test/file/commit_11e84c69/charmm19.prm");

      TestReference r(TINKER9_DIRSTR "/test/ref/partialcharge.2.txt");
      auto ref_e = r.getEnergy();
      auto ref_v = r.getVirial();
      auto ref_count = r.getCount();
      auto ref_g = r.getGradient();

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
      COMPARE_INTS(countReduce(nec), ref_count);

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
}
