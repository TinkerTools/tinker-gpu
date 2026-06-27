#include "ff/evdw.h"

#include "test.h"
#include "testrt.h"

using namespace tinker;

TEST_CASE("Vdw14-Trpcage", "[ff][evdw][vdw14][lj][trpcage]")
{
   rc_flag = calc::xyz | calc::vmask;

   const char* kname = "test_vdw14.key";
   std::string k0 = "VDWTERM ONLY\n";
   const char* xname = "test_vdw14.xyz";

   const double eps_e = 0.0001;
   const double eps_g = 0.0001;
   const double eps_v = 0.001;
   const char* argv[] = {"dummy", xname};
   int argc = 2;

   SECTION("  - elj -- no pbc, no cutoff")
   {
      TestFile fxy(TINKER9_DIRSTR "/test/file/trpcage/trp_charmm.xyz", xname);
      TestFile fke(TINKER9_DIRSTR "/test/file/trpcage/trp_charmm.key", kname, k0);
      TestFile fpr(TINKER9_DIRSTR "/test/file/commit_11e84c69/charmm19.prm");

      TestReference r(TINKER9_DIRSTR "/test/ref/vdw14.1.txt");
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
      COMPARE_INTS(countReduce(nev), ref_count);

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

   SECTION("  - elj -- pbc, cutoff")
   {
      std::string k1 = k0
         + "NEIGHBOR-LIST\n"
           "LIST-BUFFER      0.5\n"
           "CUTOFF           9.0\n"
           "A-AXIS            30\n"
           "B-AXIS            25\n"
           "C-AXIS            20\n";

      TestFile fxy(TINKER9_DIRSTR "/test/file/trpcage/trp_charmm.xyz", xname);
      TestFile fke(TINKER9_DIRSTR "/test/file/trpcage/trp_charmm.key", kname, k1);
      TestFile fpr(TINKER9_DIRSTR "/test/file/commit_11e84c69/charmm19.prm");

      TestReference r(TINKER9_DIRSTR "/test/ref/vdw14.2.txt");
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
      COMPARE_INTS(countReduce(nev), ref_count);

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
