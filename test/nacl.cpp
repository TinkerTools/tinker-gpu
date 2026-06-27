#include "ff/evdw.h"
#include "ff/modamoeba.h"

#include "test.h"
#include "testrt.h"

using namespace tinker;

#define COMPARE_CODE_BLOCK1                   \
   {                                          \
      energy(calc::v0);                       \
      COMPARE_REALS(energy_ev, ref_eng, eps); \
                                              \
      energy(calc::v1);                       \
      COMPARE_REALS(energy_ev, ref_eng, eps); \
      COMPARE_GRADIENT(ref_grad, eps);        \
      COMPARE_VIR9(virial_ev, ref_v, eps);    \
                                              \
      energy(calc::v3);                       \
      COMPARE_REALS(energy_ev, ref_eng, eps); \
      COMPARE_COUNT(nev, ref_count);          \
                                              \
      energy(calc::v4);                       \
      COMPARE_REALS(energy_ev, ref_eng, eps); \
      COMPARE_GRADIENT(ref_grad, eps);        \
                                              \
      energy(calc::v5);                       \
      COMPARE_GRADIENT(ref_grad, eps);        \
                                              \
      energy(calc::v6);                       \
      COMPARE_GRADIENT(ref_grad, eps);        \
      COMPARE_VIR9(virial_ev, ref_v, eps);    \
   }

TEST_CASE("NaCl-1", "[ff][evdw][evcorr][hal][switch][nacl]")
{
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");

   const std::string key = "vdwterm    only\n";

   int usage = 0;
   usage |= calc::xyz;
   usage |= calc::vmask;

   const double eps = 1.0e-3;

   SECTION("  - ehal -- no switch")
   {
      TestFile fke(TINKER9_DIRSTR "/test/file/nacl/nacl.key", "test_nacl.key", key);
      const char* x1 = "test_nacl.xyz";
      TestFile fx1(TINKER9_DIRSTR "/test/file/nacl/nacl1.xyz", x1);

      const char* argv[] = {"dummy", x1};
      int argc = 2;

      const double ref_eng = 51.4242;
      const int ref_count = 1;
      const double ref_grad[][3] = {{184.4899, 0.0, 0.0}, {-184.4899, 0.0, 0.0}};
      const double ref_v[][3] = {{-405.878, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

      testBeginWithArgs(argc, argv);
      rc_flag = usage;
      initialize();

      COMPARE_CODE_BLOCK1;

      finish();
      testEnd();
   }

   SECTION("  - ehal -- switch, near cut")
   {
      TestFile fke(TINKER9_DIRSTR "/test/file/nacl/nacl.key", "test_nacl.key", key);
      const char* x2 = "test_nacl.xyz_2";
      TestFile fx2(TINKER9_DIRSTR "/test/file/nacl/nacl2.xyz", x2);
      const char* argv[] = {"dummy", x2};
      int argc = 2;

      const double ref_eng = 25.8420;
      const int ref_count = 1;
      const double ref_grad[][3] = {{149.0904, 0.0, 0.0}, {-149.0904, 0.0, 0.0}};
      const double ref_v[][3] = {{-354.835, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

      testBeginWithArgs(argc, argv);
      rc_flag = usage;
      initialize();

      COMPARE_CODE_BLOCK1;

      finish();
      testEnd();
   }

   SECTION("  - ehal -- switch, near off")
   {
      TestFile fke(TINKER9_DIRSTR "/test/file/nacl/nacl.key", "test_nacl.key", key);
      const char* x3 = "test_nacl.xyz_3";
      TestFile fx3(TINKER9_DIRSTR "/test/file/nacl/nacl3.xyz", x3);

      const char* argv[] = {"dummy", x3};
      int argc = 2;

      const double ref_eng = 4.8849;
      const int ref_count = 1;
      const double ref_grad[][3] = {{127.6639, 0.0, 0.0}, {-127.6639, 0.0, 0.0}};
      const double ref_v[][3] = {{-319.160, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

      testBeginWithArgs(argc, argv);
      rc_flag = usage;
      initialize();

      COMPARE_CODE_BLOCK1;

      finish();
      testEnd();
   }

   SECTION("  - ehal -- evcorr vlambda = 1.0")
   {
      std::string key4 = key;
      // overwrite the default box size
      key4 += "b-axis            10.0\n";
      key4 += "c-axis            10.0\n";
      key4 += "vdw-correction\n";

      TestFile fke(TINKER9_DIRSTR "/test/file/nacl/nacl.key", "test_nacl.key", key4);
      const char* x4 = "test_nacl.xyz_4";
      TestFile fx4(TINKER9_DIRSTR "/test/file/nacl/nacl1.xyz", x4);

      const char* argv[] = {"dummy", x4};
      int argc = 2;

      const double ref_eng = 52.0802;
      const int ref_count = 1;
      const double ref_grad[][3] = {{184.4899, 0.0, 0.0}, {-184.4899, 0.0, 0.0}};
      const double ref_v[][3] = {{-408.398, 0.0, 0.0}, {0.0, -2.521, 0.0}, {0.0, 0.0, -2.521}};

      testBeginWithArgs(argc, argv);
      rc_flag = usage;
      initialize();

      COMPARE_CODE_BLOCK1;

      finish();
      testEnd();
   }

   SECTION("  - ehal -- evcorr vlambda = 0.9")
   {
      std::string key5 = key;
      // overwrite the default box size
      key5 += "b-axis            10.0\n";
      key5 += "c-axis            10.0\n";
      key5 += "vdw-correction\n";
      key5 += "ligand 2\n";
      key5 += "vdw-lambda 0.9\n";

      TestFile fke(TINKER9_DIRSTR "/test/file/nacl/nacl.key", "test_nacl.key", key5);
      const char* x5 = "test_nacl.xyz_5";
      TestFile fx4(TINKER9_DIRSTR "/test/file/nacl/nacl1.xyz", x5);

      const char* argv[] = {"dummy", x5};
      int argc = 2;

      const double ref_eng = 25.8947;
      const int ref_count = 1;
      const double ref_grad[][3] = {{81.9570, 0.0, 0.0}, {-81.9570, 0.0, 0.0}};
      const double ref_v[][3] = {{-182.721, 0.0, 0.0}, {0.0, -2.415, 0.0}, {0.0, 0.0, -2.415}};

      testBeginWithArgs(argc, argv);
      rc_flag = usage;
      initialize();

      COMPARE_CODE_BLOCK1;

      finish();
      testEnd();
   }
}

#define COMPARE_CODE_BLOCK2                        \
   {                                               \
      energy(calc::v0);                            \
      COMPARE_ENERGY(em, ref_eng, eps);            \
                                                   \
      energy(calc::v1);                            \
      COMPARE_ENERGY(em, ref_eng, eps);            \
      COMPARE_GRADIENT(ref_grad, eps_g);           \
      COMPARE_VIR2(vir_em, vir_trq, ref_v, eps_v); \
                                                   \
      energy(calc::v3);                            \
      COMPARE_ENERGY(em, ref_eng, eps);            \
      COMPARE_COUNT(nem, ref_count);               \
                                                   \
      energy(calc::v4);                            \
      COMPARE_ENERGY(em, ref_eng, eps);            \
      COMPARE_GRADIENT(ref_grad, eps_g);           \
                                                   \
      energy(calc::v5);                            \
      COMPARE_GRADIENT(ref_grad, eps_g);           \
                                                   \
      energy(calc::v6);                            \
      COMPARE_GRADIENT(ref_grad, eps_g);           \
      COMPARE_VIR2(vir_em, vir_trq, ref_v, eps_v); \
   }

TEST_CASE("NaCl-2", "[ff][empole][nonewald][nacl]")
{
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");

   std::string key = "multipoleterm    only\n";
   TestFile fke(TINKER9_DIRSTR "/test/file/nacl/nacl.key", "test_nacl.key", key);

   int usage = 0;
   usage |= calc::xyz;
   usage |= calc::vmask;

   const double eps = 1.0e-5;

   SECTION("empole -- non-ewald, pbc")
   {
      const char* x1 = "test_nacl.xyz";
      TestFile fx1(TINKER9_DIRSTR "/test/file/nacl/nacl1.xyz", x1);

      const char* argv[] = {"dummy", x1};
      int argc = 2;

      const double ref_eng = -150.9381;
      const int ref_count = 1;
      const double eps_g = eps;
      const double ref_grad[][3] = {{-68.6082, 0.0, 0.0}, {68.6082, 0.0, 0.0}};
      const double eps_v = eps;
      const double ref_v[][3] = {{150.938, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

      testBeginWithArgs(argc, argv);
      rc_flag = usage;
      initialize();

      COMPARE_CODE_BLOCK2;

      finish();
      testEnd();
   }
}

TEST_CASE("NaCl-3", "[ff][empole][ewald][nacl]")
{
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");

   std::string key = "multipoleterm    only\n";
   TestFile fke(TINKER9_DIRSTR "/test/file/nacl/nacl4.key", "test_nacl.key", key);

   int usage = 0;
   usage |= calc::xyz;
   usage |= calc::vmask;

   const double eps = 5.0e-4;

   SECTION("empole -- pme")
   {
      const char* x4 = "test_nacl.xyz_4";
      TestFile fx1(TINKER9_DIRSTR "/test/file/nacl/nacl4.xyz", x4);

      const char* argv[] = {"dummy", x4};
      int argc = 2;

      // real space energy 149.1782
      // recip space energy 54.7806
      // self energy -204.2734
      const double ref_eng = -0.3146;
      const int ref_count = 2;
      // total grad
      const double eps_g = eps;
      const double ref_grad[][3] = {{0.1731, 0.1921, 0.2103}, {-0.1668, -0.1917, -0.2081}};
      // self grad = 0
      // real grad
      // const double ref_grad[][3] = {{21.9459, 24.1404, 26.3789},
      //                               {-21.9459, -24.1404, -26.3789}};
      // recip grad
      // const double ref_grad[][3] = {{-21.7728, -23.9484, -26.1686},
      //                               {21.7791, 23.9487, 26.1709}};
      const double eps_v = eps;
      // total virial
      const double ref_v[][3] = {{0.059, -0.284, -0.311}, {-0.284, 0.105, -0.342}, {-0.311, -0.342, 0.155}};
      // self virial = 0
      // real virial
      // const double ref_v[][3] = {{-21.946, -24.140, -26.379},
      //                            {-24.140, -26.554, -29.017},
      //                            {-26.379, -29.017, -31.707}};
      // recip virial
      // const double ref_v[][3] = {{22.005, 23.857, 26.068},
      //                            {23.857, 26.659, 28.675},
      //                            {26.068, 28.675, 31.863}};

      testBeginWithArgs(argc, argv);
      rc_flag = usage;
      initialize();

      COMPARE_CODE_BLOCK2;

      finish();
      testEnd();
   }
}
