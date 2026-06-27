#include "ff/modamoeba.h"

#include "test.h"
#include "testrt.h"

using namespace tinker;

static int usage = calc::xyz | calc::vmask;
static const char* key0 = R"**(
parameters  amoeba09
cutoff           7.0
neighbor-list
list-buffer      0.1
a-axis          20.0


bondterm        none
angleterm       none
strbndterm      none
ureybradterm    none
torsionterm     none
vdwterm         none
)**";
static const char* kname = "test_local_frame3.key";
static const char* xname = "test_local_frame3.xyz";
static const char* argv[] = {"dummy", xname};
static int argc = 2;

TEST_CASE("Local-Frame3-1", "[ff][empole][epolar][emplar][nonewald][local-frame3]")
{
   std::string key = key0;

   TestFile fxy(TINKER9_DIRSTR "/test/file/local_frame/local_frame2.xyz", xname);
   TestFile fke("", kname, key);
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");

   testBeginWithArgs(argc, argv);
   rc_flag = usage;
   initialize();

   SECTION("  - emplar -- non-ewald pbc")
   {
      const int ref_count = 120;
      // empole -107.5885 count 120
      // epolar  -18.4063 count 120
      const double ref_eng = -125.9948;
      // empole virial
      //   83.083   6.570 -33.345
      //    6.570   2.497 -13.213
      //  -33.345 -13.213  38.666
      // epolar virial
      //   31.264  -0.788   7.542
      //   -0.788   4.922  -3.701
      //    7.542  -3.701  18.821
      const double ref_v[][3] = {{114.347, 5.782, -25.803}, {5.782, 7.419, -16.914}, {-25.803, -16.914, 57.488}};
      const double ref_g[][3] = {{21.9813, 3.0377, -6.1168}, {3.2451, -11.5492, 18.3287}, {9.1717, 3.2256, -2.2598},
         {-3.8440, -1.5729, 0.9964}, {-4.1469, -2.1011, 0.0910}, {16.3362, -14.6835, 3.6483}, {-5.9219, 5.8674, 3.6770},
         {-25.8555, 11.8977, -4.4568}, {1.1900, -0.5955, 9.1827}, {-0.5987, -0.2092, -3.9256},
         {3.4362, -0.4291, 2.6352}, {-1.6781, 0.2599, -1.3334}, {-2.3079, 0.2314, -1.4129}, {-2.0652, 0.8646, -1.6531},
         {-28.0256, -5.4435, 3.8673}, {2.0955, 2.9518, -1.4402}, {8.6696, -0.2665, -2.3909},
         {8.3182, 8.5144, -17.4371}};
      const double eps_e = 0.0001;
      const double eps_g = 0.0001;
      const double eps_v = 0.001;

      energy(calc::v0);
      COMPARE_REALS(esum, ref_eng, eps_e);

      energy(calc::v1);
      COMPARE_REALS(esum, ref_eng, eps_e);
      COMPARE_GRADIENT(ref_g, eps_g);
      for (int i = 0; i < 3; ++i)
         for (int j = 0; j < 3; ++j)
            COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_v);

      energy(calc::v3);
      COMPARE_REALS(esum, ref_eng, eps_e);
      COMPARE_INTS(countReduce(nem), ref_count);
      COMPARE_INTS(countReduce(nep), ref_count);

      energy(calc::v4);
      COMPARE_REALS(esum, ref_eng, eps_e);
      COMPARE_GRADIENT(ref_g, eps_g);

      energy(calc::v5);
      COMPARE_GRADIENT(ref_g, eps_g);

      energy(calc::v6);
      COMPARE_GRADIENT(ref_g, eps_g);
      for (int i = 0; i < 3; ++i)
         for (int j = 0; j < 3; ++j)
            COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_v);
   }

   finish();
   testEnd();
}

TEST_CASE("Local-Frame3-2", "[ff][empole][epolar][emplar][ewald][local-frame3]")
{
   std::string key = key0;
   key += "ewald\n";

   TestFile fxy(TINKER9_DIRSTR "/test/file/local_frame/local_frame2.xyz", xname);
   TestFile fke("", kname, key);
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");

   testBeginWithArgs(argc, argv);
   rc_flag = usage;
   initialize();

   SECTION("  - emplar -- ewald pbc")
   {
      const int ref_count = 138;
      // empole  -92.5088 count 138
      // epolar  -17.3296 count 138
      const double ref_eng = -109.8384;
      const double ref_v[][3] = {{102.470, 6.200, -20.734}, {6.200, 6.392, -17.112}, {-20.734, -17.112, 52.162}};
      const double ref_g[][3] = {{19.9369, 3.0207, -5.0886}, {3.4854, -11.3956, 17.6976}, {7.2089, 2.3667, -1.3915},
         {-2.9905, -1.1704, 0.6131}, {-3.2875, -1.7351, -0.2953}, {16.7804, -14.8546, 3.3768},
         {-4.8524, 5.7678, 2.6657}, {-25.8810, 12.0574, -4.3136}, {1.4852, -0.6003, 9.1247},
         {-0.7483, -0.0195, -3.9075}, {2.2061, -0.6409, 2.2389}, {-0.9777, 0.3695, -0.9381}, {-2.0167, 0.2311, -1.1756},
         {-1.3661, 0.8324, -1.4142}, {-27.9560, -5.3807, 3.5241}, {2.0808, 2.9524, -1.3329}, {8.6643, -0.3029, -2.2751},
         {8.3116, 8.4359, -17.2765}};
      const double eps_e = 0.0001;
      const double eps_g = 0.0001;
      const double eps_v = 0.001;

      energy(calc::v0);
      COMPARE_REALS(esum, ref_eng, eps_e);

      energy(calc::v1);
      COMPARE_REALS(esum, ref_eng, eps_e);
      COMPARE_GRADIENT(ref_g, eps_g);
      for (int i = 0; i < 3; ++i)
         for (int j = 0; j < 3; ++j)
            COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_v);

      energy(calc::v3);
      COMPARE_REALS(esum, ref_eng, eps_e);
      COMPARE_INTS(countReduce(nem), ref_count);
      COMPARE_INTS(countReduce(nep), ref_count);

      energy(calc::v4);
      COMPARE_REALS(esum, ref_eng, eps_e);
      COMPARE_GRADIENT(ref_g, eps_g);

      energy(calc::v5);
      COMPARE_GRADIENT(ref_g, eps_g);

      energy(calc::v6);
      COMPARE_GRADIENT(ref_g, eps_g);
      for (int i = 0; i < 3; ++i)
         for (int j = 0; j < 3; ++j)
            COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_v);
   }

   finish();
   testEnd();
}

TEST_CASE("Local-Frame3-3", "[ff][empole][epolar][emplar][ewald][local-frame3]")
{
   std::string key = key0;
   key += "ewald\n";

   TestFile fxy(TINKER9_DIRSTR "/test/file/local_frame/local_frame2.xyz", xname);
   TestFile fke("", kname, key);
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");

   testBeginWithArgs(argc, argv);
   rc_flag = usage & ~calc::analyz;
   initialize();

   SECTION("  - emplar -- ewald pbc")
   {
      // empole  -92.5088 count 138
      // epolar  -17.3296 count 138
      const double ref_eng = -109.8384;
      const double ref_v[][3] = {{102.470, 6.200, -20.734}, {6.200, 6.392, -17.112}, {-20.734, -17.112, 52.162}};
      const double ref_g[][3] = {{19.9369, 3.0207, -5.0886}, {3.4854, -11.3956, 17.6976}, {7.2089, 2.3667, -1.3915},
         {-2.9905, -1.1704, 0.6131}, {-3.2875, -1.7351, -0.2953}, {16.7804, -14.8546, 3.3768},
         {-4.8524, 5.7678, 2.6657}, {-25.8810, 12.0574, -4.3136}, {1.4852, -0.6003, 9.1247},
         {-0.7483, -0.0195, -3.9075}, {2.2061, -0.6409, 2.2389}, {-0.9777, 0.3695, -0.9381}, {-2.0167, 0.2311, -1.1756},
         {-1.3661, 0.8324, -1.4142}, {-27.9560, -5.3807, 3.5241}, {2.0808, 2.9524, -1.3329}, {8.6643, -0.3029, -2.2751},
         {8.3116, 8.4359, -17.2765}};
      const double eps_e = 0.0001;
      const double eps_g = 0.0001;
      const double eps_v = 0.001;

      energy(calc::v0);
      COMPARE_REALS(esum, ref_eng, eps_e);

      energy(calc::v1);
      COMPARE_REALS(esum, ref_eng, eps_e);
      COMPARE_GRADIENT(ref_g, eps_g);
      for (int i = 0; i < 3; ++i)
         for (int j = 0; j < 3; ++j)
            COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_v);

      energy(calc::v4);
      COMPARE_REALS(esum, ref_eng, eps_e);
      COMPARE_GRADIENT(ref_g, eps_g);

      energy(calc::v5);
      COMPARE_GRADIENT(ref_g, eps_g);

      energy(calc::v6);
      COMPARE_GRADIENT(ref_g, eps_g);
      for (int i = 0; i < 3; ++i)
         for (int j = 0; j < 3; ++j)
            COMPARE_REALS(vir[i * 3 + j], ref_v[i][j], eps_v);
   }

   finish();
   testEnd();
}
