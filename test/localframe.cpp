#include "ff/amoeba/empole.h"
#include "ff/amoeba/epolar.h"
#include "ff/amoeba/induce.h"
#include "ff/modamoeba.h"
#include <array>
#include <tinker/detail/units.hh>

#include "test.h"
#include "testrt.h"

using namespace tinker;

/**
 * \brief
 * The last molecule in the local-frame test case is a NH3, where all of its 4
 * atoms are in the xy-plane. Therefore the z-gradient components of this
 * molecule are not included in the test.
 */
static bool do_ij(int i, int /* j */)
{
   return i < n - 4;
}

TEST_CASE("Local-Frame-1", "[ff][empole][nonewald][local-frame]")
{
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");

   const char* k = "test_local_frame.key";

   const char* x1 = "test_local_frame.xyz";
   TestFile fx1(TINKER9_DIRSTR "/test/file/local_frame/local_frame.xyz", x1);

   int usage = 0;
   usage |= calc::xyz;
   usage |= calc::vmask;

   SECTION("empole -- gas phase, no cutoff")
   {
      std::string key1 = "multipoleterm    only\n";
      TestFile fke(TINKER9_DIRSTR "/test/file/local_frame/local_frame.key", k, key1);

      const char* argv[] = {"dummy", x1};
      int argc = 2;

      const double eps_e = 1.0e-5;
      const double ref_eng = -95.7432;
      const int ref_count = 201;
      const double eps_g = 0.0005;
      const double ref_grad[][3] = {
         {-20.9915, 29.4368, 0.5977},
         {12.7503, -9.9272, 3.3604},
         {15.0932, -5.2916, 0.1809},
         {-6.0154, 0.6960, -0.2064},
         {-5.0594, 4.3698, -1.3173},
         {36.4924, -26.8619, 11.0059},
         {-11.2357, 10.4015, -1.8246},
         {-26.9426, 13.9832, -5.6296},
         {12.3270, -4.4632, 9.1612},
         {-6.1247, 3.8477, -1.6673},
         {3.8365, -3.0201, -1.0674},
         {-1.5792, 0.9991, -0.5125},
         {-2.3155, 1.5935, 0.3036},
         {-2.0629, 1.7467, 0.9395},
         {0.1139, -5.1353, 9.9238},
         {-1.9284, 1.1291, -2.8500},
         {1.1353, -0.2388, -5.6115},
         {-0.2273, 6.5552, -7.8322},
         {23.9256, -41.7121, -54.8522},
         {-18.4759, 10.3134, 8.4037},
         {-6.0110, 9.7698, 14.0093},
         {3.2955, 1.8082, 25.4850},
      };
      const double eps_v = 0.005;
      const double ref_v[][3] = {{72.636, -34.992, 9.243}, {-34.992, 53.410, 6.517}, {9.243, 6.517, 4.708}};

      testBeginWithArgs(argc, argv);
      rc_flag = usage;
      initialize();

      energy(calc::v0);
      COMPARE_ENERGY(em, ref_eng, eps_e);

      energy(calc::v1);
      COMPARE_ENERGY(em, ref_eng, eps_e);
      COMPARE_GRADIENT2(ref_grad, eps_g, do_ij);
      COMPARE_VIR2(vir_em, vir_trq, ref_v, eps_v);

      energy(calc::v3);
      COMPARE_ENERGY(em, ref_eng, eps_e);
      COMPARE_COUNT(nem, ref_count);

      finish();
      testEnd();
   }
}

TEST_CASE("Local-Frame-2", "[ff][empole][ewald][local-frame]")
{
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");

   const char* k = "test_local_frame.key";

   const char* x1 = "test_local_frame.xyz";
   TestFile fx1(TINKER9_DIRSTR "/test/file/local_frame/local_frame.xyz", x1);

   int usage = 0;
   usage |= calc::xyz;
   usage |= calc::vmask;

   SECTION("empole -- pme")
   {
      std::string key1 = "multipoleterm    only\n";
      key1 += "ewald\n";
      key1 += "ewald-cutoff    7.0\n";
      key1 += "neighbor-list\n";
      key1 += "list-buffer    0.1\n";
      key1 += "a-axis    20.0\n";
      TestFile fke(TINKER9_DIRSTR "/test/file/local_frame/local_frame.key", k, key1);

      const char* argv[] = {"dummy", x1};
      int argc = 2;

      const double eps_e = 0.0001;
      // emreal 132.76851908903473
      // +recip 279.93038602122442 // recip 147.1618669321897
      // +self -99.229836789909172 // self -379.1602228111336
      const double ref_ereal = 132.76851908903473;
      const double ref_erecip = 147.1618669321897;
      const double ref_eself = -379.1602228111336;
      const double ref_eng = ref_eself + ref_erecip + ref_ereal;
      const int ref_count = 222;
      const double eps_g = 0.0005;
      const double ref_grad[][3] = {{-20.0239, 29.1238, 0.7844}, {11.8249, -9.4636, 3.1958}, {14.5683, -4.9576, 0.0931},
         {-5.7229, 0.5038, -0.1740}, {-4.7824, 4.2038, -1.2616}, {35.8953, -26.6166, 10.8531},
         {-10.9482, 10.2634, -1.7619}, {-26.6406, 13.8783, -5.5468}, {11.7934, -4.2057, 8.9439},
         {-5.7126, 3.6221, -1.4949}, {3.5447, -2.8759, -1.0921}, {-1.4544, 0.9341, -0.4978}, {-2.1774, 1.4935, 0.3237},
         {-1.9210, 1.7174, 0.9565}, {-0.6072, -4.9773, 9.5574}, {-1.8037, 1.1493, -2.7232}, {1.4169, -0.3143, -5.4280},
         {0.0561, 6.4388, -7.8332}, {22.8962, -41.3767, -55.0242}, {-17.8085, 9.9950, 8.4640},
         {-5.9679, 9.6342, 14.0679}, {3.6422, 2.0227, 25.5414}};
      const double eps_v = 0.001;
      const double ref_v[][3] = {{76.909, -31.146, 7.610}, {-31.146, 53.934, 7.245}, {7.610, 7.245, 3.181}};

      testBeginWithArgs(argc, argv);
      rc_flag = usage;
      initialize();

      energy(calc::v0);
      COMPARE_ENERGY(em, ref_eng, eps_e);

      energy(calc::v1);
      COMPARE_ENERGY(em, ref_eng, eps_e);
      COMPARE_GRADIENT(ref_grad, eps_g);
      COMPARE_VIR2(vir_em, vir_trq, ref_v, eps_v);

      energy(calc::v3);
      COMPARE_ENERGY(em, ref_eng, eps_e);
      COMPARE_COUNT(nem, ref_count);

      finish();
      testEnd();
   }
}

#define COMPARE_CODE_BLOCK2                        \
   {                                               \
      energy(calc::v0);                            \
      COMPARE_ENERGY(ep, ref_eng, eps_e);          \
                                                   \
      energy(calc::v1);                            \
      COMPARE_ENERGY(ep, ref_eng, eps_e);          \
      COMPARE_GRADIENT2(ref_grad, eps_g, do_ij);   \
      COMPARE_VIR2(vir_ep, vir_trq, ref_v, eps_v); \
                                                   \
      energy(calc::v3);                            \
      COMPARE_ENERGY(ep, ref_eng, eps_e);          \
      COMPARE_COUNT(nep, ref_count);               \
                                                   \
      energy(calc::v4);                            \
      COMPARE_ENERGY(ep, ref_eng, eps_e);          \
      COMPARE_GRADIENT2(ref_grad, eps_g, do_ij);   \
                                                   \
      energy(calc::v5);                            \
      COMPARE_GRADIENT2(ref_grad, eps_g, do_ij);   \
                                                   \
      energy(calc::v6);                            \
      COMPARE_GRADIENT2(ref_grad, eps_g, do_ij);   \
      COMPARE_VIR2(vir_ep, vir_trq, ref_v, eps_v); \
   }

TEST_CASE("Local-Frame-3", "[ff][epolar][nonewald][local-frame]")
{
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");

   const char* k = "test_local_frame.key";
   std::string key1 = "usolve-cutoff    0.01\n";
   key1 += "polarizeterm    only\n";
   TestFile fke(TINKER9_DIRSTR "/test/file/local_frame/local_frame.key", k, key1);

   const char* x1 = "test_local_frame.xyz";
   TestFile fx1(TINKER9_DIRSTR "/test/file/local_frame/local_frame.xyz", x1);

   int usage = 0;
   usage |= calc::xyz;
   usage |= calc::vmask;

   const char* argv[] = {"dummy", x1};
   int argc = 2;

   const double debye = units::debye;
   const double eps_f = 0.0001;

   testBeginWithArgs(argc, argv);
   rc_flag = usage;
   initialize();

   SECTION("dfield -- non-ewald no cutoff")
   {
      const double ref_dir_field_d[][3] = {{0.0632, -0.0886, -0.0018}, {0.0332, -0.0268, 0.0058},
         {0.0771, -0.0274, 0.0004}, {0.0567, -0.0101, -0.0015}, {0.0609, -0.0428, 0.0132}, {0.1799, -0.1501, 0.0475},
         {0.1291, -0.1097, 0.0187}, {0.2264, -0.1796, 0.0300}, {0.0617, -0.0242, 0.0407}, {0.0420, -0.0311, 0.0001},
         {0.0230, -0.0237, -0.0167}, {0.0167, -0.0148, -0.0114}, {0.0190, -0.0233, -0.0202}, {0.0215, -0.0292, -0.0307},
         {-0.0020, -0.0570, 0.0876}, {0.0402, -0.0723, 0.0618}, {-0.0272, -0.0261, 0.0970}, {0.0030, -0.0984, 0.1580},
         {0.0209, -0.1631, -0.0351}, {0.0056, -0.0713, -0.0274}, {0.1466, -0.1529, 0.0013},
         {-0.1215, -0.1945, -0.0897}};
      const double ref_dir_field_p[][3] = {{0.0632, -0.0886, -0.0018}, {0.0332, -0.0268, 0.0058},
         {0.0771, -0.0274, 0.0004}, {0.0567, -0.0101, -0.0015}, {0.0609, -0.0428, 0.0132}, {0.1799, -0.1501, 0.0475},
         {0.1291, -0.1097, 0.0187}, {0.2264, -0.1796, 0.0300}, {0.0697, -0.0275, 0.0461}, {0.0420, -0.0311, 0.0001},
         {0.0230, -0.0237, -0.0167}, {0.0369, -0.0230, 0.0194}, {0.0607, -0.0264, -0.0032}, {0.0483, -0.0565, -0.0134},
         {-0.0020, -0.0570, 0.0876}, {0.0402, -0.0723, 0.0618}, {-0.0272, -0.0261, 0.0970}, {0.0030, -0.0984, 0.1580},
         {0.0209, -0.1631, -0.0351}, {0.0056, -0.0713, -0.0274}, {0.1466, -0.1529, 0.0013},
         {-0.1215, -0.1945, -0.0897}};

      zeroEGV();
      mpoleInit(calc::v0);
      dfieldNonEwald(udir, udirp);
      std::vector<std::array<double, 3>> fieldd, fieldp;
      fieldd.resize(n);
      fieldp.resize(n);
      darray::copyout(g::q0, n, &fieldd[0][0], udir);
      darray::copyout(g::q0, n, &fieldp[0][0], udirp);
      waitFor(g::q0);
      for (int i = 0; i < n; ++i) {
         for (int j = 0; j < 3; ++j) {
            REQUIRE(fieldd[i][j] == Approx(ref_dir_field_d[i][j]).margin(eps_f));
            REQUIRE(fieldp[i][j] == Approx(ref_dir_field_p[i][j]).margin(eps_f));
         }
      }
   }

   SECTION("ufield -- non-ewald no cutoff")
   {
      const double ref_ufield_d[][3] = {{0.2171, 1.7393, 0.3937}, {-0.0768, -0.7185, 0.1626},
         {-0.4172, -0.8336, -0.7593}, {-0.3755, -0.4743, -0.5571}, {-0.3918, -1.1932, -0.6319},
         {-0.4074, -0.4488, 0.0958}, {-0.5865, -0.2294, 0.5782}, {-0.2908, -0.4478, 0.2516}, {0.5830, -0.5150, -0.4780},
         {0.1768, -1.2311, 0.0137}, {-1.3528, -1.3973, -0.7081}, {-1.0405, -0.5500, 0.6940}, {-0.0828, 0.0215, -0.2048},
         {-1.0584, 0.0414, -0.1254}, {-1.7674, -1.6456, -1.6328}, {-0.7705, -1.2322, -1.0136},
         {-1.2783, 0.2778, -0.1103}, {-0.9239, -1.1300, 0.6484}, {-1.9449, -2.2349, -2.4348},
         {-0.0062, 0.3626, -1.6759}, {0.7257, -1.4398, -1.0205}, {-1.5095, -0.8471, -1.4955}};
      const double ref_ufield_p[][3] = {{0.1942, 1.6373, 0.3969}, {-0.0770, -0.6580, 0.1638},
         {-0.3878, -0.6974, -0.5665}, {-0.3390, -0.3660, -0.4104}, {-0.3833, -1.0256, -0.4906},
         {-0.3573, -0.3907, 0.1704}, {-0.5076, -0.2187, 0.5236}, {-0.2260, -0.4438, 0.2835}, {0.5554, -0.4572, -0.4330},
         {0.1458, -1.1176, 0.0421}, {-1.3029, -1.2660, -0.6027}, {-0.9806, -0.4974, 0.6001}, {-0.0944, 0.0638, -0.1612},
         {-1.0025, 0.0203, -0.0974}, {-1.6961, -1.5330, -1.4690}, {-0.7433, -1.1496, -0.8978},
         {-1.2131, 0.2382, -0.0705}, {-0.8943, -1.0352, 0.5990}, {-1.8822, -2.1268, -2.2442},
         {-0.0331, 0.4012, -1.5516}, {0.7016, -1.3551, -0.9421}, {-1.4662, -0.8545, -1.3742}};

      zeroEGV();
      mpoleInit(calc::v0);
      std::vector<std::array<double, 3>> ud, up;
      ud.resize(n);
      up.resize(n);
      for (int i = 0; i < n; ++i) {
         for (int j = 0; j < 3; ++j) {
            ud[i][j] = 0.1 * (i + 1) + 0.03 * (j + 1);
            up[i][j] = 0.1 * (i + 1) - 0.03 * (j + 1);
         }
      }
      darray::copyin(g::q0, n, uind, &ud[0][0]);
      darray::copyin(g::q0, n, uinp, &up[0][0]);
      waitFor(g::q0);
      ufieldNonEwald(uind, uinp, udir, udirp);
      ud.resize(n);
      up.resize(n);
      darray::copyout(g::q0, n, &ud[0][0], udir);
      darray::copyout(g::q0, n, &up[0][0], udirp);
      waitFor(g::q0);
      for (int i = 0; i < n; ++i) {
         for (int j = 0; j < 3; ++j) {
            REQUIRE(ud[i][j] == Approx(ref_ufield_d[i][j]).margin(eps_f));
            REQUIRE(up[i][j] == Approx(ref_ufield_p[i][j]).margin(eps_f));
         }
      }
   }

   SECTION("default induce routine -- non-ewald no cutoff")
   {
      const double ref_ud_debye[][3] = {{0.0256, -0.0849, -0.0107}, {1.3172, -0.8359, 0.5512},
         {0.2342, -0.0712, 0.0100}, {0.0651, -0.0337, 0.0202}, {0.1076, -0.0967, 0.0939}, {0.5459, -0.5351, 0.1614},
         {0.1637, -0.1555, -0.0992}, {0.5366, -0.4246, -0.0109}, {0.2835, -0.0694, 0.1701}, {0.2570, -0.1309, -0.0448},
         {0.1316, -0.0672, -0.0213}, {0.0195, -0.0099, -0.0283}, {0.0558, -0.0130, -0.0514}, {0.0402, -0.0646, -0.0392},
         {-0.0107, -0.1520, 0.3173}, {0.1540, -0.1650, 0.1211}, {-0.0419, 0.0277, 0.1490}, {0.0006, -0.1720, 0.4521},
         {0.2376, -0.6121, -0.0925}, {0.0203, -0.1166, -0.0128}, {0.3638, -0.1926, 0.0577},
         {-0.2110, -0.4679, -0.1808}};
      const double ref_up_debye[][3] = {{0.0254, -0.0847, -0.0108}, {1.3219, -0.8336, 0.5631},
         {0.2326, -0.0713, 0.0118}, {0.0647, -0.0340, 0.0205}, {0.1068, -0.0964, 0.0940}, {0.5453, -0.5340, 0.1619},
         {0.1633, -0.1551, -0.0986}, {0.5363, -0.4239, -0.0098}, {0.3281, -0.0868, 0.2021}, {0.3042, -0.1482, -0.0182},
         {0.0552, -0.0342, -0.0785}, {0.0683, -0.0313, 0.0402}, {0.1544, -0.0293, -0.0008}, {0.1074, -0.1232, 0.0090},
         {-0.0075, -0.1480, 0.3169}, {0.1544, -0.1631, 0.1208}, {-0.0376, 0.0295, 0.1482}, {0.0018, -0.1711, 0.4538},
         {0.2329, -0.6094, -0.0903}, {0.0180, -0.1162, -0.0128}, {0.3620, -0.1921, 0.0590},
         {-0.2126, -0.4665, -0.1829}};

      zeroEGV();
      mpoleInit(calc::v0);
      induce(uind, uinp);
      std::vector<std::array<double, 3>> ud, up;
      ud.resize(n);
      up.resize(n);
      darray::copyout(g::q0, n, &ud[0][0], uind);
      darray::copyout(g::q0, n, &up[0][0], uinp);
      waitFor(g::q0);
      for (int i = 0; i < n; ++i) {
         for (int j = 0; j < 3; ++j) {
            REQUIRE(ud[i][j] * debye == Approx(ref_ud_debye[i][j]).margin(eps_f));
            REQUIRE(up[i][j] * debye == Approx(ref_up_debye[i][j]).margin(eps_f));
         }
      }
   }

   SECTION("epolar via dot product")
   {
      const double ref_eng = -37.7476;

      zeroEGV();
      epolar(calc::v0);
      COMPARE_ENERGY(ep, ref_eng, eps_f);
   }

   SECTION("various epolar versions -- non-ewald no cutoff")
   {
      const double eps_e = eps_f;
      const double ref_eng = -37.7476;
      const int ref_count = 201;
      const double eps_g = eps_f;
      const double ref_grad[][3] = {{6.5312, 20.1361, 5.8824}, {13.2027, -5.6940, 7.3885}, {-3.2986, 0.8521, 1.9560},
         {0.8378, 0.0755, -0.0391}, {1.2206, -1.5629, -2.3336}, {4.3143, -2.4724, 3.7688}, {0.8981, -0.1895, 1.4048},
         {-13.5406, 5.6213, -0.7109}, {-0.9682, 0.7014, -4.1109}, {0.4035, -1.7353, 1.4395}, {0.1023, 0.9368, -0.3637},
         {-0.0900, -0.2618, -0.0421}, {0.1241, -0.2121, 0.2100}, {-0.0689, -0.6002, -0.5924},
         {-0.6472, 1.1960, -0.4438}, {0.8111, -0.4277, -0.1703}, {-0.1691, -0.0039, 0.8109}, {2.2442, 1.1983, -6.2875},
         {0.7050, -6.5825, -2.1732}, {-1.9008, 0.9707, 1.7439}, {-2.9368, -1.0271, -1.8723},
         {-7.7748, -10.9186, -5.4650}};
      const double eps_v = 0.001;
      const double ref_v[][3] = {{49.295, 5.373, 13.458}, {5.373, 37.751, 8.791}, {13.458, 8.791, 33.516}};

      COMPARE_CODE_BLOCK2;
   }

   finish();
   testEnd();
}

TEST_CASE("Local-Frame-4", "[ff][epolar][ewald][local-frame]")
{
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");

   const char* k = "test_local_frame.key";
   std::string key1 = "usolve-cutoff    0.01\n";
   key1 += "polarizeterm    only\n";
   key1 += "ewald\n";
   key1 += "ewald-cutoff    7.0\n";
   key1 += "neighbor-list\n";
   key1 += "list-buffer    0.1\n";
   key1 += "a-axis    20.0\n";
   TestFile fke(TINKER9_DIRSTR "/test/file/local_frame/local_frame.key", k, key1);

   const char* x1 = "test_local_frame.xyz";
   TestFile fx1(TINKER9_DIRSTR "/test/file/local_frame/local_frame.xyz", x1);

   int usage = 0;
   usage |= calc::xyz;
   usage |= calc::vmask;

   const char* argv[] = {"dummy", x1};
   int argc = 2;

   const double debye = units::debye;
   const double eps_f = 0.0001;

   testBeginWithArgs(argc, argv);
   rc_flag = usage;
   initialize();

   SECTION("dfield -- pme")
   {
      const double ref_dir_field_d[][3] = {{0.0603, -0.0877, -0.0024}, {0.0304, -0.0254, 0.0053},
         {0.0743, -0.0256, -0.0002}, {0.0540, -0.0080, -0.0020}, {0.0577, -0.0410, 0.0126}, {0.1771, -0.1487, 0.0468},
         {0.1263, -0.1083, 0.0180}, {0.2235, -0.1782, 0.0293}, {0.0589, -0.0228, 0.0400}, {0.0394, -0.0297, -0.0007},
         {0.0204, -0.0223, -0.0171}, {0.0144, -0.0136, -0.0118}, {0.0160, -0.0218, -0.0204}, {0.0188, -0.0281, -0.0310},
         {-0.0052, -0.0557, 0.0866}, {0.0372, -0.0710, 0.0609}, {-0.0302, -0.0249, 0.0959}, {-0.0002, -0.0971, 0.1573},
         {0.0176, -0.1618, -0.0356}, {0.0020, -0.0699, -0.0279}, {0.1436, -0.1515, 0.0008},
         {-0.1249, -0.1932, -0.0900}};
      const double ref_dir_field_p[][3] = {{0.0603, -0.0877, -0.0024}, {0.0304, -0.0254, 0.0053},
         {0.0743, -0.0256, -0.0002}, {0.0540, -0.0080, -0.0020}, {0.0577, -0.0410, 0.0126}, {0.1771, -0.1487, 0.0468},
         {0.1263, -0.1083, 0.0180}, {0.2235, -0.1782, 0.0293}, {0.0670, -0.0261, 0.0454}, {0.0394, -0.0297, -0.0007},
         {0.0204, -0.0223, -0.0171}, {0.0345, -0.0218, 0.0190}, {0.0577, -0.0249, -0.0034}, {0.0456, -0.0554, -0.0137},
         {-0.0052, -0.0557, 0.0866}, {0.0372, -0.0710, 0.0609}, {-0.0302, -0.0249, 0.0959}, {-0.0002, -0.0971, 0.1573},
         {0.0176, -0.1618, -0.0356}, {0.0020, -0.0699, -0.0279}, {0.1436, -0.1515, 0.0008},
         {-0.1249, -0.1932, -0.0900}};

      zeroEGV();
      mpoleInit(calc::v0);
      dfieldEwald(udir, udirp);
      std::vector<std::array<double, 3>> fieldd, fieldp;
      fieldd.resize(n);
      fieldp.resize(n);
      darray::copyout(g::q0, n, &fieldd[0][0], udir);
      darray::copyout(g::q0, n, &fieldp[0][0], udirp);
      waitFor(g::q0);
      for (int i = 0; i < n; ++i) {
         for (int j = 0; j < 3; ++j) {
            REQUIRE(fieldd[i][j] == Approx(ref_dir_field_d[i][j]).margin(eps_f));
            REQUIRE(fieldp[i][j] == Approx(ref_dir_field_p[i][j]).margin(eps_f));
         }
      }
   }

   SECTION("ufield -- pme")
   {
      const double ref_ufield_d[][3] = {{0.2326, 1.7537, 0.4066}, {-0.0625, -0.7063, 0.1748},
         {-0.4040, -0.8194, -0.7421}, {-0.3634, -0.4587, -0.5382}, {-0.3778, -1.1806, -0.6153},
         {-0.3945, -0.4370, 0.1095}, {-0.5747, -0.2189, 0.5939}, {-0.2782, -0.4352, 0.2659}, {0.5975, -0.5023, -0.4645},
         {0.1915, -1.2181, 0.0277}, {-1.3381, -1.3854, -0.6932}, {-1.0270, -0.5388, 0.7093}, {-0.0693, 0.0332, -0.1906},
         {-1.0449, 0.0542, -0.1094}, {-1.7474, -1.6330, -1.6222}, {-0.7535, -1.2197, -1.0019},
         {-1.2575, 0.2898, -0.0987}, {-0.9060, -1.1180, 0.6608}, {-1.9284, -2.2206, -2.4186}, {0.0085, 0.3752, -1.6604},
         {0.7407, -1.4251, -1.0051}, {-1.4916, -0.8335, -1.4794}};
      const double ref_ufield_p[][3] = {{0.2088, 1.6503, 0.4078}, {-0.0632, -0.6470, 0.1741},
         {-0.3752, -0.6846, -0.5516}, {-0.3275, -0.3521, -0.3940}, {-0.3701, -1.0143, -0.4764},
         {-0.3450, -0.3801, 0.1822}, {-0.4963, -0.2093, 0.5370}, {-0.2139, -0.4325, 0.2956}, {0.5692, -0.4457, -0.4214},
         {0.1598, -1.1058, 0.0538}, {-1.2888, -1.2553, -0.5901}, {-0.9675, -0.4872, 0.6128}, {-0.0813, 0.0744, -0.1494},
         {-0.9896, 0.0318, -0.0839}, {-1.6770, -1.5217, -1.4601}, {-0.7271, -1.1385, -0.8880},
         {-1.1935, 0.2491, -0.0607}, {-0.8772, -1.0243, 0.6095}, {-1.8666, -2.1138, -2.2301},
         {-0.0191, 0.4125, -1.5382}, {0.7158, -1.3417, -0.9288}, {-1.4493, -0.8422, -1.3602}};

      zeroEGV();
      mpoleInit(calc::v0);
      std::vector<std::array<double, 3>> ud, up;
      ud.resize(n);
      up.resize(n);
      for (int i = 0; i < n; ++i) {
         for (int j = 0; j < 3; ++j) {
            ud[i][j] = 0.1 * (i + 1) + 0.03 * (j + 1);
            up[i][j] = 0.1 * (i + 1) - 0.03 * (j + 1);
         }
      }
      darray::copyin(g::q0, n, uind, &ud[0][0]);
      darray::copyin(g::q0, n, uinp, &up[0][0]);
      waitFor(g::q0);
      ufieldEwald(uind, uinp, udir, udirp);
      ud.resize(n);
      up.resize(n);
      darray::copyout(g::q0, n, &ud[0][0], udir);
      darray::copyout(g::q0, n, &up[0][0], udirp);
      waitFor(g::q0);

      for (int i = 0; i < n; ++i) {
         for (int j = 0; j < 3; ++j) {
            REQUIRE(ud[i][j] == Approx(ref_ufield_d[i][j]).margin(eps_f));
            REQUIRE(up[i][j] == Approx(ref_ufield_p[i][j]).margin(eps_f));
         }
      }
   }

   SECTION("default induce routine -- pme")
   {
      const double ref_ud_debye[][3] = {{0.0243, -0.0845, -0.0109}, {1.2653, -0.8136, 0.5372},
         {0.2268, -0.0683, 0.0078}, {0.0616, -0.0298, 0.0188}, {0.1019, -0.0927, 0.0913}, {0.5383, -0.5306, 0.1597},
         {0.1600, -0.1537, -0.0985}, {0.5290, -0.4213, -0.0113}, {0.2702, -0.0662, 0.1661}, {0.2420, -0.1248, -0.0516},
         {0.1211, -0.0628, -0.0252}, {0.0168, -0.0089, -0.0295}, {0.0499, -0.0125, -0.0515}, {0.0365, -0.0617, -0.0396},
         {-0.0212, -0.1500, 0.3140}, {0.1487, -0.1622, 0.1192}, {-0.0466, 0.0288, 0.1480}, {-0.0021, -0.1708, 0.4490},
         {0.2273, -0.6081, -0.0940}, {0.0143, -0.1151, -0.0139}, {0.3569, -0.1912, 0.0563},
         {-0.2176, -0.4640, -0.1812}};
      const double ref_up_debye[][3] = {{0.0241, -0.0843, -0.0110}, {1.2706, -0.8115, 0.5494},
         {0.2253, -0.0684, 0.0097}, {0.0612, -0.0301, 0.0192}, {0.1011, -0.0925, 0.0915}, {0.5377, -0.5296, 0.1603},
         {0.1596, -0.1533, -0.0979}, {0.5288, -0.4205, -0.0101}, {0.3149, -0.0836, 0.1982}, {0.2894, -0.1421, -0.0248},
         {0.0447, -0.0298, -0.0822}, {0.0657, -0.0302, 0.0391}, {0.1486, -0.0288, -0.0009}, {0.1037, -0.1202, 0.0087},
         {-0.0179, -0.1459, 0.3136}, {0.1491, -0.1603, 0.1190}, {-0.0422, 0.0305, 0.1471}, {-0.0008, -0.1699, 0.4506},
         {0.2227, -0.6054, -0.0918}, {0.0119, -0.1147, -0.0138}, {0.3551, -0.1907, 0.0577},
         {-0.2191, -0.4626, -0.1832}};

      zeroEGV();
      mpoleInit(calc::v0);
      induce(uind, uinp);
      std::vector<std::array<double, 3>> ud, up;
      ud.resize(n);
      up.resize(n);
      darray::copyout(g::q0, n, &ud[0][0], uind);
      darray::copyout(g::q0, n, &up[0][0], uinp);
      waitFor(g::q0);
      for (int i = 0; i < n; ++i) {
         for (int j = 0; j < 3; ++j) {
            REQUIRE(ud[i][j] * debye == Approx(ref_ud_debye[i][j]).margin(eps_f));
            REQUIRE(up[i][j] * debye == Approx(ref_up_debye[i][j]).margin(eps_f));
         }
      }
   }

   SECTION("epolar via dot product")
   {
      const double ref_eng = -36.5477;

      zeroEGV();
      epolar(calc::v0);
      COMPARE_ENERGY(ep, ref_eng, eps_f);
   }

   SECTION("various epolar versions -- pme")
   {
      const double eps_e = eps_f;
      const double ref_eng = -36.5477;
      const int ref_count = 222;
      const double eps_g = eps_f;
      const double ref_grad[][3] = {{6.2894, 20.2084, 5.8392}, {13.1726, -5.7387, 7.3043}, {-3.1627, 0.7221, 1.9420},
         {0.7778, 0.1382, -0.0498}, {1.1185, -1.4977, -2.3294}, {4.2613, -2.4784, 3.6942}, {0.8135, -0.1526, 1.3546},
         {-13.1983, 5.5264, -0.7154}, {-0.8940, 0.5947, -4.0091}, {0.3292, -1.6380, 1.3897}, {0.1622, 0.8818, -0.3597},
         {-0.1040, -0.2417, -0.0405}, {0.0955, -0.1813, 0.2299}, {-0.0968, -0.5834, -0.5770},
         {-0.5504, 1.0980, -0.3873}, {0.7537, -0.3906, -0.1910}, {-0.2020, 0.0218, 0.7480}, {2.1083, 1.2493, -6.2425},
         {0.8625, -6.5883, -2.0177}, {-1.9364, 1.0426, 1.6333}, {-2.8554, -1.0075, -1.8528},
         {-7.7509, -10.9759, -5.4845}};
      const double eps_v = 0.001;
      const double ref_v[][3] = {{46.473, 4.246, 13.751}, {4.246, 36.899, 8.584}, {13.751, 8.584, 33.337}};

      COMPARE_CODE_BLOCK2;
   }

   finish();
   testEnd();
}
