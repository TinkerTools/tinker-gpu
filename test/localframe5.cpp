#include "ff/energy.h"

#include "test.h"
#include "testrt.h"

using namespace tinker;

// 3-Fold local frame: the z-axis is the unit trisector of the three bond
// vectors and the x-axis is the x-axis bond made perpendicular to it, so
// the spin about z is fixed by the x-axis atom alone. The older force
// distribution rotated each frame atom rigidly about the bisector of the
// other two bonds; that is exact for torque perpendicular to the trisector
// but wrong for torque along it. The torque now uses the exact derivative
// of the frame (frame rotation p x dp + (s.dr) p).
//
// Fixture: triethylamine whose 3-Fold N (atom 3, type 403) has an in-plane
// dipole and off-diagonal quadrupole, with a Na+ (atom 23) 4 Ang from the N
// and 60 degrees off its lone-pair axis, so the torque about the trisector
// is large. Two geometries: pyramidal N (0.41 Ang above the plane of its
// neighbors) and N 0.08 Ang from that plane. The older distribution was off
// by up to about 1 kcal/mol/Ang on N and its frame atoms here.
//
// Reference values: double-precision CPU Tinker (testgrad, analytic) with
// the same 3-Fold frame and the exact torque, multipoles only and
// polarization only; they match the numerical gradient (polarization to
// its convergence).
static int usage = calc::xyz | calc::vmask;
static const char* key_mpole = R"**(
parameters  local_frame5
multipoleterm  only
)**";
static const char* key_polar = R"**(
parameters  local_frame5
polarizeterm  only
polar-eps  0.000001
)**";
static const char* kname = "test_local_frame5.key";
static const char* xname = "test_local_frame5.xyz";
static const char* argv[] = {"dummy", xname};
static int argc = 2;

static const double ref_g_pyr_mpole[][3] = {
   {-0.9549, 3.0193, 7.1919}, {0.6722, -0.8723, -2.9643}, {1.6690, 0.6622, 1.8137}, {-0.3615, 1.7324, 0.2793},
   {2.3747, -0.3995, 1.1910}, {0.0115, -4.7647, 5.9290}, {5.2568, -3.0764, 1.1087}, {0.0328, -0.3826, -0.6946},
   {-0.7700, -1.2595, -1.5283}, {0.7326, -0.3100, -4.1207}, {-0.5327, -0.0130, -0.2780}, {-0.0147, 0.1638, -0.0412},
   {0.0386, -0.0178, 0.3707}, {-0.6868, 0.1615, -0.5623}, {-0.5988, 0.0773, -0.4929}, {-0.7947, -0.1621, -0.3727},
   {-0.3879, -0.2098, -0.3357}, {1.1466, 2.4740, -2.8651}, {-0.5272, -0.6111, -0.6850}, {-0.1997, 0.3841, -0.4022},
   {-4.9894, 1.0366, -1.5102}, {-1.2084, 2.3154, -0.8036}, {0.0919, 0.0523, -0.2278}};
static const double ref_g_pyr_polar[][3] = {
   {0.5703, -1.1631, -0.3950}, {0.2889, 0.3365, 1.0929}, {-0.4716, -0.1272, -0.1887}, {0.6306, 0.0178, 1.0772},
   {0.4727, 0.5111, -0.3486}, {-0.0247, -0.2859, -0.6574}, {0.3902, 0.0990, 1.2249}, {0.1279, 0.0059, 0.0364},
   {0.0895, 0.3073, 0.3971}, {-0.4719, 1.3802, 2.5048}, {0.1930, -0.0291, -0.1429}, {-0.0583, -0.0033, -0.0858},
   {-0.1807, 0.0842, -0.0770}, {-0.3833, 0.2024, -0.2757}, {-0.2599, -0.2000, 0.0938}, {0.0450, -0.1059, 0.1054},
   {-0.3102, -0.0701, -0.0781}, {0.3957, -1.2401, 1.8833}, {0.2844, 0.2142, -0.0130}, {-0.0499, -0.2009, -0.0229},
   {3.1498, -2.1032, 1.0840}, {0.3351, -0.3791, 0.0945}, {-4.7625, 2.7493, -7.3090}};
static const double ref_g_flat_mpole[][3] = {
   {-0.8808, 3.8933, 7.5979}, {-1.1800, -1.5632, -4.0728}, {4.7009, 3.6152, 2.3492}, {-0.1213, 0.4896, 0.2248},
   {2.4440, -0.3271, 0.6892}, {-1.4630, -7.1890, 7.7634}, {6.7505, -2.4500, 0.6106}, {0.0359, -0.4658, -0.6841},
   {-0.7472, -1.4378, -1.4593}, {0.8993, -0.6689, -4.5281}, {-0.7308, 0.0220, -0.4251}, {-0.0990, 0.2037, -0.0717},
   {0.1164, -0.0562, 0.4315}, {-0.7598, 0.1005, -0.4565}, {-0.6106, 0.0259, -0.4584}, {-0.8495, -0.2213, -0.4005},
   {-0.4329, -0.2444, -0.3324}, {0.8187, 2.3315, -3.6571}, {-0.6549, -0.7128, -0.8032}, {-0.2779, 0.3827, -0.3770},
   {-5.5580, 0.9749, -1.0761}, {-1.4321, 2.5929, -0.7392}, {0.0322, 0.7044, -0.1250}};
static const double ref_g_flat_polar[][3] = {
   {0.4548, -1.3632, -0.3493}, {-0.1587, 0.2576, 0.6822}, {-0.0703, 0.5424, 1.0883}, {0.8096, -0.4550, 0.4982},
   {0.4155, 0.5349, -0.3834}, {0.4022, -0.0419, -0.7456}, {0.3456, -0.2477, 1.5208}, {0.1572, 0.0091, 0.0549},
   {0.1055, 0.3399, 0.3646}, {-0.4674, 1.7560, 3.0467}, {0.2262, -0.0092, -0.1744}, {-0.0839, -0.0078, -0.1328},
   {-0.1595, 0.0680, -0.1059}, {-0.4325, 0.1384, -0.2342}, {-0.2767, -0.2131, 0.0878}, {0.0448, -0.1255, 0.1297},
   {-0.3327, -0.0903, -0.0760}, {0.7606, -1.7333, 2.9523}, {0.3440, 0.2570, -0.0413}, {-0.1092, -0.1859, -0.0175},
   {3.6463, -2.0425, 0.9095}, {0.4387, -0.4833, 0.1234}, {-6.0599, 3.0953, -9.1977}};

// v1 and v6 run the torque kernel with the virial, v4 and v5 without
static void checkThreeFold(const char* xyz, const char* key, double ref_eng, const double (*ref_g)[3])
{
   TestFile fxy(xyz, xname);
   TestFile fke("", kname, key);
   TestFile fpr(TINKER9_DIRSTR "/test/file/local_frame/local_frame5.prm");

   TestSession session(argc, argv);
   rc_flag = usage;
   session.init();

   const double eps_e = 0.0001;
   const double eps_g = 0.001;
   energy(calc::v0);
   COMPARE_REALS(esum, ref_eng, eps_e);
   energy(calc::v1);
   COMPARE_REALS(esum, ref_eng, eps_e);
   COMPARE_GRADIENT(ref_g, eps_g);
   energy(calc::v4);
   COMPARE_REALS(esum, ref_eng, eps_e);
   COMPARE_GRADIENT(ref_g, eps_g);
   energy(calc::v5);
   COMPARE_GRADIENT(ref_g, eps_g);
   energy(calc::v6);
   COMPARE_GRADIENT(ref_g, eps_g);

   session.end();
}

TEST_CASE("Local-Frame5-1", "[ff][empole][nonewald][3-fold][local-frame5][L1]")
{
   SECTION("  - empole -- 3-Fold N, pyramidal")
   {
      checkThreeFold(TINKER9_DIRSTR "/test/file/local_frame/local_frame5.xyz", key_mpole, -3.7058, ref_g_pyr_mpole);
   }
   SECTION("  - empole -- 3-Fold N, near planar")
   {
      checkThreeFold(TINKER9_DIRSTR "/test/file/local_frame/local_frame5_flat.xyz", key_mpole, -4.0180, ref_g_flat_mpole);
   }
}

TEST_CASE("Local-Frame5-2", "[ff][epolar][nonewald][3-fold][local-frame5][L1]")
{
   SECTION("  - epolar -- 3-Fold N, pyramidal")
   {
      checkThreeFold(TINKER9_DIRSTR "/test/file/local_frame/local_frame5.xyz", key_polar, -9.8754, ref_g_pyr_polar);
   }
   SECTION("  - epolar -- 3-Fold N, near planar")
   {
      checkThreeFold(TINKER9_DIRSTR "/test/file/local_frame/local_frame5_flat.xyz", key_polar, -11.6995,
         ref_g_flat_polar);
   }
}
