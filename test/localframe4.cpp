#include "ff/energy.h"

#include "test.h"
#include "testrt.h"

using namespace tinker;

// BRD4 ligand 04 (a tBu sulfonamide) at a frame saved from NPT dynamics,
// one step before the run blew up. The N-H hydrogen (atom 59, type 446)
// uses a z-then-bisector frame whose bisector atoms are two different
// 1-3 neighbors:
//
//    multipole   446  422 -420 -424      z: N28   bisector: C24, S31
//
// Seen from H59, C24 and S31 both lie behind N28, so the projection of
// H59->C24 onto the (z, bisector) plane is nearly parallel to z here:
// cos = 0.99999999986 in double precision. In single precision it rounds
// to 1 or a little above, and the unclamped sqrt(1 - cos*cos) in the
// Z-Bisect torque returns NaN. The force itself is finite, because the
// torque divides by the sum of two sines (about 0.42 here). Before the
// clamp, the gradients on atoms 24, 31 and 59 came out as +/-2^31 (a NaN
// in fixed-point accumulation) and the ligand left the box in one step.
// The energy is the same with or without the clamp; only the gradient
// catches the bug.
//
// Reference values: double-precision CPU Tinker (testgrad), multipoles only.
static int usage = calc::xyz | calc::vmask;
static const char* key0 = R"**(
parameters  local_frame4
multipoleterm  only
)**";
static const char* kname = "test_local_frame4.key";
static const char* xname = "test_local_frame4.xyz";
static const char* argv[] = {"dummy", xname};
static int argc = 2;

TEST_CASE("Local-Frame4-1", "[ff][empole][nonewald][z-bisect][local-frame4]")
{
   TestFile fxy(TINKER9_DIRSTR "/test/file/local_frame/local_frame4.xyz", xname);
   TestFile fke("", kname, key0);
   TestFile fpr(TINKER9_DIRSTR "/test/file/local_frame/local_frame4.prm");

   TestSession session(argc, argv);
   rc_flag = usage;
   session.init();

   SECTION("  - empole -- near-singular z-bisector frame, gas phase")
   {
      const double ref_eng = 18.6100;
      const double ref_g[][3] = {{-0.2970, 0.3397, 0.1157}, {-0.6278, -0.8746, -2.0148}, {0.7655, 0.7307, 0.4144},
         {1.2706, -0.0021, 1.0567}, {-1.8827, -1.3401, -6.5137}, {1.1918, 0.8448, 0.8178}, {0.2216, 0.1246, 0.4747},
         {0.5735, -1.3576, 0.6473}, {-0.4777, 1.9716, -1.9614}, {0.3782, -0.3939, -0.1251}, {-2.0135, -0.2487, -0.2263},
         {2.1162, 0.4623, 1.0847}, {-1.3720, 2.1997, -1.5175}, {0.4456, 0.0245, 0.6104}, {-0.4391, -0.2267, 0.0159},
         {0.2325, -0.0315, -0.5958}, {0.8745, 0.6424, 1.0164}, {2.0104, -3.6316, -0.4200}, {-1.8763, -0.3574, 0.9060},
         {1.4205, 5.3414, 9.0313}, {0.4601, -0.2686, -1.7546}, {-0.2436, -0.1073, -1.7922}, {-1.0586, -0.2191, -0.6246},
         {-0.0379, 1.4976, -1.6652}, {-2.1138, -2.2010, 1.5071}, {1.9765, 0.5874, 0.7761}, {-0.8700, -2.8870, 2.6143},
         {2.0690, -4.3538, 3.9249}, {-2.0224, -1.2838, 4.1112}, {-1.0752, 0.7126, 6.9158}, {-0.9251, 2.9648, -9.7777},
         {3.2040, 3.5069, 3.8255}, {-0.6529, -2.6433, -6.3055}, {-0.5569, 0.7041, 3.8760}, {0.5785, -0.2452, -0.1052},
         {0.4397, 1.5387, 2.9955}, {-1.2512, 0.2184, -0.3174}, {-0.3419, -0.2704, 0.1340}, {-0.5559, -0.4285, -0.1793},
         {-0.3250, -0.9175, -0.4129}, {0.7227, -0.7276, -1.3702}, {-0.4317, -1.0597, 0.1147},
         {-0.1333, -0.2048, -0.3062}, {-0.8743, -0.7244, 0.2959}, {-2.2321, -0.8047, 0.2909}, {1.2407, 0.9602, 1.3679},
         {-1.3539, 1.9914, -0.3941}, {0.0524, 1.0022, -1.9004}, {-0.8440, -2.0324, -1.4156}, {0.4982, -0.0777, -0.3908},
         {0.2509, 0.1194, 0.1074}, {0.2385, -0.6797, -3.5042}, {0.6781, 0.3330, 1.1687}, {-1.0604, 0.1782, 3.0385},
         {-0.3881, -0.9053, -0.7001}, {1.0585, -0.0512, 0.5287}, {-0.9373, -0.7710, -0.0477},
         {0.7307, -0.4286, -1.2092}, {-0.0295, 5.1034, -4.0330}, {0.6198, 0.7975, -1.0707}, {0.1653, -0.0608, -0.2988},
         {-0.2778, -2.1084, 0.8375}, {-1.2680, -0.6657, 1.3733}, {-0.1907, 0.9683, 1.6992}, {0.0865, 2.0584, -0.0500},
         {1.2688, 1.0430, -1.5726}, {0.0989, -1.8376, -2.0415}, {0.7310, 1.4970, -0.4845}, {-0.8646, -0.8673, -0.0027},
         {0.3242, 0.4653, -1.1982}, {0.0621, -0.0815, 0.1620}, {1.1774, -0.2141, 0.6634}, {-0.7957, -0.2243, -0.1778},
         {2.4642, -2.1130, -0.0419}};
      const double eps_e = 0.0001;
      const double eps_g = 0.001;

      energy(calc::v0);
      COMPARE_REALS(esum, ref_eng, eps_e);

      // v1 and v6 run the torque kernel with the virial, v4 and v5 without
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
   }

   session.end();
}
