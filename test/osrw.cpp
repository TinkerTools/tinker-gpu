#include "md/osrw.h"
#include "ff/evdw.h"
#include "ff/modamoeba.h"

#include "test.h"
#include "testrt.h"

using namespace tinker;

static std::string k0 = "osrw-lambda   0.5\n";
static const char* k1 = "test_osrw.key";
static const char* x1 = "test_osrw.xyz";
static const char* argv[] = {"dummy", x1};
static int argc = 2;

static const double eps = 0.0001;
static const double veps = 0.001;
static const double geps = 0.0005;

static const double s1 = 0.5;
static const double refev_1 = 4.1749;
static const double refem_1 = -42.4972;
static const double refep_1 = -4.0921;
static const double refe_1 = -42.2312;
static const double refvir_1[][3] = {{7.948, -0.088, -0.017}, {-0.088, 7.438, 0.025}, {-0.017, 0.025, 7.874}};
static const double refg_1[][3] = {{0.0025, 0.0004, -0.0089}, {0.0031, -0.0058, -0.0063}, {-0.0037, 0.0011, 0.0042},
   {-0.0068, -0.0027, -0.0003}};

static const double s0 = 0.5;
static const double refev_0 = 0;
static const double refem_0 = -0.0234;
static const double refep_0 = -0.0000;
static const double refe_0 = 0.1598;
static const double refvir_0[][3] = {{2.888, -4.334, -1.242}, {-4.334, 6.601, 1.632}, {-1.242, 1.632, 4.096}};
static const double refg_0[][3] = {{0.0000, 0.0000, 0.0000}, {-9.1476, 13.6765, 5.6035}, {4.2502, -6.1981, -4.9165},
   {4.9047, -7.4853, -0.6896}};

TEST_CASE("K-Water", "[ff][osrw][mixcuda]")
{
   TestFile fx(TINKER9_DIRSTR "/test/file/kwater/kwater.xyz", x1);
   TestFile fk(TINKER9_DIRSTR "/test/file/kwater/kwater.key", k1, k0);
   TestFile fp(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");
   int flag = calc::xyz | calc::vmask;
   flag &= ~calc::analyz;

   rc_flag = flag;
   testBeginWithArgs(argc, argv);
   initialize();

   osrw_energy(calc::v0);
   double refe = s1 * refe_1 + s0 * refe_0;
   REQUIRE(esum == Approx(refe).margin(eps));

   double refdu1 = refe_1 - refe_0;
   REQUIRE(osrw_du1 == Approx(refdu1).margin(eps));

   osrw_energy(calc::v6);
   double gdx[4], gdy[4], gdz[4];
   copyGradient(calc::v6, gdx, gdy, gdz);

   for (int i = 0; i < 4; ++i) {
      double refgx = s1 * refg_1[i][0] + s0 * refg_0[i][0];
      double refgy = s1 * refg_1[i][1] + s0 * refg_0[i][1];
      double refgz = s1 * refg_1[i][2] + s0 * refg_0[i][2];
      REQUIRE(gdx[i] == Approx(refgx).margin(geps));
      REQUIRE(gdy[i] == Approx(refgy).margin(geps));
      REQUIRE(gdz[i] == Approx(refgz).margin(geps));
   }
   double d1x[4], d1y[4], d1z[4];
   copyGradient(calc::v6, d1x, d1y, d1z, osrw_dgx, osrw_dgy, osrw_dgz);
   for (int i = 0; i < 4; ++i) {
      double refgx1 = refg_1[i][0] - refg_0[i][0];
      double refgy1 = refg_1[i][1] - refg_0[i][1];
      double refgz1 = refg_1[i][2] - refg_0[i][2];
      REQUIRE(d1x[i] == Approx(refgx1).margin(geps));
      REQUIRE(d1y[i] == Approx(refgy1).margin(geps));
      REQUIRE(d1z[i] == Approx(refgz1).margin(geps));
   }
   for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
         int iv = 3 * i + j;
         double refv = s1 * refvir_1[i][j] + s0 * refvir_0[i][j];
         REQUIRE(vir[iv] == Approx(refv).margin(veps));

         double refv1 = refvir_1[i][j] - refvir_0[i][j];
         REQUIRE(osrw_dv1[iv] == Approx(refv1).margin(veps));
      }
   }

   finish();
   testEnd();
}

TEST_CASE("K-Water-Analyze", "[ff][osrw][mixcuda]")
{
   TestFile fx(TINKER9_DIRSTR "/test/file/kwater/kwater.xyz", x1);
   TestFile fk(TINKER9_DIRSTR "/test/file/kwater/kwater.key", k1, k0);
   TestFile fp(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");
   int flag = calc::xyz | calc::vmask;

   rc_flag = flag;
   testBeginWithArgs(argc, argv);
   initialize();

   osrw_energy(calc::v3);
   double refev = s1 * refev_1 + s0 * refev_0;
   REQUIRE(energy_ev == Approx(refev).margin(eps));
   double refem = s1 * refem_1 + s0 * refem_0;
   REQUIRE(energy_em == Approx(refem).margin(eps));
   double refep = s1 * refep_1 + s0 * refep_0;
   REQUIRE(energy_ep == Approx(refep).margin(eps));
   double refe = s1 * refe_1 + s0 * refe_0;
   REQUIRE(esum == Approx(refe).margin(eps));

   double refdu1 = refe_1 - refe_0;
   REQUIRE(osrw_du1 == Approx(refdu1).margin(eps));

   osrw_energy(calc::v6);
   double gdx[4], gdy[4], gdz[4];
   copyGradient(calc::v6, gdx, gdy, gdz);
   for (int i = 0; i < 4; ++i) {
      double refgx = s1 * refg_1[i][0] + s0 * refg_0[i][0];
      double refgy = s1 * refg_1[i][1] + s0 * refg_0[i][1];
      double refgz = s1 * refg_1[i][2] + s0 * refg_0[i][2];
      REQUIRE(gdx[i] == Approx(refgx).margin(geps));
      REQUIRE(gdy[i] == Approx(refgy).margin(geps));
      REQUIRE(gdz[i] == Approx(refgz).margin(geps));
   }
   double d1x[4], d1y[4], d1z[4];
   copyGradient(calc::v6, d1x, d1y, d1z, osrw_dgx, osrw_dgy, osrw_dgz);
   for (int i = 0; i < 4; ++i) {
      double refgx1 = refg_1[i][0] - refg_0[i][0];
      double refgy1 = refg_1[i][1] - refg_0[i][1];
      double refgz1 = refg_1[i][2] - refg_0[i][2];
      REQUIRE(d1x[i] == Approx(refgx1).margin(geps));
      REQUIRE(d1y[i] == Approx(refgy1).margin(geps));
      REQUIRE(d1z[i] == Approx(refgz1).margin(geps));
   }
   for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
         int iv = 3 * i + j;
         double refv = s1 * refvir_1[i][j] + s0 * refvir_0[i][j];
         REQUIRE(vir[iv] == Approx(refv).margin(veps));

         double refv1 = refvir_1[i][j] - refvir_0[i][j];
         REQUIRE(osrw_dv1[iv] == Approx(refv1).margin(veps));
      }
   }

   finish();
   testEnd();
}
