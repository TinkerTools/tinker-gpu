#include "ff/amoeba/induce.h"
#include "ff/atom.h"
#include "ff/energy.h"
#include "ff/modamoeba.h"
#include "ff/potent.h"
#include "md/integrator.h"
#include "md/misc.h"
#include <tinker/detail/bath.hh>
#include <tinker/detail/boxes.hh>
#include <tinker/detail/inform.hh>
#include <tinker/detail/polpot.hh>

#include "test.h"
#include "testrt.h"

#include <vector>

using namespace tinker;

static int usage_ = calc::xyz | calc::vel | calc::mass | calc::energy | calc::grad | calc::md | calc::virial;

static BasicIntegrator* intg;
static Box dup_buf_box;

TEST_CASE("NPT-Berendsen-Iso", "[ff][npt][Berendsen][iso]")
{
   const char* txt_intg = R"**(
   integrator   respa
   thermostat   nose-hoover
   barostat     berendsen
   tau-pressure 2.0
   respa-inner  4
   )**";

   double ref_ekin[] = {1630.7885, 1639.6993, 1631.9428, 1640.2466, 1648.9550};
   double ref_epot[] = {-9180.1585, -9190.3660, -9183.6644, -9192.2068, -9200.4336};
   double ref_latlen[] = {29.856794, 29.856794, 29.856794, 29.856852, 29.856852, 29.856852, 29.856588, 29.856588,
      29.856588, 29.856176, 29.856176, 29.856176, 29.855938, 29.855938, 29.855938};
   double ref_latang[] = {90.000000, 90.000000, 90.000000, 90.000000, 90.000000, 90.000000, 90.000000, 90.000000,
      90.000000, 90.000000, 90.000000, 90.000000, 90.000000, 90.000000, 90.000000};

   const char* k = "test_water30.key";
   const char* d = "test_water30.dyn";
   const char* x = "test_water30.xyz";

   std::string k0 = txt_intg;
   TestFile fke(TINKER9_DIRSTR "/test/file/water30/water30.key", k, k0);

   TestFile fd(TINKER9_DIRSTR "/test/file/water30/water30_iso.dyn", d);
   TestFile fx(TINKER9_DIRSTR "/test/file/water30/water30_iso.xyz", x);
   TestFile fp(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");

   const char* argv[] = {"dummy", x};
   int argc = 2;
   testBeginWithArgs(argc, argv);
   testMdInit(298., 1.);

   rc_flag = usage_;
   initialize();

   const double dt_ps = 0.002;
   const int nsteps = 5;
   const double eps_e = 0.0001;
   const double eps_l = 1e-6;
   std::vector<double> epots, eksums, latlen, latang;
   int old = inform::iwrite;
   inform::iwrite = 1;
   ThermostatEnum thermostat = ThermostatEnum::NHC;
   BarostatEnum barostat = BarostatEnum::BERENDSEN;
   intg = new RespaIntegrator(thermostat, barostat);
   for (int i = 1; i <= nsteps; ++i) {
      intg->dynamic(i, dt_ps);
      T_prec temp;
      kinetic(temp);
      epots.push_back(esum);
      eksums.push_back(eksum);
      boxGetCurrent(dup_buf_box);
      boxSetTinker(dup_buf_box);
      latlen.push_back(boxes::xbox);
      latlen.push_back(boxes::ybox);
      latlen.push_back(boxes::zbox);
      latang.push_back(boxes::alpha);
      latang.push_back(boxes::beta);
      latang.push_back(boxes::gamma);
   }
   inform::iwrite = old;

   delete intg;
   intg = nullptr;
   finish();
   testEnd();

   for (int i = 0; i < nsteps; ++i) {
      REQUIRE(epots[i] == Approx(ref_epot[i]).margin(eps_e));
      REQUIRE(eksums[i] == Approx(ref_ekin[i]).margin(eps_e));
   }

   for (int i = 0; i < 3 * nsteps; ++i) {
      REQUIRE(latlen[i] == Approx(ref_latlen[i]).margin(eps_l));
      REQUIRE(latang[i] == Approx(ref_latang[i]).margin(eps_l));
   }

   TestRemoveFileOnExit("test_water30.arc");
   bath::kelvin = 0.;
   bath::atmsph = 0.;
   bath::isothermal = 0;
   bath::isobaric = 0;
}

TEST_CASE("NPT-Berendsen-Semiiso", "[ff][npt][Berendsen][semiiso]")
{
   const char* txt_intg = R"**(
   integrator   respa
   thermostat   nose-hoover
   barostat     berendsen
   tau-pressure 2.0
   respa-inner  4
   pressure     semi
   )**";

   double ref_ekin[] = {2483.2144, 2490.7158, 2503.0586, 2479.1602, 2533.9223};
   double ref_epot[] = {-7260.0687, -7269.5653, -7283.1059, -7256.2822, -7311.8789};
   double ref_latlen[] = {36.250446, 36.250446, 27.983412, 36.250164, 36.250164, 27.983194, 36.249752, 36.249752,
      27.982458, 36.249252, 36.249252, 27.982252, 36.248669, 36.248669, 27.982710};
   double ref_latang[] = {90.000000, 90.000000, 90.000000, 90.000000, 90.000000, 90.000000, 90.000000, 90.000000,
      90.000000, 90.000000, 90.000000, 90.000000, 90.000000, 90.000000, 90.000000};

   const char* k = "test_water30.key";
   const char* d = "test_water30.dyn";
   const char* x = "test_water30.xyz";

   std::string k0 = txt_intg;
   TestFile fke(TINKER9_DIRSTR "/test/file/water30/water30.key", k, k0);

   TestFile fd(TINKER9_DIRSTR "/test/file/water30/water30_semiiso.dyn", d);
   TestFile fx(TINKER9_DIRSTR "/test/file/water30/water30_semiiso.xyz", x);
   TestFile fp(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");

   const char* argv[] = {"dummy", x};
   int argc = 2;
   testBeginWithArgs(argc, argv);
   testMdInit(298., 1.);

   rc_flag = usage_;
   initialize();

   const double dt_ps = 0.002;
   const int nsteps = 5;
   const double eps_e = 0.0001;
   const double eps_l = 1e-6;
   std::vector<double> epots, eksums, latlen, latang;
   int old = inform::iwrite;
   inform::iwrite = 1;
   ThermostatEnum thermostat = ThermostatEnum::NHC;
   BarostatEnum barostat = BarostatEnum::BERENDSEN;
   intg = new RespaIntegrator(thermostat, barostat);
   for (int i = 1; i <= nsteps; ++i) {
      intg->dynamic(i, dt_ps);
      T_prec temp;
      kinetic(temp);
      epots.push_back(esum);
      eksums.push_back(eksum);
      boxGetCurrent(dup_buf_box);
      boxSetTinker(dup_buf_box);
      latlen.push_back(boxes::xbox);
      latlen.push_back(boxes::ybox);
      latlen.push_back(boxes::zbox);
      latang.push_back(boxes::alpha);
      latang.push_back(boxes::beta);
      latang.push_back(boxes::gamma);
   }
   inform::iwrite = old;

   delete intg;
   intg = nullptr;
   finish();
   testEnd();

   for (int i = 0; i < nsteps; ++i) {
      REQUIRE(epots[i] == Approx(ref_epot[i]).margin(eps_e));
      REQUIRE(eksums[i] == Approx(ref_ekin[i]).margin(eps_e));
   }

   for (int i = 0; i < 3 * nsteps; ++i) {
      REQUIRE(latlen[i] == Approx(ref_latlen[i]).margin(eps_l));
      REQUIRE(latang[i] == Approx(ref_latang[i]).margin(eps_l));
   }

   TestRemoveFileOnExit("test_water30.arc");
   bath::kelvin = 0.;
   bath::atmsph = 0.;
   bath::isothermal = 0;
   bath::isobaric = 0;
}

TEST_CASE("NPT-Berendsen-Aniso", "[ff][npt][Berendsen][aniso]")
{
   const char* txt_intg = R"**(
   integrator   respa
   thermostat   nose-hoover
   barostat     berendsen
   tau-pressure 2.0
   respa-inner  4
   pressure     aniso
   )**";

   double ref_ekin[] = {2395.2272, 2375.9748, 2405.2844, 2422.3316, 2418.1192};
   double ref_epot[] = {-7792.5164, -7772.7474, -7803.1145, -7820.1069, -7814.9148};
   double ref_latlen[] = {30.546352, 26.475403, 34.816055, 30.545927, 26.475521, 34.815452, 30.545654, 26.474924,
      34.814811, 30.545559, 26.474230, 34.814269, 30.545351, 26.474248, 34.814030};
   double ref_latang[] = {92.286898, 100.717920, 88.330350, 92.287554, 100.716516, 88.331966, 92.287782, 100.715079,
      88.333800, 92.287003, 100.716232, 88.334700, 92.285767, 100.719048, 88.335112};

   const char* k = "test_water30.key";
   const char* d = "test_water30.dyn";
   const char* x = "test_water30.xyz";

   std::string k0 = txt_intg;
   TestFile fke(TINKER9_DIRSTR "/test/file/water30/water30.key", k, k0);

   TestFile fd(TINKER9_DIRSTR "/test/file/water30/water30_aniso.dyn", d);
   TestFile fx(TINKER9_DIRSTR "/test/file/water30/water30_aniso.xyz", x);
   TestFile fp(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");

   const char* argv[] = {"dummy", x};
   int argc = 2;
   testBeginWithArgs(argc, argv);
   testMdInit(298., 1.);

   rc_flag = usage_;
   initialize();

   const double dt_ps = 0.002;
   const int nsteps = 5;
   const double eps_e = 0.0001;
   const double eps_l = 1e-6;
   std::vector<double> epots, eksums, latlen, latang;
   int old = inform::iwrite;
   inform::iwrite = 1;
   ThermostatEnum thermostat = ThermostatEnum::NHC;
   BarostatEnum barostat = BarostatEnum::BERENDSEN;
   intg = new RespaIntegrator(thermostat, barostat);
   for (int i = 1; i <= nsteps; ++i) {
      intg->dynamic(i, dt_ps);
      T_prec temp;
      kinetic(temp);
      epots.push_back(esum);
      eksums.push_back(eksum);
      boxGetCurrent(dup_buf_box);
      boxSetTinker(dup_buf_box);
      latlen.push_back(boxes::xbox);
      latlen.push_back(boxes::ybox);
      latlen.push_back(boxes::zbox);
      latang.push_back(boxes::alpha);
      latang.push_back(boxes::beta);
      latang.push_back(boxes::gamma);
   }
   inform::iwrite = old;

   delete intg;
   intg = nullptr;
   finish();
   testEnd();

   for (int i = 0; i < nsteps; ++i) {
      REQUIRE(epots[i] == Approx(ref_epot[i]).margin(eps_e));
      REQUIRE(eksums[i] == Approx(ref_ekin[i]).margin(eps_e));
   }

   for (int i = 0; i < 3 * nsteps; ++i) {
      REQUIRE(latlen[i] == Approx(ref_latlen[i]).margin(eps_l));
      REQUIRE(latang[i] == Approx(ref_latang[i]).margin(eps_l));
   }

   TestRemoveFileOnExit("test_water30.arc");
   bath::kelvin = 0.;
   bath::atmsph = 0.;
   bath::isothermal = 0;
   bath::isobaric = 0;
}

TEST_CASE("NPT-Bussi-Iso", "[ff][npt][Bussi][iso]")
{
   const char* txt_intg = R"**(
   integrator   respa
   thermostat   nose-hoover
   barostat     bussi
   tau-pressure 2.0
   respa-inner  4
   randomseed   1
   )**";

   double ref_ekin[] = {1630.8349, 1640.1864, 1632.0956, 1640.3652, 1649.5452};
   double ref_epot[] = {-9180.1586, -9190.3978, -9183.9795, -9192.1187, -9200.9539};
   double ref_latlen[] = {29.856340, 29.856340, 29.856340, 29.852356, 29.852356, 29.852356, 29.854733, 29.854733,
      29.854733, 29.850239, 29.850239, 29.850239, 29.850168, 29.850168, 29.850168};
   double ref_latang[] = {90.000000, 90.000000, 90.000000, 90.000000, 90.000000, 90.000000, 90.000000, 90.000000,
      90.000000, 90.000000, 90.000000, 90.000000, 90.000000, 90.000000, 90.000000};

   const char* k = "test_water30.key";
   const char* d = "test_water30.dyn";
   const char* x = "test_water30.xyz";

   std::string k0 = txt_intg;
   TestFile fke(TINKER9_DIRSTR "/test/file/water30/water30.key", k, k0);

   TestFile fd(TINKER9_DIRSTR "/test/file/water30/water30_iso.dyn", d);
   TestFile fx(TINKER9_DIRSTR "/test/file/water30/water30_iso.xyz", x);
   TestFile fp(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");

   const char* argv[] = {"dummy", x};
   int argc = 2;
   testBeginWithArgs(argc, argv);
   testMdInit(298., 1.);

   rc_flag = usage_;
   initialize();

   const double dt_ps = 0.002;
   const int nsteps = 5;
   const double eps_e = 0.0001;
   const double eps_l = 1e-6;
   std::vector<double> epots, eksums, latlen, latang;
   int old = inform::iwrite;
   inform::iwrite = 1;
   ThermostatEnum thermostat = ThermostatEnum::NHC;
   BarostatEnum barostat = BarostatEnum::BUSSI;
   intg = new RespaIntegrator(thermostat, barostat);
   for (int i = 1; i <= nsteps; ++i) {
      intg->dynamic(i, dt_ps);
      T_prec temp;
      kinetic(temp);
      epots.push_back(esum);
      eksums.push_back(eksum);
      boxGetCurrent(dup_buf_box);
      boxSetTinker(dup_buf_box);
      latlen.push_back(boxes::xbox);
      latlen.push_back(boxes::ybox);
      latlen.push_back(boxes::zbox);
      latang.push_back(boxes::alpha);
      latang.push_back(boxes::beta);
      latang.push_back(boxes::gamma);
   }
   inform::iwrite = old;

   delete intg;
   intg = nullptr;
   finish();
   testEnd();

   for (int i = 0; i < nsteps; ++i) {
      REQUIRE(epots[i] == Approx(ref_epot[i]).margin(eps_e));
      REQUIRE(eksums[i] == Approx(ref_ekin[i]).margin(eps_e));
   }

   for (int i = 0; i < 3 * nsteps; ++i) {
      REQUIRE(latlen[i] == Approx(ref_latlen[i]).margin(eps_l));
      REQUIRE(latang[i] == Approx(ref_latang[i]).margin(eps_l));
   }

   TestRemoveFileOnExit("test_water30.arc");
   bath::kelvin = 0.;
   bath::atmsph = 0.;
   bath::isothermal = 0;
   bath::isobaric = 0;
}

TEST_CASE("NPT-Bussi-Semiiso", "[ff][npt][Bussi][semiiso]")
{
   const char* txt_intg = R"**(
   integrator   respa
   thermostat   nose-hoover
   barostat     bussi
   tau-pressure 2.0
   respa-inner  4
   randomseed   1
   pressure     semi
   )**";

   double ref_ekin[] = {2483.6151, 2491.2342, 2502.6389, 2479.2860, 2534.5894};
   double ref_epot[] = {-7260.0687, -7269.9012, -7282.9407, -7256.1530, -7312.3325};
   double ref_latlen[] = {36.249870, 36.249870, 27.977816, 36.252842, 36.252842, 27.972122, 36.252560, 36.252560,
      27.975256, 36.252182, 36.252182, 27.972260, 36.253807, 36.253807, 27.973774};
   double ref_latang[] = {90.000000, 90.000000, 90.000000, 90.000000, 90.000000, 90.000000, 90.000000, 90.000000,
      90.000000, 90.000000, 90.000000, 90.000000, 90.000000, 90.000000, 90.000000};

   const char* k = "test_water30.key";
   const char* d = "test_water30.dyn";
   const char* x = "test_water30.xyz";

   std::string k0 = txt_intg;
   TestFile fke(TINKER9_DIRSTR "/test/file/water30/water30.key", k, k0);

   TestFile fd(TINKER9_DIRSTR "/test/file/water30/water30_semiiso.dyn", d);
   TestFile fx(TINKER9_DIRSTR "/test/file/water30/water30_semiiso.xyz", x);
   TestFile fp(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");

   const char* argv[] = {"dummy", x};
   int argc = 2;
   testBeginWithArgs(argc, argv);
   testMdInit(298., 1.);

   rc_flag = usage_;
   initialize();

   const double dt_ps = 0.002;
   const int nsteps = 5;
   const double eps_e = 0.0001;
   const double eps_l = 1e-6;
   std::vector<double> epots, eksums, latlen, latang;
   int old = inform::iwrite;
   inform::iwrite = 1;
   ThermostatEnum thermostat = ThermostatEnum::NHC;
   BarostatEnum barostat = BarostatEnum::BUSSI;
   intg = new RespaIntegrator(thermostat, barostat);
   for (int i = 1; i <= nsteps; ++i) {
      intg->dynamic(i, dt_ps);
      T_prec temp;
      kinetic(temp);
      epots.push_back(esum);
      eksums.push_back(eksum);
      boxGetCurrent(dup_buf_box);
      boxSetTinker(dup_buf_box);
      latlen.push_back(boxes::xbox);
      latlen.push_back(boxes::ybox);
      latlen.push_back(boxes::zbox);
      latang.push_back(boxes::alpha);
      latang.push_back(boxes::beta);
      latang.push_back(boxes::gamma);
   }
   inform::iwrite = old;

   delete intg;
   intg = nullptr;
   finish();
   testEnd();

   for (int i = 0; i < nsteps; ++i) {
      REQUIRE(epots[i] == Approx(ref_epot[i]).margin(eps_e));
      REQUIRE(eksums[i] == Approx(ref_ekin[i]).margin(eps_e));
   }

   for (int i = 0; i < 3 * nsteps; ++i) {
      REQUIRE(latlen[i] == Approx(ref_latlen[i]).margin(eps_l));
      REQUIRE(latang[i] == Approx(ref_latang[i]).margin(eps_l));
   }

   TestRemoveFileOnExit("test_water30.arc");
   bath::kelvin = 0.;
   bath::atmsph = 0.;
   bath::isothermal = 0;
   bath::isobaric = 0;
}

TEST_CASE("NPT-Bussi-Aniso", "[ff][npt][Bussi][aniso]")
{
   const char* txt_intg = R"**(
   integrator   respa
   thermostat   nose-hoover
   barostat     bussi
   tau-pressure 2.0
   respa-inner  4
   randomseed   1
   pressure     aniso
   )**";

   double ref_ekin[] = {2395.1267, 2375.2113, 2405.8904, 2424.0741, 2417.4446};
   double ref_epot[] = {-7792.5163, -7772.7773, -7802.1268, -7820.9907, -7814.5243};
   double ref_latlen[] = {30.545563, 26.475237, 34.818195, 30.546438, 26.480258, 34.827250, 30.540039, 26.467727,
      34.826973, 30.535694, 26.463077, 34.828659, 30.543438, 26.456017, 34.841805};
   double ref_latang[] = {92.279012, 100.707328, 88.356677, 92.296240, 100.704388, 88.320018, 92.288354, 100.719542,
      88.360654, 92.274500, 100.698158, 88.357584, 92.275968, 100.709553, 88.342381};

   const char* k = "test_water30.key";
   const char* d = "test_water30.dyn";
   const char* x = "test_water30.xyz";

   std::string k0 = txt_intg;
   TestFile fke(TINKER9_DIRSTR "/test/file/water30/water30.key", k, k0);

   TestFile fd(TINKER9_DIRSTR "/test/file/water30/water30_aniso.dyn", d);
   TestFile fx(TINKER9_DIRSTR "/test/file/water30/water30_aniso.xyz", x);
   TestFile fp(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");

   const char* argv[] = {"dummy", x};
   int argc = 2;
   testBeginWithArgs(argc, argv);
   testMdInit(298., 1.);

   rc_flag = usage_;
   initialize();

   const double dt_ps = 0.002;
   const int nsteps = 5;
   const double eps_e = 0.0001;
   const double eps_l = 1e-6;
   std::vector<double> epots, eksums, latlen, latang;
   int old = inform::iwrite;
   inform::iwrite = 1;
   ThermostatEnum thermostat = ThermostatEnum::NHC;
   BarostatEnum barostat = BarostatEnum::BUSSI;
   intg = new RespaIntegrator(thermostat, barostat);
   for (int i = 1; i <= nsteps; ++i) {
      intg->dynamic(i, dt_ps);
      T_prec temp;
      kinetic(temp);
      epots.push_back(esum);
      eksums.push_back(eksum);
      boxGetCurrent(dup_buf_box);
      boxSetTinker(dup_buf_box);
      latlen.push_back(boxes::xbox);
      latlen.push_back(boxes::ybox);
      latlen.push_back(boxes::zbox);
      latang.push_back(boxes::alpha);
      latang.push_back(boxes::beta);
      latang.push_back(boxes::gamma);
   }
   inform::iwrite = old;

   delete intg;
   intg = nullptr;
   finish();
   testEnd();

   for (int i = 0; i < nsteps; ++i) {
      REQUIRE(epots[i] == Approx(ref_epot[i]).margin(eps_e));
      REQUIRE(eksums[i] == Approx(ref_ekin[i]).margin(eps_e));
   }

   for (int i = 0; i < 3 * nsteps; ++i) {
      REQUIRE(latlen[i] == Approx(ref_latlen[i]).margin(eps_l));
      REQUIRE(latang[i] == Approx(ref_latang[i]).margin(eps_l));
   }

   TestRemoveFileOnExit("test_water30.arc");
   bath::kelvin = 0.;
   bath::atmsph = 0.;
   bath::isothermal = 0;
   bath::isobaric = 0;
}

TEST_CASE("NPT-Monte-Iso", "[ff][npt][Monte][iso]")
{
   const char* txt_intg = R"**(
   integrator   respa
   thermostat   nose-hoover
   barostat     montecarlo
   volume-trial 1
   respa-inner  4
   randomseed   1
   )**";

   double ref_ekin[] = {1630.7885, 1639.6912, 1632.2556, 1640.9378, 1650.5557};
   double ref_epot[] = {-9181.1709, -9191.2824, -9186.0726, -9193.9167, -9202.1321};
   double ref_latlen[] = {29.835733, 29.835733, 29.835733, 29.837189, 29.837189, 29.837189, 29.800278, 29.800278,
      29.800278, 29.832556, 29.832556, 29.832556, 29.847620, 29.847620, 29.847620};
   double ref_latang[] = {90.000000, 90.000000, 90.000000, 90.000000, 90.000000, 90.000000, 90.000000, 90.000000,
      90.000000, 90.000000, 90.000000, 90.000000, 90.000000, 90.000000, 90.000000};

   const char* k = "test_water30.key";
   const char* d = "test_water30.dyn";
   const char* x = "test_water30.xyz";

   std::string k0 = txt_intg;
   TestFile fke(TINKER9_DIRSTR "/test/file/water30/water30.key", k, k0);

   TestFile fd(TINKER9_DIRSTR "/test/file/water30/water30_iso.dyn", d);
   TestFile fx(TINKER9_DIRSTR "/test/file/water30/water30_iso.xyz", x);
   TestFile fp(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");

   const char* argv[] = {"dummy", x};
   int argc = 2;
   testBeginWithArgs(argc, argv);
   testMdInit(298., 1.);

   rc_flag = usage_ & ~calc::md;
   initialize();
   rc_flag = usage_;

   const double dt_ps = 0.002;
   const int nsteps = 5;
   const double eps_e = 0.0001;
   const double eps_l = 1e-6;
   std::vector<double> epots, eksums, latlen, latang;
   int old = inform::iwrite;
   inform::iwrite = 1;
   ThermostatEnum thermostat = ThermostatEnum::NHC;
   BarostatEnum barostat = BarostatEnum::MONTECARLO;
   intg = new RespaIntegrator(thermostat, barostat);
   for (int i = 1; i <= nsteps; ++i) {
      intg->dynamic(i, dt_ps);
      T_prec temp;
      kinetic(temp);
      epots.push_back(esum);
      eksums.push_back(eksum);
      boxGetCurrent(dup_buf_box);
      boxSetTinker(dup_buf_box);
      latlen.push_back(boxes::xbox);
      latlen.push_back(boxes::ybox);
      latlen.push_back(boxes::zbox);
      latang.push_back(boxes::alpha);
      latang.push_back(boxes::beta);
      latang.push_back(boxes::gamma);
   }
   inform::iwrite = old;

   delete intg;
   intg = nullptr;
   rc_flag = usage_ & ~calc::md;
   finish();
   testEnd();

   for (int i = 0; i < nsteps; ++i) {
      REQUIRE(epots[i] == Approx(ref_epot[i]).margin(eps_e));
      REQUIRE(eksums[i] == Approx(ref_ekin[i]).margin(eps_e));
   }

   for (int i = 0; i < 3 * nsteps; ++i) {
      REQUIRE(latlen[i] == Approx(ref_latlen[i]).margin(eps_l));
      REQUIRE(latang[i] == Approx(ref_latang[i]).margin(eps_l));
   }

   TestRemoveFileOnExit("test_water30.arc");
   bath::kelvin = 0.;
   bath::atmsph = 0.;
   bath::isothermal = 0;
   bath::isobaric = 0;
}

TEST_CASE("NPT-Monte-Semiiso", "[ff][npt][Monte][semiiso]")
{
   const char* txt_intg = R"**(
   integrator   respa
   thermostat   nose-hoover
   barostat     montecarlo
   volume-trial 1
   respa-inner  4
   randomseed   1
   pressure     semi
   )**";

   double ref_ekin[] = {2483.2144, 2490.7202, 2503.0821, 2479.2142, 2533.9880};
   double ref_epot[] = {-7260.0686, -7269.3926, -7282.7994, -7256.8270, -7313.7929};
   double ref_latlen[] = {36.250908, 36.250908, 27.983082, 36.252861, 36.252861, 27.984591, 36.252861, 36.252861,
      27.988688, 36.235928, 36.235928, 27.988688, 36.235928, 36.235928, 27.923601};
   double ref_latang[] = {90.000000, 90.000000, 90.000000, 90.000000, 90.000000, 90.000000, 90.000000, 90.000000,
      90.000000, 90.000000, 90.000000, 90.000000, 90.000000, 90.000000, 90.000000};

   const char* k = "test_water30.key";
   const char* d = "test_water30.dyn";
   const char* x = "test_water30.xyz";

   std::string k0 = txt_intg;
   TestFile fke(TINKER9_DIRSTR "/test/file/water30/water30.key", k, k0);

   TestFile fd(TINKER9_DIRSTR "/test/file/water30/water30_semiiso.dyn", d);
   TestFile fx(TINKER9_DIRSTR "/test/file/water30/water30_semiiso.xyz", x);
   TestFile fp(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");

   const char* argv[] = {"dummy", x};
   int argc = 2;
   testBeginWithArgs(argc, argv);
   testMdInit(298., 1.);

   rc_flag = usage_ & ~calc::md;
   initialize();
   rc_flag = usage_;

   const double dt_ps = 0.002;
   const int nsteps = 5;
   const double eps_e = 0.0001;
   const double eps_l = 1e-6;
   std::vector<double> epots, eksums, latlen, latang;
   int old = inform::iwrite;
   inform::iwrite = 1;
   ThermostatEnum thermostat = ThermostatEnum::NHC;
   BarostatEnum barostat = BarostatEnum::MONTECARLO;
   intg = new RespaIntegrator(thermostat, barostat);
   for (int i = 1; i <= nsteps; ++i) {
      intg->dynamic(i, dt_ps);
      T_prec temp;
      kinetic(temp);
      epots.push_back(esum);
      eksums.push_back(eksum);
      boxGetCurrent(dup_buf_box);
      boxSetTinker(dup_buf_box);
      latlen.push_back(boxes::xbox);
      latlen.push_back(boxes::ybox);
      latlen.push_back(boxes::zbox);
      latang.push_back(boxes::alpha);
      latang.push_back(boxes::beta);
      latang.push_back(boxes::gamma);
   }
   inform::iwrite = old;

   delete intg;
   intg = nullptr;
   rc_flag = usage_ & ~calc::md;
   finish();
   testEnd();

   for (int i = 0; i < nsteps; ++i) {
      REQUIRE(epots[i] == Approx(ref_epot[i]).margin(eps_e));
      REQUIRE(eksums[i] == Approx(ref_ekin[i]).margin(eps_e));
   }

   for (int i = 0; i < 3 * nsteps; ++i) {
      REQUIRE(latlen[i] == Approx(ref_latlen[i]).margin(eps_l));
      REQUIRE(latang[i] == Approx(ref_latang[i]).margin(eps_l));
   }

   TestRemoveFileOnExit("test_water30.arc");
   bath::kelvin = 0.;
   bath::atmsph = 0.;
   bath::isothermal = 0;
   bath::isobaric = 0;
}

TEST_CASE("NPT-Monte-Aniso", "[ff][npt][Monte][aniso]")
{
   const char* txt_intg = R"**(
   integrator   respa
   thermostat   nose-hoover
   barostat     montecarlo
   volume-trial 1
   respa-inner  4
   randomseed   1
   pressure     aniso
   )**";

   double ref_ekin[] = {2395.2272, 2375.9868, 2405.2958, 2422.3834, 2418.1759};
   double ref_epot[] = {-7792.5164, -7772.5469, -7802.8408, -7819.7612, -7814.4291};
   double ref_latlen[] = {30.546417, 26.474856, 34.816662, 30.548605, 26.476753, 34.819156, 30.548605, 26.476753,
      34.819156, 30.548605, 26.476753, 34.819156, 30.548605, 26.476753, 34.819156};
   double ref_latang[] = {92.286064, 100.715959, 88.330294, 92.286064, 100.715959, 88.330294, 92.286064, 100.715959,
      88.330294, 92.286064, 100.715959, 88.330294, 92.286064, 100.715959, 88.330294};

   const char* k = "test_water30.key";
   const char* d = "test_water30.dyn";
   const char* x = "test_water30.xyz";

   std::string k0 = txt_intg;
   TestFile fke(TINKER9_DIRSTR "/test/file/water30/water30.key", k, k0);

   TestFile fd(TINKER9_DIRSTR "/test/file/water30/water30_aniso.dyn", d);
   TestFile fx(TINKER9_DIRSTR "/test/file/water30/water30_aniso.xyz", x);
   TestFile fp(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");

   const char* argv[] = {"dummy", x};
   int argc = 2;
   testBeginWithArgs(argc, argv);
   testMdInit(298., 1.);

   rc_flag = usage_ & ~calc::md;
   initialize();
   rc_flag = usage_;

   const double dt_ps = 0.002;
   const int nsteps = 5;
   const double eps_e = 0.0001;
   const double eps_l = 1e-6;
   std::vector<double> epots, eksums, latlen, latang;
   int old = inform::iwrite;
   inform::iwrite = 1;
   ThermostatEnum thermostat = ThermostatEnum::NHC;
   BarostatEnum barostat = BarostatEnum::MONTECARLO;
   intg = new RespaIntegrator(thermostat, barostat);
   for (int i = 1; i <= nsteps; ++i) {
      intg->dynamic(i, dt_ps);
      T_prec temp;
      kinetic(temp);
      epots.push_back(esum);
      eksums.push_back(eksum);
      boxGetCurrent(dup_buf_box);
      boxSetTinker(dup_buf_box);
      latlen.push_back(boxes::xbox);
      latlen.push_back(boxes::ybox);
      latlen.push_back(boxes::zbox);
      latang.push_back(boxes::alpha);
      latang.push_back(boxes::beta);
      latang.push_back(boxes::gamma);
   }
   inform::iwrite = old;

   delete intg;
   intg = nullptr;
   rc_flag = usage_ & ~calc::md;
   finish();
   testEnd();

   for (int i = 0; i < nsteps; ++i) {
      REQUIRE(epots[i] == Approx(ref_epot[i]).margin(eps_e));
      REQUIRE(eksums[i] == Approx(ref_ekin[i]).margin(eps_e));
   }

   for (int i = 0; i < 3 * nsteps; ++i) {
      REQUIRE(latlen[i] == Approx(ref_latlen[i]).margin(eps_l));
      REQUIRE(latang[i] == Approx(ref_latang[i]).margin(eps_l));
   }

   TestRemoveFileOnExit("test_water30.arc");
   bath::kelvin = 0.;
   bath::atmsph = 0.;
   bath::isothermal = 0;
   bath::isobaric = 0;
}

namespace {
// Loads water30 with the induced dipole predictor, and fills its history with
// solves at nearby geometries, past maxualt so that the predictor is in use and
// the next solve overwrites a real entry.
void loadWater30Predictor()
{
   const char* k = "test_water30.key";
   const char* x = "test_water30.xyz";
   TestFile fke(TINKER9_DIRSTR "/test/file/water30/water30.key", k, "\nintegrator verlet\n");
   TestFile fx(TINKER9_DIRSTR "/test/file/water30/water30_iso.xyz", x);
   TestFile fp(TINKER9_DIRSTR "/test/file/commit_6fe8e913/water03.prm");

   const char* argv[] = {"dummy", x};
   int argc = 2;
   testBeginWithArgs(argc, argv);
   testMdInit(298., 1.);
   rc_flag = usage_;
   initialize();
   REQUIRE(use(Potent::POLAR));
   REQUIRE(maxualt > 0);
   REQUIRE_FALSE(polpot::use_tholed);

   std::vector<pos_prec> xb(n);
   for (int step = 0; step < maxualt + 3; ++step) {
      darray::copyout(g::q0, n, xb.data(), xpos);
      waitFor(g::q0);
      for (int i = 0; i < n; ++i)
         xb[i] += 0.01 * ((i + step) % 3 - 1);
      darray::copyin(g::q0, n, xpos, xb.data());
      waitFor(g::q0);
      copyPosToXyz(true);
      energy(calc::v1);
   }
}

std::vector<real> grab(real (*dev)[3])
{
   std::vector<real> v(3 * n);
   darray::copyout(g::q0, n, reinterpret_cast<real(*)[3]>(v.data()), dev);
   waitFor(g::q0);
   return v;
}

// A rejected Monte Carlo volume move: the trial energy solves the induced
// dipoles at the trial geometry and adds them to the predictor history, and the
// reject restores both along with the coordinates.
void checkRejectKeepsInduced()
{
   real(*slots[])[3] = {udalt_00, udalt_01, udalt_02, udalt_03, udalt_04, udalt_05, udalt_06, udalt_07, udalt_08,
      udalt_09, udalt_10, udalt_11, udalt_12, udalt_13, udalt_14, udalt_15};
   real(*pslots[])[3] = {upalt_00, upalt_01, upalt_02, upalt_03, upalt_04, upalt_05, upalt_06, upalt_07, upalt_08,
      upalt_09, upalt_10, upalt_11, upalt_12, upalt_13, upalt_14, upalt_15};
   REQUIRE(nualt >= maxualt);
   const auto uind0 = grab(uind);
   const auto uinp0 = grab(uinp);
   const auto udir0 = grab(udir);
   const auto udirp0 = grab(udirp);
   std::vector<std::vector<real>> slots0, pslots0;
   for (int i = 0; i < maxualt; ++i) {
      slots0.push_back(grab(slots[i]));
      pslots0.push_back(grab(pslots[i]));
   }
   const int nualt0 = nualt;
   std::vector<pos_prec> xb(n);
   darray::copyout(g::q0, n, xb.data(), xpos);
   waitFor(g::q0);

   {
      MonteCarloBarostat mc;
      InducedSnapshot snapshot;
      // an old energy far below any trial energy rejects the move for certain
      monteCarloBarostat(-1.0e30, 298., false, false, &snapshot);
   }

   std::vector<pos_prec> xa(n);
   darray::copyout(g::q0, n, xa.data(), xpos);
   waitFor(g::q0);
   REQUIRE(xa == xb);
   REQUIRE(grab(uind) == uind0);
   REQUIRE(grab(uinp) == uinp0);
   REQUIRE(grab(udir) == udir0);
   REQUIRE(grab(udirp) == udirp0);
   REQUIRE(nualt == nualt0);
   for (int i = 0; i < maxualt; ++i) {
      REQUIRE(grab(slots[i]) == slots0[i]);
      REQUIRE(grab(pslots[i]) == pslots0[i]);
   }
}
}

TEST_CASE("NPT-Monte-Reject-Keeps-Induced", "[ff][npt][Monte][reject]")
{
   loadWater30Predictor();
   checkRejectKeepsInduced();
   finish();
   testEnd();
}

TEST_CASE("NPT-Monte-Reject-After-Expol", "[ff][npt][Monte][reject]")
{
   // A HIPPO system with exchange polarization first. Its polinv and polscale
   // stay dangling after finish(), so a snapshot that picked its buffers by
   // pointer rather than by use_expol read freed memory in the AMOEBA system.
   {
      const char* x = "water30_2.xyz";
      TestFile fx(TINKER9_DIRSTR "/test/file/tinkernist/water30_2.xyz", x);
      TestFile fk(TINKER9_DIRSTR "/test/file/tinkernist/water30_2.key");
      TestFile fp(TINKER9_DIRSTR "/test/file/commit_aad9340c/water21.prm");

      const char* argv[] = {"dummy", x};
      int argc = 2;
      testBeginWithArgs(argc, argv);
      rc_flag = calc::xyz | calc::vmask;
      initialize();
      REQUIRE(polpot::use_expol);
      energy(calc::v0);
      finish();
      testEnd();
   }

   loadWater30Predictor();
   REQUIRE_FALSE(polpot::use_expol);
   checkRejectKeepsInduced();
   finish();
   testEnd();
}
