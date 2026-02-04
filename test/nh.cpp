#include "md/integrator.h"
#include "md/misc.h"
#include <tinker/detail/bath.hh>
#include <tinker/detail/inform.hh>

#include "test.h"
#include "testrt.h"

using namespace tinker;

#if TINKER_REAL_SIZE == 8
static int usage_ = calc::xyz | calc::vel | calc::mass | calc::energy | calc::grad | calc::md | calc::virial;

static const char* nh_intg = R"**(
integrator  nose-hoover
thermostat  nose-hoover
barostat    nose-hoover
)**";

static double arbox_kin[] = {64.648704, 64.666848, 64.685167, 64.703666, 64.722346, 64.741211, 64.760262, 64.779502,
   64.798934, 64.818559, 64.838379, 64.858397, 64.878612, 64.899027, 64.919642, 64.940458, 64.961475, 64.982692,
   65.004108, 65.025723};
static double arbox_pot[] = {-277.037089, -277.055107, -277.073220, -277.091430, -277.109741, -277.128153, -277.146671,
   -277.165296, -277.184031, -277.202877, -277.221837, -277.240911, -277.260102, -277.279410, -277.298835, -277.318379,
   -277.338042, -277.357822, -277.377719, -277.397732};
#endif

TEST_CASE("NPT-NoseHoover-ArBox", "[ff][npt][nosehoover][arbox]")
{
#if TINKER_REAL_SIZE == 8
   const char* k = "test_arbox.key";
   const char* d = "test_arbox.dyn";
   const char* x = "test_arbox.xyz";

   // std::string k0 = nh_intg;
   std::string k0 = nh_intg;
   TestFile fke(TINKER9_DIRSTR "/test/file/arbox/arbox.key", k, k0);

   TestFile fd(TINKER9_DIRSTR "/test/file/arbox/arbox.dyn", d);
   TestFile fx(TINKER9_DIRSTR "/test/file/arbox/arbox.xyz", x);
   TestFile fp(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");

   const char* argv[] = {"dummy", x};
   int argc = 2;
   testBeginWithArgs(argc, argv);
   testMdInit(298., 1.);

   rc_flag = usage_;
   initialize();

   const double dt_ps = 0.001;
   const int nsteps = 20;
   const double eps_e = 0.0001;
   std::vector<double> epots, eksums;
   int old = inform::iwrite;
   inform::iwrite = 1;
   Nhc96Integrator nhi;
   for (int i = 1; i <= nsteps; ++i) {
      nhi.dynamic(i, dt_ps);
      epots.push_back(esum);
      eksums.push_back(eksum);
   }
   inform::iwrite = old;

   finish();
   testEnd();

   for (int i = 0; i < nsteps; ++i) {
      REQUIRE(epots[i] == Approx(arbox_pot[i]).margin(eps_e));
      REQUIRE(eksums[i] == Approx(arbox_kin[i]).margin(eps_e));
   }

   TestRemoveFileOnExit("test_arbox.arc");
   bath::kelvin = 0.;
   bath::atmsph = 0.;
   bath::isothermal = 0;
   bath::isobaric = 0;
#else
   REQUIRE(true);
#endif
}
