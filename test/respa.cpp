#include "tool/iofortstr.h"
#include <tinker/detail/bath.hh>
#include <tinker/detail/mdstuf.hh>

#include "test.h"
#include "testrt.h"

using namespace tinker;

static const char* respa_intg = R"**(
integrator  vrespa
thermostat  bussi
barostat    montecarlo
)**";

TEST_CASE("NPT-VRESPA", "[ff][npt][respa]")
{
   const char* k = "test_arbox.key";
   const char* d = "test_arbox.dyn";
   const char* x = "test_arbox.xyz";

   std::string k0 = respa_intg;
   TestFile fke(TINKER9_DIRSTR "/test/file/arbox/arbox.key", k, k0);

   TestFile fd(TINKER9_DIRSTR "/test/file/arbox/arbox.dyn", d);
   TestFile fx(TINKER9_DIRSTR "/test/file/arbox/arbox.xyz", x);
   TestFile fp(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");

   const char* argv[] = {"dummy", x};
   int argc = 2;
   testBeginWithArgs(argc, argv);
   testMdInit(298., 1);

   rc_flag = calc::xyz | calc::vel | calc::mass | calc::energy | calc::grad | calc::md | calc::virial;
   initialize();

   COMPARE_INTS(mdstuf::nrespa, 2);
   COMPARE_INTS(bath::voltrial, 25);
   COMPARE_INTS(mdstuf::nfree, 645);
   COMPARE_REALS(bath::volmove, 100, 1e-12);
   COMPARE_REALS(bath::tautemp, 0.2, 1e-12);
   COMPARE_REALS(bath::taupres, 2.0, 1e-12);
   FstrView itg = mdstuf::integrate;
   REQUIRE(itg=="VRESPA");

   finish();
   testEnd();
   bath::kelvin = 0.;
   bath::atmsph = 0.;
   bath::isothermal = 0;
   bath::isobaric = 0;
}
