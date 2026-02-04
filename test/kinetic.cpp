#include "md/misc.h"

#include "test.h"
#include "testrt.h"

using namespace tinker;

static const char* verlet_intg = "integrator  verlet\nbarostat  berendsen\n";
static int usage_ = calc::xyz | calc::vel | calc::mass | calc::vmask | calc::md;

TEST_CASE("Kinetic-ArBox", "[ff][kinetic][arbox]")
{
   const char* k = "test_arbox.key";
   const char* d = "test_arbox.dyn";
   const char* x = "test_arbox.xyz";

   std::string k0 = verlet_intg;
   TestFile fke(TINKER9_DIRSTR "/test/file/arbox/arbox.key", k, k0);

   TestFile fd(TINKER9_DIRSTR "/test/file/arbox/arbox.dyn_2", d);
   TestFile fx(TINKER9_DIRSTR "/test/file/arbox/arbox.xyz", x);
   TestFile fp(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");

   const char* argv[] = {"dummy", x};
   int argc = 2;
   testBeginWithArgs(argc, argv);
   testMdInit(0, 0);

   rc_flag = usage_;
   initialize();

   T_prec temp;
   kinetic(temp);

   const double ref_eksum = 100446.40376;
   const double ref_temp = 156008.001336;
   const double eps_e = 0.0001;

   REQUIRE(eksum == Approx(ref_eksum).margin(eps_e));
   REQUIRE(temp == Approx(ref_temp).margin(eps_e));

   finish();
   testEnd();
}
