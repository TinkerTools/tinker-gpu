#include "tool/argkey.h"

#include "test.h"
#include "testrt.h"

using namespace tinker;

TEST_CASE("GetKV", "[util]")
{
   const char* key = R"(
parameters           amoeba09.prm

rotatable-bond            1 2 3 4
restrain-groups   1 2 2.5 4.5 6.0
epsilonrule              abc x3-Y

aniso-pressure
pme-order                       5
angle-cubic                 -0.25
rattle                      water
)";

   const char* x1 = "test_nacl.xyz";
   TestFile fke("", "test_nacl.key", key);
   TestFile fx1(TINKER9_DIRSTR "/test/file/nacl/nacl1.xyz", x1);
   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");

   const char* argv[] = {"dummy", x1};
   int argc = 2;

   TestSession session(argc, argv);

   std::vector<int> vint;
   getKV("ROTATABLE-BOND", vint);
   REQUIRE(vint.size() == 4);
   REQUIRE(vint[0] == 1);
   REQUIRE(vint[1] == 2);
   REQUIRE(vint[2] == 3);
   REQUIRE(vint[3] == 4);

   std::vector<double> vdbl;
   getKV("RESTRAIN-GROUPS", vdbl);
   REQUIRE(vdbl.size() == 5);
   REQUIRE(vdbl[0] == 1.0);
   REQUIRE(vdbl[1] == 2.0);
   REQUIRE(vdbl[2] == 2.5);
   REQUIRE(vdbl[3] == 4.5);
   REQUIRE(vdbl[4] == 6.0);

   std::vector<std::string> vstr;
   getKV("EPSILONRULE", vstr);
   REQUIRE(vstr.size() == 2);
   REQUIRE(vstr[0] == "ABC");
   REQUIRE(vstr[1] == "X3-Y");

   bool aniso = false, archive = true;
   getKV("ANISO-PRESSURE", aniso, false);
   getKV("ARCHIVE", archive, false);
   REQUIRE(aniso == true);
   REQUIRE(archive == false);

   int po = 0, dig = 0;
   getKV("PME-ORDER", po, 4);
   getKV("DIGITS", dig, 4);
   REQUIRE(po == 5);
   REQUIRE(dig == 4);

   double ac = 1.0, aq = 1.0;
   getKV("ANGLE-CUBIC", ac, 0.0);
   getKV("ANGLE-QUARTIC", aq, 0.0);
   REQUIRE(ac == -0.25);
   REQUIRE(aq == 0.0);

   std::string rw = "bonds", rd = "hhg";
   getKV("RATTLE", rw, "");
   getKV("RADIUS-RULE", rd, "Cubic-Mean"); // intentionally not all caps
   REQUIRE(rw == "WATER");
   REQUIRE(rd == "Cubic-Mean");

   session.end();
}
