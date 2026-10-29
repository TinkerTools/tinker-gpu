#include "ff/evdw.h"

#include "test.h"
#include "testrt.h"

using namespace tinker;

static const char* triclinic_box = R"**(
ewald
ewald-cutoff  7.0
vdw-cutoff    7.0
neighbor-list
list-buffer  0.01

a-axis   28.0
b-axis   22.0
c-axis   18.0
alpha   105.0
beta    110.0
gamma    80.0
)**";

static const char* monoclinic_box = R"**(
ewald
ewald-cutoff  7.0
vdw-cutoff    7.0
neighbor-list
list-buffer  0.01

a-axis   28.0
b-axis   22.0
c-axis   18.0
alpha    90.0
beta    110.0
gamma    90.0
)**";

static int usage = calc::xyz | calc::vmask;
static const char* k = "test_local_frame2.key";
static const char* x1 = "test_local_frame2.xyz";
static const char* argv[] = {"dummy", x1};
static int argc = 2;

TEST_CASE("Local-Frame2-1", "[ff][triclinic][evdw][hal][local-frame2]")
{
   std::string k0 = triclinic_box;
   k0 += "vdwterm  only\n";
   TestFile fke(TINKER9_DIRSTR "/test/file/local_frame/local_frame.key", k, k0);

   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");
   TestFile fx1(TINKER9_DIRSTR "/test/file/local_frame/local_frame2.xyz", x1);

   TestSession session(argc, argv);
   rc_flag = usage;
   session.init();

   SECTION("ehal -- pbc, cutoff")
   {
      const double eps_e = 0.0001;
      const double ref_eng = 206.5670;
      const int ref_count = 129;

      zeroEGV();
      energy(calc::v3);
      COMPARE_REALS(energy_ev, ref_eng, eps_e);
      COMPARE_COUNT(nev, ref_count);
   }

   session.end();
}

TEST_CASE("Local-Frame2-2", "[ff][monoclinic][evdw][hal][local-frame2]")
{
   std::string k0 = monoclinic_box;
   k0 += "vdwterm  only\n";
   TestFile fke(TINKER9_DIRSTR "/test/file/local_frame/local_frame.key", k, k0);

   TestFile fpr(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");
   TestFile fx1(TINKER9_DIRSTR "/test/file/local_frame/local_frame2.xyz", x1);

   TestSession session(argc, argv);
   rc_flag = usage;
   session.init();

   SECTION("ehal -- pbc, cutoff")
   {
      const double eps_e = 0.0001;
      const double ref_eng = 182.5400;
      const int ref_count = 125;

      zeroEGV();
      energy(calc::v3);
      COMPARE_REALS(energy_ev, ref_eng, eps_e);
      COMPARE_COUNT(nev, ref_count);
   }

   session.end();
}
