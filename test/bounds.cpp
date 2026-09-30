#include "ff/box.h"
#include "ff/rwcrd.h"
#include "tool/iofortstr.h"
#include <tinker/detail/files.hh>

#include "test.h"
#include "testrt.h"

static const char* coord = R"**(   2
   1 Na+    51.000000   -83.000000   164.000000     7
   2 Cl-    63.000000    95.000000    47.000000    14
)**";

static const char* keyfile = R"**(
parameters                 amoeba09
a-axis                       20.000

bondterm                       only
)**";

using namespace tinker;

TEST_CASE("Bounds", "[ff][box]")
{
   const char* xn = "test_bounds.xyz";
   TestFile xfile("", xn, coord);
   TestFile kfile("", "test_bounds.key", keyfile);
   TestFile pfile(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");

   // box: cubic 20 20 20
   // 51, -83, 164 -> -9, -3, 4
   // 63,  95,  47 ->  3, -5, 7

   const char* argv[] = {"dummy", xn};
   int argc = 2;
   TestSession session(argc, argv);
   rc_flag = calc::xyz | calc::mass;
   session.init();

   FstrView fsw = files::filename;
   std::string fname = fsw.trim();
   auto ipt = CrdReader(fname);
   ipt.readCurrent();
   bounds();

   double eps = 1.0e-5;
   double xref[] = {-9, 3};
   double yref[] = {-3, -5};
   double zref[] = {4, 7};
   real xans[2], yans[2], zans[2];
   darray::copyout(g::q0, 2, xans, x);
   darray::copyout(g::q0, 2, yans, y);
   darray::copyout(g::q0, 2, zans, z);
   waitFor(g::q0);

   COMPARE_REALS(xans[0], xref[0], eps);
   COMPARE_REALS(yans[0], yref[0], eps);
   COMPARE_REALS(zans[0], zref[0], eps);
   COMPARE_REALS(xans[1], xref[1], eps);
   COMPARE_REALS(yans[1], yref[1], eps);
   COMPARE_REALS(zans[1], zref[1], eps);

   session.end();
}
