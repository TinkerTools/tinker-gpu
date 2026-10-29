#include "ff/box.h"
#include "ff/rwcrd.h"
#include "tool/error.h"
#include "tool/iofortstr.h"
#include <tinker/detail/atoms.hh>
#include <tinker/detail/files.hh>

#include "test.h"
#include "testrt.h"

namespace {
const char* frame1 = R"**(     2  Frame 1
     1  Na+     1.000000     2.000000     3.000000     7
     2  Cl-     4.000000     5.000000     6.000000    14
)**";

const char* frame2 = R"**(     2  Frame 2
     1  Na+    11.000000    12.000000    13.000000     7
     2  Cl-    14.000000    15.000000    16.000000    14
)**";

const char* keyfile = R"**(
parameters                 amoeba09
bondterm                       only
)**";
}

using namespace tinker;

// An archive that ends with a blank line has only its two frames, and reading
// the second one reports the end of the file. A blank line used to be read as a
// third frame, with an atom index that was never set.
TEST_CASE("CrdReader-trailing-blank", "[ff][rwcrd]")
{
   const char* xn = "test_rwcrd.arc";
   TestFile xfile("", xn, std::string(frame1) + frame2 + "\n\n");
   TestFile kfile("", "test_rwcrd.key", keyfile);
   TestFile pfile(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");

   const char* argv[] = {"dummy", xn};
   int argc = 2;
   TestSession session(argc, argv);
   rc_flag = calc::xyz | calc::mass;
   session.init();

   FstrView fsw = files::filename;
   std::string fname = fsw.trim();
   auto ipt = CrdReader(fname);
   REQUIRE(ipt.readCurrent() == 0);
   REQUIRE(atoms::x[0] == 1.0);
   REQUIRE(atoms::z[1] == 6.0);
   REQUIRE(ipt.readCurrent() == 1);
   REQUIRE(atoms::x[0] == 11.0);
   REQUIRE(atoms::z[1] == 16.0);
}

// An atom line that cannot be read stops the reading instead of writing an
// atom at an index that was never set.
TEST_CASE("CrdReader-bad-atom-line", "[ff][rwcrd]")
{
   const char* xn = "test_rwcrd_bad.arc";
   std::string bad = std::string(frame1) + "     2  Frame 2\n     1  Na+    11.0    12.0    13.0     7\n garbage\n";
   TestFile xfile("", xn, bad);
   TestFile kfile("", "test_rwcrd_bad.key", keyfile);
   TestFile pfile(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");

   const char* argv[] = {"dummy", xn};
   int argc = 2;
   TestSession session(argc, argv);
   rc_flag = calc::xyz | calc::mass;
   session.init();

   FstrView fsw = files::filename;
   std::string fname = fsw.trim();
   auto ipt = CrdReader(fname);
   REQUIRE(ipt.readCurrent() == 0);
   REQUIRE_THROWS_AS(ipt.readCurrent(), FatalError);
}
