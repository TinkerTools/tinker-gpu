#include "test.h"
#include "testrt.h"
#include "tinker9.h"

#include <tinker/routines.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace tinker;

// CUDA builds only. The bar driver does not depend on the platform, and the
// OpenACC build would either rerun the CUDA kernels, its default for a key with
// a neighbor list and a box, or, under GPU_PACKAGE=OPENACC, spend minutes per
// case on the 12501 atom box.
#if TINKER_GPULANG_CUDA

namespace {
/// One trajectory block of a BAR file: its header, then one row per frame.
struct BarBlock
{
   int nfrm = 0;
   double temp = 0;
   std::vector<double> u0, u1, vol; ///< Energies in states 0 and 1, and the box volume (0 if none).
};

/// Reads the two trajectory blocks "bar" mode 1 writes.
std::vector<BarBlock> readBar(const std::string& path)
{
   std::ifstream fin(path);
   REQUIRE(fin);
   std::vector<BarBlock> blocks;
   std::string line;
   while (std::getline(fin, line)) {
      BarBlock b;
      std::istringstream hdr(line);
      if (not(hdr >> b.nfrm >> b.temp))
         continue;
      for (int i = 0; i < b.nfrm; ++i) {
         REQUIRE(std::getline(fin, line));
         std::istringstream row(line);
         int idx;
         double u0, u1, vol = 0;
         REQUIRE((row >> idx >> u0 >> u1));
         row >> vol;
         REQUIRE(idx == i + 1);
         b.u0.push_back(u0);
         b.u1.push_back(u1);
         b.vol.push_back(vol);
      }
      blocks.push_back(b);
   }
   return blocks;
}

/// Ends the Fortran runtime however the case ends.
struct FortranRuntimeScope
{
   FortranRuntimeScope(int argc, const char** argv)
   {
      tinkerFortranRuntimeBegin(argc, const_cast<char**>(argv));
      tinker_f_command();
   }
   ~FortranRuntimeScope() { tinkerFortranRuntimeEnd(); }
};

// Runs "bar 1 <A>.arc 298 <B>.arc 298 N", which re-evaluates both trajectories
// in both states and writes <A>.bar, and compares that file with the Tinker
// reference (test_bar.f). Each AMOEBA water box trajectory decouples one water
// molecule between two neighboring lambda windows.
void runBar(const std::string& basea, const std::string& baseb)
{
   TestLmdaFlagReset lmdaReset;

   const std::string files = TINKER9_DIRSTR "/test/file/bar/";
   const std::string arca = basea + ".arc", arcb = baseb + ".arc";
   TestFile fprm(files + "hostsG4.prm");
   TestFile fkeya(files + basea + ".key");
   TestFile fkeyb(files + baseb + ".key");
   TestFile farca(files + arca);
   TestFile farcb(files + arcb);

   // A leftover output would make "bar" write <A>.bar_2 instead.
   const std::string barfile = basea + ".bar";
   std::remove(barfile.c_str());
   TestRemoveFileOnExit rmbar(barfile);

   const char* argv[] = {"dummy", "1", arca.c_str(), "298", arcb.c_str(), "298", "N"};
   const int argc = 7;
   {
      FortranRuntimeScope scope(argc, argv);
      xBar(argc, const_cast<char**>(argv));
   }

   auto got = readBar(barfile);
   auto ref = readBar(std::string(TINKER9_DIRSTR "/test/ref/bar/") + barfile);
   REQUIRE(got.size() == 2);
   REQUIRE(ref.size() == 2);

   // Total energies of 12501 atoms near -37000 kcal/mol. Catch2's Approx keeps
   // a relative slack of about 0.4 kcal/mol at that size, more than the state 1
   // minus state 0 differences BAR works from, so the comparisons are absolute.
   // Mixed precision lands about 0.02 from the reference in total energy and
   // 2e-4 in the difference, and the volume, computed in single precision, up
   // to 0.01 away; double precision agrees to the printed 1e-4. The difference
   // is the sensitive check: moving VDW-LAMBDA by 0.001 shifts it by 3e-3 or
   // more on every frame.
   const double eps_e = testGetEps(1.0e-1, 1.0e-3);
   const double eps_d = testGetEps(2.0e-3, 5.0e-4);
   const double eps_vol = testGetEps(5.0e-2, 1.0e-3);
   for (int k = 0; k < 2; ++k) {
      const auto &g = got[k], &r = ref[k];
      REQUIRE(g.nfrm == r.nfrm);
      REQUIRE(g.temp == r.temp);
      for (int i = 0; i < r.nfrm; ++i) {
         INFO("block " << k + 1 << " frame " << i + 1);
         REQUIRE(std::fabs(g.u0[i] - r.u0[i]) <= eps_e);
         REQUIRE(std::fabs(g.u1[i] - r.u1[i]) <= eps_e);
         REQUIRE(std::fabs((g.u1[i] - g.u0[i]) - (r.u1[i] - r.u0[i])) <= eps_d);
         REQUIRE(std::fabs(g.vol[i] - r.vol[i]) <= eps_vol);
      }
   }
}
}

TEST_CASE("BAR-water-00-0000-0950", "[ff][bar]") { runBar("water-00-0000-0950", "water-00-0000-0900"); }
TEST_CASE("BAR-water-00-0950-1000", "[ff][bar]") { runBar("water-00-0950-1000", "water-00-0900-1000"); }

#endif
