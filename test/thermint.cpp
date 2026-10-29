#include "ff/ethrmint.h"
#include "ff/dlmda.h"
#include "ff/elec.h"
#include "ff/energy.h"
#include "ff/evdw.h"
#include "ff/ost.h"
#include "md/integrator.h"
#include "md/misc.h"
#include "tool/iofortstr.h"

#include "test.h"
#include "testrt.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <tinker/detail/bath.hh>
#include <tinker/detail/dlmda.hh>
#include <tinker/detail/files.hh>
#include <tinker/detail/mutant.hh>
#include <tinker/detail/thrmint.hh>
#include <tinker/routines.h>

// Unit tests for the sequential lambda windows (src/dlmda.cpp) and the
// thermodynamic integration that samples them (src/ethrmint.cpp), following the
// Fortran suite in tinker/test/test_thermint.f. Most cases are pure host math:
// no molecular system, no Fortran runtime, no GPU. etidyn samples the plain
// host global dedl, so the accumulation logic is driven by assigning dedl
// directly rather than by evaluating any energy. Like the Fortran helpers, they
// seed lmdawinend/lmdawinlist directly and bypass init_tidyn, so the
// accumulation cases do not depend on how the step budget was divided.
//
// THERMINT-dedl-survives-trial sets a system up, to hold the energy calls to
// what etidyn expects of them.
//
// THERMINT-save is the exception: it brings up the Fortran runtime, because the
// schedule is precomputed by mutate.f/setlmdasched and the .ti file is written
// by ethrmint.f/prttihead and saveti. It is also where the adoption performed
// by lmdaWinData(INIT) and thermintData(INIT) is checked, since those read the
// Fortran dlmda and thrmint modules.
//
// IMPORTANT: use_ti is a process-wide global and all.tests is one process. If a
// case here leaves it true, a later test's initialize() would see it in the
// TermBuffer::manage() calls in empole/epolar/evdw and silently switch those
// terms into private-buffer chain-rule mode. Every TEST_CASE below resets
// use_ti to false before returning.

using namespace tinker;

namespace {
// mapSubLambda() reads the one main lambda, so drive it by setting that first.
void mapAt(double lmda)
{
   lambda = lmda;
   mapSubLambda();
}

// Puts the sub-lambda maps on the power-law branch, which is plain C++. The
// QNT/NONE taper branch calls tinker_f_switch and needs the Fortran runtime.
void useExpMaps(int eexp, int pexp, int vexp)
{
   use_rel = false;
   elmdamap.type = Lmdamap::EXP;
   plmdamap.type = Lmdamap::EXP;
   vlmdamap.type = Lmdamap::EXP;
   elmdamap.exp = eexp;
   plmdamap.exp = pexp;
   vlmdamap.exp = vexp;
   // Every sub-lambda follows the main lambda here.
   elmdamap.on = true;
   plmdamap.on = true;
   vlmdamap.on = true;
}

// resettisched -- install an arbitrary schedule and size the accumulators the
// way settiblocks would, then rewind to the first window.
void resettisched(const std::vector<double>& lam, const std::vector<int>& winend, int nstepavg,
   double eqratio)
{
   use_ti = true;
   useExpMaps(1, 1, 1);

   nlmdawin = (int)lam.size();
   tinstepavg = nstepavg;
   lmdawinratio = eqratio;
   lmdawinlist = lam;
   lmdawinfrac.assign(nlmdawin, 1.0 / (double)nlmdawin);
   lmdawinend = winend;

   tinbtot = 0;
   for (int i = 1; i <= nlmdawin; ++i) {
      int nw, ne;
      lmdaWinSize(i, nw, ne);
      tinbtot += (nw - ne) / nstepavg;
   }

   int cap = std::max(1, tinbtot);
   tilmdahist.assign(cap, 0.0);
   tilmdadedl.assign(cap, 0.0);
   tilmdadedlstd.assign(cap, 0.0);
   tidedllist.assign(nstepavg, 0.0);
   tinbcount = 0;

   lmdawin = 1;
   lambda = lmdawinlist[0];
   lmdaWinSize(lmdawin, lmdawinlen, lmdawineq);
   dedl = 0;
}

// tidynstep -- does for one dynamics step what the integrator and mdPropagate
// do between them, sampling the current lambda window and then advancing at a
// window boundary.
void tidynstep(int istep)
{
   lmdaWinDyn(istep);
   lmdaWinStep(istep);
}

// The equal-window special case, with the default schedule descending 1 -> 0.
void resetti(int nbin, int nstepavg, int window, int nequil)
{
   std::vector<int> winend(nbin);
   std::vector<double> lam(nbin);
   for (int i = 0; i < nbin; ++i) {
      winend[i] = (i + 1) * window;
      lam[i] = (nbin > 1) ? 1.0 - (double)i / (double)(nbin - 1) : 1.0;
   }
   if (nbin > 1)
      lam[nbin - 1] = 0.0;
   resettisched(lam, winend, nstepavg, (double)nequil / (double)window);
}

// Restores the process-wide flags and drops the accumulators. The maps go back
// to the defaults mutate.f sets, so nothing downstream sees a half-configured
// state.
void clearti()
{
   use_ti = false;
   lmdawin = 0;
   lmdawinlist.clear();
   lmdawinfrac.clear();
   lmdawinend.clear();
   tilmdahist.clear();
   tilmdadedl.clear();
   tilmdadedlstd.clear();
   tidedllist.clear();
   elmdamap.type = Lmdamap::QNT;
   plmdamap.type = Lmdamap::QNT;
   vlmdamap.type = Lmdamap::QNT;
   elmdamap.on = false;
   plmdamap.on = false;
   vlmdamap.on = false;
}

// True for a line that holds nothing but blanks.
bool tiBlankLine(const std::string& line)
{
   return line.find_first_not_of(" \t\r") == std::string::npos;
}

// Number of data records. prttihead writes the standard Tinker banner and
// plain text, so there is no comment character to key on: the header runs
// through the line labeling the columns, and every nonblank line after it is
// a block average appended by saveti.
int tiCountRows(const std::string& fname)
{
   std::ifstream fs(fname);
   std::string line;
   bool header = false;
   int n = 0;
   while (std::getline(fs, line)) {
      if (not header) {
         if (line.find("Index") != std::string::npos)
            header = true;
      } else if (not tiBlankLine(line)) {
         ++n;
      }
   }
   return n;
}

// Lambda column of the nth (1-based) data record, counted the same way.
double tiRowLambda(const std::string& fname, int nth)
{
   std::ifstream fs(fname);
   std::string line;
   bool header = false;
   int n = 0;
   while (std::getline(fs, line)) {
      if (not header) {
         if (line.find("Index") != std::string::npos)
            header = true;
         continue;
      }
      if (tiBlankLine(line))
         continue;
      if (++n == nth) {
         std::istringstream ss(line);
         int idx;
         double lam;
         ss >> idx >> lam;
         return lam;
      }
   }
   return -1.0;
}
}

TEST_CASE("THERMINT-avgstd", "[ff][thermint]")
{
   const double eps = 1.0e-12;
   // population standard deviation of ten consecutive integers
   const double sd10 = std::sqrt(8.25);

   double avg = -1.0, sd = -1.0;

   std::vector<double> v10;
   for (int i = 1; i <= 10; ++i)
      v10.push_back((double)i);

   avgstd(v10, 0, 10, avg, sd);
   COMPARE_REALS(avg, 5.5, eps);
   COMPARE_REALS(sd, sd10, eps);

   // count must be honored: only the first five entries participate
   avgstd(v10, 0, 5, avg, sd);
   COMPARE_REALS(avg, 3.0, eps);
   COMPARE_REALS(sd, std::sqrt(2.0), eps);

   // a constant list must give exactly zero, not a denormal from round-off
   std::vector<double> vc(4, 7.0);
   avgstd(vc, 0, 4, avg, sd);
   COMPARE_REALS(avg, 7.0, eps);
   REQUIRE(sd == 0.0);

   // single sample
   std::vector<double> v1{42.0};
   avgstd(v1, 0, 1, avg, sd);
   COMPARE_REALS(avg, 42.0, eps);
   REQUIRE(sd == 0.0);

   // empty: early return leaves both at zero
   avg = -1.0;
   sd = -1.0;
   avgstd(v10, 0, 0, avg, sd);
   REQUIRE(avg == 0.0);
   REQUIRE(sd == 0.0);

   // Large common offset. The shifted-mean form keeps this accurate; a naive
   // sum(x*x) - mean*mean would lose roughly seven digits of the variance.
   std::vector<double> vbig;
   for (int i = 1; i <= 10; ++i)
      vbig.push_back(1.0e8 + (double)i);
   avgstd(vbig, 0, 10, avg, sd);
   COMPARE_REALS(avg, 1.0e8 + 5.5, 1.0e-6);
   COMPARE_REALS(sd, sd10, 1.0e-9);
}

TEST_CASE("THERMINT-schedule", "[ff][thermint]")
{
   const double eps = 1.0e-12;

   // 21 bins: lambda = 1.00, 0.95, ..., 0.05, 0.00. lmdawin is a 1-based
   // window number, so the schedule starts at 1 rather than 0.
   resetti(21, 10, 100, 50);
   REQUIRE(lmdawin == 1);
   REQUIRE(lambda == 1.0);
   for (int k = 1; k <= 20; ++k) {
      nextLmdaWin();
      REQUIRE(lmdawin == k + 1);
      COMPARE_REALS(lambda, 1.0 - (double)k / 20.0, eps);
   }
   // the final window must sit exactly on the endpoint
   REQUIRE(lambda == 0.0);

   // one call past the end leaves the lambda where it is, and any further
   // call changes nothing at all
   nextLmdaWin();
   REQUIRE(lambda == 0.0);
   REQUIRE(lmdawin == 22);
   nextLmdaWin();
   REQUIRE(lambda == 0.0);
   REQUIRE(lmdawin == 22);

   // 5 bins: 1.00, 0.75, 0.50, 0.25, 0.00
   resetti(5, 10, 40, 20);
   const double lref5[] = {0.75, 0.50, 0.25, 0.00};
   for (int k = 0; k < 4; ++k) {
      nextLmdaWin();
      COMPARE_REALS(lambda, lref5[k], eps);
   }
   REQUIRE(lambda == 0.0);

   // 2 bins: just the two endpoints
   resetti(2, 10, 40, 20);
   REQUIRE(lambda == 1.0);
   nextLmdaWin();
   REQUIRE(lambda == 0.0);
   REQUIRE(lmdawin == 2);

   // An ascending schedule is legal and must hold at its own last value rather
   // than being clamped back toward zero.
   resettisched({0.0, 0.25, 0.5, 0.75, 1.0}, {40, 80, 120, 160, 200}, 10, 0.5);
   REQUIRE(lambda == 0.0);
   for (int k = 1; k <= 4; ++k)
      nextLmdaWin();
   REQUIRE(lambda == 1.0);
   nextLmdaWin();
   REQUIRE(lambda == 1.0);
   REQUIRE(lmdawin == 6);

   // An interior-only schedule never touches either endpoint.
   resettisched({0.75, 0.70, 0.20}, {40, 80, 120}, 10, 0.5);
   COMPARE_REALS(lambda, 0.75, eps);
   nextLmdaWin();
   COMPARE_REALS(lambda, 0.70, eps);
   nextLmdaWin();
   COMPARE_REALS(lambda, 0.20, eps);

   clearti();
}

TEST_CASE("THERMINT-phase", "[ff][thermint]")
{
   int iprod;

   // three windows of forty steps, the first ten equilibration
   resetti(3, 10, 40, 10);
   REQUIRE(lmdawinlen == 40);
   REQUIRE(lmdawineq == 10);

   // equilibration steps are not production, and the count starts at one on
   // the first step after equilibration
   lmdaWinPhase(1, iprod);
   REQUIRE(iprod == 0);
   lmdaWinPhase(10, iprod);
   REQUIRE(iprod == 0);
   lmdaWinPhase(11, iprod);
   REQUIRE(iprod == 1);
   lmdaWinPhase(40, iprod);
   REQUIRE(iprod == 30);

   // looking at a step does not move the schedule
   REQUIRE(lmdawin == 1);
   REQUIRE(lambda == 1.0);

   // the window advances on its final step and on no other
   lmdaWinStep(39);
   REQUIRE(lmdawin == 1);
   lmdaWinStep(40);
   REQUIRE(lmdawin == 2);
   REQUIRE(lambda == 0.5);
   lmdaWinStep(40);
   REQUIRE(lmdawin == 2);

   // positions in a later window are measured from its own start
   lmdaWinPhase(50, iprod);
   REQUIRE(iprod == 0);
   lmdaWinPhase(51, iprod);
   REQUIRE(iprod == 1);
   lmdaWinPhase(80, iprod);
   REQUIRE(iprod == 30);

   // once the schedule has run out no step is in production
   lmdaWinStep(80);
   lmdaWinStep(120);
   REQUIRE(lmdawin == 4);
   lmdaWinPhase(121, iprod);
   REQUIRE(iprod == 0);
   lmdaWinStep(121);
   REQUIRE(lmdawin == 4);
   REQUIRE(lambda == 0.0);

   // A schedule that was built but never laid out over a run, as in a program
   // without lambda windows, is left alone.
   resetti(3, 10, 40, 10);
   lmdawinend.clear();
   tinbcount = 0;
   dedl = 5;
   lmdaWinPhase(11, iprod);
   REQUIRE(iprod == 0);
   etidyn(11);
   REQUIRE(tinbcount == 0);
   lmdaWinStep(40);
   REQUIRE(lmdawin == 1);

   clearti();
}

TEST_CASE("THERMINT-data", "[ff][thermint]")
{
   // thermintData(INIT) and lmdaWinData(INIT) read the Fortran modules, so the
   // adoption they perform is checked in THERMINT-save where the runtime is up.
   // What is testable here is that thermintData is inert while use_ti is false,
   // and that DEALLOC releases the accumulators.
   use_ti = false;
   tinstepavg = 13;
   tilmdadedl.assign(3, 1.0);
   tilmdadedlstd.assign(3, 3.0);
   tidedllist.assign(2, 9.0);
   thermintData(RcOp::ALLOC | RcOp::INIT);
   REQUIRE(tilmdadedl.size() == 3);
   REQUIRE(tilmdadedlstd.size() == 3);
   REQUIRE(tidedllist.size() == 2);

   use_ti = true;
   thermintData(RcOp::DEALLOC);
   REQUIRE(tilmdadedl.empty());
   REQUIRE(tilmdadedlstd.empty());
   REQUIRE(tidedllist.empty());
   REQUIRE(tilmdahist.empty());

   // The window schedule is released whether or not a method is still active.
   use_ti = false;
   lmdawinend.assign(3, 40);
   lmdawinfrac.assign(3, 1.0 / 3.0);
   lmdawinlist.assign(3, 0.5);
   lmdaWinData(RcOp::DEALLOC);
   REQUIRE(lmdawinend.empty());
   REQUIRE(lmdawinfrac.empty());
   REQUIRE(lmdawinlist.empty());

   clearti();
}

TEST_CASE("THERMINT-etidyn", "[ff][thermint]")
{
   const double eps = testGetEps(1.0e-4, 1.0e-12);
   const double sd10 = std::sqrt(8.25);

   // 5 windows of 40 steps: 20 equilibration, 20 production, 2 blocks of 10.
   resetti(5, 10, 40, 20);
   REQUIRE(tinbtot == 10);

   std::vector<double> lambda_seen(201, -1.0);
   for (int istep = 1; istep <= 200; ++istep) {
      dedl = (energy_prec)istep;
      lambda_seen[istep] = lambda;
      tidynstep(istep);
   }

   // means of the ten consecutive integers in each production block, in record
   // order rather than one row per window
   const double ref[10] = {25.5, 35.5, 65.5, 75.5, 105.5, 115.5, 145.5, 155.5, 185.5, 195.5};

   REQUIRE(tinbcount == 10);
   for (int b = 0; b < 10; ++b) {
      COMPARE_REALS(tilmdadedl[b], ref[b], eps);
      COMPARE_REALS(tilmdadedlstd[b], sd10, eps);
      // each block is tagged with the lambda that produced it
      COMPARE_REALS(tilmdahist[b], 1.0 - (double)(b / 2) / 4.0, 1.0e-12);
   }

   // the schedule must advance at the window boundary, not one step off
   for (int istep = 1; istep <= 200; ++istep) {
      double lref = 1.0 - (double)((istep - 1) / 40) / 4.0;
      COMPARE_REALS(lambda_seen[istep], lref, 1.0e-12);
   }
   REQUIRE(lmdawin == 6);
   REQUIRE(lambda == 0.0);

   // Same run, but poison every equilibration step. The block averages must be
   // untouched, which makes the "discard while equilibrating" rule explicit.
   resetti(5, 10, 40, 20);
   for (int istep = 1; istep <= 200; ++istep) {
      int tistep = (istep - 1) % 40 + 1;
      dedl = (tistep <= lmdawineq) ? (energy_prec)-1.0e9 : (energy_prec)istep;
      tidynstep(istep);
   }
   REQUIRE(tinbcount == 10);
   for (int b = 0; b < 10; ++b)
      COMPARE_REALS(tilmdadedl[b], ref[b], eps);

   clearti();
}

TEST_CASE("THERMINT-etidyn-partialblock", "[ff][thermint]")
{
   const double eps = testGetEps(1.0e-4, 1.0e-12);

   // 25 production steps per window with blocks of 10: two blocks flush and
   // five samples are stranded in tidedllist. They must not leak into the next
   // window, which overwrites every index before its first flush.
   resetti(5, 10, 40, 15);
   REQUIRE(lmdawineq == 15);
   REQUIRE(tinbtot == 10);

   for (int istep = 1; istep <= 200; ++istep) {
      dedl = (energy_prec)istep;
      tidynstep(istep);
   }

   REQUIRE(tinbcount == 10);
   // window 1: steps 16-25 and 26-35; steps 36-40 orphaned
   COMPARE_REALS(tilmdadedl[0], 20.5, eps);
   COMPARE_REALS(tilmdadedl[1], 30.5, eps);
   // window 2 production starts at step 56; a leak from window 1 would drag
   // this below 60.5
   COMPARE_REALS(tilmdadedl[2], 60.5, eps);
   COMPARE_REALS(tilmdadedl[3], 70.5, eps);

   clearti();
}

TEST_CASE("THERMINT-etidyn-trailing", "[ff][thermint]")
{
   // 210 steps over 5 windows of 40: the last 10 steps fall past the schedule
   // and must hit the lmdawin > nlmdawin early returns rather than index out
   // of bounds on lmdawinend[lmdawin-1].
   resetti(5, 10, 40, 20);

   for (int istep = 1; istep <= 210; ++istep) {
      dedl = (energy_prec)istep;
      tidynstep(istep);
   }

   REQUIRE(lmdawin == 6);
   REQUIRE(tinbcount == 10);
   REQUIRE(tinbcount == tinbtot);
   REQUIRE((int)tilmdadedl.size() == 10);

   clearti();
}

TEST_CASE("THERMINT-uneven", "[ff][thermint]")
{
   const double eps = testGetEps(1.0e-4, 1.0e-12);

   // Windows of very different lengths, which a fixed-width nstep/nlmdawin
   // layout could not express at all: 50%, 48% and 2% of a 1000 step run.
   resettisched({1.0, 0.5, 0.0}, {500, 980, 1000}, 10, 0.5);
   REQUIRE(tinbtot == 50); // 25 + 24 + 1

   // each window is measured against the boundary before it
   const int lenref[] = {500, 480, 20}, eqref[] = {250, 240, 10};
   for (int i = 1; i <= 3; ++i) {
      int nlen = 0, neq = 0;
      lmdaWinSize(i, nlen, neq);
      REQUIRE(nlen == lenref[i - 1]);
      REQUIRE(neq == eqref[i - 1]);
   }

   for (int istep = 1; istep <= 1000; ++istep) {
      dedl = (energy_prec)istep;
      tidynstep(istep);
   }

   REQUIRE(tinbcount == 50);
   // window 1 production starts at step 251
   COMPARE_REALS(tilmdadedl[0], 255.5, eps);
   // the 2% window holds exactly one block, steps 991-1000
   COMPARE_REALS(tilmdadedl[49], 995.5, eps);
   COMPARE_REALS(tilmdahist[0], 1.0, 1.0e-12);
   COMPARE_REALS(tilmdahist[24], 1.0, 1.0e-12);
   COMPARE_REALS(tilmdahist[25], 0.5, 1.0e-12);
   COMPARE_REALS(tilmdahist[48], 0.5, 1.0e-12);
   COMPARE_REALS(tilmdahist[49], 0.0, 1.0e-12);

   // A window shorter than one block records nothing, but the run still visits
   // its lambda and the windows around it are unaffected.
   resettisched({1.0, 0.5, 0.0}, {100, 105, 205}, 10, 0.5);
   REQUIRE(tinbtot == 10); // 5 + 0 + 5

   std::vector<double> lambda_seen(206, -1.0);
   for (int istep = 1; istep <= 205; ++istep) {
      dedl = (energy_prec)istep;
      lambda_seen[istep] = lambda;
      tidynstep(istep);
   }

   REQUIRE(tinbcount == 10);
   COMPARE_REALS(lambda_seen[101], 0.5, 1.0e-12); // the short window is visited
   COMPARE_REALS(lambda_seen[105], 0.5, 1.0e-12);
   COMPARE_REALS(lambda_seen[106], 0.0, 1.0e-12);
   COMPARE_REALS(tilmdahist[4], 1.0, 1.0e-12); // last block of window 1
   COMPARE_REALS(tilmdahist[5], 0.0, 1.0e-12); // first block of window 3
   COMPARE_REALS(tilmdadedl[4], 95.5, eps);    // steps 91-100
   COMPARE_REALS(tilmdadedl[5], 160.5, eps);   // steps 156-165

   clearti();
}

TEST_CASE("THERMINT-save", "[ff][thermint]")
{
   // The one case that needs the Fortran runtime: the schedule comes from
   // setlmdasched and the .ti file is written by prttihead and saveti.
   const char* argv[] = {"dummy"};
   tinkerFortranRuntimeBegin(1, (char**)argv);
   initial();

   // Ends the runtime and puts back the base name and the TI state as the case
   // leaves scope, so that a failed check leaves none of them to the cases that
   // run after it.
   struct Teardown
   {
      char savedname[240];
      int savedleng;
      std::string tifile;
      ~Teardown()
      {
         try {
            if (tifile.size())
               fileExistsAndDelete(tifile);
            std::memcpy(files::filename, savedname, 240);
            files::leng = savedleng;
            dlmda::use_ti = 0;
            clearti();
            testEnd();
         } catch (...) {
         }
      }
   } teardown;

   // write into a scratch base name rather than whatever the runtime picked
   std::memcpy(teardown.savedname, files::filename, 240);
   teardown.savedleng = files::leng;
   FstrView fname = files::filename;
   fname = "tisave_tmp";
   files::leng = 10;

   // mutate.f never ran, so seed what it would have parsed and let
   // setlmdasched build the schedule itself
   dlmda::use_ti = 1;
   mutant::use_rel = 0;
   FstrView(dlmda::elmdamap) = "EXP";
   FstrView(dlmda::plmdamap) = "EXP";
   FstrView(dlmda::vlmdamap) = "EXP";
   dlmda::elmdaexp = 1;
   dlmda::plmdaexp = 1;
   dlmda::vlmdaexp = 1;
   dlmda::nlmdawin = 5;
   dlmda::lmdawinratio = 0.5;
   thrmint::tinstepavg = 10;
   int nwin = 0, nwinset = 0;
   tinker_f_setlmdasched(&nwin, &nwinset);

   use_ti = true;
   useExpMaps(1, 1, 1);

   // lmdaWinData adopts the precomputed schedule and thermintData the block
   // size; the boundaries wait for the length of the run
   lmdaWinData(RcOp::INIT);
   thermintData(RcOp::INIT);
   REQUIRE(nlmdawin == 5);
   REQUIRE(tinstepavg == 10);
   COMPARE_REALS(lmdawinratio, 0.5, 1.0e-12);
   REQUIRE((int)lmdawinlist.size() == 5);
   COMPARE_REALS(lmdawinlist[0], 1.0, 1.0e-12);
   COMPARE_REALS(lmdawinlist[2], 0.5, 1.0e-12);
   REQUIRE(lmdawinlist[4] == 0.0);
   REQUIRE((int)lmdawinfrac.size() == 5);
   COMPARE_REALS(lmdawinfrac[0], 0.2, 1.0e-12);
   REQUIRE(lmdawin == 1);
   REQUIRE(lmdawinend.empty());

   // An odd step count is absorbed by the boundaries rather than left as a
   // trailing remainder, and 41*0.5 truncates to 20. inittidyn alone does not
   // touch the file system.
   int ns = 205;
   tinker_f_inittidyn(&ns);
   lmdaWinFromFortran();
   REQUIRE((int)lmdawinend.size() == 5);
   REQUIRE(lmdawinend[0] == 41);
   REQUIRE(lmdawinend[4] == 205);
   REQUIRE(lmdawinlen == 41);
   REQUIRE(lmdawineq == 20);
   REQUIRE(thrmint::tinbtot == 10);

   // init_tidyn divides the run among the windows and starts the file
   init_tidyn(200);
   REQUIRE((int)lmdawinend.size() == 5);
   REQUIRE(lmdawinend[0] == 40);
   REQUIRE(lmdawinend[4] == 200);
   REQUIRE(lmdawinlen == 40);
   REQUIRE(lmdawineq == 20);
   REQUIRE((lmdawinlen - lmdawineq) / tinstepavg == 2);
   REQUIRE(tinbtot == 10);
   REQUIRE(lmdawin == 1);
   REQUIRE(lambda == 1.0);
   REQUIRE(tinbcount == 0);

   // prttihead claims a new version, so take the name it actually used
   std::string tifile = FstrView(dlmda::lmdasavefile).trim();
   teardown.tifile = tifile;
   REQUIRE(tifile.size() > 0);
   REQUIRE(tiCountRows(tifile) == 0); // header only

   for (int istep = 1; istep <= 200; ++istep) {
      dedl = (energy_prec)istep;
      tidynstep(istep);
      if (istep == 80) {
         mdsaveLmdaFinal(istep);
         REQUIRE(tiCountRows(tifile) == 4); // two windows of two blocks
      } else if (istep == 160) {
         mdsaveLmdaFinal(istep);
         REQUIRE(tiCountRows(tifile) == 8);
         // nothing new has completed, so a second call must not append
         mdsaveLmdaFinal(istep);
         REQUIRE(tiCountRows(tifile) == 8);
      }
   }

   mdsaveLmdaFinal(200);
   REQUIRE(tiCountRows(tifile) == 10);
   REQUIRE(tiCountRows(tifile) == tinbcount);
   REQUIRE(thrmint::tinbsave == tinbcount);

   // the lambda column must carry the schedule, not the row number
   COMPARE_REALS(tiRowLambda(tifile, 1), 1.0, 1.0e-8);
   COMPARE_REALS(tiRowLambda(tifile, 5), 0.5, 1.0e-8);
   COMPARE_REALS(tiRowLambda(tifile, 10), 0.0, 1.0e-8);
}

TEST_CASE("THERMINT-mapsublambda", "[ff][thermint]")
{
   const double epsv = testGetEps(1.0e-6, 1.0e-12); // elam/plam/vlam are real
   const double epsd = 1.0e-12;                     // the derivatives are double

   // ele uses elmdamap.exp, pol uses plmdamap.exp, vdw uses vlmdamap.exp.
   useExpMaps(2, 3, 1);

   // The maps below are exercised from the one main lambda; DLMDA-one-main-lambda
   // pins that mapSubLambda reads it and nothing else.
   mapAt(0.5);

   COMPARE_REALS(elam, 0.25, epsv); // 0.5^2
   COMPARE_REALS(deldlmda, 1.0, epsd);
   COMPARE_REALS(d2eldlmda2, 2.0, epsd);

   COMPARE_REALS(plam, 0.125, epsv); // 0.5^3
   COMPARE_REALS(dpldlmda, 0.75, epsd);
   COMPARE_REALS(d2pldlmda2, 3.0, epsd);

   COMPARE_REALS(vlam, 0.5, epsv); // 0.5^1
   COMPARE_REALS(dvldlmda, 1.0, epsd);
   COMPARE_REALS(d2vldlmda2, 0.0, epsd);

   // Lower boundary clamp of the power-law map.
   mapAt(0.0);
   COMPARE_REALS(vlam, 0.0, epsv);
   COMPARE_REALS(dvldlmda, 1.0, epsd); // exponent 1
   COMPARE_REALS(d2vldlmda2, 0.0, epsd);
   COMPARE_REALS(elam, 0.0, epsv);
   COMPARE_REALS(deldlmda, 0.0, epsd); // exponent 2
   COMPARE_REALS(d2eldlmda2, 2.0, epsd);
   COMPARE_REALS(plam, 0.0, epsv);
   COMPARE_REALS(dpldlmda, 0.0, epsd); // exponent 3
   COMPARE_REALS(d2pldlmda2, 0.0, epsd);

   // Upper boundary clamp.
   mapAt(1.0);
   COMPARE_REALS(elam, 1.0, epsv);
   COMPARE_REALS(deldlmda, 2.0, epsd);
   COMPARE_REALS(d2eldlmda2, 2.0, epsd);
   COMPARE_REALS(plam, 1.0, epsv);
   COMPARE_REALS(dpldlmda, 3.0, epsd);
   COMPARE_REALS(d2pldlmda2, 6.0, epsd);

   // Shifted inverse-power map, identity case.
   vlmdamap.type = Lmdamap::INV;
   vlmdamap.invn = 1;
   vlmdamap.inveps = 0.3;
   mapAt(0.25);
   COMPARE_REALS(vlam, 0.25, epsv);
   COMPARE_REALS(dvldlmda, 1.0, epsd);
   COMPARE_REALS(d2vldlmda2, 0.0, epsd);

   // Shifted inverse-power map, n = 4. The endpoints are exact by construction.
   vlmdamap.invn = 4;
   mapAt(0.0);
   COMPARE_REALS(vlam, 0.0, epsv);
   mapAt(1.0);
   COMPARE_REALS(vlam, 1.0, epsv);

   // Check the reported first derivative against a central difference of the
   // mapped value -- independent of the closed form in src/ost.cpp.
   const double h = 0.01;
   mapAt(0.5 + h);
   double vhi = vlam;
   mapAt(0.5 - h);
   double vlo = vlam;
   mapAt(0.5);
   COMPARE_REALS(dvldlmda, (vhi - vlo) / (2.0 * h), 1.0e-3);
   REQUIRE(dvldlmda > 0.0);

   // Leave the maps as the defaults set by mutate.f so nothing downstream sees
   // a half-configured state.
   clearti();
}

#if TINKER_GPULANG_CUDA
// etidyn samples dU/dlambda once the step is complete, after whatever energy
// calls a barostat trial made of its own. Those carry no lambda derivative, and
// are to leave the one of the force call of the step in place.
TEST_CASE("THERMINT-dedl-survives-trial", "[ff][thermint]")
{
   // Puts the bath back as the case leaves scope, however it leaves.
   struct BathReset
   {
      ~BathReset()
      {
         bath::kelvin = 0.;
         bath::atmsph = 0.;
         bath::isothermal = 0;
         bath::isobaric = 0;
      }
   } bathReset;
   TestLmdaFlagReset lmdaReset;

   const std::string dir = TINKER9_DIRSTR "/test/file/";
   TestFile fxyz(dir + "mutate/water2.xyz", "water2.xyz");
   TestFile fkey(dir + "mutate/128_water_ast_ye_l05.key", "tidedl.key", "\nintegrator verlet\n");
   TestFile fprm(dir + "commit_6fe8e913/water03.prm");
   const char* argv[] = {"dummy", "water2.xyz", "-k", "tidedl.key"};
   TestSession session(4, argv);
   testMdInit(298., 1.);
   rc_flag = calc::xyz | calc::vel | calc::mass | calc::energy | calc::grad | calc::md | calc::virial;
   session.init();
   REQUIRE(use_dlmda);

   energy(calc::v1);
   const energy_prec dedl0 = dedl;
   const energy_prec e0 = esum;
   REQUIRE(dedl0 != 0);

   // an energy call with no lambda derivative
   energy(calc::energy);
   REQUIRE(dedl == dedl0);

   {
      MonteCarloBarostat mc;
      // an old energy far below any trial energy rejects the move for certain
      monteCarloBarostat(-1.0e30, 298., false, false);
      REQUIRE(dedl == dedl0);
      // and a move that is accepted or not as it falls
      monteCarloBarostat(e0, 298., false, false);
      REQUIRE(dedl == dedl0);
   }

   session.end();
}
#endif
