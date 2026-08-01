#include "ff/thermint.h"
#include "ff/dlmda.h"
#include "ff/elec.h"
#include "ff/evdw.h"
#include "ff/ost.h"
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

#include <tinker/detail/dlmda.hh>
#include <tinker/detail/files.hh>
#include <tinker/detail/thrmint.hh>
#include <tinker/routines.h>

// Unit tests for thermodynamic integration (src/thermint.cpp), following the
// Fortran suite in tinker/test/test_thermint.f. Most cases are pure host math:
// no molecular system, no Fortran runtime, no GPU. etidyn samples the plain
// host global dedl, so the accumulation logic is driven by assigning dedl
// directly rather than by evaluating any energy. Like the Fortran helpers, they
// seed tiwinend/tilmdalist directly and bypass init_tidyn, so the accumulation
// cases do not depend on how the step budget was divided.
//
// THERMINT-save is the exception: it brings up the Fortran runtime, because the
// schedule is precomputed by mutate.f/settisched and the .ti file is written by
// thermint.f/prttihead and saveti. It is also where the adoption performed by
// thermintData(INIT) is checked, since that reads the Fortran thrmint module.
//
// IMPORTANT: use_ti is a process-wide global and all.tests is one process. If a
// case here leaves it true, a later test's initialize() would see it in the
// TermBuffer::manage() calls in empole/epolar/evdw and silently switch those
// terms into private-buffer chain-rule mode. Every TEST_CASE below resets
// use_ti to false before returning.

using namespace tinker;

namespace {
// Puts the sub-lambda maps on the power-law branch, which is plain C++. The
// QNT/NONE taper branch calls tinker_f_switch and needs the Fortran runtime.
void useExpMaps(int eexp, int pexp, int vexp)
{
   use_relstage = false;
   elmdamap = Lmdamap::EXP;
   plmdamap = Lmdamap::EXP;
   vlmdamap = Lmdamap::EXP;
   elmdaexp = eexp;
   plmdaexp = pexp;
   vlmdaexp = vexp;
}

// resettisched -- install an arbitrary schedule and size the accumulators the
// way settiblocks would, then rewind to the first window. tischedule and
// init_tidyn both end in mapSubLambda, so the maps are pinned here too.
void resettisched(const std::vector<double>& lam, const std::vector<int>& winend, int nstepavg,
   double eqratio)
{
   use_ti = true;
   useExpMaps(1, 1, 1);

   tinbin = (int)lam.size();
   tinstepavg = nstepavg;
   tieqratio = eqratio;
   tilmdalist = lam;
   tifraclist.assign(tinbin, 1.0 / (double)tinbin);
   tiwinend = winend;

   tinbtot = 0;
   int istart = 0;
   for (int i = 0; i < tinbin; ++i) {
      int nw = winend[i] - istart;
      int ne = (int)((double)nw * eqratio);
      tinbtot += (nw - ne) / nstepavg;
      istart = winend[i];
   }

   int cap = std::max(1, tinbtot);
   tilmdahist.assign(cap, 0.0);
   tilmdadedl.assign(cap, 0.0);
   tilmdadedlstd.assign(cap, 0.0);
   tidedllist.assign(nstepavg, 0.0);
   tinbcount = 0;

   tibin = 1;
   tilmda = tilmdalist[0];
   tiwindow = tiwinend[0];
   tinequil = (int)((double)tiwindow * tieqratio);
   tinblock = (tiwindow - tinequil) / tinstepavg;
   dedl = 0;
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
   tilmdalist.clear();
   tifraclist.clear();
   tiwinend.clear();
   tilmdahist.clear();
   tilmdadedl.clear();
   tilmdadedlstd.clear();
   tidedllist.clear();
   elmdamap = Lmdamap::QNT;
   plmdamap = Lmdamap::QNT;
   vlmdamap = Lmdamap::QNT;
}

// Number of data records, i.e. lines that prttihead did not write.
int tiCountRows(const std::string& fname)
{
   std::ifstream fs(fname);
   std::string line;
   int n = 0;
   while (std::getline(fs, line))
      if (not line.empty() and line[0] != '#')
         ++n;
   return n;
}

// Lambda column of the nth (1-based) data record.
double tiRowLambda(const std::string& fname, int nth)
{
   std::ifstream fs(fname);
   std::string line;
   int n = 0;
   while (std::getline(fs, line)) {
      if (line.empty() or line[0] == '#')
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

   // 21 bins: lambda = 1.00, 0.95, ..., 0.05, 0.00. tibin is a 1-based window
   // number, so the schedule starts at 1 rather than 0.
   resetti(21, 10, 100, 50);
   REQUIRE(tibin == 1);
   REQUIRE(tilmda == 1.0);
   for (int k = 1; k <= 20; ++k) {
      tischedule();
      REQUIRE(tibin == k + 1);
      COMPARE_REALS(tilmda, 1.0 - (double)k / 20.0, eps);
   }
   // the final window must sit exactly on the endpoint
   REQUIRE(tilmda == 0.0);

   // one call past the end leaves the lambda where it is
   tischedule();
   REQUIRE(tilmda == 0.0);
   REQUIRE(tibin == 22);

   // 5 bins: 1.00, 0.75, 0.50, 0.25, 0.00
   resetti(5, 10, 40, 20);
   const double lref5[] = {0.75, 0.50, 0.25, 0.00};
   for (int k = 0; k < 4; ++k) {
      tischedule();
      COMPARE_REALS(tilmda, lref5[k], eps);
   }
   REQUIRE(tilmda == 0.0);

   // 2 bins: just the two endpoints
   resetti(2, 10, 40, 20);
   REQUIRE(tilmda == 1.0);
   tischedule();
   REQUIRE(tilmda == 0.0);
   REQUIRE(tibin == 2);

   // An ascending schedule is legal and must hold at its own last value rather
   // than being clamped back toward zero.
   resettisched({0.0, 0.25, 0.5, 0.75, 1.0}, {40, 80, 120, 160, 200}, 10, 0.5);
   REQUIRE(tilmda == 0.0);
   for (int k = 1; k <= 4; ++k)
      tischedule();
   REQUIRE(tilmda == 1.0);
   tischedule();
   REQUIRE(tilmda == 1.0);
   REQUIRE(tibin == 6);

   // An interior-only schedule never touches either endpoint.
   resettisched({0.75, 0.70, 0.20}, {40, 80, 120}, 10, 0.5);
   COMPARE_REALS(tilmda, 0.75, eps);
   tischedule();
   COMPARE_REALS(tilmda, 0.70, eps);
   tischedule();
   COMPARE_REALS(tilmda, 0.20, eps);

   clearti();
}

TEST_CASE("THERMINT-data", "[ff][thermint]")
{
   // thermintData(INIT) reads the Fortran thrmint module, so the adoption it
   // performs is checked in THERMINT-save where the runtime is up. What is
   // testable here is that the whole routine is inert while use_ti is false,
   // and that DEALLOC releases the accumulators.
   use_ti = false;
   tinbin = 7;
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
   REQUIRE(tiwinend.empty());

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
      lambda_seen[istep] = tilmda;
      etidyn(istep);
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
   REQUIRE(tibin == 6);
   REQUIRE(tilmda == 0.0);

   // Same run, but poison every equilibration step. The block averages must be
   // untouched, which makes the "discard while equilibrating" rule explicit.
   resetti(5, 10, 40, 20);
   for (int istep = 1; istep <= 200; ++istep) {
      int tistep = (istep - 1) % 40 + 1;
      dedl = (tistep <= tinequil) ? (energy_prec)-1.0e9 : (energy_prec)istep;
      etidyn(istep);
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
   REQUIRE(tinequil == 15);
   REQUIRE(tinbtot == 10);

   for (int istep = 1; istep <= 200; ++istep) {
      dedl = (energy_prec)istep;
      etidyn(istep);
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
   // and must hit the tibin > tinbin early return rather than index out of
   // bounds on tiwinend[tibin-1].
   resetti(5, 10, 40, 20);

   for (int istep = 1; istep <= 210; ++istep) {
      dedl = (energy_prec)istep;
      etidyn(istep);
   }

   REQUIRE(tibin == 6);
   REQUIRE(tinbcount == 10);
   REQUIRE(tinbcount == tinbtot);
   REQUIRE((int)tilmdadedl.size() == 10);

   clearti();
}

TEST_CASE("THERMINT-uneven", "[ff][thermint]")
{
   const double eps = testGetEps(1.0e-4, 1.0e-12);

   // Windows of very different lengths, which the old fixed-width nstep/tinbin
   // layout could not express at all: 50%, 48% and 2% of a 1000 step run.
   resettisched({1.0, 0.5, 0.0}, {500, 980, 1000}, 10, 0.5);
   REQUIRE(tinbtot == 50); // 25 + 24 + 1

   for (int istep = 1; istep <= 1000; ++istep) {
      dedl = (energy_prec)istep;
      etidyn(istep);
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
      lambda_seen[istep] = tilmda;
      etidyn(istep);
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
   // settisched and the .ti file is written by prttihead and saveti.
   const char* argv[] = {"dummy"};
   tinkerFortranRuntimeBegin(1, (char**)argv);
   initial();

   // write into a scratch base name rather than whatever the runtime picked
   char savedname[240];
   std::memcpy(savedname, files::filename, 240);
   int savedleng = files::leng;
   FstrView fname = files::filename;
   fname = "tisave_tmp";
   files::leng = 10;

   // mutate.f never ran, so seed what it would have parsed and let settisched
   // build the schedule itself
   dlmda::use_ti = 1;
   dlmda::use_relstage = 0;
   FstrView(dlmda::elmdamap) = "EXP";
   FstrView(dlmda::plmdamap) = "EXP";
   FstrView(dlmda::vlmdamap) = "EXP";
   dlmda::elmdaexp = 1;
   dlmda::plmdaexp = 1;
   dlmda::vlmdaexp = 1;
   thrmint::tinbin = 5;
   thrmint::tinstepavg = 10;
   thrmint::tieqratio = 0.5;
   int ntiwin = 0, tinbinset = 0;
   tinker_f_settisched(&ntiwin, &tinbinset);

   use_ti = true;
   useExpMaps(1, 1, 1);

   // thermintData adopts the precomputed schedule
   thermintData(RcOp::INIT);
   REQUIRE(tinbin == 5);
   REQUIRE(tinstepavg == 10);
   COMPARE_REALS(tieqratio, 0.5, 1.0e-12);
   REQUIRE((int)tilmdalist.size() == 5);
   COMPARE_REALS(tilmdalist[0], 1.0, 1.0e-12);
   COMPARE_REALS(tilmdalist[2], 0.5, 1.0e-12);
   REQUIRE(tilmdalist[4] == 0.0);
   REQUIRE((int)tifraclist.size() == 5);
   COMPARE_REALS(tifraclist[0], 0.2, 1.0e-12);
   REQUIRE(tibin == 1);

   // init_tidyn divides the run among the windows and starts the file
   init_tidyn(200);
   REQUIRE((int)tiwinend.size() == 5);
   REQUIRE(tiwinend[0] == 40);
   REQUIRE(tiwinend[4] == 200);
   REQUIRE(tiwindow == 40);
   REQUIRE(tinequil == 20);
   REQUIRE(tinblock == 2);
   REQUIRE(tinbtot == 10);
   REQUIRE(tibin == 1);
   REQUIRE(tilmda == 1.0);
   REQUIRE(tinbcount == 0);

   // prttihead claims a new version, so take the name it actually used
   std::string tifile = FstrView(thrmint::tifile).trim();
   REQUIRE(tifile.size() > 0);
   REQUIRE(tiCountRows(tifile) == 0); // header only

   for (int istep = 1; istep <= 200; ++istep) {
      dedl = (energy_prec)istep;
      etidyn(istep);
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

   fileExistsAndDelete(tifile);
   std::memcpy(files::filename, savedname, 240);
   files::leng = savedleng;
   dlmda::use_ti = 0;
   clearti();
   testEnd();
}

TEST_CASE("THERMINT-mapsublambda", "[ff][thermint]")
{
   const double epsv = testGetEps(1.0e-6, 1.0e-12); // elam/plam/vlam are real
   const double epsd = 1.0e-12;                     // the derivatives are double

   // ele uses elmdaexp, pol uses plmdaexp, vdw uses vlmdaexp.
   useExpMaps(2, 3, 1);

   // The lambda comes from the argument, not from ostlambda. Before the
   // refactor mapSubLambda read ostlambda implicitly; this decoy pins the new
   // contract and is the single assertion this case exists for.
   ostlambda = 0.9;
   mapSubLambda(0.5);

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
   mapSubLambda(0.0);
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
   mapSubLambda(1.0);
   COMPARE_REALS(elam, 1.0, epsv);
   COMPARE_REALS(deldlmda, 2.0, epsd);
   COMPARE_REALS(d2eldlmda2, 2.0, epsd);
   COMPARE_REALS(plam, 1.0, epsv);
   COMPARE_REALS(dpldlmda, 3.0, epsd);
   COMPARE_REALS(d2pldlmda2, 6.0, epsd);

   // Shifted inverse-power map, identity case.
   vlmdamap = Lmdamap::INV;
   vlmdainvn = 1;
   vlmdainveps = 0.3;
   mapSubLambda(0.25);
   COMPARE_REALS(vlam, 0.25, epsv);
   COMPARE_REALS(dvldlmda, 1.0, epsd);
   COMPARE_REALS(d2vldlmda2, 0.0, epsd);

   // Shifted inverse-power map, n = 4. The endpoints are exact by construction.
   vlmdainvn = 4;
   mapSubLambda(0.0);
   COMPARE_REALS(vlam, 0.0, epsv);
   mapSubLambda(1.0);
   COMPARE_REALS(vlam, 1.0, epsv);

   // Check the reported first derivative against a central difference of the
   // mapped value -- independent of the closed form in src/ost.cpp.
   const double h = 0.01;
   mapSubLambda(0.5 + h);
   double vhi = vlam;
   mapSubLambda(0.5 - h);
   double vlo = vlam;
   mapSubLambda(0.5);
   COMPARE_REALS(dvldlmda, (vhi - vlo) / (2.0 * h), 1.0e-3);
   REQUIRE(dvldlmda > 0.0);

   // Leave the maps as the defaults set by mutate.f so nothing downstream sees
   // a half-configured state.
   clearti();
}
