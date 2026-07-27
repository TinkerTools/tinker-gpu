#include "ff/thermint.h"
#include "ff/dlmda.h"
#include "ff/elec.h"
#include "ff/evdw.h"
#include "ff/ost.h"

#include "test.h"
#include "testrt.h"

#include <cmath>
#include <vector>

// Unit tests for thermodynamic integration (src/thermint.cpp). Pure host math:
// no molecular system, no Fortran runtime, no GPU. etidyn samples the plain
// host global dedl, so the accumulation logic is driven by assigning dedl
// directly rather than by evaluating any energy.
//
// IMPORTANT: use_ti is a process-wide global and all.tests is one process. If a
// case here leaves it true, a later test's initialize() would see it in the
// TermBuffer::manage() calls in empole/epolar/evdw and silently switch those
// terms into private-buffer chain-rule mode. Every TEST_CASE below resets
// use_ti to false before returning.

using namespace tinker;

namespace {
// resetti -- set the TI scalars and size the accumulators directly, bypassing
// thermintData so the accumulation cases do not depend on it. Keeps the
// invariant tinbin == tilmdadedl.size(), which etidyn's bounds check relies on.
void resetti(int nbin, int nstepavg, int window, int nequil)
{
   use_ti = true;
   tinbin = nbin;
   tinstepavg = nstepavg;
   tiwindow = window;
   tinequil = nequil;
   tieqratio = (window > 0) ? (double)nequil / (double)window : 0.0;
   tibin = 0;
   tilmda = 1.0;
   tilmdadedl.assign(nbin, {});
   tilmdadedlstd.assign(nbin, {});
   tidedllist.assign(nstepavg, 0.0);
   dedl = 0;
}

// Restores the process-wide flag and drops the accumulators.
void clearti()
{
   use_ti = false;
   tilmdadedl.clear();
   tilmdadedlstd.clear();
   tidedllist.clear();
}

// Puts the sub-lambda maps on the power-law branch, which is plain C++. The
// QNT/NONE taper branch calls tinker_f_switch and needs the Fortran runtime.
void useExpMaps(int eexp, int pexp, int vexp)
{
   elmdamap = Lmdamap::EXP;
   plmdamap = Lmdamap::EXP;
   vlmdamap = Lmdamap::EXP;
   elmdaexp = eexp;
   plmdaexp = pexp;
   vlmdaexp = vexp;
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

   // 21 bins: lambda = 1.00, 0.95, ..., 0.05, 0.00 in steps of 1/(tinbin-1).
   resetti(21, 10, 100, 50);
   for (int k = 1; k <= 20; ++k) {
      tischedule();
      REQUIRE(tibin == k);
      COMPARE_REALS(tilmda, 1.0 - (double)k / 20.0, eps);
   }
   // the final window must sit exactly on the endpoint
   REQUIRE(tilmda == 0.0);

   // one call past the end clamps instead of going negative
   tischedule();
   REQUIRE(tilmda == 0.0);
   REQUIRE(tibin == 21);

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
   REQUIRE(tibin == 1);

   clearti();
}

TEST_CASE("THERMINT-data", "[ff][thermint]")
{
   // use_ti false: thermintData is a no-op and must not touch the vectors
   use_ti = false;
   tinbin = 7;
   tinstepavg = 13;
   tilmdadedl.assign(3, std::vector<double>{1.0, 2.0});
   tilmdadedlstd.assign(3, std::vector<double>{3.0});
   tidedllist.assign(2, 9.0);
   thermintData(RcOp::ALLOC | RcOp::INIT);
   REQUIRE(tilmdadedl.size() == 3);
   REQUIRE(tilmdadedl[0].size() == 2);
   REQUIRE(tilmdadedlstd.size() == 3);
   REQUIRE(tidedllist.size() == 2);

   // use_ti true: INIT sizes one row per lambda window
   use_ti = true;
   thermintData(RcOp::INIT);
   REQUIRE((int)tilmdadedl.size() == 7);
   REQUIRE((int)tilmdadedlstd.size() == 7);
   REQUIRE((int)tidedllist.size() == 13);
   for (int i = 0; i < 7; ++i) {
      REQUIRE(tilmdadedl[i].empty());
      REQUIRE(tilmdadedlstd[i].empty());
   }
   REQUIRE(tilmda == 1.0);
   REQUIRE(tibin == 0);

   // re-initializing clears whatever the rows had accumulated
   tilmdadedl[2].push_back(5.0);
   tilmdadedlstd[2].push_back(6.0);
   tibin = 4;
   tilmda = 0.25;
   thermintData(RcOp::INIT);
   REQUIRE((int)tilmdadedl.size() == 7);
   REQUIRE(tilmdadedl[2].empty());
   REQUIRE(tilmdadedlstd[2].empty());
   REQUIRE(tibin == 0);
   REQUIRE(tilmda == 1.0);

   thermintData(RcOp::DEALLOC);
   REQUIRE(tilmdadedl.empty());
   REQUIRE(tilmdadedlstd.empty());
   REQUIRE(tidedllist.empty());

   clearti();
}

TEST_CASE("THERMINT-init_tidyn", "[ff][thermint]")
{
   // init_tidyn ends with mapSubLambda(tilmda); keep it on the power-law branch.
   useExpMaps(1, 1, 1);

   use_ti = true;
   tinstepavg = 10;

   tinbin = 5;
   tieqratio = 0.5;
   init_tidyn(200);
   REQUIRE(tiwindow == 40);
   REQUIRE(tinequil == 20);
   REQUIRE(tibin == 0);
   REQUIRE(tilmda == 1.0);

   tinbin = 21;
   tieqratio = 0.25;
   init_tidyn(2100);
   REQUIRE(tiwindow == 100);
   REQUIRE(tinequil == 25);

   // integer truncation: 205/5 = 41 steps per window, 41*0.5 = 20.5 -> 20
   tinbin = 5;
   tieqratio = 0.5;
   init_tidyn(205);
   REQUIRE(tiwindow == 41);
   REQUIRE(tinequil == 20);

   clearti();
}

TEST_CASE("THERMINT-etidyn", "[ff][thermint]")
{
   const double eps = testGetEps(1.0e-4, 1.0e-12);
   const double sd10 = std::sqrt(8.25);

   // 5 windows of 40 steps: 20 equilibration, 20 production, 2 blocks of 10.
   resetti(5, 10, 40, 20);

   std::vector<double> lambda_seen(201, -1.0);
   for (int istep = 1; istep <= 200; ++istep) {
      dedl = (energy_prec)istep;
      lambda_seen[istep] = tilmda;
      etidyn(istep);
   }

   // means of the ten consecutive integers in each production block
   const double ref[5][2] = {
      {25.5, 35.5}, {65.5, 75.5}, {105.5, 115.5}, {145.5, 155.5}, {185.5, 195.5}};

   REQUIRE((int)tilmdadedl.size() == 5);
   REQUIRE((int)tilmdadedlstd.size() == 5);
   for (int w = 0; w < 5; ++w) {
      REQUIRE((int)tilmdadedl[w].size() == 2);
      REQUIRE((int)tilmdadedlstd[w].size() == 2);
      for (int b = 0; b < 2; ++b) {
         COMPARE_REALS(tilmdadedl[w][b], ref[w][b], eps);
         COMPARE_REALS(tilmdadedlstd[w][b], sd10, eps);
      }
   }

   // the schedule must advance at the window boundary, not one step off
   for (int istep = 1; istep <= 200; ++istep) {
      double lref = 1.0 - (double)((istep - 1) / 40) / 4.0;
      COMPARE_REALS(lambda_seen[istep], lref, 1.0e-12);
   }
   REQUIRE(tibin == 5);
   REQUIRE(tilmda == 0.0);

   // Same run, but poison every equilibration step. The block averages must be
   // untouched, which makes the "discard while equilibrating" rule explicit.
   resetti(5, 10, 40, 20);
   for (int istep = 1; istep <= 200; ++istep) {
      int tistep = (istep - 1) % tiwindow + 1;
      dedl = (tistep <= tinequil) ? (energy_prec)-1.0e9 : (energy_prec)istep;
      etidyn(istep);
   }
   for (int w = 0; w < 5; ++w) {
      REQUIRE((int)tilmdadedl[w].size() == 2);
      for (int b = 0; b < 2; ++b)
         COMPARE_REALS(tilmdadedl[w][b], ref[w][b], eps);
   }

   clearti();
}

TEST_CASE("THERMINT-etidyn-partialblock", "[ff][thermint]")
{
   const double eps = testGetEps(1.0e-4, 1.0e-12);

   // 25 production steps per window with blocks of 10: two blocks flush and
   // five samples are stranded in tidedllist. They must not leak into the next
   // window, which overwrites every index before its first flush.
   resetti(5, 10, 40, 15);

   for (int istep = 1; istep <= 200; ++istep) {
      dedl = (energy_prec)istep;
      etidyn(istep);
   }

   REQUIRE((int)tilmdadedl.size() == 5);
   for (int w = 0; w < 5; ++w)
      REQUIRE((int)tilmdadedl[w].size() == 2);

   // window 0: steps 16-25 and 26-35; steps 36-40 orphaned
   COMPARE_REALS(tilmdadedl[0][0], 20.5, eps);
   COMPARE_REALS(tilmdadedl[0][1], 30.5, eps);
   // window 1 production starts at step 56; a leak from window 0 would drag
   // this below 60.5
   COMPARE_REALS(tilmdadedl[1][0], 60.5, eps);
   COMPARE_REALS(tilmdadedl[1][1], 70.5, eps);

   clearti();
}

TEST_CASE("THERMINT-etidyn-trailing", "[ff][thermint]")
{
   // 210 steps over 5 windows of 40: the last 10 steps fall past the schedule
   // and must hit the tibin >= tinbin early return rather than index out of
   // bounds on tilmdadedl[tibin].
   resetti(5, 10, 40, 20);

   for (int istep = 1; istep <= 210; ++istep) {
      dedl = (energy_prec)istep;
      etidyn(istep);
   }

   REQUIRE(tibin == 5);
   REQUIRE((int)tilmdadedl.size() == 5);
   for (int w = 0; w < 5; ++w) {
      REQUIRE((int)tilmdadedl[w].size() == 2);
      REQUIRE((int)tilmdadedlstd[w].size() == 2);
   }

   clearti();
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
   elmdamap = Lmdamap::QNT;
   plmdamap = Lmdamap::QNT;
   vlmdamap = Lmdamap::QNT;
   clearti();
}
