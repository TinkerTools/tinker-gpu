#include "ff/dlmda.h"
#include "ff/elec.h"
#include "ff/evdw.h"
#include "test.h"
#include "testrt.h"

#include <cmath>
#include <tinker/detail/mutant.hh>

// Host-only checks of the lambda-mapping math in src/dlmda.cpp: the quintic
// taper and the staged relative free energy schedule that mapSubLambda()
// builds from it. No molecular system, no GPU.

using namespace tinker;

namespace {
// mapSubLambda() reads the one main lambda, so drive it by setting that first.
void mapAt(double lmda)
{
   lambda = lmda;
   mapSubLambda();
}

double taperAt(double x, double cut, double off)
{
   double t, dt, d2t;
   quinticTaper(x, cut, off, t, dt, d2t);
   return t;
}

double dtaperAt(double x, double cut, double off)
{
   double t, dt, d2t;
   quinticTaper(x, cut, off, t, dt, d2t);
   return dt;
}
}

TEST_CASE("DLMDA-one-main-lambda", "[ff][dlmda]")
{
   // Without a sampling method the main lambda is the one from the key file.
   bool oldUseOst = use_ost, oldUseMeta = use_meta, oldUseTi = use_ti;
   bool oldRel = use_relstage;
   bool oldE = use_elmdamap, oldP = use_plmdamap, oldV = use_vlmdamap;
   Lmdamap oldEm = elmdamap, oldPm = plmdamap, oldVm = vlmdamap;
   int oldEx = elmdaexp, oldPx = plmdaexp, oldVx = vlmdaexp;
   double oldMutant = mutant::lambda, oldLambda = lambda;

   // A main lambda claims all three sub-lambdas, so map all three.
   use_relstage = false;
   elmdamap = Lmdamap::EXP;
   plmdamap = Lmdamap::EXP;
   vlmdamap = Lmdamap::EXP;
   elmdaexp = 2;
   plmdaexp = 2;
   vlmdaexp = 2;
   use_elmdamap = true;
   use_plmdamap = true;
   use_vlmdamap = true;

   // Decoy: were the Fortran copy consulted, elam would come out 0.81.
   mutant::lambda = 0.9;

   // 0 = a fixed lambda from the key file, then each sampling method in turn.
   for (int method = 0; method < 4; ++method) {
      use_ost = (method == 1);
      use_meta = (method == 2);
      use_ti = (method == 3);
      CAPTURE(method);
      mapAt(0.5);
      COMPARE_REALS(elam, 0.25, 1.0e-7); // 0.5^2
   }

   use_ost = oldUseOst;
   use_meta = oldUseMeta;
   use_ti = oldUseTi;
   use_relstage = oldRel;
   use_elmdamap = oldE;
   use_plmdamap = oldP;
   use_vlmdamap = oldV;
   elmdamap = oldEm;
   plmdamap = oldPm;
   vlmdamap = oldVm;
   elmdaexp = oldEx;
   plmdaexp = oldPx;
   vlmdaexp = oldVx;
   mutant::lambda = oldMutant;
   lambda = oldLambda;
}

TEST_CASE("DLMDA-taper-shape", "[ff][dlmda]")
{
   const double cut = 0.3, off = 0.7;

   // Flat and saturated outside the window, including well outside it.
   COMPARE_REALS(taperAt(-2.0, cut, off), 1.0, 1.0e-14);
   COMPARE_REALS(taperAt(-1.0, cut, off), 1.0, 1.0e-14);
   COMPARE_REALS(taperAt(0.0, cut, off), 1.0, 1.0e-14);
   COMPARE_REALS(taperAt(cut, cut, off), 1.0, 1.0e-14);
   COMPARE_REALS(taperAt(off, cut, off), 0.0, 1.0e-14);
   COMPARE_REALS(taperAt(1.0, cut, off), 0.0, 1.0e-14);
   COMPARE_REALS(taperAt(2.0, cut, off), 0.0, 1.0e-14);

   // Monotonically decreasing across the window, and symmetric about the
   // midpoint where it passes through one half.
   COMPARE_REALS(taperAt(0.5 * (cut + off), cut, off), 0.5, 1.0e-12);
   double prev = 1.0;
   for (int k = 1; k < 40; ++k) {
      double x = cut + (off - cut) * k / 40.0;
      double t = taperAt(x, cut, off);
      CAPTURE(x);
      REQUIRE(t < prev);
      REQUIRE(t >= 0.0);
      REQUIRE(t <= 1.0);
      prev = t;
   }

   // The first and second derivative vanish at both ends. This is what makes
   // dU/dlambda continuous where one staged leg hands over to the next.
   const double eps = 1.0e-9;
   double t, dt, d2t;
   quinticTaper(cut + eps, cut, off, t, dt, d2t);
   COMPARE_REALS(dt, 0.0, 1.0e-15);
   COMPARE_REALS(d2t, 0.0, 1.0e-6);
   quinticTaper(off - eps, cut, off, t, dt, d2t);
   COMPARE_REALS(dt, 0.0, 1.0e-15);
   COMPARE_REALS(d2t, 0.0, 1.0e-6);
}

TEST_CASE("DLMDA-taper-derivatives", "[ff][dlmda]")
{
   const double cut = 0.3, off = 0.7;
   const double h = 1.0e-6;

   // Analytic derivatives against central finite differences of the taper.
   for (int k = 1; k < 20; ++k) {
      double x = cut + (off - cut) * k / 20.0;
      CAPTURE(x);
      double t, dt, d2t;
      quinticTaper(x, cut, off, t, dt, d2t);

      double fd = (taperAt(x + h, cut, off) - taperAt(x - h, cut, off)) / (2 * h);
      COMPARE_REALS(dt, fd, 1.0e-7);

      double fd2 = (dtaperAt(x + h, cut, off) - dtaperAt(x - h, cut, off)) / (2 * h);
      COMPARE_REALS(d2t, fd2, 1.0e-5);
   }
}

TEST_CASE("DLMDA-taper-matches-switch", "[ff][dlmda]")
{
   // quinticTaper evaluates in a normalized variable; switch.f:128-136 builds
   // the same polynomial as c0..c5 in powers of x. Check they agree away from
   // the window edges, where the c0..c5 form is well conditioned.
   const double cut = 0.3, off = 0.7;
   double off2 = off * off, cut2 = cut * cut;
   double dif = off - cut;
   double denom = dif * dif * dif * dif * dif;
   double c0 = off * off2 * (off2 - 5.0 * off * cut + 10.0 * cut2) / denom;
   double c1 = -30.0 * off2 * cut2 / denom;
   double c2 = 30.0 * (off2 * cut + off * cut2) / denom;
   double c3 = -10.0 * (off2 + 4.0 * off * cut + cut2) / denom;
   double c4 = 15.0 * (off + cut) / denom;
   double c5 = -6.0 / denom;

   for (int k = 1; k < 20; ++k) {
      double x = cut + dif * k / 20.0;
      CAPTURE(x);
      double x2 = x * x, x3 = x2 * x, x4 = x2 * x2, x5 = x2 * x3;
      double ref = c5 * x5 + c4 * x4 + c3 * x3 + c2 * x2 + c1 * x + c0;
      double dref = 5.0 * c5 * x4 + 4.0 * c4 * x3 + 3.0 * c3 * x2 + 2.0 * c2 * x + c1;
      double d2ref = 20.0 * c5 * x3 + 12.0 * c4 * x2 + 6.0 * c3 * x + 2.0 * c2;

      double t, dt, d2t;
      quinticTaper(x, cut, off, t, dt, d2t);
      COMPARE_REALS(t, ref, 1.0e-14);
      COMPARE_REALS(dt, dref, 1.0e-13);
      COMPARE_REALS(d2t, d2ref, 1.0e-12);
   }
}

TEST_CASE("DLMDA-taper-degenerate", "[ff][dlmda]")
{
   // A zero-width or inverted window collapses to a step at cut, with no
   // division by zero.
   double t, dt, d2t;
   quinticTaper(0.4, 0.5, 0.5, t, dt, d2t);
   COMPARE_REALS(t, 1.0, 1.0e-14);
   COMPARE_REALS(dt, 0.0, 1.0e-14);
   quinticTaper(0.6, 0.5, 0.5, t, dt, d2t);
   COMPARE_REALS(t, 0.0, 1.0e-14);
   COMPARE_REALS(dt, 0.0, 1.0e-14);
}

TEST_CASE("DLMDA-relstage-schedule", "[ff][dlmda]")
{
   // Drive mapSubLambda() through the staged schedule directly. Only the
   // scalar mapping is exercised here; the endpoint mixing needs a system.
   use_relstage = true;
   relstg2lmda0 = 0.7;
   relstg2lmda1 = 1.0;
   relstg1lmda0 = 0.0;
   relstg1lmda1 = 0.3;
   vlmdamap = Lmdamap::QNT;
   qntvlmda0 = 0.3;
   qntvlmda1 = 0.7;

   // lambda = 1: ligand 1 fully coupled, van der Waals at ligand 1.
   mapAt(1.0);
   REQUIRE(relstage == RelStage::LIG1_ELE);
   COMPARE_REALS(elam, 1.0, 1.0e-14);
   REQUIRE(relstagemix == false);
   COMPARE_REALS(vlam, 1.0, 1.0e-14);
   COMPARE_REALS(deldlmda, 0.0, 1.0e-14);

   // Interior of the ligand 1 discharge leg.
   mapAt(0.85);
   REQUIRE(relstage == RelStage::LIG1_ELE);
   COMPARE_REALS(elam, 0.5, 1.0e-12);
   REQUIRE(relstagemix == true);
   REQUIRE(deldlmda > 0.0); // the weight grows with lambda
   COMPARE_REALS(vlam, 1.0, 1.0e-14);

   // The van der Waals morph leg: both ligands electrostatically decoupled.
   for (double lambda : {0.7, 0.5, 0.3}) {
      CAPTURE(lambda);
      mapAt(lambda);
      REQUIRE(relstage == RelStage::VDW_MORPH);
      COMPARE_REALS(elam, 0.0, 1.0e-14);
      REQUIRE(relstagemix == false);
      COMPARE_REALS(deldlmda, 0.0, 1.0e-14);
      COMPARE_REALS(d2eldlmda2, 0.0, 1.0e-14);
   }
   mapAt(0.5);
   COMPARE_REALS(vlam, 0.5, 1.0e-12);

   // Interior of the ligand 0 recharge leg.
   mapAt(0.15);
   REQUIRE(relstage == RelStage::LIG0_ELE);
   COMPARE_REALS(elam, 0.5, 1.0e-12);
   REQUIRE(relstagemix == true);
   REQUIRE(deldlmda < 0.0); // the weight grows as lambda falls
   COMPARE_REALS(vlam, 0.0, 1.0e-14);

   // lambda = 0: ligand 0 fully coupled.
   mapAt(0.0);
   REQUIRE(relstage == RelStage::LIG0_ELE);
   COMPARE_REALS(elam, 1.0, 1.0e-14);
   REQUIRE(relstagemix == false);
   COMPARE_REALS(vlam, 0.0, 1.0e-14);
   COMPARE_REALS(deldlmda, 0.0, 1.0e-14);

   use_relstage = false;
}

TEST_CASE("DLMDA-relstage-collapsed-weight", "[ff][dlmda]")
{
   // Just inside a leg, the weight is built by cancellation and collapses onto
   // zero -- or a little past it -- for about 5e-7 of main lambda past the
   // decoupled edge. The schedule has to clamp that to zero and report the
   // morph leg: a LIG leg with no mix means "the weight is 1" to the term
   // routines, which would switch the entire ligand interaction on inside a
   // window a lambda dynamics run can wander into. Both legs approach zero from
   // their own side, so both are checked.
   use_relstage = true;
   relstg2lmda0 = 0.7;
   relstg2lmda1 = 1.0;
   relstg1lmda0 = 0.0;
   relstg1lmda1 = 0.3;
   vlmdamap = Lmdamap::QNT;
   qntvlmda0 = 0.3;
   qntvlmda1 = 0.7;

   for (double lambda : {0.7 + 1.0e-7, 0.7 + 5.0e-7, 0.3 - 1.0e-7, 0.3 - 5.0e-7}) {
      CAPTURE(lambda);
      mapAt(lambda);
      // Clamped up from zero or from a slightly negative cancellation result.
      REQUIRE(elam == 0.0);
      REQUIRE(relstage == RelStage::VDW_MORPH);
      REQUIRE(relstagemix == false);
   }

   // Past the collapse the weight survives, and the ordinary leg resumes with
   // a mix rather than a bare endpoint.
   for (double lambda : {0.7 + 1.0e-5, 0.3 - 1.0e-5}) {
      CAPTURE(lambda);
      mapAt(lambda);
      REQUIRE(elam > 0.0);
      REQUIRE(relstage != RelStage::VDW_MORPH);
      REQUIRE(relstagemix == true);
   }

   use_relstage = false;
}

TEST_CASE("DLMDA-relstage-continuity", "[ff][dlmda]")
{
   use_relstage = true;
   relstg2lmda0 = 0.7;
   relstg2lmda1 = 1.0;
   relstg1lmda0 = 0.0;
   relstg1lmda1 = 0.3;
   vlmdamap = Lmdamap::QNT;
   qntvlmda0 = 0.3;
   qntvlmda1 = 0.7;

   // Polarization tracks the multipoles exactly, so emplar stays usable.
   for (double lambda : {0.95, 0.85, 0.72, 0.5, 0.28, 0.15, 0.05}) {
      CAPTURE(lambda);
      mapAt(lambda);
      COMPARE_REALS(plam, elam, 1.0e-15);
      COMPARE_REALS(dpldlmda, deldlmda, 1.0e-15);
      COMPARE_REALS(d2pldlmda2, d2eldlmda2, 1.0e-15);
   }

   // The electrostatic weight and its derivative approach each leg boundary
   // continuously from the inside, so dU/dlambda has no step where one leg
   // hands over to the next. The weight is flat to machine precision; the
   // derivative vanishes quadratically, so it is checked against the analytic
   // bound |dw/dl| <= 30 (eps/w)^2 / w for a window of width w rather than a
   // fixed tolerance.
   const double eps = 1.0e-7;
   const double window = 0.3;
   const double dbound = 30.0 * (eps / window) * (eps / window) / window;
   for (double edge : {0.7, 0.3}) {
      CAPTURE(edge);
      mapAt(edge + eps);
      double w_hi = elam, d_hi = deldlmda;
      mapAt(edge - eps);
      double w_lo = elam, d_lo = deldlmda;
      COMPARE_REALS(w_hi, w_lo, 1.0e-14);
      REQUIRE(std::fabs(d_hi - d_lo) <= dbound);
      REQUIRE(std::fabs(d_hi) <= dbound);
      REQUIRE(std::fabs(d_lo) <= dbound);
   }

   // The staged weight is C1 in the main lambda across each leg.
   auto legMatchesTaper = [](double lambda, double lo, double hi, double sign) {
      CAPTURE(lambda);
      mapAt(lambda);
      double t, dt, d2t;
      quinticTaper(lambda, lo, hi, t, dt, d2t);
      // elam is real, so it holds the weight only to float round-off.
      COMPARE_REALS(elam, sign < 0.0 ? 1.0 - t : t, 1.0e-7);
      COMPARE_REALS(deldlmda, sign * dt, 1.0e-14);
      COMPARE_REALS(d2eldlmda2, sign * d2t, 1.0e-14);
   };
   for (double lambda : {0.75, 0.85, 0.95})
      legMatchesTaper(lambda, relstg2lmda0, relstg2lmda1, -1.0);
   for (double lambda : {0.05, 0.15, 0.25})
      legMatchesTaper(lambda, relstg1lmda0, relstg1lmda1, 1.0);

   use_relstage = false;
}
