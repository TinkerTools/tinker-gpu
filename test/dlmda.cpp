#include "ff/dlmda.h"
#include "ff/elec.h"
#include "ff/evdw.h"
#include "test.h"
#include "testrt.h"

#include <cmath>

// Host-only checks of the lambda-mapping math in src/dlmda.cpp: the quintic
// taper and the staged relative free energy schedule that mapSubLambda()
// builds from it. No molecular system, no GPU.

using namespace tinker;

namespace {
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
   relstage1lmda0 = 0.7;
   relstage1lmda1 = 1.0;
   relstage0lmda0 = 0.0;
   relstage0lmda1 = 0.3;
   vlmdamap = Lmdamap::QNT;
   qntvlmda0 = 0.3;
   qntvlmda1 = 0.7;

   // lambda = 1: ligand 1 fully coupled, van der Waals at ligand 1.
   mapSubLambda(1.0);
   REQUIRE(relstage == RelStage::LIG1_ELE);
   COMPARE_REALS(relstagew, 1.0, 1.0e-14);
   REQUIRE(relstagemix == false);
   COMPARE_REALS(vlam, 1.0, 1.0e-14);
   COMPARE_REALS(deldlmda, 0.0, 1.0e-14);

   // Interior of the ligand 1 discharge leg.
   mapSubLambda(0.85);
   REQUIRE(relstage == RelStage::LIG1_ELE);
   COMPARE_REALS(relstagew, 0.5, 1.0e-12);
   REQUIRE(relstagemix == true);
   REQUIRE(deldlmda > 0.0); // the weight grows with lambda
   COMPARE_REALS(vlam, 1.0, 1.0e-14);

   // The van der Waals morph leg: both ligands electrostatically decoupled.
   for (double lambda : {0.7, 0.5, 0.3}) {
      CAPTURE(lambda);
      mapSubLambda(lambda);
      REQUIRE(relstage == RelStage::VDW_MORPH);
      COMPARE_REALS(relstagew, 0.0, 1.0e-14);
      REQUIRE(relstagemix == false);
      COMPARE_REALS(deldlmda, 0.0, 1.0e-14);
      COMPARE_REALS(d2eldlmda2, 0.0, 1.0e-14);
   }
   mapSubLambda(0.5);
   COMPARE_REALS(vlam, 0.5, 1.0e-12);

   // Interior of the ligand 0 recharge leg.
   mapSubLambda(0.15);
   REQUIRE(relstage == RelStage::LIG0_ELE);
   COMPARE_REALS(relstagew, 0.5, 1.0e-12);
   REQUIRE(relstagemix == true);
   REQUIRE(deldlmda < 0.0); // the weight grows as lambda falls
   COMPARE_REALS(vlam, 0.0, 1.0e-14);

   // lambda = 0: ligand 0 fully coupled.
   mapSubLambda(0.0);
   REQUIRE(relstage == RelStage::LIG0_ELE);
   COMPARE_REALS(relstagew, 1.0, 1.0e-14);
   REQUIRE(relstagemix == false);
   COMPARE_REALS(vlam, 0.0, 1.0e-14);
   COMPARE_REALS(deldlmda, 0.0, 1.0e-14);

   use_relstage = false;
}

TEST_CASE("DLMDA-relstage-continuity", "[ff][dlmda]")
{
   use_relstage = true;
   relstage1lmda0 = 0.7;
   relstage1lmda1 = 1.0;
   relstage0lmda0 = 0.0;
   relstage0lmda1 = 0.3;
   vlmdamap = Lmdamap::QNT;
   qntvlmda0 = 0.3;
   qntvlmda1 = 0.7;

   // Polarization tracks the multipoles exactly, so emplar stays usable.
   for (double lambda : {0.95, 0.85, 0.72, 0.5, 0.28, 0.15, 0.05}) {
      CAPTURE(lambda);
      mapSubLambda(lambda);
      COMPARE_REALS(plam, elam, 1.0e-15);
      COMPARE_REALS(dpldlmda, deldlmda, 1.0e-15);
      COMPARE_REALS(d2pldlmda2, d2eldlmda2, 1.0e-15);
      COMPARE_REALS(elam, relstagew, 1.0e-15);
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
      mapSubLambda(edge + eps);
      double w_hi = relstagew, d_hi = deldlmda;
      mapSubLambda(edge - eps);
      double w_lo = relstagew, d_lo = deldlmda;
      COMPARE_REALS(w_hi, w_lo, 1.0e-14);
      REQUIRE(std::fabs(d_hi - d_lo) <= dbound);
      REQUIRE(std::fabs(d_hi) <= dbound);
      REQUIRE(std::fabs(d_lo) <= dbound);
   }

   // The staged weight is C1 in the main lambda across each leg.
   const double h = 1.0e-6;
   auto weightAt = [](double lambda) {
      mapSubLambda(lambda);
      return relstagew;
   };
   for (double lambda : {0.75, 0.85, 0.95, 0.05, 0.15, 0.25}) {
      CAPTURE(lambda);
      mapSubLambda(lambda);
      double analytic = deldlmda;
      double fd = (weightAt(lambda + h) - weightAt(lambda - h)) / (2 * h);
      COMPARE_REALS(analytic, fd, 1.0e-7);
   }

   use_relstage = false;
}
