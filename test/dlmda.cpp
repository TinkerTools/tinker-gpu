#include "ff/dlmda.h"
#include "ff/elec.h"
#include "ff/evdw.h"
#include "test.h"
#include "testrt.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <tinker/detail/bath.hh>
#include <tinker/detail/dlmda.hh>
#include <tinker/detail/mutant.hh>
#include <tinker/detail/units.hh>
#include <tinker/routines.h>
#include <utility>

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

TEST_CASE("DLMDA-one-main-lambda", "[ff][dlmda][L2]")
{
   // Without a sampling method the main lambda is the one from the key file.
   bool oldUseOst = use_ost, oldUseMeta = use_meta, oldUseTi = use_ti;
   bool oldRel = use_rel;
   const LmdaMap oldEm = elmdamap, oldPm = plmdamap, oldVm = vlmdamap;
   double oldMutant = mutant::lambda, oldLambda = lambda;

   // A main lambda claims all three sub-lambdas, so map all three.
   use_rel = false;
   elmdamap.type = Lmdamap::EXP;
   plmdamap.type = Lmdamap::EXP;
   vlmdamap.type = Lmdamap::EXP;
   elmdamap.exp = 2;
   plmdamap.exp = 2;
   vlmdamap.exp = 2;
   elmdamap.on = true;
   plmdamap.on = true;
   vlmdamap.on = true;

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
   use_rel = oldRel;
   elmdamap = oldEm;
   plmdamap = oldPm;
   vlmdamap = oldVm;
   mutant::lambda = oldMutant;
   lambda = oldLambda;
}

TEST_CASE("DLMDA-taper-shape", "[ff][dlmda][L2]")
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

TEST_CASE("DLMDA-taper-derivatives", "[ff][dlmda][L2]")
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

// The soft core coefficients against finite differences of v^n T(h) with
// h = alpha v^m (1-v)^p and v = x + 0.3 x^2 of the main lambda x, including the
// endpoints v = 0 and 1, where n or m below 2 used to lose b2 (one-sided
// differences there).
TEST_CASE("DLMDA-softcore-coefficients", "[ff][dlmda][L2]")
{
   // turns the lambda flags back off however the case ends
   TestLmdaFlagReset lmdaReset;
   use_d2lmda = true;

   const double alpha = 0.5, h = 1.0e-4;
   auto vof = [](double x) { return x + 0.3 * x * x; };
   auto tof = [](double s, int k) {
      double q = 1 / (1 + s);
      return k == 0 ? q * q : (k == 1 ? -2 * q * q * q : 6 * q * q * q * q);
   };
   const double x1 = (std::sqrt(1 + 1.2) - 1) / 0.6; // v(x1) = 1
   for (double n : {1.0, 2.0, 5.0})
      for (double m : {0.0, 1.0, 2.0})
         for (double p : {1.0, 2.0})
            for (double x : {0.0, 0.25, x1}) {
               auto e = [&](double y) {
                  double v = vof(y);
                  return std::pow(v, n) * tof(alpha * std::pow(v, m) * std::pow(1 - v, p), 0);
               };
               double d1, d2;
               if (x == 0) {
                  d1 = (-3 * e(0) + 4 * e(h) - e(2 * h)) / (2 * h);
                  d2 = (2 * e(0) - 5 * e(h) + 4 * e(2 * h) - e(3 * h)) / (h * h);
               } else if (x == x1) {
                  d1 = (3 * e(x) - 4 * e(x - h) + e(x - 2 * h)) / (2 * h);
                  d2 = (2 * e(x) - 5 * e(x - h) + 4 * e(x - 2 * h) - e(x - 3 * h)) / (h * h);
               } else {
                  d1 = (e(x + h) - e(x - h)) / (2 * h);
                  d2 = (e(x + h) - 2 * e(x) + e(x - h)) / (h * h);
               }
               const SoftCoreLmda s = softCoreLmda(vof(x), n, alpha, m, p, 1 + 0.6 * x, 0.6, true);
               const double t0 = tof(s.sc, 0), t1 = tof(s.sc, 1), t2 = tof(s.sc, 2);
               CAPTURE(n, m, p, x);
               COMPARE_REALS(s.p0 * t0, e(x), 1.0e-6);
               COMPARE_REALS(s.a1 * t0 + s.b1 * t1, d1, 1.0e-5 * std::max(1.0, std::fabs(d1)));
               COMPARE_REALS(s.a2 * t0 + s.b2 * t1 + s.c2 * t2, d2, 1.0e-4 * std::max(1.0, std::fabs(d2)));
            }
}

TEST_CASE("DLMDA-taper-matches-switch", "[ff][dlmda][L2]")
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

TEST_CASE("DLMDA-taper-degenerate", "[ff][dlmda][L2]")
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

// Two maps are the same only with the same type and the parameters that type
// reads; the fused single topology kernel relies on it (polTracksEle).
TEST_CASE("DLMDA-same-map", "[ff][dlmda][L2]")
{
   const LmdaMap exp2 = {Lmdamap::EXP, true, 2, 1, 0.1, 2, 1.5, 0.0, 1.0};
   LmdaMap m = exp2;
   REQUIRE(sameLmdaMap(m, exp2));
   m.exp = 3;
   REQUIRE_FALSE(sameLmdaMap(m, exp2));
   // parameters of the other types do not matter
   m = exp2;
   m.invn = 4;
   m.qnt1 = 0.5;
   REQUIRE(sameLmdaMap(m, exp2));
   m = exp2;
   m.type = Lmdamap::QNT;
   REQUIRE_FALSE(sameLmdaMap(m, exp2));
   LmdaMap q = m;
   REQUIRE(sameLmdaMap(m, q));
   q.qnt1 = 0.5;
   REQUIRE_FALSE(sameLmdaMap(m, q));
   m.type = q.type = Lmdamap::INV;
   q.inveps = 0.3;
   REQUIRE_FALSE(sameLmdaMap(m, q));
   m.type = q.type = Lmdamap::APM;
   q.inveps = m.inveps;
   q.apmrho = 1.8;
   REQUIRE_FALSE(sameLmdaMap(m, q));
}

TEST_CASE("DLMDA-dtneed", "[ff][dlmda][L2]")
{
   bool need0, need1;

   // An endpoint is dead only when it contributes neither weight nor either
   // lambda derivative (dlambda.f:relneed).
   dtNeed(lmdaChain(0.0, 0.0, 0.0, 0.0, 0.0), need0, need1);
   REQUIRE(need0);
   REQUIRE_FALSE(need1);

   dtNeed(lmdaChain(1.0, 0.0, 0.0, 0.0, 0.0), need0, need1);
   REQUIRE_FALSE(need0);
   REQUIRE(need1);

   dtNeed(lmdaChain(0.5, 0.0, 0.0, 0.0, 0.0), need0, need1);
   REQUIRE(need0);
   REQUIRE(need1);

   // A flat weight with a live first derivative keeps both endpoints, because
   // the mix still has to supply dE/dL.
   dtNeed(lmdaChain(0.0, 1.0, 0.0, 1.0, 0.0), need0, need1);
   REQUIRE(need0);
   REQUIRE(need1);
   dtNeed(lmdaChain(1.0, 1.0, 0.0, 1.0, 0.0), need0, need1);
   REQUIRE(need0);
   REQUIRE(need1);

   // A zero chain rule factor kills the derivative even when dw is nonzero:
   // this is the pinned sub-lambda that lets a term drop an endpoint.
   dtNeed(lmdaChain(0.0, 1.0, 0.0, 0.0, 0.0), need0, need1);
   REQUIRE(need0);
   REQUIRE_FALSE(need1);

   // Only the second derivative survives: c2 = d2w*chain^2 + dw*d2chain. It
   // keeps both endpoints when second derivatives are requested, and is ignored
   // otherwise (dlambda.f:relneed).
   const bool saved_d2 = use_d2lmda;
   use_d2lmda = true;
   dtNeed(lmdaChain(1.0, 0.0, 2.0, 1.0, 0.0), need0, need1);
   REQUIRE(need0);
   REQUIRE(need1);
   dtNeed(lmdaChain(1.0, 1.0, 0.0, 0.0, 3.0), need0, need1);
   REQUIRE(need0);
   REQUIRE(need1);
   use_d2lmda = false;
   dtNeed(lmdaChain(1.0, 0.0, 2.0, 1.0, 0.0), need0, need1);
   REQUIRE_FALSE(need0);
   REQUIRE(need1);
   dtNeed(lmdaChain(1.0, 1.0, 0.0, 0.0, 3.0), need0, need1);
   REQUIRE_FALSE(need0);
   REQUIRE(need1);
   use_d2lmda = saved_d2;

   // The weight zeroes a derivative whose coefficient vanishes, so a zero
   // sub-lambda never reaches a negative power (dlambda.f:lmdapwr).
   double w, dw, d2w;
   lmdaPwr(0.0, 1, true, w, dw, d2w);
   COMPARE_REALS(w, 0.0, 1.0e-14);
   COMPARE_REALS(dw, 1.0, 1.0e-14);
   COMPARE_REALS(d2w, 0.0, 1.0e-14);
   lmdaPwr(0.5, 3, true, w, dw, d2w);
   COMPARE_REALS(w, 0.125, 1.0e-14);
   COMPARE_REALS(dw, 0.75, 1.0e-14);
   COMPARE_REALS(d2w, 3.0, 1.0e-14);
}

// A quadratic dual topology weight at zero polarization lambda has curvature
// and no slope, so without second derivatives the coupled polarization
// endpoint and the ligand-ligand charge transfer pairs carry nothing and are
// left out; with them, both come back.
TEST_CASE("DLMDA-curvature-gate", "[ff][dlmda][L2]")
{
   const bool saved_d2 = use_d2lmda, saved_pdl = use_pdlmda, saved_epdt = use_epdt, saved_rel = use_rel;
   const double saved_plam = plam;
   const int saved_exp = epdtexp;
   const double saved_dpl = dpldlmda, saved_d2pl = d2pldlmda2;
   use_pdlmda = true;
   use_rel = false;
   plam = 0;
   dpldlmda = 1;
   d2pldlmda2 = 0;
   epdtexp = 2;

   DtPass pass[2];
   use_epdt = true;
   use_d2lmda = false;
   dtPasses(pass);
   REQUIRE(pass[0].need);
   REQUIRE_FALSE(pass[1].need);
   REQUIRE(dtPassIsIdle(calc::v7, pass[1], false));
   use_d2lmda = true;
   dtPasses(pass);
   REQUIRE(pass[1].need);
   COMPARE_REALS(pass[1].wc, 2.0, 1.0e-14);
   REQUIRE_FALSE(dtPassIsIdle(calc::v9, pass[1], false));

   // single topology charge transfer: a ligand-ligand pair goes as plam^2
   use_epdt = false;
   use_d2lmda = false;
   REQUIRE_FALSE(ctScale().live & (1u << (3 * 1 + 1)));
   use_d2lmda = true;
   const GrpPairScale cs = ctScale();
   REQUIRE(cs.live & (1u << (3 * 1 + 1)));
   COMPARE_REALS(cs.d2l[1][1], 2.0, 1.0e-14);

   use_d2lmda = saved_d2;
   use_pdlmda = saved_pdl;
   use_epdt = saved_epdt;
   use_rel = saved_rel;
   plam = saved_plam;
   epdtexp = saved_exp;
   dpldlmda = saved_dpl;
   d2pldlmda2 = saved_d2pl;
}

namespace {
// Puts back everything a staged leg walks on, so the schedule does not leak
// into later cases.
struct StagedScope
{
   bool rel, mainl;
   RelStage leg;
   bool stg[3];
   double stglo[3], stghi[3];
   LmdaMap em, pm, vm;
   double mainlmda;
   int dtexp;

   StagedScope()
      : rel(use_rel)
      , mainl(use_mainlmda)
      , leg(relstage)
      , em(elmdamap)
      , pm(plmdamap)
      , vm(vlmdamap)
      , mainlmda(lambda)
      , dtexp(epdtexp)
   {
      for (int i = 0; i < 3; ++i) {
         stg[i] = use_relstg[i];
         stglo[i] = relstglo[i];
         stghi[i] = relstghi[i];
      }
   }

   ~StagedScope()
   {
      use_rel = rel;
      use_mainlmda = mainl;
      relstage = leg;
      for (int i = 0; i < 3; ++i) {
         use_relstg[i] = stg[i];
         relstglo[i] = stglo[i];
         relstghi[i] = stghi[i];
      }
      elmdamap = em;
      plmdamap = pm;
      vlmdamap = vm;
      lambda = mainlmda;
      epdtexp = dtexp;
   }
};

// Declares the main lambda ranges of the LIG2, VDWM and LIG1 legs of the staged
// relative schedule, where a leg given a zero width is left undeclared
// (test_eostmap.f:test_eostmap_setstg). Every map is the quintic one, the only
// map on which legs that share a main lambda meet with a flat slope.
void setStages(double lo2, double hi2, double lov, double hiv, double lo1, double hi1)
{
   use_rel = true;
   // A staged schedule drives every sub-lambda off the main lambda (mutate.f).
   use_mainlmda = true;
   elmdamap.on = true;
   plmdamap.on = true;
   vlmdamap.on = true;
   // mutate.f floors this at 1; nothing has run dlmda_mech() here.
   epdtexp = 1;
   elmdamap.type = Lmdamap::QNT;
   plmdamap.type = Lmdamap::QNT;
   vlmdamap.type = Lmdamap::QNT;
   const double lo[3] = {lo2, lov, lo1};
   const double hi[3] = {hi2, hiv, hi1};
   for (int i = 0; i < 3; ++i) {
      relstglo[i] = lo[i];
      relstghi[i] = hi[i];
      use_relstg[i] = hi[i] > lo[i];
   }
}

// The three-simulation staged protocol: one run per leg, each declaring its
// leg alone over its range of the main lambda.
void setStagedLeg(RelStage leg)
{
   const double lo[3] = {0.0, 0.3, 0.7};
   const double hi[3] = {0.3, 0.7, 1.0};
   setStages(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
   const int k = static_cast<int>(leg);
   relstglo[k] = lo[k];
   relstghi[k] = hi[k];
   use_relstg[k] = true;
}
}

TEST_CASE("DLMDA-relstage-schedule", "[ff][dlmda][L2]")
{
   StagedScope scope;
   // Each leg run as its own simulation, declared alone over its range.

   // Ligand 2 is discharged as the main lambda rises; van der Waals sits on it.
   setStagedLeg(RelStage::LIG2);
   mapAt(0.0);
   COMPARE_REALS(elam, 1.0, 1.0e-7);
   COMPARE_REALS(vlam, 0.0, 1.0e-14);
   COMPARE_REALS(dvldlmda, 0.0, 1.0e-14);
   mapAt(0.15);
   COMPARE_REALS(elam, 0.5, 1.0e-6);
   REQUIRE(deldlmda < 0.0); // the coupled weight falls as lambda rises
   mapAt(0.3);
   COMPARE_REALS(elam, 0.0, 1.0e-7);
   COMPARE_REALS(deldlmda, 0.0, 1.0e-14);

   // Both ligands decoupled while van der Waals morphs from 2 onto 1.
   setStagedLeg(RelStage::VDWM);
   for (double lambda : {0.3, 0.5, 0.7}) {
      CAPTURE(lambda);
      mapAt(lambda);
      COMPARE_REALS(elam, 0.0, 1.0e-14);
      COMPARE_REALS(deldlmda, 0.0, 1.0e-14);
      COMPARE_REALS(d2eldlmda2, 0.0, 1.0e-14);
   }
   mapAt(0.3);
   COMPARE_REALS(vlam, 0.0, 1.0e-7);
   mapAt(0.5);
   COMPARE_REALS(vlam, 0.5, 1.0e-6);
   mapAt(0.7);
   COMPARE_REALS(vlam, 1.0, 1.0e-7);

   // Ligand 1 is charged as the main lambda rises, van der Waals already on it.
   setStagedLeg(RelStage::LIG1);
   mapAt(0.7);
   COMPARE_REALS(elam, 0.0, 1.0e-7);
   COMPARE_REALS(vlam, 1.0, 1.0e-14);
   COMPARE_REALS(dvldlmda, 0.0, 1.0e-14);
   mapAt(0.85);
   COMPARE_REALS(elam, 0.5, 1.0e-6);
   REQUIRE(deldlmda > 0.0); // the coupled weight grows with lambda
   mapAt(1.0);
   COMPARE_REALS(elam, 1.0, 1.0e-7);
   COMPARE_REALS(deldlmda, 0.0, 1.0e-14);
}

TEST_CASE("DLMDA-relstage-pinned-endpoints", "[ff][dlmda][L2]")
{
   StagedScope scope;
   // A leg pins the sub-lambdas it is not walking, and a pinned sub-lambda has
   // a flat chain rule. dtNeed() therefore drops the coupled polarization
   // endpoint on the morph leg. The multipoles and van der Waals are single
   // passes with no endpoints to skip.
   bool need0, need1;

   setStagedLeg(RelStage::VDWM);
   mapAt(0.5);
   dtNeed(lmdaPwrChain(plam, epdtexp, dpldlmda, d2pldlmda2, true), need0, need1);
   REQUIRE(need0);
   REQUIRE_FALSE(need1);
}

TEST_CASE("DLMDA-relstage-continuity", "[ff][dlmda][L2]")
{
   StagedScope scope;
   // Polarization stages with the multipoles exactly, on the leg and off it.
   for (RelStage leg : {RelStage::LIG2, RelStage::VDWM, RelStage::LIG1}) {
      setStagedLeg(leg);
      for (double lambda : {0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95}) {
         CAPTURE(lambda);
         mapAt(lambda);
         COMPARE_REALS(plam, elam, 1.0e-15);
         COMPARE_REALS(dpldlmda, deldlmda, 1.0e-15);
         COMPARE_REALS(d2pldlmda2, d2eldlmda2, 1.0e-15);
         REQUIRE(polTracksEle());
         // The map complement on the ligand 2 leg is clamped into range.
         REQUIRE(elam >= 0.0);
         REQUIRE(elam <= 1.0);
      }
   }

   // The three simulations hand over to each other continuously: the leg
   // boundaries agree in every sub-lambda, so a free energy assembled from the
   // three legs has no step (test_eostmap.f).
   auto sampleAt = [](RelStage leg, double lmda, double& e, double& de, double& v, double& dv) {
      setStagedLeg(leg);
      mapAt(lmda);
      e = elam;
      de = deldlmda;
      v = vlam;
      dv = dvldlmda;
   };
   double e0, de0, v0, dv0, e1, de1, v1, dv1;

   sampleAt(RelStage::LIG2, 0.3, e0, de0, v0, dv0);
   sampleAt(RelStage::VDWM, 0.3, e1, de1, v1, dv1);
   COMPARE_REALS(e0, e1, 1.0e-7);
   COMPARE_REALS(de0, de1, 1.0e-12);
   COMPARE_REALS(v0, v1, 1.0e-7);
   COMPARE_REALS(dv0, dv1, 1.0e-12);

   sampleAt(RelStage::VDWM, 0.7, e0, de0, v0, dv0);
   sampleAt(RelStage::LIG1, 0.7, e1, de1, v1, dv1);
   COMPARE_REALS(e0, e1, 1.0e-7);
   COMPARE_REALS(de0, de1, 1.0e-12);
   COMPARE_REALS(v0, v1, 1.0e-7);
   COMPARE_REALS(dv0, dv1, 1.0e-12);

   // Each leg's electrostatic weight is the quintic taper of its own range,
   // complemented on the ligand 2 leg where the ligand is being discharged.
   auto legMatchesTaper = [](RelStage leg, double lambda, double lo, double hi, double sign) {
      CAPTURE(lambda);
      setStagedLeg(leg);
      mapAt(lambda);
      double t, dt, d2t;
      quinticTaper(lambda, lo, hi, t, dt, d2t);
      // elam is real, so it holds the weight only to float round-off.
      COMPARE_REALS(elam, sign < 0.0 ? t : 1.0 - t, 1.0e-6);
      COMPARE_REALS(deldlmda, sign < 0.0 ? dt : -dt, 1.0e-14);
      COMPARE_REALS(d2eldlmda2, sign < 0.0 ? d2t : -d2t, 1.0e-14);
   };
   for (double lambda : {0.75, 0.85, 0.95})
      legMatchesTaper(RelStage::LIG1, lambda, 0.7, 1.0, 1.0);
   for (double lambda : {0.05, 0.15, 0.25})
      legMatchesTaper(RelStage::LIG2, lambda, 0.0, 0.3, -1.0);
}

TEST_CASE("DLMDA-relstage-active-leg", "[ff][dlmda][L2]")
{
   // All three legs declared, so one main lambda walks them in turn: ligand 2
   // discharging over the lower range, van der Waals morphing over the middle
   // one and ligand 1 charging over the upper one. A junction belongs to the
   // upper leg, at its flat end (test_eostmap.f).
   StagedScope scope;
   setStages(0.0, 0.3, 0.3, 0.7, 0.7, 1.0);

   mapAt(1.0);
   REQUIRE(relstage == RelStage::LIG1);
   COMPARE_REALS(elam, 1.0, 1.0e-7);
   COMPARE_REALS(vlam, 1.0, 1.0e-14);
   mapAt(0.85);
   REQUIRE(relstage == RelStage::LIG1);
   COMPARE_REALS(elam, 0.5, 1.0e-6);
   REQUIRE(deldlmda > 0.0);
   COMPARE_REALS(dvldlmda, 0.0, 1.0e-14);
   mapAt(0.7);
   REQUIRE(relstage == RelStage::LIG1);
   COMPARE_REALS(elam, 0.0, 1.0e-7);
   COMPARE_REALS(deldlmda, 0.0, 1.0e-14);
   COMPARE_REALS(vlam, 1.0, 1.0e-14);

   mapAt(0.5);
   REQUIRE(relstage == RelStage::VDWM);
   COMPARE_REALS(vlam, 0.5, 1.0e-6);
   REQUIRE(dvldlmda > 0.0);
   REQUIRE(elam == 0.0);
   REQUIRE(deldlmda == 0.0);
   mapAt(0.3);
   REQUIRE(relstage == RelStage::VDWM);
   COMPARE_REALS(vlam, 0.0, 1.0e-7);
   COMPARE_REALS(dvldlmda, 0.0, 1.0e-14);

   mapAt(0.15);
   REQUIRE(relstage == RelStage::LIG2);
   COMPARE_REALS(elam, 0.5, 1.0e-6);
   REQUIRE(deldlmda < 0.0);
   COMPARE_REALS(vlam, 0.0, 1.0e-14);
   mapAt(0.0);
   REQUIRE(relstage == RelStage::LIG2);
   COMPARE_REALS(elam, 1.0, 1.0e-7);
}

TEST_CASE("DLMDA-relstage-pol-carries-ele", "[ff][dlmda][L2]")
{
   // Polarization carries the electrostatic weight on every leg, and neither a
   // map of its own nor the sub-lambda windows of the unstaged maps can move a
   // staged leg (test_eostmap.f).
   StagedScope scope;
   setStages(0.0, 0.3, 0.3, 0.7, 0.7, 1.0);
   plmdamap.type = Lmdamap::EXP;
   elmdamap.qnt0 = 0.9;
   vlmdamap.qnt1 = 0.1;

   for (double lambda : {0.1, 0.5, 0.85}) {
      CAPTURE(lambda);
      mapAt(lambda);
      REQUIRE(plam == elam);
      REQUIRE(dpldlmda == deldlmda);
      REQUIRE(d2pldlmda2 == d2eldlmda2);
      REQUIRE(polTracksEle());
   }
   COMPARE_REALS(elam, 0.5, 1.0e-6);
   mapAt(0.5);
   COMPARE_REALS(vlam, 0.5, 1.0e-6);
}

TEST_CASE("DLMDA-relstage-leg-width", "[ff][dlmda][L2]")
{
   // Legs of unequal width carry the slope of their own local coordinate, here
   // twice the quintic slope of 15/8 on a leg of half width (test_eostmap.f).
   StagedScope scope;
   setStages(0.0, 0.0, 0.0, 0.5, 0.5, 1.0);

   mapAt(0.75);
   REQUIRE(relstage == RelStage::LIG1);
   COMPARE_REALS(elam, 0.5, 1.0e-6);
   COMPARE_REALS(deldlmda, 3.75, 1.0e-12);
   COMPARE_REALS(vlam, 1.0, 1.0e-14);

   mapAt(0.25);
   REQUIRE(relstage == RelStage::VDWM);
   COMPARE_REALS(vlam, 0.5, 1.0e-6);
   COMPARE_REALS(dvldlmda, 3.75, 1.0e-12);
   REQUIRE(elam == 0.0);
}

TEST_CASE("DLMDA-relstage-gap-and-ends", "[ff][dlmda][L2]")
{
   StagedScope scope;

   // A main lambda in a gap between legs holds the end state of the leg below,
   // with every slope an exact zero (test_eostmap.f).
   setStages(0.0, 0.2, 0.3, 0.7, 0.7, 1.0);
   mapAt(0.25);
   REQUIRE(relstage == RelStage::LIG2);
   REQUIRE(elam == 0.0);
   REQUIRE(vlam == 0.0);
   REQUIRE(deldlmda == 0.0);
   REQUIRE(d2eldlmda2 == 0.0);
   REQUIRE(dvldlmda == 0.0);

   // A main lambda below or above every declared leg holds the end state of
   // the nearest one.
   use_relstg[0] = false;
   mapAt(0.1);
   REQUIRE(relstage == RelStage::VDWM);
   REQUIRE(vlam == 0.0);
   REQUIRE(dvldlmda == 0.0);

   use_relstg[0] = true;
   use_relstg[1] = false;
   use_relstg[2] = false;
   mapAt(0.6);
   REQUIRE(relstage == RelStage::LIG2);
   REQUIRE(elam == 0.0);
   REQUIRE(deldlmda == 0.0);
}

namespace {
// The asymmetric power map is host math like the taper above, but mapOne() is
// static, so drive it through mapSubLambda() and read the shape back off the
// globals. This puts back everything that walks on.
struct ApmScope
{
   bool rel, mainl;
   LmdaMap em, pm, vm;
   double mainlmda;

   ApmScope()
      : rel(use_rel)
      , mainl(use_mainlmda)
      , em(elmdamap)
      , pm(plmdamap)
      , vm(vlmdamap)
      , mainlmda(lambda)
   {
      // One main lambda claiming all three sub-lambdas, each on the apm map.
      use_rel = false;
      use_mainlmda = true;
      elmdamap.on = true;
      plmdamap.on = true;
      vlmdamap.on = true;
      elmdamap.type = Lmdamap::APM;
      plmdamap.type = Lmdamap::APM;
      vlmdamap.type = Lmdamap::APM;
   }

   ~ApmScope()
   {
      use_rel = rel;
      use_mainlmda = mainl;
      elmdamap = em;
      plmdamap = pm;
      vlmdamap = vm;
      lambda = mainlmda;
   }
};

// The same power and slope ratio on all three sub-lambdas.
void setApm(int nn, double rho)
{
   elmdamap.apmn = nn;
   plmdamap.apmn = nn;
   vlmdamap.apmn = nn;
   elmdamap.apmrho = rho;
   plmdamap.apmrho = rho;
   vlmdamap.apmrho = rho;
}
}

// The values and chain rule derivatives of each map type at one main lambda,
// each sub-lambda on its own parameters so that a crossed argument would land
// on the wrong one (test_eostmap.f:test_eostmap_mapsub).
TEST_CASE("DLMDA-map-values", "[ff][dlmda][L2]")
{
   ApmScope scope;
   auto setType = [](Lmdamap t) { elmdamap.type = plmdamap.type = vlmdamap.type = t; };
   // the sub-lambdas carry float round-off; the derivatives are double
   auto check = [](double p, double dp, double d2p, double e, double de, double d2e, double v, double dv, double d2v) {
      COMPARE_REALS(plam, p, 1.0e-6);
      COMPARE_REALS(dpldlmda, dp, 1.0e-12);
      COMPARE_REALS(d2pldlmda2, d2p, 1.0e-12);
      COMPARE_REALS(elam, e, 1.0e-6);
      COMPARE_REALS(deldlmda, de, 1.0e-12);
      COMPARE_REALS(d2eldlmda2, d2e, 1.0e-12);
      COMPARE_REALS(vlam, v, 1.0e-6);
      COMPARE_REALS(dvldlmda, dv, 1.0e-12);
      COMPARE_REALS(d2vldlmda2, d2v, 1.0e-12);
   };

   setType(Lmdamap::EXP);
   plmdamap.exp = 2, elmdamap.exp = 3, vlmdamap.exp = 4;
   mapAt(0.25);
   check(0.0625, 0.5, 2.0, 0.015625, 0.1875, 1.5, 0.00390625, 0.0625, 0.75);

   setType(Lmdamap::INV);
   plmdamap.invn = 2, elmdamap.invn = 3, vlmdamap.invn = 4;
   plmdamap.inveps = 0.01, elmdamap.inveps = 0.02, vlmdamap.inveps = 0.03;
   mapAt(0.25);
   check(0.452936557937477, 1.08352945028593, -2.08371048131911, 0.509927040983045, 1.08536378202464,
      -2.67991057290036, 0.526434441030379, 1.09852311505228, -2.94247262960432);

   setType(Lmdamap::APM);
   plmdamap.apmn = 2, elmdamap.apmn = 3, vlmdamap.apmn = 4;
   plmdamap.apmrho = 1.5, elmdamap.apmrho = 2.0, vlmdamap.apmrho = 2.5;
   mapAt(0.25);
   check(0.3203125, 1.09375, -1.25, 0.3818359375, 1.140625, -2.4375, 0.43017578125, 1.134765625, -3.34375);

   // the quintic taper, each sub-lambda in its own window: one half at the
   // middle of all three, with different slopes
   setType(Lmdamap::QNT);
   plmdamap.qnt0 = 0.2, plmdamap.qnt1 = 0.8;
   elmdamap.qnt0 = 0.3, elmdamap.qnt1 = 0.7;
   vlmdamap.qnt0 = 0.1, vlmdamap.qnt1 = 0.9;
   mapAt(0.5);
   check(0.5, 3.125, 0.0, 0.5, 4.6875, 0.0, 0.5, 2.34375, 0.0);
   // off center, the complement of the taper and of its derivatives
   mapAt(0.35);
   COMPARE_REALS(plam, 1.0 - 0.896484375, 1.0e-6);
   COMPARE_REALS(dpldlmda, 1.7578125, 1.0e-12);
   COMPARE_REALS(d2pldlmda2, 15.625, 1.0e-12);
   // off below the window and on above it, edges included
   for (double l : {0.1, 0.2}) {
      mapAt(l);
      COMPARE_REALS(plam, 0.0, 1.0e-12);
      COMPARE_REALS(dpldlmda, 0.0, 1.0e-12);
   }
   for (double l : {0.8, 0.9}) {
      mapAt(l);
      COMPARE_REALS(plam, 1.0, 1.0e-12);
      COMPARE_REALS(dpldlmda, 0.0, 1.0e-12);
   }
}

// The endpoints of the power law, whose first and second derivatives survive
// at zero for exponents of one and two, and the inverse power map, the
// identity for an exponent of one or less and with a shift of 0.1 for a
// nonpositive one (test_eostmap.f:test_eostmap_sublmda).
TEST_CASE("DLMDA-power-endpoints", "[ff][dlmda][L2]")
{
   struct Case
   {
      double x, a, f, df, d2f;
   };
   const Case cases[] = {{0.0, 1.0, 0.0, 1.0, 0.0}, {0.0, 2.0, 0.0, 0.0, 2.0}, {0.0, 3.0, 0.0, 0.0, 0.0},
      {1.0, 3.0, 1.0, 3.0, 6.0}, {0.5, 1.0, 0.5, 1.0, 0.0}};
   for (const auto& c : cases) {
      double f, df, d2f;
      lmdaPwr(c.x, c.a, true, f, df, d2f);
      CAPTURE(c.x, c.a);
      COMPARE_REALS(f, c.f, 1.0e-12);
      COMPARE_REALS(df, c.df, 1.0e-12);
      COMPARE_REALS(d2f, c.d2f, 1.0e-12);
   }

   ApmScope scope;
   elmdamap.type = plmdamap.type = vlmdamap.type = Lmdamap::INV;
   for (int nn : {1, 0}) {
      plmdamap.invn = nn, plmdamap.inveps = 0.1;
      mapAt(0.4);
      COMPARE_REALS(plam, 0.4, 1.0e-6);
      COMPARE_REALS(dpldlmda, 1.0, 1.0e-12);
      COMPARE_REALS(d2pldlmda2, 0.0, 1.0e-12);
   }
   plmdamap.invn = 2, plmdamap.inveps = 0.1;
   mapAt(0.25);
   const double p0 = plam, dp0 = dpldlmda, d2p0 = d2pldlmda2;
   for (double eps : {0.0, -1.0}) {
      plmdamap.inveps = eps;
      mapAt(0.25);
      COMPARE_REALS(plam, p0, 1.0e-12);
      COMPARE_REALS(dpldlmda, dp0, 1.0e-12);
      COMPARE_REALS(d2pldlmda2, d2p0, 1.0e-12);
   }
}

TEST_CASE("DLMDA-apm-endpoints", "[ff][dlmda][L2]")
{
   // The map pins both endpoint values and carries the slope ratio at the
   // decoupled end against a unit slope at the coupled end, which is what the
   // normalization exists to enforce. The sub-lambdas are real, so they hold
   // their value only to float round-off; the derivatives are double.
   ApmScope scope;
   setApm(12, 4.0);

   mapAt(0.0);
   COMPARE_REALS(vlam, 0.0, 1.0e-6);
   COMPARE_REALS(dvldlmda, 4.0, 1.0e-12);
   COMPARE_REALS(d2vldlmda2, -39.272727272727273, 1.0e-12);

   mapAt(1.0);
   COMPARE_REALS(vlam, 1.0, 1.0e-6);
   COMPARE_REALS(dvldlmda, 1.0, 1.0e-12);
   COMPARE_REALS(d2vldlmda2, 3.272727272727273, 1.0e-12);

   // The interior runs ahead of the linear schedule, its slope having already
   // fallen below one by the midpoint.
   mapAt(0.5);
   COMPARE_REALS(vlam, 0.6153564453125, 1.0e-6);
   COMPARE_REALS(dvldlmda, 0.728138316761364, 1.0e-12);
   COMPARE_REALS(d2vldlmda2, -0.017578125, 1.0e-12);
}

TEST_CASE("DLMDA-apm-exact-ends", "[ff][dlmda][L2]")
{
   // The end states map exactly onto themselves: at a power of two and a slope
   // ratio of 1.0325 the normalized sum rounds an ulp past one, where a soft
   // core power of one minus the sub-lambda is undefined (dlambda.f:sublmdaapm).
   ApmScope scope;
   setApm(2, 1.0325);
   mapAt(1.0);
   REQUIRE(vlam == 1);
   REQUIRE(elam == 1);
   REQUIRE(plam == 1);
   mapAt(0.0);
   REQUIRE(vlam == 0);
   REQUIRE(elam == 0);
   REQUIRE(plam == 0);
}

TEST_CASE("DLMDA-apm-degenerate", "[ff][dlmda][L2]")
{
   // A power below two or a slope ratio of one or less leaves no asymmetry to
   // impose, and a slope ratio at or above the power sits on the pole of the
   // normalization. All three fall back to the identity map.
   ApmScope scope;

   for (auto shape : {std::make_pair(1, 4.0), std::make_pair(12, 1.0), std::make_pair(4, 4.0)}) {
      CAPTURE(shape.first);
      CAPTURE(shape.second);
      setApm(shape.first, shape.second);
      mapAt(0.4);
      COMPARE_REALS(vlam, 0.4, 1.0e-6);
      COMPARE_REALS(dvldlmda, 1.0, 1.0e-12);
      COMPARE_REALS(d2vldlmda2, 0.0, 1.0e-12);
   }
}

TEST_CASE("DLMDA-apm-per-term-shape", "[ff][dlmda][L2]")
{
   // Three terms on three different powers and slope ratios, so a crossed
   // argument at a mapOne() call site would land the wrong shape on the wrong
   // sub-lambda instead of going unnoticed.
   ApmScope scope;
   plmdamap.apmn = 2;
   elmdamap.apmn = 3;
   vlmdamap.apmn = 4;
   plmdamap.apmrho = 1.5;
   elmdamap.apmrho = 2.0;
   vlmdamap.apmrho = 2.5;

   mapAt(0.25);
   COMPARE_REALS(plam, 0.3203125, 1.0e-6);
   COMPARE_REALS(dpldlmda, 1.09375, 1.0e-12);
   COMPARE_REALS(d2pldlmda2, -1.25, 1.0e-12);
   COMPARE_REALS(elam, 0.3818359375, 1.0e-6);
   COMPARE_REALS(deldlmda, 1.140625, 1.0e-12);
   COMPARE_REALS(d2eldlmda2, -2.4375, 1.0e-12);
   COMPARE_REALS(vlam, 0.43017578125, 1.0e-6);
   COMPARE_REALS(dvldlmda, 1.134765625, 1.0e-12);
   COMPARE_REALS(d2vldlmda2, -3.34375, 1.0e-12);
}

TEST_CASE("DLMDA-apm-pol-tracks-ele", "[ff][dlmda][L2]")
{
   // Polarization rides the multipole weight only while both maps carry the
   // same shape; either half of the shape parting company parts the weights.
   ApmScope scope;
   setApm(12, 4.0);
   REQUIRE(polTracksEle());

   plmdamap.apmn = 8;
   REQUIRE_FALSE(polTracksEle());

   plmdamap.apmn = 12;
   plmdamap.apmrho = 2.0;
   REQUIRE_FALSE(polTracksEle());
}

namespace {
// The theta map of the lambda particle is host math on process globals, and
// Catch2 randomizes case order, so every case that drives the particle puts
// the whole particle state back (test_eostmap.f:test_eostmap_theta).
struct ThetaScope
{
   LmdaThMap map;
   double alpha, theta, vtheta, mass, fric, dt, eff, mainlmda, kelvin, lkelvin;

   ThetaScope()
      : map(lmdathmap)
      , alpha(lmdathalpha)
      , theta(lmdatheta)
      , vtheta(lmdavtheta)
      , mass(lmdamass)
      , fric(lmdafric)
      , dt(lmdadt)
      , eff(deffdl)
      , mainlmda(lambda)
      , kelvin(bath::kelvin)
      , lkelvin(lmdakelvin)
   {
      // the defaults mutate.f sets, which every case below overrides as it needs
      lmdathmap = LmdaThMap::TRI;
      lmdathalpha = 0.999999999;
   }

   ~ThetaScope()
   {
      lmdathmap = map;
      lmdathalpha = alpha;
      lmdatheta = theta;
      lmdavtheta = vtheta;
      lmdamass = mass;
      lmdafric = fric;
      lmdadt = dt;
      deffdl = eff;
      lambda = mainlmda;
      bath::kelvin = kelvin;
      lmdakelvin = lkelvin;
   }
};

// The same for the Fortran module state, which the differential case below
// drives directly.
struct FortranThetaScope
{
   char map[3];
   double alpha, theta, vtheta, mass, fric, dt, eff, mainlmda, lkelvin;

   FortranThetaScope()
      : alpha(dlmda::lmdathalpha)
      , theta(dlmda::lmdatheta)
      , vtheta(dlmda::lmdavtheta)
      , mass(dlmda::lmdamass)
      , fric(dlmda::lmdafric)
      , dt(dlmda::lmdadt)
      , eff(dlmda::deffdl)
      , mainlmda(mutant::lambda)
      , lkelvin(dlmda::lmdakelvin)
   {
      std::memcpy(map, dlmda::lmdathmap, 3);
   }

   ~FortranThetaScope()
   {
      std::memcpy(dlmda::lmdathmap, map, 3);
      dlmda::lmdathalpha = alpha;
      dlmda::lmdatheta = theta;
      dlmda::lmdavtheta = vtheta;
      dlmda::lmdamass = mass;
      dlmda::lmdafric = fric;
      dlmda::lmdadt = dt;
      dlmda::deffdl = eff;
      mutant::lambda = mainlmda;
      dlmda::lmdakelvin = lkelvin;
   }
};

// tinker::pi is only a float in a mixed precision build, and these cases hold
// the map to 1e-14, so the period has to come from M_PI.
constexpr double dpi = M_PI;

double mapLmda(double theta)
{
   double l, d;
   lmdaThetaMap(theta, l, d);
   return l;
}

double mapDldth(double theta)
{
   double l, d;
   lmdaThetaMap(theta, l, d);
   return d;
}

// Roundoff in the map at theta. At a triangle extremum dlambda/dcos(2*theta)
// is about 7e3 at the sharpness selectMap uses, so one ulp of 2*theta or of
// cos moves lambda by ~1e-12, and nvc++ (-fast in particular) rounds sin and
// cos differently from glibc and gfortran.
double mapNoise(double theta)
{
   double eps = std::numeric_limits<double>::epsilon() * (1.0 + std::fabs(2.0 * theta));
   if (lmdathmap != LmdaThMap::TRI)
      return eps;
   double y = lmdathalpha * std::cos(2.0 * theta);
   return eps * (1.0 + 1.0 / (2.0 * std::asin(lmdathalpha) * std::sqrt(1.0 - y * y)));
}

// The two maps as the Fortran cases drive them, sine squared at a sharpness it
// ignores and the smoothed triangle at one close to a flat traversal.
void selectMap(int j)
{
   lmdathmap = (j == 0) ? LmdaThMap::SIN : LmdaThMap::TRI;
   lmdathalpha = (j == 0) ? 0.9 : 0.999999999;
}
}

TEST_CASE("DLMDA-theta-map-sin", "[ff][dlmda][L2]")
{
   // the sine squared map is lambda = sin(theta)^2
   ThetaScope scope;
   lmdathmap = LmdaThMap::SIN;

   COMPARE_REALS(mapLmda(0.3), std::sin(0.3) * std::sin(0.3), 1.0e-15);
   COMPARE_REALS(mapDldth(0.3), std::sin(0.6), 1.0e-15);
}

TEST_CASE("DLMDA-theta-map-tri", "[ff][dlmda][L2]")
{
   // the smoothed triangle reaches both endpoints and the midpoint, and it
   // comes into the lambda endpoints flat, which is the point of the map
   ThetaScope scope;
   lmdathmap = LmdaThMap::TRI;
   lmdathalpha = 0.999999999;

   COMPARE_REALS(mapLmda(0.0), 0.0, 1.0e-12);
   COMPARE_REALS(mapDldth(0.0), 0.0, 1.0e-12);
   COMPARE_REALS(mapLmda(0.5 * dpi), 1.0, 1.0e-12);
   COMPARE_REALS(mapLmda(-0.5 * dpi), 1.0, 1.0e-12);
   COMPARE_REALS(mapLmda(-0.25 * dpi), 0.5, 1.0e-12);
}

TEST_CASE("DLMDA-theta-map-periodic", "[ff][dlmda][L2]")
{
   // both maps close on themselves over a period of pi in theta, which is what
   // lets lmdaLangevin() wrap theta into [0,pi) without moving lambda
   ThetaScope scope;

   for (int j = 0; j < 2; ++j) {
      selectMap(j);
      for (int i = 0; i <= 12; ++i) {
         double theta = -dpi + dpi * (double)i / 12.0;
         CAPTURE(j, theta);
         COMPARE_REALS(mapLmda(theta + dpi), mapLmda(theta), 1.0e-12);
         COMPARE_REALS(mapDldth(theta + dpi), mapDldth(theta), 1.0e-9);
      }
   }
}

TEST_CASE("DLMDA-theta-map-small-alpha", "[ff][dlmda][L2]")
{
   // a vanishing sharpness recovers the sine squared map
   ThetaScope scope;

   for (int i = 0; i <= 12; ++i) {
      double theta = -dpi + 2.0 * dpi * (double)i / 12.0;
      CAPTURE(theta);

      lmdathmap = LmdaThMap::TRI;
      lmdathalpha = 1.0e-6;
      double lmda = mapLmda(theta), dldth = mapDldth(theta);

      lmdathmap = LmdaThMap::SIN;
      COMPARE_REALS(lmda, mapLmda(theta), 1.0e-8);
      COMPARE_REALS(dldth, mapDldth(theta), 1.0e-8);
   }
}

TEST_CASE("DLMDA-theta-map-derivative", "[ff][dlmda][L2]")
{
   // theta derivatives match central differences over a full period
   ThetaScope scope;
   const double h = 1.0e-6;

   for (int j = 0; j < 2; ++j) {
      selectMap(j);
      for (int i = 0; i <= 40; ++i) {
         double theta = -dpi + 2.0 * dpi * (double)i / 40.0;
         double dnum = (mapLmda(theta + h) - mapLmda(theta - h)) / (2.0 * h);
         CAPTURE(j, theta);
         COMPARE_REALS(mapDldth(theta), dnum, 1.0e-8 + 2.0 * mapNoise(theta) / h);
      }
   }
}

TEST_CASE("DLMDA-theta-inverse", "[ff][dlmda][L2]")
{
   // the inverse maps round trip lambda through theta, on the principal branch
   ThetaScope scope;
   const double llist[6] = {0.0, 0.001, 0.25, 0.5, 0.9, 1.0};

   for (int j = 0; j < 2; ++j) {
      selectMap(j);
      lmdathalpha = 0.999999999;
      for (int i = 0; i < 6; ++i) {
         double theta;
         lmdaThetaInv(llist[i], theta);
         CAPTURE(j, llist[i], theta);
         COMPARE_REALS(mapLmda(theta), llist[i], 1.0e-12);
         REQUIRE(theta >= 0.0);
         REQUIRE(theta <= 0.5 * dpi);
      }
   }
}

TEST_CASE("DLMDA-theta-langevin-tri", "[ff][dlmda][L2]")
{
   // a frictionless step with the smoothed triangle map takes its theta force
   // from the map derivative, not from sin(2*theta)
   ThetaScope scope;
   const double alpha = 0.999999999;
   lmdathmap = LmdaThMap::TRI;
   lmdathalpha = alpha;
   const double asina = std::asin(alpha);

   lmdadt = 0.1;
   lmdamass = 2.0;
   lmdafric = 0.0;
   deffdl = 1.5;
   const double th0 = 0.7, v0 = 0.3;
   lmdatheta = th0;
   lmdavtheta = v0;
   lmdaLangevin();

   double dldth =
      alpha * std::sin(2.0 * th0) / (asina * std::sqrt(1.0 - std::pow(alpha * std::cos(2.0 * th0), 2)));
   double vref = v0 - lmdadt * deffdl * dldth / lmdamass;
   double thref = th0 + lmdadt * vref;
   COMPARE_REALS(lmdavtheta, vref, 1.0e-14);
   COMPARE_REALS(lmdatheta, thref, 1.0e-14);
   COMPARE_REALS(lambda, 0.5 - std::asin(alpha * std::cos(2.0 * thref)) / (2.0 * asina), 1.0e-14);
}

TEST_CASE("DLMDA-theta-langevin-friction", "[ff][dlmda][L2]")
{
   // a step with friction and a given random kick; the kick is sized by the
   // thermal energy in kcal/mole, as is the force
   ThetaScope scope;
   const double alpha = 0.999999999;
   lmdathmap = LmdaThMap::TRI;
   lmdathalpha = alpha;
   const double asina = std::asin(alpha);

   bath::kelvin = 300.0;
   lmdakelvin = 0.0;
   lmdadt = 0.1;
   lmdamass = 2.0;
   lmdafric = 0.5;
   deffdl = 1.5;
   const double th0 = 0.7, v0 = 0.3, gauss = 0.8;
   lmdatheta = th0;
   lmdavtheta = v0;
   lmdaLangStep(gauss);

   double dldth =
      alpha * std::sin(2.0 * th0) / (asina * std::sqrt(1.0 - std::pow(alpha * std::cos(2.0 * th0), 2)));
   double decay = std::exp(-lmdafric * lmdadt);
   double sigma = std::sqrt(units::gasconst * bath::kelvin * (1.0 - decay * decay) / lmdamass);
   double vkick = v0 - lmdadt * deffdl * dldth / lmdamass;
   double vref = decay * vkick + sigma * gauss;
   double thref = th0 + 0.5 * lmdadt * vkick + 0.5 * lmdadt * vref;
   COMPARE_REALS(lmdavtheta, vref, 1.0e-14);
   COMPARE_REALS(lmdatheta, thref, 1.0e-14);

   // a unit kick from rest with no force leaves the equipartition value, which
   // is gasconst times 300 K in kcal/mole
   deffdl = 0.0;
   lmdatheta = th0;
   lmdavtheta = 0.0;
   lmdaLangStep(1.0);
   COMPARE_REALS(lmdamass * lmdavtheta * lmdavtheta / (1.0 - decay * decay), 0.59616127758, 1.0e-12);

   // LAMBDA-TEMPERATURE sizes the kick by its own thermal energy, three times
   // the bath value at 900 K
   lmdakelvin = 900.0;
   lmdatheta = th0;
   lmdavtheta = 0.0;
   lmdaLangStep(1.0);
   COMPARE_REALS(lmdamass * lmdavtheta * lmdavtheta / (1.0 - decay * decay), 3.0 * 0.59616127758, 1.0e-12);
}

TEST_CASE("DLMDA-theta-langevin-sampling", "[ff][dlmda][L2]")
{
   // A step with friction samples a harmonic well in theta at the bath
   // temperature, or at its own once LAMBDA-TEMPERATURE sets one, for small and
   // large friction times the time step. The step is linear in the well, so the
   // stationary covariance of theta and its velocity follows exactly from the
   // step matrix (test_eostmap.f:test_eostmap_lfmid).
   ThetaScope scope;

   // a well centered where the sine squared map has unit slope and as stiff as
   // omega times the time step equal to one
   bath::kelvin = 300.0;
   lmdathmap = LmdaThMap::SIN;
   lmdadt = 0.01;
   lmdamass = 2.0;
   const double kspr = lmdamass / (lmdadt * lmdadt);
   const double th0 = 0.25 * dpi, h = 0.01;

   // one step from theta offset d and velocity v with kick g, the effective
   // lambda derivative giving a force -kspr*d on theta
   auto step = [&](double d, double v, double g, double& dout, double& vout) {
      double lmda, dldth;
      lmdatheta = th0 + d;
      lmdavtheta = v;
      lmdaThetaMap(lmdatheta, lmda, dldth);
      deffdl = kspr * d / dldth;
      lmdaLangStep(g);
      dout = lmdatheta - th0;
      vout = lmdavtheta;
   };

   for (double lk : {0.0, 900.0}) {
      for (double gdt : {0.5, 5.0, 20.0}) {
         lmdakelvin = lk;
         const double kt = units::gasconst * (lk > 0.0 ? lk : bath::kelvin);
         lmdafric = gdt / lmdadt;

         // unit theta offset, velocity and kick give the step matrix and the
         // kick vector
         double q1, q2;
         step(h, 0.0, 0.0, q1, q2);
         const double a11 = q1 / h, a21 = q2 / h;
         step(0.0, h, 0.0, q1, q2);
         const double a12 = q1 / h, a22 = q2 / h;
         step(0.0, 0.0, h, q1, q2);
         const double b1 = q1 / h, b2 = q2 / h;

         // iterate the covariance under unit normal kicks to its stationary value
         double pdd = 0.0, pdv = 0.0, pvv = 0.0;
         for (int i = 0; i < 2000; ++i) {
            double cdd = a11 * a11 * pdd + 2.0 * a11 * a12 * pdv + a12 * a12 * pvv + b1 * b1;
            double cdv = a11 * a21 * pdd + (a11 * a22 + a12 * a21) * pdv + a12 * a22 * pvv + b1 * b2;
            double cvv = a21 * a21 * pdd + 2.0 * a21 * a22 * pdv + a22 * a22 * pvv + b2 * b2;
            pdd = cdd;
            pdv = cdv;
            pvv = cvv;
         }
         CAPTURE(lk, gdt);
         COMPARE_REALS(kspr * pdd / kt, 1.0, 1.0e-8);
         COMPARE_REALS(lmdamass * pvv / kt, 1.0, 1.0e-8);
      }
   }
}

TEST_CASE("DLMDA-theta-langevin-wrap", "[ff][dlmda][L2]")
{
   // a step past the end of the period wraps theta into [0,pi), and the period
   // of the map leaves lambda where the unwrapped theta would have put it
   ThetaScope scope;

   for (int j = 0; j < 2; ++j) {
      selectMap(j);
      lmdathalpha = 0.9;
      deffdl = 0.0;
      lmdafric = 0.0;
      lmdadt = 0.1;
      lmdamass = 2.0;
      const double th0 = 2.0, v0 = 30.0;
      lmdatheta = th0;
      lmdavtheta = v0;
      lmdaLangevin();

      double thref = th0 + lmdadt * v0;
      CAPTURE(j);
      COMPARE_REALS(lmdatheta, thref - dpi, 1.0e-14);
      COMPARE_REALS(lambda, mapLmda(thref), 1.0e-12);

      // many steps of a fast particle stay inside the period
      for (int i = 1; i <= 200; ++i) {
         lmdaLangevin();
         CAPTURE(i, lmdatheta);
         REQUIRE(lmdatheta >= 0.0);
         REQUIRE(lmdatheta < dpi);
      }
   }
}

TEST_CASE("DLMDA-theta-matches-fortran", "[ff][dlmda][L2]")
{
   // The whole point of the port is that a tinker9 run and a Tinker run reading
   // the same key file propagate the same lambda particle. Both engines are
   // linked into this binary, so hold the C++ routines against dlambda.f
   // directly rather than against transcribed numbers.
   ThetaScope scope;
   FortranThetaScope fscope;

   for (int j = 0; j < 2; ++j) {
      selectMap(j);
      std::memcpy(dlmda::lmdathmap, (j == 0) ? "SIN" : "TRI", 3);
      dlmda::lmdathalpha = lmdathalpha;

      // the forward map and its theta derivative, over a full period
      for (int i = 0; i <= 24; ++i) {
         double theta = -dpi + 2.0 * dpi * (double)i / 24.0;
         double l9, d9, lf, df;
         lmdaThetaMap(theta, l9, d9);
         tinker_f_lmdathetamap(&theta, &lf, &df);
         CAPTURE(j, theta);
         COMPARE_REALS(l9, lf, 1.0e-14 + 4.0 * mapNoise(theta));
         COMPARE_REALS(d9, df, 1.0e-14 + 4.0 * mapNoise(theta));
      }

      // the inverse, including the clipped endpoints
      for (double lval : {-0.5, 0.0, 0.001, 0.25, 0.5, 0.9, 1.0, 1.5}) {
         double t9, tf, lin = lval;
         lmdaThetaInv(lval, t9);
         tinker_f_lmdathetainv(&lin, &tf);
         CAPTURE(j, lval);
         COMPARE_REALS(t9, tf, 1.0e-14);
      }

      // and a frictionless trajectory, long enough to wrap the period twice
      lmdadt = 0.1;
      lmdamass = 2.0;
      lmdafric = 0.0;
      deffdl = 1.5;
      lmdatheta = 0.7;
      lmdavtheta = 3.0;
      dlmda::lmdadt = lmdadt;
      dlmda::lmdamass = lmdamass;
      dlmda::lmdafric = lmdafric;
      dlmda::deffdl = deffdl;
      dlmda::lmdatheta = lmdatheta;
      dlmda::lmdavtheta = lmdavtheta;
      for (int i = 1; i <= 50; ++i) {
         lmdaLangevin();
         tinker_f_lmdalangevin();
         CAPTURE(j, i);
         COMPARE_REALS(lmdatheta, dlmda::lmdatheta, 1.0e-13);
         COMPARE_REALS(lmdavtheta, dlmda::lmdavtheta, 1.0e-13);
         COMPARE_REALS(lambda, mutant::lambda, 1.0e-13);
      }

      // the lambda particle temperature follows the bath unless the keyword
      // sets its own
      bath::kelvin = 300.0;
      lmdakelvin = 0.0;
      dlmda::lmdakelvin = 0.0;
      REQUIRE(lmdaTemp() == tinker_f_lmdatemp());
      REQUIRE(lmdaTemp() == 300.0);
      lmdakelvin = 900.0;
      dlmda::lmdakelvin = 900.0;
      REQUIRE(lmdaTemp() == tinker_f_lmdatemp());
      REQUIRE(lmdaTemp() == 900.0);

      // and a trajectory with friction at that temperature, each engine given
      // the same random kicks
      lmdafric = 0.5;
      lmdatheta = 0.7;
      lmdavtheta = 3.0;
      dlmda::lmdafric = lmdafric;
      dlmda::lmdatheta = lmdatheta;
      dlmda::lmdavtheta = lmdavtheta;
      for (int i = 1; i <= 50; ++i) {
         double gauss = 2.0 * std::sin(1.7 * (double)i);
         lmdaLangStep(gauss);
         tinker_f_lmdalangstep(&gauss);
         CAPTURE(j, i);
         COMPARE_REALS(lmdatheta, dlmda::lmdatheta, 1.0e-13);
         COMPARE_REALS(lmdavtheta, dlmda::lmdavtheta, 1.0e-13);
         COMPARE_REALS(lambda, mutant::lambda, 1.0e-13);
      }
   }
}

namespace {
// Saves the interval convergence gate on both the C++ and the Fortran side and
// restores it on exit.
struct ConvGateScope
{
   bool cv = use_lmdacv;
   double cvstd = lmdacvstd, cvrat = lmdacvrat;
   int fcv = dlmda::use_lmdacv;
   double fcvstd = dlmda::lmdacvstd, fcvrat = dlmda::lmdacvrat;

   ~ConvGateScope()
   {
      use_lmdacv = cv;
      lmdacvstd = cvstd;
      lmdacvrat = cvrat;
      dlmda::use_lmdacv = fcv;
      dlmda::lmdacvstd = fcvstd;
      dlmda::lmdacvrat = fcvrat;
   }
};
}

TEST_CASE("DLMDA-depcriteria-matches-fortran", "[ff][dlmda][L2]")
{
   // tinker9 and Tinker must accept or reject the same interval samples, so hold
   // the C++ convergence gate against dlambda.f:depcriteria with the gate both
   // off and on, over tolerances with absolute and relative parts.
   ConvGateScope scope;

   const double tols[][2] = {{50.0, 0.5}, {10.0, 0.2}, {1.0, 0.0}, {0.0, 0.1}, {0.0, 0.0}};
   const double avgs[] = {-60.0, -1.0, 0.0, 0.5, 3.0, 100.0};
   const double stds[] = {0.0, 0.3, 1.0, 9.99, 10.0, 25.0, 1.0e6};
   for (int on = 0; on < 2; ++on) {
      for (const auto& tol : tols) {
         use_lmdacv = (on != 0);
         lmdacvstd = tol[0];
         lmdacvrat = tol[1];
         dlmda::use_lmdacv = on;
         dlmda::lmdacvstd = tol[0];
         dlmda::lmdacvrat = tol[1];
         for (double avg : avgs) {
            for (double sd : stds) {
               double a = avg, s = sd;
               CAPTURE(on, tol[0], tol[1], avg, sd);
               REQUIRE(depcriteria(avg, sd) == (tinker_f_depcriteria(&a, &s) != 0));
            }
         }
      }
   }
}
