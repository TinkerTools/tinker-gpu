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

TEST_CASE("DLMDA-relslot", "[ff][dlmda]")
{
   // The subsystem table of mutate.f:relslot, and the three coupling states
   // built out of it.
   static const RdtMask kMask[nRelSlot] = {RdtMask::AE, RdtMask::BE, RdtMask::ENV, RdtMask::LIGA, RdtMask::LIGB};
   for (int k = 0; k < nRelSlot; ++k) {
      CAPTURE(k);
      RdtMask mask;
      bool in0, in1;
      relSlot(k, RelState::LIG1, RelState::LIG2, mask, in0, in1);
      REQUIRE(mask == kMask[k]);
      // LIG1 = slots 0 and 4, LIG2 = slots 1 and 3.
      REQUIRE(in0 == (k == 0 or k == 4));
      REQUIRE(in1 == (k == 1 or k == 3));

      relSlot(k, RelState::NONE, RelState::NONE, mask, in0, in1);
      // NONE = slots 2, 3 and 4.
      REQUIRE(in0 == (k == 2 or k == 3 or k == 4));
      REQUIRE(in1 == in0);
   }

   // Only a ligand bound to the environment carries the reported count.
   REQUIRE(relSlotIsCoupled(0));
   REQUIRE(relSlotIsCoupled(1));
   for (int k = 2; k < nRelSlot; ++k)
      REQUIRE_FALSE(relSlotIsCoupled(k));

   // Each state is disjoint from the other coupled one but shares a lone
   // ligand with the decoupled reference; that overlap is what the relative
   // driver hoists past the mix.
   auto slotsOf = [](RelState st) {
      int bits = 0;
      for (int k = 0; k < nRelSlot; ++k) {
         RdtMask mask;
         bool in0, in1;
         relSlot(k, st, st, mask, in0, in1);
         if (in0)
            bits |= 1 << k;
      }
      return bits;
   };
   REQUIRE((slotsOf(RelState::LIG1) & slotsOf(RelState::LIG2)) == 0);
   REQUIRE((slotsOf(RelState::LIG1) & slotsOf(RelState::NONE)) == (1 << 4));
   REQUIRE((slotsOf(RelState::LIG2) & slotsOf(RelState::NONE)) == (1 << 3));
}

TEST_CASE("DLMDA-dtneed", "[ff][dlmda]")
{
   bool need0, need1;

   // An endpoint is dead only when it contributes neither weight nor either
   // lambda derivative (dlambda.f:relneed).
   dtNeed(0.0, 0.0, 0.0, 0.0, 0.0, need0, need1);
   REQUIRE(need0);
   REQUIRE_FALSE(need1);

   dtNeed(1.0, 0.0, 0.0, 0.0, 0.0, need0, need1);
   REQUIRE_FALSE(need0);
   REQUIRE(need1);

   dtNeed(0.5, 0.0, 0.0, 0.0, 0.0, need0, need1);
   REQUIRE(need0);
   REQUIRE(need1);

   // A flat weight with a live first derivative keeps both endpoints, because
   // the mix still has to supply dE/dL.
   dtNeed(0.0, 1.0, 0.0, 1.0, 0.0, need0, need1);
   REQUIRE(need0);
   REQUIRE(need1);
   dtNeed(1.0, 1.0, 0.0, 1.0, 0.0, need0, need1);
   REQUIRE(need0);
   REQUIRE(need1);

   // A zero chain rule factor kills the derivative even when dw is nonzero:
   // this is the pinned sub-lambda that lets a term drop an endpoint.
   dtNeed(0.0, 1.0, 0.0, 0.0, 0.0, need0, need1);
   REQUIRE(need0);
   REQUIRE_FALSE(need1);

   // Only the second derivative survives: c2 = d2w*chain^2 + dw*d2chain.
   dtNeed(1.0, 0.0, 2.0, 1.0, 0.0, need0, need1);
   REQUIRE(need0);
   REQUIRE(need1);
   dtNeed(1.0, 1.0, 0.0, 0.0, 3.0, need0, need1);
   REQUIRE(need0);
   REQUIRE(need1);

   // dtWeight takes the linear case separately so a zero sub-lambda never
   // reaches a zero power.
   double w, dw, d2w;
   dtWeight(0.0, 1, w, dw, d2w);
   COMPARE_REALS(w, 0.0, 1.0e-14);
   COMPARE_REALS(dw, 1.0, 1.0e-14);
   COMPARE_REALS(d2w, 0.0, 1.0e-14);
   dtWeight(0.5, 3, w, dw, d2w);
   COMPARE_REALS(w, 0.125, 1.0e-14);
   COMPARE_REALS(dw, 0.75, 1.0e-14);
   COMPARE_REALS(d2w, 3.0, 1.0e-14);
}

namespace {
// The reference three-simulation staged protocol: one run per leg, each
// declaring its leg and walking its own window over the full main lambda.
void setStagedLeg(RelStage leg)
{
   use_relstage = true;
   relstage = leg;
   // mutate.f floors these at 1; nothing has run dlmda_mech() here.
   emdtexp = 1;
   epdtexp = 1;
   evdtexp = 1;
   elmdamap = Lmdamap::QNT;
   vlmdamap = Lmdamap::QNT;
   qntelmda0 = (leg == RelStage::LIG1) ? 0.7 : 0.0;
   qntelmda1 = (leg == RelStage::LIG1) ? 1.0 : 0.3;
   qntvlmda0 = 0.3;
   qntvlmda1 = 0.7;
}
}

TEST_CASE("DLMDA-relstage-schedule", "[ff][dlmda]")
{
   // Each leg is its own simulation now, so the leg is declared rather than
   // derived from where the main lambda happens to sit.

   // Ligand 2 is discharged as the main lambda rises; van der Waals sits on it.
   setStagedLeg(RelStage::LIG2);
   mapAt(0.0);
   REQUIRE(emrelst0 == RelState::NONE);
   REQUIRE(emrelst1 == RelState::LIG2);
   REQUIRE(evrelst0 == RelState::LIG2);
   REQUIRE(evrelst1 == RelState::LIG1);
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
      REQUIRE(emrelst0 == RelState::NONE);
      REQUIRE(emrelst1 == RelState::NONE);
      REQUIRE(evrelst0 == RelState::LIG2);
      REQUIRE(evrelst1 == RelState::LIG1);
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
   REQUIRE(emrelst0 == RelState::NONE);
   REQUIRE(emrelst1 == RelState::LIG1);
   COMPARE_REALS(elam, 0.0, 1.0e-7);
   COMPARE_REALS(vlam, 1.0, 1.0e-14);
   COMPARE_REALS(dvldlmda, 0.0, 1.0e-14);
   mapAt(0.85);
   COMPARE_REALS(elam, 0.5, 1.0e-6);
   REQUIRE(deldlmda > 0.0); // the coupled weight grows with lambda
   mapAt(1.0);
   COMPARE_REALS(elam, 1.0, 1.0e-7);
   COMPARE_REALS(deldlmda, 0.0, 1.0e-14);

   use_relstage = false;
}

TEST_CASE("DLMDA-relstage-pinned-endpoints", "[ff][dlmda]")
{
   // A leg pins the sub-lambdas it is not walking, and a pinned sub-lambda has
   // a flat chain rule. dtNeed() therefore drops one van der Waals endpoint on
   // both charging legs, which is where the staged schedule gets most of its
   // speed: two subsystem evaluations instead of four.
   double w, dw, d2w;
   bool need0, need1;

   setStagedLeg(RelStage::LIG1);
   mapAt(0.85);
   dtWeightNeed(vlam, evdtexp, dvldlmda, d2vldlmda2, w, dw, d2w, need0, need1);
   REQUIRE_FALSE(need0); // vlam pinned at 1, so only the coupled endpoint runs
   REQUIRE(need1);

   setStagedLeg(RelStage::LIG2);
   mapAt(0.15);
   dtWeightNeed(vlam, evdtexp, dvldlmda, d2vldlmda2, w, dw, d2w, need0, need1);
   REQUIRE(need0); // vlam pinned at 0, so only the reference endpoint runs
   REQUIRE_FALSE(need1);

   // On the morph leg the electrostatics are the pinned pair instead.
   setStagedLeg(RelStage::VDWM);
   mapAt(0.5);
   dtWeightNeed(elam, emdtexp, deldlmda, d2eldlmda2, w, dw, d2w, need0, need1);
   REQUIRE(need0);
   REQUIRE_FALSE(need1);
   dtWeightNeed(vlam, evdtexp, dvldlmda, d2vldlmda2, w, dw, d2w, need0, need1);
   REQUIRE(need0);
   REQUIRE(need1);

   use_relstage = false;
}

TEST_CASE("DLMDA-relstage-continuity", "[ff][dlmda]")
{
   // Polarization stages with the multipoles exactly, so emplar stays usable.
   for (RelStage leg : {RelStage::LIG2, RelStage::VDWM, RelStage::LIG1}) {
      setStagedLeg(leg);
      for (double lambda : {0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95}) {
         CAPTURE(lambda);
         mapAt(lambda);
         COMPARE_REALS(plam, elam, 1.0e-15);
         COMPARE_REALS(dpldlmda, deldlmda, 1.0e-15);
         COMPARE_REALS(d2pldlmda2, d2eldlmda2, 1.0e-15);
         REQUIRE(eprelst0 == emrelst0);
         REQUIRE(eprelst1 == emrelst1);
         // The map complement on the ligand 2 leg is clamped into range.
         REQUIRE(elam >= 0.0);
         REQUIRE(elam <= 1.0);
      }
   }

   // The three simulations hand over to each other continuously: the leg
   // boundaries agree in every sub-lambda, so a free energy assembled from the
   // three legs has no step (test_eostmap.f:1005-1045).
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

   // Each leg's electrostatic weight is the quintic taper of its own window,
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

   use_relstage = false;
}
