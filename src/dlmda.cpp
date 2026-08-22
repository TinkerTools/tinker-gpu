#include "ff/dlmda.h"
#include "ff/ost.h"
#include "ff/thermint.h"
#include "ff/atom.h"
#include "ff/egvop.h"
#include "ff/elec.h"
#include "ff/evdw.h"
#include "ff/potent.h"
#include "md/osrw.h"
#include "seq/ost.h"
#include "tool/darray.h"
#include "tool/error.h"
#include "tool/externfunc.h"
#include "tool/ioprint.h"
#include <tinker/detail/dlmda.hh>
#include <tinker/detail/mplpot.hh>
#include <tinker/detail/mutant.hh>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace tinker {
void dlmdaData(RcOp op)
{
   if (!use_rel)
      return;

   if (op & RcOp::DEALLOC)
      darray::deallocate(rdt_group);

   if (op & RcOp::ALLOC)
      darray::allocate(n, &rdt_group);

   if (op & RcOp::INIT) {
      std::vector<int> group(n);
      for (int i = 0; i < n; ++i)
         group[i] = mutant::mutg[i];
      darray::copyin(g::q0, n, rdt_group, group.data());
      waitFor(g::q0);
   }
}

void dlmdaData2(RcOp op)
{
   if (op & RcOp::INIT) {
      bool lambda_dynamics = use_dlmda or use_ost or use_meta or use_emdt //
         or use_epdt or use_evdt or use_rel;
      if (lambda_dynamics and not(pltfm_config & Platform::CUDA))
         TINKER_THROW("OST  --  Lambda dynamics requires the CUDA platform");
   }
}

Lmdamap lmdamapFrom(const char* s)
{
   // s is a Fortran character*3 buffer (no null terminator).
   if (std::strncmp(s, "EXP", 3) == 0)
      return Lmdamap::EXP;
   if (std::strncmp(s, "INV", 3) == 0)
      return Lmdamap::INV;
   if (std::strncmp(s, "QNT", 3) == 0)
      return Lmdamap::QNT;
   return Lmdamap::QNT;
}

RelStage relStageFrom(const char* s)
{
   // s is a Fortran character*4 buffer (no null terminator).
   if (std::strncmp(s, "LIG1", 4) == 0)
      return RelStage::LIG1;
   if (std::strncmp(s, "LIG2", 4) == 0)
      return RelStage::LIG2;
   return RelStage::VDWM;
}

void relSlot(int k, RelState ist0, RelState ist1, RdtMask& mask, bool& in0, bool& in1)
{
   // Only five subsystems are reachable, and the three coupling states of a
   // relative dual topology are sums of them (mutate.f:relslot):
   //
   //    slot   la     lb     le     subsystem
   //      0    T      F      T      ligand 1 with environment
   //      1    F      T      T      ligand 2 with environment
   //      2    F      F      T      environment alone
   //      3    T      F      F      ligand 1 alone
   //      4    F      T      F      ligand 2 alone
   //
   //    LIG1 = slots 0 and 4 ,   ligand 1 bound, ligand 2 free
   //    LIG2 = slots 1 and 3 ,   ligand 2 bound, ligand 1 free
   //    NONE = slots 2, 3 and 4 ,  neither ligand bound
   static constexpr RdtMask kMask[nRelSlot] = {
      RdtMask::AE, RdtMask::BE, RdtMask::ENV, RdtMask::LIGA, RdtMask::LIGB};
   // kMember[slot][state - 1].
   static constexpr bool kMember[nRelSlot][3] = {
      {true, false, false},  //
      {false, true, false},  //
      {false, false, true},  //
      {false, true, true},   //
      {true, false, true}};  //

   mask = kMask[k];
   in0 = kMember[k][(int)ist0 - 1];
   in1 = kMember[k][(int)ist1 - 1];
}

void dtCellSet(unsigned& bits, int gi, int gk)
{
   bits |= 1u << (3 * gi + gk);
   bits |= 1u << (3 * gk + gi);
}

// A state's slots are disjoint (relSlot: LIG1 = {env+ligA} u {ligB}, LIG2 =
// {env+ligB} u {ligA}, NONE = {env} u {ligA} u {ligB}), so the union here is
// the same total the slot-by-slot evaluation used to accumulate.
unsigned dtStateBits(RelState ist)
{
   unsigned bits = 0;
   for (int k = 0; k < nRelSlot; ++k) {
      RdtMask mask;
      bool in, dup;
      relSlot(k, ist, ist, mask, in, dup);
      if (not in)
         continue;
      for (int gi = 0; gi < 3; ++gi)
         for (int gk = gi; gk < 3; ++gk)
            if (rdtPairActive(mask, gi, gk))
               dtCellSet(bits, gi, gk);
   }
   return bits;
}

// Mirrors the count gating the dual topology drivers apply. Within the reported endpoint
// a single ligand-plus-environment subsystem carries the whole count if there
// is one; a decoupled endpoint has none, so its subsystems sum.
unsigned dtCountBits(RelState reported)
{
   bool coupled = false;
   for (int k = 0; k < nRelSlot and not coupled; ++k) {
      RdtMask mask;
      bool in, dup;
      relSlot(k, reported, reported, mask, in, dup);
      coupled = in and relSlotIsCoupled(k);
   }

   unsigned bits = 0;
   for (int k = 0; k < nRelSlot; ++k) {
      RdtMask mask;
      bool in, dup;
      relSlot(k, reported, reported, mask, in, dup);
      if (not in or (coupled and not relSlotIsCoupled(k)))
         continue;
      for (int gi = 0; gi < 3; ++gi)
         for (int gk = gi; gk < 3; ++gk)
            if (rdtPairActive(mask, gi, gk))
               dtCellSet(bits, gi, gk);
   }
   return bits;
}

// How many subsystems the reported endpoint counts, by the same gating. A term
// whose count does not depend on the subsystem's parameters -- the Ewald self
// energy counts every atom whether or not its multipoles were zeroed -- reports
// its whole count once per counted pass, so it needs the number of passes
// rather than the set of pair types.
int dtCountSlots(RelState reported)
{
   bool coupled = false;
   for (int k = 0; k < nRelSlot and not coupled; ++k) {
      RdtMask mask;
      bool in, dup;
      relSlot(k, reported, reported, mask, in, dup);
      coupled = in and relSlotIsCoupled(k);
   }

   int nslot = 0;
   for (int k = 0; k < nRelSlot; ++k) {
      RdtMask mask;
      bool in, dup;
      relSlot(k, reported, reported, mask, in, dup);
      if (in and (not coupled or relSlotIsCoupled(k)))
         ++nslot;
   }
   return nslot;
}

// E = w*E1 + (1-w)*E0, so dE/dl = dw*(E1-E0) and d2E/dl2 = d2w*(E1-E0), which
// is what the endpoint mixing used to form from two finished endpoints. Both terms of
// the second-derivative chain rule are proportional to E1-E0, so the whole of
// it collapses onto one scalar per endpoint. need0/need1 are not consulted: a
// dead endpoint already falls out with every weight at zero.
void dtWeightsToCoef(DtCoef& c, double w, double dw, double d2w, double chain, double d2chain, bool driven)
{
   c.a0 = 1 - w;
   c.a1 = w;
   double b = driven ? dw * chain : 0;
   double d2 = driven ? d2w * chain * chain + dw * d2chain : 0;
   c.b0 = -b;
   c.b1 = b;
   c.c0 = -d2;
   c.c1 = d2;
}

void dtWeight(double x, int nexp, double& w, double& dw, double& d2w)
{
   // The linear case is taken separately so that a zero sub-lambda never
   // reaches a zero power.
   w = std::pow(x, nexp);
   dw = 0.0;
   d2w = 0.0;
   if (nexp == 1) {
      dw = 1.0;
   } else if (nexp >= 2) {
      dw = (double)nexp * std::pow(x, nexp - 1);
      d2w = (double)nexp * (double)(nexp - 1) * std::pow(x, nexp - 2);
   }
}

void dtNeed(double w, double dw, double d2w, double chain, double d2chain, bool& need0, bool& need1)
{
   double c1 = dw * chain;
   double c2 = d2w * chain * chain + dw * d2chain;
   need1 = (w != 0.0) or (c1 != 0.0) or (c2 != 0.0);
   need0 = (w != 1.0) or (c1 != 0.0) or (c2 != 0.0);
}

void dtWeightNeed(double sublmda, int dtexp, double chain, double d2chain, //
   double& w, double& dw, double& d2w, bool& need0, bool& need1)
{
   dtWeight(sublmda, dtexp, w, dw, d2w);
   dtNeed(w, dw, d2w, chain, d2chain, need0, need1);
}

void LmdaBuffer::manage(RcOp op, int flag, bool driven, EnergyBuffer* dl1, EnergyBuffer* dl2,
   energy_prec* term1, energy_prec* term2)
{
   const bool rc_a = flag & calc::analyz;

   if (op & RcOp::DEALLOC) {
      if (rc_a and mDl1)
         darray::deallocate(*mDl1, *mDl2);
      if (mDl1) {
         *mDl1 = nullptr;
         *mDl2 = nullptr;
      }
      mDl1 = nullptr;
      mDl2 = nullptr;
      mTerm1 = nullptr;
      mTerm2 = nullptr;
      mFlag = 0;
      mDriven = false;
   }

   if (op & RcOp::ALLOC) {
      mDl1 = dl1;
      mDl2 = dl2;
      mTerm1 = term1;
      mTerm2 = term2;
      mFlag = flag;
      mDriven = driven;

      const int mask = lmdaDerivMask(flag, driven);
      *dl1 = nullptr;
      *dl2 = nullptr;
      if (mask & calc::energy_dlmda1) {
         if (rc_a)
            darray::allocate(bufferSize(), dl1);
         else
            *dl1 = dedl_buf;
      }
      if (mask & calc::energy_dlmda2) {
         if (rc_a)
            darray::allocate(bufferSize(), dl2);
         else
            *dl2 = d2edl2_buf;
      }
   }
}

void LmdaBuffer::zero(int vers) const
{
   if (not(mFlag & calc::analyz) or not mDl1)
      return;
   const int mask = lmdaDerivMask(vers, mDriven);
   if (mask & calc::energy_dlmda1)
      darray::zero(g::q0, bufferSize(), *mDl1);
   if (mask & calc::energy_dlmda2)
      darray::zero(g::q0, bufferSize(), *mDl2);
}

void LmdaBuffer::flush(int vers) const
{
   if (not(mFlag & calc::analyz) or not mDl1)
      return;
   const int mask = lmdaDerivMask(vers, mDriven);
   if (mask & calc::energy_dlmda1) {
      energy_prec e = energyReduce(*mDl1);
      *mTerm1 += e;
      dedl += e;
   }
   if (mask & calc::energy_dlmda2) {
      energy_prec e = energyReduce(*mDl2);
      *mTerm2 += e;
      d2edl2 += e;
   }
}

int dtPassList(bool relative, RelState ist0, RelState ist1, DtPass out[nRelSlot])
{
   if (not relative) {
      // The absolute schedule zeroes the mutated atoms for endpoint 0 and keeps
      // the whole system for endpoint 1; there is nothing the two share.
      out[0] = {RdtMask::ENV, true, false, -1};
      out[1] = {RdtMask::ALL, false, true, -1};
      return 2;
   }

   int npass = 0;
   for (int k = 0; k < nRelSlot; ++k) {
      RdtMask mask;
      bool in0, in1;
      relSlot(k, ist0, ist1, mask, in0, in1);
      if (in0 or in1)
         out[npass++] = {mask, in0, in1, k};
   }
   return npass;
}

void dtPassWeights(const DtCoef& c, const DtPass& p, real& wa, real& wb, real& wc)
{
   wa = (p.in0 ? c.a0 : 0) + (p.in1 ? c.a1 : 0);
   wb = (p.in0 ? c.b0 : 0) + (p.in1 ? c.b1 : 0);
   wc = (p.in0 ? c.c0 : 0) + (p.in1 ? c.c1 : 0);
}


void dlmda_mech()
{
   lambda = mutant::lambda;

   use_dlmda = dlmda::use_dlmda;
   use_emdt = dlmda::use_emdt;
   use_epdt = dlmda::use_epdt;
   use_evdt = dlmda::use_evdt;
   use_plmda = dlmda::use_plmda and not use_osrw;
   use_mainlmda = dlmda::use_mainlmda;
   use_rel = mutant::use_rel;

   use_edlmda = (dlmda::use_edlmda != 0);
   use_pdlmda = (dlmda::use_pdlmda != 0);
   use_vdlmda = (dlmda::use_vdlmda != 0);

   use_emadt = use_emdt && !use_rel;
   use_emast = use_edlmda && !use_emdt && !use_rel;
   use_emrdt = use_emdt && use_rel;
   use_epadt = use_epdt && !use_rel;
   use_eprdt = use_epdt && use_rel;
   use_evadt = use_evdt && !use_rel;
   use_evast = use_vdlmda && !use_evdt && !use_rel;
   use_evrdt = use_evdt && use_rel;

   emdtexp = dlmda::emdtexp;
   epdtexp = dlmda::epdtexp;
   evdtexp = dlmda::evdtexp;

   // which lambda-dynamics method owns the main lambda.
   use_ost = dlmda::use_ost;
   use_meta = dlmda::use_meta;
   use_ti = dlmda::use_ti;
   dlmda::use_ostdyn = use_ost;
   dlmda::use_metadyn = use_meta;

   elmdaexp = dlmda::elmdaexp;
   plmdaexp = dlmda::plmdaexp;
   vlmdaexp = dlmda::vlmdaexp;
   elmdainvn = dlmda::elmdainvn;
   plmdainvn = dlmda::plmdainvn;
   vlmdainvn = dlmda::vlmdainvn;
   elmdainveps = dlmda::elmdainveps;
   plmdainveps = dlmda::plmdainveps;
   vlmdainveps = dlmda::vlmdainveps;

   elmdamap = lmdamapFrom(dlmda::elmdamap);
   plmdamap = lmdamapFrom(dlmda::plmdamap);
   vlmdamap = lmdamapFrom(dlmda::vlmdamap);

   use_elmdamap = (dlmda::use_elmdamap != 0);
   use_plmdamap = (dlmda::use_plmdamap != 0);
   use_vlmdamap = (dlmda::use_vlmdamap != 0);

   deldlmda = dlmda::deldlmda;
   dpldlmda = dlmda::dpldlmda;
   dvldlmda = dlmda::dvldlmda;
   d2eldlmda2 = 0.0;
   d2pldlmda2 = 0.0;
   d2vldlmda2 = 0.0;

   qntelmda0 = dlmda::qntelmda0;
   qntelmda1 = dlmda::qntelmda1;
   qntplmda0 = dlmda::qntplmda0;
   qntplmda1 = dlmda::qntplmda1;
   qntvlmda0 = dlmda::qntvlmda0;
   qntvlmda1 = dlmda::qntvlmda1;

   use_relstage = (dlmda::use_relstage != 0);
   relstage = relStageFrom(dlmda::relstage);

   // Plain relative interpolates between the two coupled states; mapRelStage()
   // overrides the electrostatic pair on a staged leg (mutate.f:404-409).
   emrelst0 = RelState::LIG2;
   emrelst1 = RelState::LIG1;
   eprelst0 = RelState::LIG2;
   eprelst1 = RelState::LIG1;
   evrelst0 = RelState::LIG2;
   evrelst1 = RelState::LIG1;
}

void avgstd(const std::vector<double>& v, int begin, int count, double& avg, double& sd)
{
   if (count < 1) {
      avg = 0.0;
      sd = 0.0;
      return;
   }

   // Shifted mean, to keep the sum of squares well conditioned.
   const double k = v[begin];
   double total = 0.0, totalsq = 0.0;
   for (int i = begin; i < begin + count; ++i) {
      double d = v[i] - k;
      total += d;
      totalsq += d * d;
   }
   avg = k + total / (double)count;
   double var = (totalsq - total * total / (double)count) / (double)count;
   sd = std::sqrt(var > 0.0 ? var : 0.0);
}

}

namespace tinker {
// Power-law sub-lambda map, lmda = x^exponent.
static void sublmdaExp(double x, int exponent, double& lmda, double& dlmda, double& d2lmda)
{
   lmda = std::pow(x, exponent);
   if (exponent == 1) {
      dlmda = 1.0;
      d2lmda = 0.0;
   } else if (exponent == 2) {
      dlmda = 2.0 * x;
      d2lmda = 2.0;
   } else {
      double expnt = (double)exponent;
      dlmda = expnt * std::pow(x, exponent - 1);
      d2lmda = expnt * (expnt - 1.0) * std::pow(x, exponent - 2);
   }
}

// Shifted inverse-power sub-lambda map onto [0,1] (dlambda.f:sublmdainvpower).
static void sublmdaInvPower(double x, int nn, double eps, double& lmda, double& dlmda, double& d2lmda)
{
   double xval = x;
   if (nn <= 1) {
      lmda = xval;
      dlmda = 1.0;
      d2lmda = 0.0;
      return;
   }
   double shift = eps;
   if (shift <= 0.0)
      shift = 0.1;
   double power = 1.0 / (double)nn;
   double root0 = std::pow(shift, power);
   double denom = std::pow(1.0 + shift, power) - root0;
   double base = xval + shift;
   lmda = (std::pow(base, power) - root0) / denom;
   dlmda = power * std::pow(base, power - 1.0) / denom;
   d2lmda = power * (power - 1.0) * std::pow(base, power - 2.0) / denom;
}

// Quintic switching polynomial
void quinticTaper(double x, double cut, double off, double& taper, double& dtaper, double& d2taper)
{
   dtaper = 0.0;
   d2taper = 0.0;
   if (x <= cut) {
      taper = 1.0;
      return;
   } else if (x >= off) {
      taper = 0.0;
      return;
   }

   double rinv = 1.0 / (off - cut);
   double u = (x - cut) * rinv;
   double u2 = u * u;
   double v = 1.0 - u;

   taper = 1.0 - u2 * u * (10.0 - 15.0 * u + 6.0 * u2);
   dtaper = -30.0 * u2 * v * v * rinv;
   d2taper = -60.0 * u * v * (1.0 - 2.0 * u) * rinv * rinv;
}

// Maps one sub-lambda from main lambda to its mapping type.
static void mapOne(double lmda, Lmdamap map, double qnt0, double qnt1, int expExp, int invN, double invEps,
   double& value, double& dvalue, double& d2value)
{
   if (map == Lmdamap::EXP) {
      sublmdaExp(lmda, expExp, value, dvalue, d2value);
   } else if (map == Lmdamap::INV) {
      sublmdaInvPower(lmda, invN, invEps, value, dvalue, d2value);
   } else { // Lmdamap::QNT
      // quantized map: sublambda = 1 - taper.
      double taper, dtaper, d2taper;
      quinticTaper(lmda, qnt0, qnt1, taper, dtaper, d2taper);
      value = 1.0 - taper;
      dvalue = -dtaper;
      d2value = -d2taper;
   }
}

// Maps the main lambda onto the sub-lambdas of the one declared staged
// relative leg (dlambda.f:maprelstage):
//
//    LIG2   charge ligand 2 against the decoupled reference, its weight
//             rising with the main lambda
//    VDWM   both ligands electrostatically decoupled while van der Waals
//             morphs from ligand 2 onto ligand 1
//    LIG1   charge ligand 1 against the decoupled reference, its weight
//             rising with the main lambda
static void mapRelStage(double lmda)
{
   double eval, vval;

   // van der Waals interpolates between the two coupled states on every leg,
   // morphing over its own map in the middle and held at one end or the other
   // while a ligand is being charged.
   evrelst0 = RelState::LIG2;
   evrelst1 = RelState::LIG1;

   if (relstage == RelStage::VDWM) {
      // The middle leg holds both ligands decoupled, so electrostatics and
      // polarization sit at the reference state and leave the chain rule
      // while van der Waals morphs across its map.
      emrelst0 = RelState::NONE;
      emrelst1 = RelState::NONE;
      eval = 0.0;
      deldlmda = 0.0;
      d2eldlmda2 = 0.0;
      mapOne(lmda, vlmdamap, qntvlmda0, qntvlmda1, vlmdaexp, vlmdainvn, vlmdainveps, vval, dvldlmda, d2vldlmda2);
      vlam = vval;
   } else if (relstage == RelStage::LIG1) {
      // The ligand 1 leg charges ligand 1 against the decoupled reference
      // with van der Waals already morphed onto it.
      emrelst0 = RelState::NONE;
      emrelst1 = RelState::LIG1;
      mapOne(lmda, elmdamap, qntelmda0, qntelmda1, elmdaexp, elmdainvn, elmdainveps, eval, deldlmda, d2eldlmda2);
      vlam = 1.0;
      dvldlmda = 0.0;
      d2vldlmda2 = 0.0;
   } else {
      // The ligand 2 leg discharges ligand 2 as the main lambda rises, so its
      // weight is the complement of the map, with van der Waals still on it.
      emrelst0 = RelState::NONE;
      emrelst1 = RelState::LIG2;
      mapOne(lmda, elmdamap, qntelmda0, qntelmda1, elmdaexp, elmdainvn, elmdainveps, eval, deldlmda, d2eldlmda2);
      eval = 1.0 - eval;
      deldlmda = -deldlmda;
      d2eldlmda2 = -d2eldlmda2;
      vlam = 0.0;
      dvldlmda = 0.0;
      d2vldlmda2 = 0.0;
   }

   // Numerical guard on the map complement.
   elam = std::min(1.0, std::max(0.0, eval));

   // Polarization stages with the multipoles: same states, same weight.
   eprelst0 = emrelst0;
   eprelst1 = emrelst1;
   plam = elam;
   dpldlmda = deldlmda;
   d2pldlmda2 = d2eldlmda2;
}

bool polTracksEle()
{
   constexpr double eps = 1.0e-6;
   auto sameValue = [](double a, double b) { return std::fabs(a - b) <= eps; };

   // The staged schedule drives polarization off the multipole weight by
   // construction, so the per-map comparisons below do not apply to it.
   if (use_relstage)
      return true;
   // One sub-lambda driven and the other frozen: they part company as soon as
   // the main lambda moves off the value they happen to share now.
   if (use_elmdamap != use_plmdamap)
      return false;
   // Neither is driven, so today's values are the only values.
   if (not(use_mainlmda and use_elmdamap))
      return sameValue(elam, plam);

   if (plmdamap != elmdamap)
      return false;

   if (elmdamap == Lmdamap::EXP) {
      return plmdaexp == elmdaexp;
   } else if (elmdamap == Lmdamap::INV) {
      return plmdainvn == elmdainvn and sameValue(plmdainveps, elmdainveps);
   } else {
      return sameValue(qntplmda0, qntelmda0) and sameValue(qntplmda1, qntelmda1);
   }
}

void mapSubLambda()
{
   if (use_relstage) {
      mapRelStage(lambda);
      return;
   }

   // Map the main lambda onto each sub-lambda that follows it.
   if (use_plmdamap) {
      double pval;
      mapOne(lambda, plmdamap, qntplmda0, qntplmda1, plmdaexp, plmdainvn, plmdainveps, pval, dpldlmda, d2pldlmda2);
      plam = pval;
   }
   if (use_elmdamap) {
      double eval;
      mapOne(lambda, elmdamap, qntelmda0, qntelmda1, elmdaexp, elmdainvn, elmdainveps, eval, deldlmda, d2eldlmda2);
      elam = eval;
   }
   if (use_vlmdamap) {
      double vval;
      mapOne(lambda, vlmdamap, qntvlmda0, qntvlmda1, vlmdaexp, vlmdainvn, vlmdainveps, vval, dvldlmda, d2vldlmda2);
      vlam = vval;
   }

}

}
