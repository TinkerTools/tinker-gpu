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

// Mirrors the count gating of relDualDrive below. Within the reported endpoint
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
// is what adtMix used to form from the two finished endpoints. Both terms of
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

void relDualDrive(int vers, RelState ist0, RelState ist1, bool need0, bool need1, const RelDualOps& ops)
{
   // dtNeed() never clears both endpoints: w is either 1, making need1 true,
   // or not 1, making need0 true.
   const auto do_a = vers & calc::analyz;
   const int qvers = vers & ~calc::analyz; // evaluate, but do not count

   int only0[nRelSlot], only1[nRelSlot], both[nRelSlot];
   int n0 = 0, n1 = 0, nb = 0;
   for (int k = 0; k < nRelSlot; ++k) {
      RdtMask mask;
      bool in0, in1;
      relSlot(k, ist0, ist1, mask, in0, in1);
      in0 = in0 and need0;
      in1 = in1 and need1;
      if (in0 and in1)
         both[nb++] = k;
      else if (in0)
         only0[n0++] = k;
      else if (in1)
         only1[n1++] = k;
   }

   // The count is reported from endpoint 1 when it is live, else endpoint 0.
   // Within that endpoint a single ligand-plus-environment subsystem carries
   // the whole count if there is one; a decoupled endpoint has none, so its
   // subsystems sum instead (empole3.f:2584-2650).
   const RelState reported = need1 ? ist1 : ist0;
   bool coupled = false;
   for (int k = 0; k < nRelSlot and not coupled; ++k) {
      RdtMask mask;
      bool in0, in1;
      relSlot(k, reported, reported, mask, in0, in1);
      coupled = in0 and relSlotIsCoupled(k);
   }

   bool first = true;
   auto run = [&](int k, bool reportedPass) {
      RdtMask mask;
      bool in0, in1;
      relSlot(k, ist0, ist1, mask, in0, in1);
      bool counts = do_a and reportedPass and (not coupled or relSlotIsCoupled(k));
      ops.state(counts ? vers : qvers, mask, first);
      first = false;
   };

   // Endpoint 0 runs first so that the reported pass is the one whose counts
   // survive the zeroWork() between the two.
   if (need0) {
      for (int i = 0; i < n0; ++i)
         run(only0[i], not need1);
      ops.save(vers);
   }
   if (need1) {
      if (need0)
         ops.zeroWork(vers);
      for (int i = 0; i < n1; ++i)
         run(only1[i], true);
      // Alias the dead endpoint 0 onto endpoint 1 so the mix is an identity.
      if (not need0)
         ops.save(vers);
   }
   ops.mix(vers);

   // The shared subsystems ride on top of the mixed result. Their counts are
   // unaffected by the mix, so the gating above still applies to them.
   for (int i = 0; i < nb; ++i)
      run(both[i], true);
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

TINKER_FVOID2(acc0, cu1, adtMix, int, bool, int, size_t, double, double, double, const EnergyBufferTraits::type*,
   EnergyBuffer, EnergyBuffer, EnergyBuffer, VirialBuffer, VirialBuffer, VirialBuffer, const grad_prec*,
   const grad_prec*, const grad_prec*, grad_prec*, grad_prec*, grad_prec*, grad_prec*, grad_prec*, grad_prec*);
void adtMix(int vers, bool do_dlmda, int n, size_t buffer_size, double weight1, double dweight1, double d2weight1,
   const EnergyBufferTraits::type* e0, EnergyBuffer e1, EnergyBuffer dedl, EnergyBuffer d2edl2, VirialBuffer v0,
   VirialBuffer v1, VirialBuffer dvdl, const grad_prec* gx0, const grad_prec* gy0, const grad_prec* gz0,
   grad_prec* gx1, grad_prec* gy1, grad_prec* gz1, grad_prec* dgxdl, grad_prec* dgydl, grad_prec* dgzdl)
{
   TINKER_FCALL2(acc0, cu1, adtMix, vers, do_dlmda, n, buffer_size, weight1, dweight1, d2weight1, e0, e1, dedl,
      d2edl2, v0, v1, dvdl, gx0, gy0, gz0, gx1, gy1, gz1, dgxdl, dgydl, dgzdl);
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
