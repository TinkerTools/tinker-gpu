#include "ff/dlmda.h"
#include "ff/ost.h"
#include "ff/thermint.h"
#include "ff/atom.h"
#include "ff/egvop.h"
#include "ff/elec.h"
#include "ff/evdw.h"
#include "ff/potent.h"
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
         or use_epdt or use_evdt or use_plmda or use_rel;
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

void dlmda_mech()
{
   use_dlmda = dlmda::use_dlmda;
   use_emdt = dlmda::use_emdt;
   use_epdt = dlmda::use_epdt;
   use_evdt = dlmda::use_evdt;
   use_plmda = dlmda::use_plmda;
   use_mainlmda = dlmda::use_mainlmda;
   use_rel = mutant::use_rel;

   use_emadt = use_emdt && !use_rel;
   use_emast = use_dlmda && !use_emdt && !use_rel;
   use_emrdt = use_emdt && use_rel;
   use_epadt = use_epdt && !use_rel;
   use_eprdt = use_epdt && use_rel;
   use_evadt = use_evdt && !use_rel;
   use_evast = use_dlmda && !use_evdt && !use_rel;
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

   use_ele4i = true;
   use_ele4f = true;
   use_pol4i = true;
   use_pol4f = true;
   use_vdw4i = true;
   use_vdw4f = true;

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

   // A sub-lambda not driven by the main lambda has an identity chain.
   deldlmda = 1.0;
   dpldlmda = 1.0;
   dvldlmda = 1.0;
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
   relstg1lmda0 = dlmda::relstg1lmda0;
   relstg1lmda1 = dlmda::relstg1lmda1;
   relstg2lmda0 = dlmda::relstg2lmda0;
   relstg2lmda1 = dlmda::relstg2lmda1;
   relstage = RelStage::VDW_MORPH;
   relstagemix = false;
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

void adtWeight(double lambda, int exponent, double& weight, double& dweight, double& d2weight)
{
   weight = std::pow(lambda, exponent);
   dweight = 0;
   d2weight = 0;
   if (exponent >= 2) {
      dweight = exponent * std::pow(lambda, exponent - 1);
      d2weight = exponent * (exponent - 1) * std::pow(lambda, exponent - 2);
   } else {
      dweight = 1;
   }
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
   double expnt = (double)exponent;
   if (x <= 0.0) {
      lmda = 0.0;
      if (exponent == 1) {
         dlmda = 1.0;
         d2lmda = 0.0;
      } else if (exponent == 2) {
         dlmda = 0.0;
         d2lmda = 2.0;
      } else {
         dlmda = 0.0;
         d2lmda = 0.0;
      }
      return;
   } else if (x >= 1.0) {
      lmda = 1.0;
      dlmda = expnt;
      d2lmda = expnt * (expnt - 1.0);
      return;
   }
   lmda = std::pow(x, exponent);
   dlmda = expnt * std::pow(x, exponent - 1);
   if (exponent == 1)
      d2lmda = 0.0;
   else
      d2lmda = expnt * (expnt - 1.0) * std::pow(x, exponent - 2);
}

// Shifted inverse-power sub-lambda map onto [0,1] (eost.f:sublmdainvpower).
static void sublmdaInvPower(double x, int nn, double eps, double& lmda, double& dlmda, double& d2lmda)
{
   double xval = x;
   if (xval < 0.0)
      xval = 0.0;
   if (xval > 1.0)
      xval = 1.0;
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
static void mapOne(double lambda, Lmdamap map, double qnt0, double qnt1, int expExp, int invN, double invEps,
   double& value, double& dvalue, double& d2value)
{
   if (map == Lmdamap::EXP) {
      sublmdaExp(lambda, expExp, value, dvalue, d2value);
   } else if (map == Lmdamap::INV) {
      sublmdaInvPower(lambda, invN, invEps, value, dvalue, d2value);
   } else { // Lmdamap::QNT
      // quantized map: sublambda = 1 - taper.
      double taper, dtaper, d2taper;
      quinticTaper(lambda, qnt0, qnt1, taper, dtaper, d2taper);
      value = 1.0 - taper;
      dvalue = -dtaper;
      d2value = -d2taper;
   }
}

// Staged relative free energy schedule
static void mapRelStage(double lambda)
{
   double w, taper, dtaper, d2taper;
   if (lambda > relstg2lmda0) {
      // High-lambda electrostatics leg: weight runs 0 -> 1 across the window.
      quinticTaper(lambda, relstg2lmda0, relstg2lmda1, taper, dtaper, d2taper);
      relstage = RelStage::LIG1_ELE;
      w = 1.0 - taper;
      deldlmda = -dtaper;
      d2eldlmda2 = -d2taper;
   } else if (lambda < relstg1lmda1) {
      // Low-lambda electrostatics leg: weight runs 1 -> 0 across the window.
      quinticTaper(lambda, relstg1lmda0, relstg1lmda1, taper, dtaper, d2taper);
      relstage = RelStage::LIG0_ELE;
      w = taper;
      deldlmda = dtaper;
      d2eldlmda2 = d2taper;
   } else {
      // Both ligands are electrostatically decoupled from the environment.
      relstage = RelStage::VDW_MORPH;
      w = 0.0;
      deldlmda = 0.0;
      d2eldlmda2 = 0.0;
   }

   elam = std::min(1.0, std::max(0.0, w));

   if (elam == 0.0)
      relstage = RelStage::VDW_MORPH;

   relstagemix = (elam > 0.0 and elam < 1.0);

   // Polarization stages with the multipoles: same states, same weight.
   plam = elam;
   dpldlmda = deldlmda;
   d2pldlmda2 = d2eldlmda2;

   // van der Waals keeps its ordinary map.
   double taper2, dtaper2, d2taper2;
   quinticTaper(lambda, qntvlmda0, qntvlmda1, taper2, dtaper2, d2taper2);
   vlam = 1.0 - taper2;
   dvldlmda = -dtaper2;
   d2vldlmda2 = -d2taper2;
   use_vdw4i = (lambda <= qntvlmda1);
   use_vdw4f = (lambda >= qntvlmda0);
}

double mainLambda()
{
   if (use_ost or use_meta)
      return ostlambda;
   else if (use_ti)
      return tilmda;
   else
      return mutant::lambda;
}

void mapSubLambda(double lambda)
{
   if (use_relstage) {
      mapRelStage(lambda);
      return;
   }

   // Map the main lambda onto each sub-lambda that follows it; the rest keep
   // their fixed values and their identity chain factors.
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

   if (use_elmdamap and elmdamap == Lmdamap::QNT) {
      use_ele4i = (lambda <= qntelmda1);
      use_ele4f = (lambda >= qntelmda0);
   }
   if (use_plmdamap and plmdamap == Lmdamap::QNT) {
      use_pol4i = (lambda <= qntplmda1);
      use_pol4f = (lambda >= qntplmda0);
   }
   if (use_vlmdamap and vlmdamap == Lmdamap::QNT) {
      use_vdw4i = (lambda <= qntvlmda1);
      use_vdw4f = (lambda >= qntvlmda0);
   }
}

void lmdachain(int vers)
{
   // Applies the chain rule using the sub-lambda derivatives computed by mapSubLambda.
   auto do_e = vers & calc::energy;
   auto do_v = vers & calc::virial;
   auto do_g = vers & calc::grad;

   // Chain rule for the scalar energy derivatives.
   if (do_e) {
      d2emdl2 = d2emdl2 * deldlmda * deldlmda + demdl * d2eldlmda2;
      demdl = demdl * deldlmda;
      d2epdl2 = d2epdl2 * dpldlmda * dpldlmda + depdl * d2pldlmda2;
      depdl = depdl * dpldlmda;
      d2evdl2 = d2evdl2 * dvldlmda * dvldlmda + devdl * d2vldlmda2;
      devdl = devdl * dvldlmda;

      dedl = demdl + depdl + devdl;
      d2edl2 = d2emdl2 + d2epdl2 + d2evdl2;
   }

   // Chain rule for the virial derivative (first derivative only).
   if (do_v) {
      for (int k = 0; k < 9; ++k) {
         demvirdl[k] *= deldlmda;
         depvirdl[k] *= dpldlmda;
         devvirdl[k] *= dvldlmda;
         dvirdl[k] = demvirdl[k] + depvirdl[k] + devvirdl[k];
      }
   }

   // Chain rule for the per-atom force derivatives.
   if (do_g) {
      darray::zero(g::q0, n, dfsumdlx, dfsumdly, dfsumdlz);
      if (dfvdlx)
         sumGradient(dvldlmda, dfsumdlx, dfsumdly, dfsumdlz, dfvdlx, dfvdly, dfvdlz);
      if (dfmdlx)
         sumGradient(deldlmda, dfsumdlx, dfsumdly, dfsumdlz, dfmdlx, dfmdly, dfmdlz);
      if (dfpdlx)
         sumGradient(dpldlmda, dfsumdlx, dfsumdly, dfsumdlz, dfpdlx, dfpdly, dfpdlz);
   }
}
}
