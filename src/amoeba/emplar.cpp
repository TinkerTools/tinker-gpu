#include "ff/amoeba/emplar.h"
#include "ff/amoeba/empole.h"
#include "ff/amoeba/epolar.h"
#include "ff/amoeba/induce.h"
#include "ff/dlmda.h"
#include "ff/termbuf.h"
#include "ff/elec.h"
#include "ff/energy.h"
#include "ff/evdw.h"
#include "ff/modamoeba.h"
#include "ff/nblist.h"
#include "ff/ost.h"
#include "ff/potent.h"
#include "math/zero.h"
#include "tool/error.h"
#include "tool/externfunc.h"
#include <tinker/detail/extfld.hh>
#include <tinker/detail/mplpot.hh>

#include <cassert>
#include <cmath>

namespace tinker {
static int emplar_flag = -1;

/// compare double
static bool doubleEq(double a, double b)
{
   constexpr double eps = 1.0e-6;
   return std::fabs(a - b) <= eps;
}

/// Determines whether emplar can be used for dual topology.
static bool emplarDualMatched()
{
   if (not polTracksEle())
      return false;
   // The fused kernel evaluates one set of subsystems and mixes them once, so
   // both terms must name the same endpoints and interpolate identically. This
   // holds on a staged leg too, where mapRelStage() copies the multipole
   // coupling states onto polarization but the two exponents stay independent.
   if (emrelst0 != eprelst0 or emrelst1 != eprelst1)
      return false;
   return emdtexp == epdtexp;
}

static bool emplarDecide()
{
   if (mplpot::use_chgpen)
      return false;
   if (use_plmda and not polTracksEle())
      return false;
   if (use_emast and not use_epast)
      return false;
   if (rc_flag & calc::analyz)
      return false;
   if (not(use(Potent::MPOLE) and use(Potent::POLAR)))
      return false;
   if (not(mlistVersion() & Nbl::SPATIAL))
      return false;

   if (use_emdt != use_epdt)
      return false;
   if (use_emdt and not emplarDualMatched())
      return false;

   return true;
}

bool useEmplar()
{
   assert(emplar_flag >= 0);
   return emplar_flag == 1;
}

void emplarData(RcOp op)
{
   if (op & RcOp::ALLOC)
      emplar_flag = emplarDecide() ? 1 : 0;
}
}

namespace tinker {
TINKER_FVOID2(acc0, cu1, emplar, int);
static void emplarKernel(int vers)
{
   TINKER_FCALL2(acc0, cu1, emplar, vers);
}

void emplar(int vers)
{
   auto do_v = vers & calc::virial;

   zeroOnHost(energy_em, virial_em);
   zeroOnHost(energy_ep, virial_ep);

   mpoleInit(vers, use_emast);
   emplarKernel(vers);
   exfield(vers, 1);
   // epolarPairwiseExtfield(vers, uind); // emplar uses the dot product version
   torque(vers, demx, demy, demz);
   if (do_v) {
      VirialBuffer u2 = vir_trq;
      virial_prec v2[9];
      virialReduce(v2, u2);
      for (int iv = 0; iv < 9; ++iv)
         virial_elec[iv] += v2[iv];
   }
}


TINKER_FVOID2(acc0, cu1, emplarAst, int);
static void emplarAstKernel(int vers)
{
   TINKER_FCALL2(acc0, cu1, emplarAst, vers);
}

void emplarAst(int vers)
{
   if (not doubleEq(elam, plam))
      TINKER_THROW("The electrostatic and polarization lambda values have drifted apart; "
                   "the fused multipole/polarization single topology needs them to be equal.");

   const int dvers = lmdaDerivVers(vers, use_edlmda);
   auto do_v = vers & calc::virial;

   zeroOnHost(energy_em, virial_em);
   zeroOnHost(energy_ep, virial_ep);

   // Lambda scaled state, and the solve.
   mpoleScale(plam);
   polarState(RdtMask::ALL, mut, plam);
   mpoleInit(vers, false);
   induce(uind, uinp);

   // Unscaled state: the permanent multipole terms and their lambda derivative.
   // empoleEwaldRecip and exfield both decorate the version themselves once
   // use_emast is set, so they take the undecorated one.
   mpoleInitAst();
   emplarAstKernel(dvers);
   if (useEwald())
      empoleEwaldRecip(vers);
   exfield(vers, 1);

   // Back to the lambda scaled state for the polarization reciprocal term, which
   // reads the permanent dipole straight out of rpole. Its version stays
   // undecorated too, but for the opposite reason: it has no lambda derivative
   // channel, and a version it does not recognize leaves it doing nothing at all.
   mpoleRefresh();
   if (useEwald()) {
      const AccumRef out = em_buf.ref();
      epolarEwaldRecipSelf(vers & ~calc::energy, out.e, out.v, out.gx, out.gy, out.gz);
   }
   if (vers & calc::energy)
      epolar0DotProd(uind, udirp, em_buf.ref().e);

   const bool do_astdl = lmdaDerivMask(vers, use_pdlmda) & calc::energy_dlmda1;
   if (do_astdl)
      epolarAstDeriv(vers);

   torque(vers, demx, demy, demz);
   if (do_v) {
      VirialBuffer u2 = vir_trq;
      virial_prec v2[9];
      virialReduce(v2, u2);
      for (int iv = 0; iv < 9; ++iv)
         virial_elec[iv] += v2[iv];
   }

   // Leave pole where epolar() leaves it, and undo the masking epolarAstDeriv
   // left behind so that whatever runs next sees the whole system again.
   mpoleScale(elam);
   if (do_astdl)
      mpoleRefresh();
}


TINKER_FVOID2(acc0, cu1, emplarDt, int, real, real, real);
static void emplarDt(int vers, real wa, real wb, real wc)
{
   TINKER_FCALL2(acc0, cu1, emplarDt, vers, wa, wb, wc);
}

TINKER_FVOID2(acc0, cu1, epolar0DotProdDt, int, const real (*)[3], const real (*)[3], EnergyBuffer, real, real,
   real);
TINKER_FVOID2(acc0, cu1, exfieldDipoleDt, int, const DtCoef&);

static void emplarBegin(int vers)
{
   empoleBegin(vers);
   zeroOnHost(energy_ep, virial_ep);
}

/// Evaluates one dual topology subsystem straight into the fused accumulators.
/// The pass masks rpole and polarity first, so one set of weights covers
/// everything it touches, and nothing is reduced or copied aside between passes.
static void emplarState(int vers, RdtMask mask, const int* group, bool first_state, //
   real wa, real wb, real wc)
{
   // cmp has to be built here rather than left to empoleEwaldRecipDt below:
   // induce() gets the reciprocal part of its direct field from it.
   mpoleInitStateDt(vers, mask, group, first_state);
   polarState(mask, group);

   induce(uind, uinp);

   // permanent multipole real space and self, plus polarization real space
   emplarDt(vers, wa, wb, wc);
   if (useEwald()) {
      empoleEwaldRecipDt(vers, mask, wa, wb, wc);
      const AccumRef out = em_buf.ref();
      epolarEwaldRecipSelfDt(vers, out.e, out.v, out.gx, out.gy, out.gz, dtRecipSinks(vers, wa, wb));
   }
   // the polarization energy, and with it both of its lambda derivatives
   if (vers & calc::energy)
      TINKER_FCALL2(acc0, cu1, epolar0DotProdDt, vers, uind, udirp, em_buf.ref().e, wa, wb, wc);
   if (extfld::use_exfld)
      TINKER_FCALL2(acc0, cu1, exfieldDipoleDt, vers, dtCoefUniform(wa, wb, wc));
}

void emplar_dt(int vers)
{
   if (not doubleEq(elam, plam))
      TINKER_THROW("The electrostatic and polarization lambda values have drifted apart; "
                   "the fused multipole/polarization dual topology needs them to be equal.");

   const int dvers = lmdaDerivVers(vers, use_edlmda);
   const bool relative = use_emrdt;
   const int* group = relative ? rdt_group : mut;
   auto do_g = vers & calc::grad;

   double w, dw, d2w;
   bool need0, need1;
   dtWeightNeed(elam, emdtexp, deldlmda, d2eldlmda2, w, dw, d2w, need0, need1);

   emplarBegin(vers);

   DtCoef c;
   dtWeightsToCoef(c, w, dw, d2w, deldlmda, d2eldlmda2, use_edlmda);

   DtPass pass[nRelSlot];
   const int npass = dtPassList(relative, emrelst0, emrelst1, pass);

   // emplar is never reached under analyz, so no pass has an interaction count
   // to keep and each one is evaluated purely for its weight.
   bool first = true;
   for (int k = 0; k < npass; ++k) {
      real wa, wb, wc;
      dtPassWeights(c, pass[k], wa, wb, wc);
      // emplar never counts, so nothing obliges a weightless pass to run.
      if (dtPassIsIdle(dvers, wa, wb, wc, false))
         continue;
      emplarState(dvers, pass[k].mask, group, first, wa, wb, wc);
      first = false;
   }

   // Every pass added its torque unconverted, so one conversion covers them all.
   if (do_g) {
      torque(vers, demx, demy, demz, trqx, trqy, trqz, vir_em);
      if (dvers & (calc::grad_dlmda | calc::virial_dlmda))
         torque(vers, dfdlx, dfdly, dfdlz, dltrqx, dltrqy, dltrqz, dvirdl_buf);
   }

   empoleFinish(vers);

   // The last pass leaves rpole and polarity masked down to a subsystem.
   dtRestoreFullState(group);
}
}
