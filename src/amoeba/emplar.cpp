#include "ff/amoeba/emplar.h"
#include "ff/amoeba/empole.h"
#include "ff/amoeba/epolar.h"
#include "ff/dlmda.h"
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
   if (use_emast)
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

/// Evaluates one dual topology state
static void emplarState(int vers, RdtMask mask, const int* group, bool first_state)
{
   mpoleInitState(vers, mask, group, first_state, true);
   polarState(mask, group);
   emplarKernel(vers);
   exfield(vers, 1);
   torque(vers, demx, demy, demz, trqx, trqy, trqz, vir_em);
}

static void emplarBegin(int vers)
{
   empoleBegin(vers);
   zeroOnHost(energy_ep, virial_ep);
   ep_dl.zero(vers);
}

static void emplarMixEndpoints(int vers)
{
   if (not doubleEq(elam, plam))
      TINKER_THROW("The electrostatic and polarization lambda values have drifted apart; "
                   "the fused multipole/polarization dual topology needs them to be equal.");
   empoleMixEndpoints(vers);
}

void emplar_adt(int vers)
{
   double w, dw, d2w;
   bool need0, need1;
   dtWeightNeed(elam, emdtexp, deldlmda, d2eldlmda2, w, dw, d2w, need0, need1);

   emplarBegin(vers);

   // emplar is never reached under analyz, so it carries no count run.
   bool first = true;
   if (need0) {
      emplarState(vers, RdtMask::ENV, mut, first);
      first = false;
      empoleSaveEndpoint0(vers);
   }
   if (need1) {
      if (need0)
         empoleZeroWork(vers);
      emplarState(vers, RdtMask::ALL, mut, first);
      if (not need0)
         empoleSaveEndpoint0(vers);
   }

   emplarMixEndpoints(vers);
   empoleFinish(vers);
}

void emplar_rdt(int vers)
{
   double w, dw, d2w;
   bool need0, need1;
   dtWeightNeed(elam, emdtexp, deldlmda, d2eldlmda2, w, dw, d2w, need0, need1);

   emplarBegin(vers);

   const RelDualOps ops = {
      [](int v, RdtMask mask, bool first) { emplarState(v, mask, rdt_group, first); },
      [](int v) { empoleZeroWork(v); },
      [](int v) { empoleSaveEndpoint0(v); },
      [](int v) { emplarMixEndpoints(v); },
   };
   relDualDrive(vers, emrelst0, emrelst1, need0, need1, ops);

   empoleFinish(vers);

   // Restore the full system for whatever runs next.
   mpoleInitState(calc::v0, RdtMask::ALL, rdt_group, false);
   polarState(RdtMask::ALL, rdt_group);
}
}
