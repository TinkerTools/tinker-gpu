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
   if (emdtexp != epdtexp)
      return false;
   if (not doubleEq(elam, plam))
      return false;
   if (not use_dlmda)
      return true;

   if (ostpmap != ostemap)
      return false;

   if (ostemap == Ostmap::EXP) {
      return ostepexp == ostemexp;
   } else if (ostemap == Ostmap::INV) {
      return ostinvepn == ostinvemn and doubleEq(ostinvepeps, ostinvemeps);
   } else {
      return doubleEq(ostplmda0, ostelmda0) and doubleEq(ostplmda1, ostelmda1);
   }
}

static bool emplarDecide()
{
   if (mplpot::use_chgpen)
      return false;
   if (use_plmda)
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

   mpoleInit(vers);
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
   emplarBegin(vers);

   emplarState(vers, RdtMask::ENV, mut, true);
   empoleSaveEndpoint0(vers);

   empoleZeroWork(vers);
   emplarState(vers, RdtMask::ALL, mut, false);

   emplarMixEndpoints(vers);
   empoleFinish(vers);
}

void emplar_rdt(int vers)
{
   emplarBegin(vers);

   // E0 = E(B+environment) + E(A).
   emplarState(vers, RdtMask::BE, rdt_group, true);
   emplarState(vers, RdtMask::A, rdt_group, false);
   empoleSaveEndpoint0(vers);

   // E1 = E(A+environment) + E(B).
   empoleZeroWork(vers);
   emplarState(vers, RdtMask::AE, rdt_group, false);
   emplarState(vers, RdtMask::B, rdt_group, false);

   emplarMixEndpoints(vers);
   empoleFinish(vers);

   // Restore the full system for whatever runs next.
   mpoleInitState(calc::v0, RdtMask::ALL, rdt_group, false);
   polarState(RdtMask::ALL, rdt_group);
}
}
