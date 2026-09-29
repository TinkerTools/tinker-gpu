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
#include <tinker/detail/mplpot.hh>

#include <cassert>

namespace tinker {
static int emplar_flag = -1;

static bool emplarDecide()
{
   if (mplpot::use_chgpen)
      return false;
   if (use_prst and not polTracksEle())
      return false;
   if (use_emast and not use_epast)
      return false;
   if (rc_flag & calc::analyz)
      return false;
   if (not(use(Potent::MPOLE) and use(Potent::POLAR)))
      return false;
   if (not(mlistVersion() & Nbl::SPATIAL))
      return false;

   // The fused kernel has no dual topology form, and the multipoles never
   // use dual topology, so dual topology polarization always runs apart.
   if (use_epdt)
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

   mpoleEnsureElec();
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
   if (not lmdaSameValue(elam, plam))
      TINKER_THROW("The electrostatic and polarization lambda values have drifted apart; "
                   "the fused multipole/polarization single topology needs them to be equal.");

   const int dvers = lmdaDerivVers(vers, use_edlmda);
   auto do_v = vers & calc::virial;
   // The solve sees multipoles scaled by plam and the permanent terms by elam;
   // only when the two are the same number is the solve's multipole potential
   // theirs too, and the scaled state left behind the one to restore.
   const bool same_state = (elam == plam);

   zeroOnHost(energy_em, virial_em);
   zeroOnHost(energy_ep, virial_ep);

   // Lambda scaled state, and the solve.
   mpoleScale(plam);
   polarState(coupledMask(), emGroup(), plam);
   mpoleInit(vers, false);
   induce(uind, uinp);

   // Unscaled state: the permanent multipole terms and their lambda derivative.
   // empoleEwaldRecip and exfield both decorate the version themselves once
   // use_emast is set, so they take the undecorated one.
   mpoleUseOrig(true);
   emplarAstKernel(dvers);
   if (useEwald())
      empoleEwaldRecip(vers, same_state);
   exfield(vers, 1);

   // Back to the lambda scaled state for the polarization reciprocal term, which
   // reads the permanent dipole straight out of rpole. Its version stays
   // undecorated too, but for the opposite reason: it has no lambda derivative
   // channel, and a version it does not recognize leaves it doing nothing at all.
   // cmp is already the scaled one when the two lambdas agree, so only rpole is
   // rotated back.
   if (same_state)
      mpoleRotateScaled();
   else
      mpoleUsePole();
   if (useEwald()) {
      const AccumRef out = em_buf.ref();
      epolarEwaldRecipSelf(vers & ~calc::energy, out.e, out.v, out.gx, out.gy, out.gz);
   }
   if (vers & calc::energy)
      epolar0DotProd(uind, udirp, em_buf.ref().e);

   const bool do_astdl = lmdaDerivVers(vers, use_pdlmda) & calc::energy_dlmda1;
   if (do_astdl)
      epolarAstDeriv();

   torque(vers, demx, demy, demz);
   if (do_v) {
      VirialBuffer u2 = vir_trq;
      virial_prec v2[9];
      virialReduce(v2, u2);
      for (int iv = 0; iv < 9; ++iv)
         virial_elec[iv] += v2[iv];
   }
}
}
