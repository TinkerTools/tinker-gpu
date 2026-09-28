#include "ff/atom.h"
#include "ff/dlmda.h"
#include "ff/elec.h"
#include "ff/modamoeba.h"
#include "ff/pme.h"
#include "tool/externfunc.h"

#include <limits>

namespace tinker {
TINKER_FVOID2(acc1, cu1, torque, int, grad_prec*, grad_prec*, grad_prec*);
void torque(int vers, grad_prec* dx, grad_prec* dy, grad_prec* dz)
{
   TINKER_FCALL2(acc1, cu1, torque, vers, dx, dy, dz);
}

TINKER_FVOID2(acc1, cu1, torque, int, grad_prec*, grad_prec*, grad_prec*, const real*, const real*,
   const real*, VirialBuffer);
void torque(int vers, grad_prec* dx, grad_prec* dy, grad_prec* dz, const real* tqx, const real* tqy,
   const real* tqz, VirialBuffer vbuf)
{
   TINKER_FCALL2(acc1, cu1, torque, vers, dx, dy, dz, tqx, tqy, tqz, vbuf);
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, chkpole);
static void chkpole()
{
   TINKER_FCALL2(acc1, cu1, chkpole);
}

TINKER_FVOID2(acc1, cu1, rotpole, bool);
static void rotpole(bool use_orig)
{
   TINKER_FCALL2(acc1, cu1, rotpole, use_orig);
}

TINKER_FVOID2(acc0, cu1, rotpoleState, RdtMask, const int*);
static void rotpoleState(RdtMask mask, const int* group)
{
   TINKER_FCALL2(acc0, cu1, rotpoleState, mask, group);
}

// The lambda pole was last scaled to, or NaN when unknown. chkpole inverts pole
// and poleorig together, so it keeps pole the scaled poleorig and leaves this
// valid; only a wholesale rewrite of pole invalidates it.
static double pole_scale = std::numeric_limits<double>::quiet_NaN();

void mpoleScaleInvalidate()
{
   pole_scale = std::numeric_limits<double>::quiet_NaN();
}

// Reinstalls pole only when it is not already scaled to lmda, so polarization
// at a plambda equal to elambda reuses the multipole state (mutate.f:altepset).
TINKER_FVOID2(acc0, cu1, mpoleScale, const int*, GrpScale);
void mpoleScale(double lmda)
{
   if (lmda == pole_scale)
      return;
   TINKER_FCALL2(acc0, cu1, mpoleScale, emGroup(), grpScale(lmda));
   pole_scale = lmda;
}

static void mpoleInitBuffers(int vers, bool use_vir_trq)
{
   if (vers & calc::grad) {
      darray::zero(g::q0, n, trqx, trqy, trqz);
      if (lmdaDerivVers(vers, use_edlmda or use_pdlmda) & (calc::grad_dlmda | calc::virial_dlmda)) {
         darray::zero(g::q0, n, dltrqx, dltrqy, dltrqz);
      }
   }
   if (use_vir_trq && (vers & calc::virial))
      darray::zero(g::q0, bufferSize(), vir_trq);
}

static void mpoleZeroRecipVirial()
{
   if (vir_m)
      darray::zero(g::q0, bufferSize(), vir_m);
}

// With do_dlmda, cmp is lambda scaled, and build_dl also fills dlcmp (d cmp / d
// lambda), which only a version carrying a lambda derivative reads.
static void mpoleInitEwald(bool do_dlmda, bool build_dl, bool prepare_splines, bool prepare_polar_splines)
{
   if (do_dlmda)
      rpoleToCmpDlmda(build_dl);
   else
      rpoleToCmp();
   if (prepare_splines && (pltfm_config & Platform::CUDA)) {
      bool precompute_theta = (!TINKER_CU_THETA_ON_THE_FLY_GRID_MPOLE) || (!TINKER_CU_THETA_ON_THE_FLY_GRID_UIND);
      if (epme_unit.valid()) {
         if (precompute_theta)
            bsplineFill(epme_unit, 3);
      }
      // The lambda derivative grid is spread and gathered with on-the-fly
      // B-splines, so its precomputed ones are only needed if that changes.
      if (do_dlmda && dlpme_unit.valid() && !TINKER_CU_THETA_ON_THE_FLY_GRID_MPOLE)
         bsplineFill(dlpme_unit, 3);
      if (prepare_polar_splines && ppme_unit.valid() && (ppme_unit != epme_unit)) {
         if (precompute_theta)
            bsplineFill(ppme_unit, 2);
      }
      if (prepare_polar_splines && pvpme_unit.valid()) {
         if (precompute_theta)
            bsplineFill(pvpme_unit, 2);
      }
   }
}

void mpoleInit(int vers, bool do_dlmda)
{
   mpoleInitBuffers(vers, true);
   chkpole();
   rotpole(do_dlmda);

   if (useEwald()) {
      constexpr int dlbits = calc::energy_dlmda1 | calc::energy_dlmda2 | calc::grad_dlmda | calc::virial_dlmda;
      const bool build_dl = do_dlmda and (lmdaDerivVers(vers, do_dlmda) & dlbits);
      mpoleZeroRecipVirial();
      mpoleInitEwald(do_dlmda, build_dl, true, true);
   }
}

// The B-splines were already filled by the mpoleInit() that precedes this
// call for the same coordinates.
void mpoleInitAst()
{
   rotpole(true);
   if (useEwald())
      mpoleInitEwald(true, true, false, false);
}

// A dual topology driver accumulates torque over every subsystem and converts it
// once at the end, so the torque buffers are cleared on the first pass only.
void mpoleInitStateDt(int vers, RdtMask mask, const int* group, bool first_state)
{
   if (first_state) {
      mpoleInitBuffers(vers, false);
      chkpole();
   }
   rotpoleState(mask, group);
   if (useEwald()) {
      mpoleZeroRecipVirial();
      mpoleInitEwald(false, false, first_state, first_state);
   }
}

// Rotates the lambda scaled pole into rpole and leaves cmp alone, for callers
// whose cmp already holds the same scaled multipoles.
void mpoleRotateScaled()
{
   rotpole(false);
}

void mpoleRefresh()
{
   rotpole(false);
   if (useEwald())
      mpoleInitEwald(false, false, false, false);
}

// Undoes the masking a dual topology run leaves behind, so whatever runs next
// sees the whole system again. rpole is rebuilt from the full set of atoms, and
// with it cmp, which the next term's reciprocal space work reads.
void mpoleRestoreFullState(const int* group)
{
   rotpoleState(RdtMask::ALL, group);
   if (useEwald())
      mpoleInitEwald(false, false, false, false);
}
}
