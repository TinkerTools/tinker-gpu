#include "ff/atom.h"
#include "ff/dlmda.h"
#include "ff/elec.h"
#include "ff/modamoeba.h"
#include "ff/pme.h"
#include "tool/externfunc.h"

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

TINKER_FVOID2(acc0, cu1, mpoleScale, real);
void mpoleScale(real factor)
{
   TINKER_FCALL2(acc0, cu1, mpoleScale, factor);
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

static void mpoleInitEwald(bool do_dlmda, bool prepare_splines, bool prepare_polar_splines, bool build_cmp = true)
{
   if (build_cmp) {
      if (do_dlmda)
         rpoleToCmpDlmda(); // fills both cmp (lambda scaled) and dlcmp (d cmp / d lambda)
      else
         rpoleToCmp();
   }
   if (prepare_splines && (pltfm_config & Platform::CUDA)) {
      bool precompute_theta = (!TINKER_CU_THETA_ON_THE_FLY_GRID_MPOLE) || (!TINKER_CU_THETA_ON_THE_FLY_GRID_UIND);
      if (epme_unit.valid()) {
         if (precompute_theta)
            bsplineFill(epme_unit, 3);
      }
      if (do_dlmda && dlpme_unit.valid()) {
         if (precompute_theta)
            bsplineFill(dlpme_unit, 3);
      }
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
      mpoleZeroRecipVirial();
      mpoleInitEwald(do_dlmda, true, true);
   }
}

void mpoleInitDt(int vers)
{
   mpoleInitBuffers(vers, false);
   chkpole();
   rotpole(true);
   if (useEwald()) {
      mpoleZeroRecipVirial();
      mpoleInitEwald(false, true, false, false);
   }
}

void mpoleInitAst()
{
   rotpole(true);
   if (useEwald())
      mpoleInitEwald(true, true, false);
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
      mpoleInitEwald(false, first_state, first_state);
   }
}

void mpoleRefresh()
{
   rotpole(false);
   if (useEwald())
      mpoleInitEwald(false, false, false);
}

// Undoes the masking a dual topology run leaves behind, so whatever runs next
// sees the whole system again. rpole is rebuilt from the full set of atoms, and
// with it cmp, which the next term's reciprocal space work reads.
void mpoleRestoreFullState(const int* group)
{
   rotpoleState(RdtMask::ALL, group);
   if (useEwald())
      mpoleInitEwald(false, false, false);
}
}
