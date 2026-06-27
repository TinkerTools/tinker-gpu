#include "ff/atom.h"
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
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, chkpole);
static void chkpole()
{
   TINKER_FCALL2(acc1, cu1, chkpole);
}

TINKER_FVOID2(acc1, cu1, rotpole);
static void rotpole()
{
   TINKER_FCALL2(acc1, cu1, rotpole);
}

void mpoleInit(int vers)
{
   if (vers & calc::grad)
      darray::zero(g::q0, n, trqx, trqy, trqz);
   if (vers & calc::virial)
      darray::zero(g::q0, bufferSize(), vir_trq);

   chkpole();
   rotpole();

   if (useEwald()) {
      rpoleToCmp();
      if (vir_m)
         darray::zero(g::q0, bufferSize(), vir_m);
      if (pltfm_config & Platform::CUDA) {
         bool precompute_theta = (!TINKER_CU_THETA_ON_THE_FLY_GRID_MPOLE) || (!TINKER_CU_THETA_ON_THE_FLY_GRID_UIND);
         if (epme_unit.valid()) {
            if (precompute_theta)
               bsplineFill(epme_unit, 3);
         }
         if (ppme_unit.valid() && (ppme_unit != epme_unit)) {
            if (precompute_theta)
               bsplineFill(ppme_unit, 2);
         }
         if (pvpme_unit.valid()) {
            if (precompute_theta)
               bsplineFill(pvpme_unit, 2);
         }
      }
   }
}
}
