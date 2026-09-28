#include "ff/amoeba/empole.h"
#include "ff/dlmda.h"
#include "ff/elec.h"
#include "ff/energy.h"
#include "ff/evdw.h"
#include "ff/hippo/empole.h"
#include "ff/modamoeba.h"
#include "ff/nblist.h"
#include "ff/ost.h"
#include "ff/potent.h"
#include "ff/termbuf.h"
#include "math/zero.h"
#include "tool/darray.h"
#include "tool/error.h"
#include "tool/externfunc.h"
#include "tool/platform.h"
#include <tinker/detail/mplpot.hh>

namespace tinker {
static LmdaBuffer em_dl;

void empoleData(RcOp op)
{
   if (not use(Potent::MPOLE))
      return;
   if (mplpot::use_chgpen)
      return;

   auto rc_a = rc_flag & calc::analyz;

   if (op & RcOp::DEALLOC) {
      if (rc_a)
         bufferDeallocate(rc_flag, nem);
      em_buf.manage(op, rc_flag, {}, {}, false);
      em_dl.manage(op, rc_flag, use_edlmda, &demdl_buf, &d2emdl2_buf, &demdl, &d2emdl2);
      nem = nullptr;
   }

   if (op & RcOp::ALLOC) {
      nem = nullptr;
      em_buf.manage(op, rc_flag, {&em, &vir_em, &demx, &demy, &demz},
         {eng_buf_elec, vir_buf_elec, gx_elec, gy_elec, gz_elec}, rc_a, //
         {&energy_em, &virial_em}, {&energy_elec, &virial_elec});

      em_dl.manage(op, rc_flag, use_edlmda, &demdl_buf, &d2emdl2_buf, &demdl, &d2emdl2);

      if (rc_a)
         bufferAllocate(rc_flag, &nem);
   }

   if (op & RcOp::INIT) {}
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, empoleNonEwald, int);
static void empoleNonEwald(int vers)
{
   TINKER_FCALL2(acc1, cu1, empoleNonEwald, lmdaDerivVers(vers, use_emast));
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, empoleEwaldRealSelf, int);
static void empoleEwaldRealSelf(int vers)
{
   TINKER_FCALL2(acc1, cu1, empoleEwaldRealSelf, lmdaDerivVers(vers, use_emast));
}

TINKER_FVOID2(acc0, cu1, empoleEwaldRecipDlmda, int, bool);
TINKER_FVOID2(acc0, cu1, empoleEwaldRecipReuse, int);
void empoleEwaldRecip(int vers, bool reuse_pot)
{
   if (use_emast) {
      TINKER_FCALL2(acc0, cu1, empoleEwaldRecipDlmda, lmdaDerivVers(vers, use_emast), reuse_pot);
      return;
   }
   if (reuse_pot) {
      TINKER_FCALL2(acc0, cu1, empoleEwaldRecipReuse, vers);
      return;
   }
   int use_cf = 0;
   empoleChgpenEwaldRecip(vers, use_cf);
}

static void empoleEwald(int vers)
{
   empoleEwaldRealSelf(vers);
   empoleEwaldRecip(vers);
}
}

namespace tinker {
void empoleZeroWork(int vers)
{
   auto rc_a = rc_flag & calc::analyz;
   auto do_a = vers & calc::analyz;
   if (rc_a and do_a)
      darray::zero(g::q0, bufferSize(), nem);
   em_buf.zero(vers);
}

void empoleBegin(int vers)
{
   zeroOnHost(energy_em, virial_em);
   empoleZeroWork(vers);
   em_dl.zero(vers);
}

static void empoleKernel(int vers)
{
   if (useEwald())
      empoleEwald(vers);
   else
      empoleNonEwald(vers);
}

void empoleFinish(int vers)
{
   em_buf.flush(vers);
   em_dl.flush(vers);
}

void empole(int vers)
{
   auto do_v = vers & calc::virial;

   empoleBegin(vers);

   mpoleInit(vers, use_emast);
   empoleKernel(vers);
   exfield(vers, 1);
   torque(vers, demx, demy, demz);
   if (lmdaDerivVers(vers, use_emast) & (calc::grad_dlmda | calc::virial_dlmda))
      torque(vers, dfdlx, dfdly, dfdlz, dltrqx, dltrqy, dltrqz, dvirdl_buf);
   if (do_v) {
      VirialBuffer u2 = vir_trq;
      virial_prec v2[9];
      virialReduce(v2, u2);
      for (int iv = 0; iv < 9; ++iv) {
         virial_em[iv] += v2[iv];
         virial_elec[iv] += v2[iv];
      }
   }

   empoleFinish(vers);
}
}
