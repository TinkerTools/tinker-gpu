#include "ff/amoeba/empole.h"
#include "ff/dlmda.h"
#include "ff/elec.h"
#include "ff/energy.h"
#include "ff/hippo/empole.h"
#include "ff/modamoeba.h"
#include "ff/nblist.h"
#include "ff/potent.h"
#include "math/zero.h"
#include "tool/externfunc.h"
#include <tinker/detail/dlmda.hh>
#include <tinker/detail/mplpot.hh>

namespace tinker {
void empoleData(RcOp op)
{
   if (not use(Potent::MPOLE))
      return;
   if (mplpot::use_chgpen)
      return;

   auto rc_a = rc_flag & calc::analyz;

   if (op & RcOp::DEALLOC) {
      if (rc_a) {
         bufferDeallocate(rc_flag, nem);
         bufferDeallocate(rc_flag, em, vir_em, demx, demy, demz);
         bufferDeallocate(rc_flag, demdl_buf, demvirdl_buf, dfmdlx, dfmdly, dfmdlz);
         if (rc_flag & calc::energy)
            darray::deallocate(d2emdl2_buf);
      }
      nem = nullptr;
      em = nullptr;
      vir_em = nullptr;
      demx = nullptr;
      demy = nullptr;
      demz = nullptr;
      demdl_buf = nullptr;
      d2emdl2_buf = nullptr;
      demvirdl_buf = nullptr;
      dfmdlx = nullptr;
      dfmdly = nullptr;
      dfmdlz = nullptr;
   }

   if (op & RcOp::ALLOC) {
      nem = nullptr;
      em = eng_buf_elec;
      vir_em = vir_buf_elec;
      demx = gx_elec;
      demy = gy_elec;
      demz = gz_elec;
      if (dlmda::use_dlmda) {
         demdl_buf = dedl_buf;
         d2emdl2_buf = d2edl2_buf;
         demvirdl_buf = dvirdl_buf;
         dfmdlx = dfsumdlx;
         dfmdly = dfsumdly;
         dfmdlz = dfsumdlz;
      }
      if (rc_a) {
         bufferAllocate(rc_flag, &nem);
         bufferAllocate(rc_flag, &em, &vir_em, &demx, &demy, &demz);
         if (dlmda::use_dlmda) {
            bufferAllocate(rc_flag, &demdl_buf, &demvirdl_buf, &dfmdlx, &dfmdly, &dfmdlz);
            if (rc_flag & calc::energy)
               darray::allocate(bufferSize(), &d2emdl2_buf);
         }
      }
   }

   if (op & RcOp::INIT) {}
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, empoleNonEwald, int);
static void empoleNonEwald(int vers)
{
   TINKER_FCALL2(acc1, cu1, empoleNonEwald, vers);
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, empoleEwaldRealSelf, int);
static void empoleEwaldRealSelf(int vers)
{
   TINKER_FCALL2(acc1, cu1, empoleEwaldRealSelf, vers);
}

TINKER_FVOID2(acc0, cu1, empoleEwaldRecipDlmda, int);
void empoleEwaldRecip(int vers)
{
   if (dlmda::use_dlmda) {
      TINKER_FCALL2(acc0, cu1, empoleEwaldRecipDlmda, vers);
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
void empole(int vers)
{
   auto rc_a = rc_flag & calc::analyz;
   auto do_a = vers & calc::analyz;
   auto do_e = vers & calc::energy;
   auto do_v = vers & calc::virial;
   auto do_g = vers & calc::grad;

   zeroOnHost(energy_em, virial_em);
   if (dlmda::use_dlmda)
      zeroOnHost(demdl, d2emdl2, demvirdl);
   size_t bsize = bufferSize();
   if (rc_a) {
      if (do_a)
         darray::zero(g::q0, bsize, nem);
      if (do_e)
         darray::zero(g::q0, bsize, em);
      if (do_v)
         darray::zero(g::q0, bsize, vir_em);
      if (do_g)
         darray::zero(g::q0, n, demx, demy, demz);
      if (dlmda::use_dlmda) {
         if (do_e)
            darray::zero(g::q0, bsize, demdl_buf, d2emdl2_buf);
         if (do_v)
            darray::zero(g::q0, bsize, demvirdl_buf);
         if (do_g)
            darray::zero(g::q0, n, dfmdlx, dfmdly, dfmdlz);
      }
   }

   mpoleInit(vers);
   if (useEwald())
      empoleEwald(vers);
   else
      empoleNonEwald(vers);
   exfield(vers, 1);
   torque(vers, demx, demy, demz);
   if (dlmda::use_dlmda)
      torque(vers, dfmdlx, dfmdly, dfmdlz, dltrqx, dltrqy, dltrqz, demvirdl_buf);
   if (do_v) {
      VirialBuffer u2 = vir_trq;
      virial_prec v2[9];
      virialReduce(v2, u2);
      for (int iv = 0; iv < 9; ++iv) {
         virial_em[iv] += v2[iv];
         virial_elec[iv] += v2[iv];
      }
   }

   if (rc_a) {
      if (do_e) {
         EnergyBuffer u = em;
         energy_prec e = energyReduce(u);
         energy_em += e;
         energy_elec += e;
      }
      if (do_v) {
         VirialBuffer u1 = vir_em;
         virial_prec v1[9];
         virialReduce(v1, u1);
         for (int iv = 0; iv < 9; ++iv) {
            virial_em[iv] += v1[iv];
            virial_elec[iv] += v1[iv];
         }
      }
      if (do_g)
         sumGradient(gx_elec, gy_elec, gz_elec, demx, demy, demz);

      if (dlmda::use_dlmda) {
         if (do_e) {
            energy_prec e1 = energyReduce(demdl_buf);
            demdl += e1;
            dedl += e1;
            energy_prec e2 = energyReduce(d2emdl2_buf);
            d2emdl2 += e2;
            d2edl2 += e2;
         }
         if (do_v) {
            virial_prec v[9];
            virialReduce(v, demvirdl_buf);
            for (int iv = 0; iv < 9; ++iv) {
               demvirdl[iv] += v[iv];
               dvirdl[iv] += v[iv];
            }
         }
         if (do_g)
            sumGradient(dfsumdlx, dfsumdly, dfsumdlz, dfmdlx, dfmdly, dfmdlz);
      }
   }
}
}
