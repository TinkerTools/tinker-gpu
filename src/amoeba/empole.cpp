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
#include <tinker/detail/extfld.hh>
#include <tinker/detail/mplpot.hh>
#include <tinker/detail/mutant.hh>

namespace tinker {
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
      if (rc_a)
         darray::deallocate(demdl_buf, d2emdl2_buf);
      demdl_buf = nullptr;
      d2emdl2_buf = nullptr;
      nem = nullptr;
   }

   if (op & RcOp::ALLOC) {
      nem = nullptr;
      em_buf.manage(op, rc_flag, {&em, &vir_em, &demx, &demy, &demz},
         {eng_buf_elec, vir_buf_elec, gx_elec, gy_elec, gz_elec}, rc_a, //
         {&energy_em, &virial_em}, {&energy_elec, &virial_elec});

      const int dlmask = lmdaDerivMask(rc_flag, use_edlmda);
      demdl_buf = nullptr;
      d2emdl2_buf = nullptr;
      if (dlmask & calc::energy_dlmda1) {
         if (rc_a)
            darray::allocate(bufferSize(), &demdl_buf);
         else
            demdl_buf = dedl_buf;
      }
      if (dlmask & calc::energy_dlmda2) {
         if (rc_a)
            darray::allocate(bufferSize(), &d2emdl2_buf);
         else
            d2emdl2_buf = d2edl2_buf;
      }

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

TINKER_FVOID2(acc0, cu1, empoleEwaldRecipDlmda, int);
void empoleEwaldRecip(int vers)
{
   if (use_emast) {
      TINKER_FCALL2(acc0, cu1, empoleEwaldRecipDlmda, lmdaDerivVers(vers, use_emast));
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
   const int dlmask = lmdaDerivMask(vers, use_edlmda);
   zeroOnHost(energy_em, virial_em);
   empoleZeroWork(vers);
   if (rc_flag & calc::analyz) {
      if (dlmask & calc::energy_dlmda1)
         darray::zero(g::q0, bufferSize(), demdl_buf);
      if (dlmask & calc::energy_dlmda2)
         darray::zero(g::q0, bufferSize(), d2emdl2_buf);
   }
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
   const int dlmask = lmdaDerivMask(vers, use_edlmda);
   em_buf.flush(vers);
   if (rc_flag & calc::analyz) {
      if (dlmask & calc::energy_dlmda1) {
         energy_prec e = energyReduce(demdl_buf);
         demdl += e;
         dedl += e;
      }
      if (dlmask & calc::energy_dlmda2) {
         energy_prec e = energyReduce(d2emdl2_buf);
         d2emdl2 += e;
         d2edl2 += e;
      }
   }
}

void empole(int vers)
{
   auto do_v = vers & calc::virial;

   empoleBegin(vers);

   mpoleInit(vers, use_emast);
   empoleKernel(vers);
   exfield(vers, 1);
   torque(vers, demx, demy, demz);
   if (lmdaDerivMask(vers, use_emast) & (calc::grad_dlmda | calc::virial_dlmda))
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


TINKER_FVOID2(acc0, cu1, empoleDt, int, const DtCoef&, int);
void empoleDt(int vers, const DtCoef& coef, int nself)
{
   TINKER_FCALL2(acc0, cu1, empoleDt, vers, coef, nself);
}

TINKER_FVOID2(acc0, cu1, empoleEwaldRecipDt, int, RdtMask, real, real, real);
void empoleEwaldRecipDt(int vers, RdtMask mask, real wa, real wb, real wc)
{
   TINKER_FCALL2(acc0, cu1, empoleEwaldRecipDt, vers, mask, wa, wb, wc);
}

TINKER_FVOID2(acc0, cu1, exfieldDipoleDt, int, const DtCoef&);
static void exfieldDt(int vers, const DtCoef& coef)
{
   if (not extfld::use_exfld)
      return;
   TINKER_FCALL2(acc0, cu1, exfieldDipoleDt, vers, coef);
}

static DtCoef empoleDtCoefAdt(double w, double dw, double d2w)
{
   DtCoef c;
   dtWeightsToCoef(c, w, dw, d2w, deldlmda, d2eldlmda2, use_edlmda);
   c.in0bits = 0;
   c.in1bits = 0;
   dtCellSet(c.in0bits, 0, 0);
   for (int gi = 0; gi < 2; ++gi)
      for (int gk = gi; gk < 2; ++gk)
         dtCellSet(c.in1bits, gi, gk);
   c.cntbits = c.in1bits;
   return c;
}

// Relative dual topology. Groups are the ternary rdt_group labels.
static DtCoef empoleDtCoefRdt(double w, double dw, double d2w, bool need1)
{
   DtCoef c;
   dtWeightsToCoef(c, w, dw, d2w, deldlmda, d2eldlmda2, use_edlmda);
   c.in0bits = dtStateBits(emrelst0);
   c.in1bits = dtStateBits(emrelst1);
   c.cntbits = dtCountBits(need1 ? emrelst1 : emrelst0);
   return c;
}

static void empoleRecipDt(int vers, const DtCoef& c)
{
   DtPass pass[nRelSlot];
   const int npass = dtPassList(use_emrdt, emrelst0, emrelst1, pass);

   for (int k = 0; k < npass; ++k) {
      real wa, wb, wc;
      dtPassWeights(c, pass[k], wa, wb, wc);
      if (wa == 0 and wb == 0 and wc == 0)
         continue;
      empoleEwaldRecipDt(vers, pass[k].mask, wa, wb, wc);
   }
}

void empole_dt(int vers)
{
   const int dvers = lmdaDerivVers(vers, use_edlmda);
   auto do_g = vers & calc::grad;

   double w, dw, d2w;
   bool need0, need1;
   dtWeightNeed(elam, emdtexp, deldlmda, d2eldlmda2, w, dw, d2w, need0, need1);

   empoleBegin(vers);

   mpoleInitDt(vers);

   const DtCoef coef =
      use_emrdt ? empoleDtCoefRdt(w, dw, d2w, need1) : empoleDtCoefAdt(w, dw, d2w);

   const int nself = use_emrdt ? dtCountSlots(need1 ? emrelst1 : emrelst0) : 1;

   empoleDt(dvers, coef, nself);
   if (useEwald())
      empoleRecipDt(dvers, coef);
   exfieldDt(dvers, coef);

   if (do_g) {
      torque(vers, demx, demy, demz, trqx, trqy, trqz, vir_em);
      if (dvers & (calc::grad_dlmda | calc::virial_dlmda))
         torque(vers, dfdlx, dfdly, dfdlz, dltrqx, dltrqy, dltrqz, dvirdl_buf);
   }

   empoleFinish(vers);
}
}
