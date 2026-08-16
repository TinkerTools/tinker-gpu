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
#include "tool/error.h"
#include "tool/externfunc.h"
#include "tool/platform.h"
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
      em_dl.manage(op, rc_flag, {}, {}, false);
      em_snap.manage(op, rc_flag, false);
      nem = nullptr;
   }

   if (op & RcOp::ALLOC) {
      nem = nullptr;
      em_buf.manage(op, rc_flag, {&em, &vir_em, &demx, &demy, &demz},
         {eng_buf_elec, vir_buf_elec, gx_elec, gy_elec, gz_elec}, rc_a or use_emdt, //
         {&energy_em, &virial_em}, {&energy_elec, &virial_elec});
      em_dl.manage(op, rc_flag, {&demdl_buf, &demvirdl_buf, &dfmdlx, &dfmdly, &dfmdlz, &d2emdl2_buf},
         {dedl_buf, dvirdl_buf, dfsumdlx, dfsumdly, dfsumdlz, d2edl2_buf},
         use_edlmda, //
         {&demdl, &demvirdl, &d2emdl2}, {&dedl, &dvirdl, &d2edl2}, use_edlmda);
      em_snap.manage(op, rc_flag, use_emdt);
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
   if (use_emast) {
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
   if (use_edlmda)
      zeroOnHost(demdl, d2emdl2, demvirdl);
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

static void empoleState(int vers, RdtMask mask, const int* group, bool prepare_splines)
{
   mpoleInitState(vers, mask, group, prepare_splines);
   empoleKernel(vers);
   exfield(vers, 1);
   torque(vers, demx, demy, demz, trqx, trqy, trqz, vir_em);
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
   if (use_emast)
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

   empoleFinish(vers);
}

void empoleSaveEndpoint0(int vers)
{
   em_snap.save(vers, em_buf);
}

void empoleMixEndpoints(int vers)
{
   em_snap.mix(vers, elam, emdtexp, use_edlmda, em_buf, em_dl);
}

void empole_adt(int vers)
{
   double w, dw, d2w;
   bool need0, need1;
   dtWeightNeed(elam, emdtexp, deldlmda, d2eldlmda2, w, dw, d2w, need0, need1);

   empoleBegin(vers);

   // Analysis reports the interaction count of the fully coupled endpoint even
   // when that endpoint carries no weight, so it is built for the count alone
   // and its energy discarded (empole3.f:2419-2427). em_buf.zero() rather than
   // empoleZeroWork(), which would clear the count this exists to keep.
   int wvers = vers;
   bool first = true;
   if ((vers & calc::analyz) and not need1) {
      empoleState(vers, RdtMask::ALL, mut, true);
      em_buf.zero(vers);
      wvers = vers & ~calc::analyz;
      first = false;
   }

   if (need0) {
      empoleState(wvers, RdtMask::ENV, mut, first);
      first = false;
      empoleSaveEndpoint0(wvers);
   }
   if (need1) {
      if (need0)
         empoleZeroWork(wvers);
      empoleState(wvers, RdtMask::ALL, mut, first);
      if (not need0)
         empoleSaveEndpoint0(wvers);
   }

   empoleMixEndpoints(vers);
   empoleFinish(vers);
}

void empole_rdt(int vers)
{
   double w, dw, d2w;
   bool need0, need1;
   dtWeightNeed(elam, emdtexp, deldlmda, d2eldlmda2, w, dw, d2w, need0, need1);

   empoleBegin(vers);

   const RelDualOps ops = {
      [](int v, RdtMask mask, bool first) { empoleState(v, mask, rdt_group, first); },
      [](int v) { empoleZeroWork(v); },
      [](int v) { empoleSaveEndpoint0(v); },
      [](int v) { empoleMixEndpoints(v); },
   };
   relDualDrive(vers, emrelst0, emrelst1, need0, need1, ops);

   empoleFinish(vers);

   mpoleInitState(calc::v0, RdtMask::ALL, rdt_group, false);
}
}
