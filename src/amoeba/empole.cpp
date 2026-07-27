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
         (rc_a or useLmdaChain()) and use_dlmda, //
         {&demdl, &demvirdl, &d2emdl2}, {&dedl, &dvirdl, &d2edl2}, useLmdaChain());
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
   if (use_dlmda)
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

   mpoleInit(vers);
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
   em_snap.mix(vers, elam, emdtexp, use_dlmda, em_buf, em_dl);
}

void empole_adt(int vers)
{
   empoleBegin(vers);

   if (use_ele4i) {
      empoleState(vers, RdtMask::ENV, mut, true);
      empoleSaveEndpoint0(vers);
   }
   if (use_ele4f) {
      if (use_ele4i)
         empoleZeroWork(vers);
      empoleState(vers, RdtMask::ALL, mut, not use_ele4i);
   }
   if (not use_ele4i)
      empoleSaveEndpoint0(vers);

   empoleMixEndpoints(vers);
   empoleFinish(vers);
}

void empole_rdt(int vers)
{
   empoleBegin(vers);

   // E0 = E(B+environment) + E(A).
   if (use_ele4i) {
      empoleState(vers, RdtMask::BE, rdt_group, true);
      empoleState(vers, RdtMask::A, rdt_group, false);
      empoleSaveEndpoint0(vers);
   }
   // E1 = E(A+environment) + E(B).
   if (use_ele4f) {
      if (use_ele4i)
         empoleZeroWork(vers);
      empoleState(vers, RdtMask::AE, rdt_group, not use_ele4i);
      int bvers = (vers == calc::v3 ? calc::v0 : vers);
      empoleState(bvers, RdtMask::B, rdt_group, false);
   }
   if (not use_ele4i)
      empoleSaveEndpoint0(vers);

   empoleMixEndpoints(vers);
   empoleFinish(vers);

   mpoleInitState(calc::v0, RdtMask::ALL, rdt_group, false);
}
}
