#include "ff/amoeba/empole.h"
#include "ff/dlmda.h"
#include "ff/elec.h"
#include "ff/energy.h"
#include "ff/evdw.h"
#include "ff/hippo/empole.h"
#include "ff/modamoeba.h"
#include "ff/nblist.h"
#include "ff/potent.h"
#include "math/zero.h"
#include "tool/externfunc.h"
#include <tinker/detail/mplpot.hh>
#include <tinker/detail/mutant.hh>

namespace tinker {
static EnergyBuffer em0;
static VirialBuffer vir_em0;
static grad_prec* dem0x;
static grad_prec* dem0y;
static grad_prec* dem0z;

void empoleData(RcOp op)
{
   if (not use(Potent::MPOLE))
      return;
   if (mplpot::use_chgpen)
      return;

   auto rc_a = rc_flag & calc::analyz;
   auto use_private = rc_a || use_emdt;

   if (op & RcOp::DEALLOC) {
      if (rc_a)
         bufferDeallocate(rc_flag, nem);
      if (use_private)
         bufferDeallocate(rc_flag, em, vir_em, demx, demy, demz);
      if (rc_a) {
         bufferDeallocate(rc_flag, demdl_buf, demvirdl_buf, dfmdlx, dfmdly, dfmdlz);
         if (rc_flag & calc::energy)
            darray::deallocate(d2emdl2_buf);
      }
      if (use_emdt)
         bufferDeallocate(rc_flag | calc::analyz, em0, vir_em0, dem0x, dem0y, dem0z);
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
      em0 = nullptr;
      vir_em0 = nullptr;
      dem0x = nullptr;
      dem0y = nullptr;
      dem0z = nullptr;
   }

   if (op & RcOp::ALLOC) {
      nem = nullptr;
      em = eng_buf_elec;
      vir_em = vir_buf_elec;
      demx = gx_elec;
      demy = gy_elec;
      demz = gz_elec;
      if (use_dlmda) {
         demdl_buf = dedl_buf;
         d2emdl2_buf = d2edl2_buf;
         demvirdl_buf = dvirdl_buf;
         dfmdlx = dfsumdlx;
         dfmdly = dfsumdly;
         dfmdlz = dfsumdlz;
      }
      if (rc_a)
         bufferAllocate(rc_flag, &nem);
      if (use_private)
         bufferAllocate(rc_flag | calc::analyz, &em, &vir_em, &demx, &demy, &demz);
      if (rc_a) {
         if (use_dlmda) {
            bufferAllocate(rc_flag, &demdl_buf, &demvirdl_buf, &dfmdlx, &dfmdly, &dfmdlz);
            if (rc_flag & calc::energy)
               darray::allocate(bufferSize(), &d2emdl2_buf);
         }
      }
      if (use_emdt)
         bufferAllocate(rc_flag | calc::analyz, &em0, &vir_em0, &dem0x, &dem0y, &dem0z);
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
static void empoleZeroWork(int vers)
{
   auto do_a = vers & calc::analyz;
   auto do_e = vers & calc::energy;
   auto do_v = vers & calc::virial;
   auto do_g = vers & calc::grad;
   size_t bsize = bufferSize();

   if (do_a)
      darray::zero(g::q0, bsize, nem);
   if (do_e)
      darray::zero(g::q0, bsize, em);
   if (do_v)
      darray::zero(g::q0, bsize, vir_em);
   if (do_g)
      darray::zero(g::q0, n, demx, demy, demz);
}

static void empoleBegin(int vers, bool dual)
{
   auto rc_a = rc_flag & calc::analyz;
   auto do_e = vers & calc::energy;
   auto do_v = vers & calc::virial;
   auto do_g = vers & calc::grad;

   zeroOnHost(energy_em, virial_em);
   if (use_dlmda)
      zeroOnHost(demdl, d2emdl2, demvirdl);
   size_t bsize = bufferSize();
   if (rc_a || dual) {
      empoleZeroWork(vers);
      if (rc_a && use_dlmda) {
         if (do_e)
            darray::zero(g::q0, bsize, demdl_buf, d2emdl2_buf);
         if (do_v)
            darray::zero(g::q0, bsize, demvirdl_buf);
         if (do_g)
            darray::zero(g::q0, n, dfmdlx, dfmdly, dfmdlz);
      }
   }
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

static void empoleFinish(int vers, bool dual)
{
   auto rc_a = rc_flag & calc::analyz;
   auto do_e = vers & calc::energy;
   auto do_v = vers & calc::virial;
   auto do_g = vers & calc::grad;
   size_t bsize = bufferSize();

   if (dual && !rc_a) {
      if (do_e)
         sumEnergyBuffer(bsize, eng_buf_elec, em);
      if (do_v) {
         auto size = bsize * VirialBufferTraits::value;
         sumVirialBuffer(size, vir_buf_elec, vir_em);
      }
      if (do_g)
         sumGradient(gx_elec, gy_elec, gz_elec, demx, demy, demz);
   }

   if (rc_a) {
      if (do_e) {
         energy_prec e = energyReduce(em);
         energy_em += e;
         energy_elec += e;
      }
      if (do_v) {
         virial_prec v[9];
         virialReduce(v, vir_em);
         for (int iv = 0; iv < 9; ++iv) {
            virial_em[iv] += v[iv];
            virial_elec[iv] += v[iv];
         }
      }
      if (do_g)
         sumGradient(gx_elec, gy_elec, gz_elec, demx, demy, demz);

      if (use_dlmda) {
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

void empole(int vers)
{
   auto do_v = vers & calc::virial;

   empoleBegin(vers, false);

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

   empoleFinish(vers, false);
}

static void empoleSaveEndpoint0(int vers)
{
   size_t bsize = bufferSize();
   if (vers & calc::energy)
      darray::copy(g::q0, bsize, em0, em);
   if (vers & calc::virial)
      darray::copy(g::q0, bsize, vir_em0, vir_em);
   if (vers & calc::grad) {
      darray::copy(g::q0, n, dem0x, demx);
      darray::copy(g::q0, n, dem0y, demy);
      darray::copy(g::q0, n, dem0z, demz);
   }
}

static void empoleMixEndpoints(int vers)
{
   double weight1, dweight1, d2weight1;
   adtWeight(mutant::elambda, emdtexp, weight1, dweight1, d2weight1);
   adtMix(vers, use_dlmda, n, bufferSize(), weight1, dweight1, d2weight1, em0, em, demdl_buf,
      d2emdl2_buf, vir_em0, vir_em, demvirdl_buf, dem0x, dem0y, dem0z, demx, demy, demz, dfmdlx,
      dfmdly, dfmdlz);
}

void empole_adt(int vers)
{
   empoleBegin(vers, true);

   empoleState(vers, RdtMask::ENV, mut, true);
   empoleSaveEndpoint0(vers);

   empoleZeroWork(vers);
   empoleState(vers, RdtMask::ALL, mut, false);

   empoleMixEndpoints(vers);
   empoleFinish(vers, true);
}

void empole_rdt(int vers)
{
   empoleBegin(vers, true);

   // E0 = E(B+environment) + E(A).
   empoleState(vers, RdtMask::BE, rdt_group, true);
   empoleState(vers, RdtMask::A, rdt_group, false);
   empoleSaveEndpoint0(vers);

   // E1 = E(A+environment) + E(B).
   empoleZeroWork(vers);
   empoleState(vers, RdtMask::AE, rdt_group, false);
   int bvers = (vers == calc::v3 ? calc::v0 : vers);
   empoleState(bvers, RdtMask::B, rdt_group, false);

   empoleMixEndpoints(vers);
   empoleFinish(vers, true);
}
}
