#include "ff/amoeba/empole.h"
#include "ff/elec.h"
#include "ff/energy.h"
#include "ff/hippo/cflux.h"
#include "ff/hippo/empole.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
#include "ff/potent.h"
#include "math/zero.h"
#include "tool/externfunc.h"
#include <tinker/detail/mplpot.hh>

namespace tinker {
void empoleChgpenData(RcOp op)
{
   if (not use(Potent::MPOLE) and not use(Potent::CHGTRN))
      return;
   if (not mplpot::use_chgpen)
      return;

   auto rc_a = rc_flag & calc::analyz;

   if (op & RcOp::DEALLOC) {
      if (rc_a) {
         bufferDeallocate(rc_flag, nem);
         bufferDeallocate(rc_flag, em, vir_em, demx, demy, demz);
      }
      nem = nullptr;
      em = nullptr;
      vir_em = nullptr;
      demx = nullptr;
      demy = nullptr;
      demz = nullptr;
   }

   if (op & RcOp::ALLOC) {
      nem = nullptr;
      em = eng_buf_elec;
      vir_em = vir_buf_elec;
      demx = gx_elec;
      demy = gy_elec;
      demz = gz_elec;

      if (rc_a) {
         bufferAllocate(rc_flag, &nem);
         bufferAllocate(rc_flag, &em, &vir_em, &demx, &demy, &demz);
      }
   }

   if (op & RcOp::INIT) {}
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, empoleChgpenNonEwald, int, int);
static void empoleChgpenNonEwald(int vers, int use_cf)
{
   TINKER_FCALL2(acc1, cu1, empoleChgpenNonEwald, vers, use_cf);
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, empoleChgpenEwaldRealSelf, int, int);
static void empoleChgpenEwaldRealSelf(int vers, int use_cf)
{
   TINKER_FCALL2(acc1, cu1, empoleChgpenEwaldRealSelf, vers, use_cf);
}

TINKER_FVOID2(acc1, cu1, empoleChgpenEwaldRecip, int, int);
void empoleChgpenEwaldRecip(int vers, int use_cf)
{
   TINKER_FCALL2(acc1, cu1, empoleChgpenEwaldRecip, vers, use_cf);
}

static void empoleChgpenEwald(int vers, int use_cf)
{
   empoleChgpenEwaldRealSelf(vers, use_cf);
   empoleChgpenEwaldRecip(vers, use_cf);
}
}

namespace tinker {
void empoleChgpen(int vers)
{
   auto rc_a = rc_flag & calc::analyz;
   auto do_a = vers & calc::analyz;
   auto do_e = vers & calc::energy;
   auto do_v = vers & calc::virial;
   auto do_g = vers & calc::grad;
   auto use_cf = use(Potent::CHGFLX);
   int use_cfgrad = use_cf and do_g;

   zeroOnHost(energy_em, virial_em);
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
   }

   if (use_cf)
      alterchg();
   mpoleInit(vers);
   if (use_cfgrad) {
      cfluxZeroPot();
   }
   if (useEwald()) {
      if (pentyp == Chgpen::GORDON1)
         empoleChgpenEwald(vers, use_cfgrad);
      else if (pentyp == Chgpen::GORDON2)
         empoleAplusEwald(vers, use_cfgrad);
   } else {
      if (pentyp == Chgpen::GORDON1)
         empoleChgpenNonEwald(vers, use_cfgrad);
      else if (pentyp == Chgpen::GORDON2)
         empoleAplusNonEwald(vers, use_cfgrad);
   }
   exfield(vers, 1);
   torque(vers, demx, demy, demz);
   if (use_cfgrad)
      dcflux(vers, demx, demy, demz, vir_em);
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
   }
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, empoleAplusEwaldRealSelf, int, int);
static void empoleAplusEwaldRealSelf(int vers, int useCF)
{
   TINKER_FCALL2(acc1, cu1, empoleAplusEwaldRealSelf, vers, useCF);
}

static void empoleAplusEwaldRecip(int vers, int useCF)
{
   empoleChgpenEwaldRecip(vers, useCF);
}

void empoleAplusEwald(int vers, int useCF)
{
   empoleAplusEwaldRealSelf(vers, useCF);
   empoleAplusEwaldRecip(vers, useCF);
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, empoleAplusNonEwald, int, int);
void empoleAplusNonEwald(int vers, int useCF)
{
   TINKER_FCALL2(acc1, cu1, empoleAplusNonEwald, vers, useCF);
}
}
