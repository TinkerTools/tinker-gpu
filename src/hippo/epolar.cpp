#include "ff/amoeba/epolar.h"
#include "ff/amoeba/empole.h"
#include "ff/modamoeba.h"
#include "ff/elec.h"
#include "ff/energy.h"
#include "ff/hippo/cflux.h"
#include "ff/hippo/expol.h"
#include "ff/hippo/induce.h"
#include "ff/potent.h"
#include "math/zero.h"
#include "tool/externfunc.h"
#include "tool/iofortstr.h"
#include <tinker/detail/mplpot.hh>
#include <tinker/detail/polar.hh>
#include <tinker/detail/polpot.hh>
#include <tinker/detail/uprior.hh>

namespace tinker {
void epolarChgpenData(RcOp op)
{
   if (not use(Potent::POLAR))
      return;
   if (polpot::use_tholed) // AMOEBA Plus
      return;
   if (not mplpot::use_chgpen)
      return;

   auto rc_a = rc_flag & calc::analyz;

   if (op & RcOp::DEALLOC) {
      darray::deallocate(polarity, polarity_inv);

      if (rc_a) {
         bufferDeallocate(rc_flag, nep);
         bufferDeallocate(rc_flag, ep, vir_ep, depx, depy, depz);
      }
      nep = nullptr;
      ep = nullptr;
      vir_ep = nullptr;
      depx = nullptr;
      depy = nullptr;
      depz = nullptr;

      darray::deallocate(ufld, dufld);
      darray::deallocate(work01_, work02_, work03_, work04_, work05_);

      if (polpred == UPred::ASPC) {
         darray::deallocate(udalt_00, udalt_01, udalt_02, udalt_03, udalt_04, udalt_05, udalt_06, udalt_07, udalt_08,
            udalt_09, udalt_10, udalt_11, udalt_12, udalt_13, udalt_14, udalt_15);
      } else if (polpred == UPred::GEAR) {
         darray::deallocate(udalt_00, udalt_01, udalt_02, udalt_03, udalt_04, udalt_05);
      } else if (polpred == UPred::LSQR) {
         darray::deallocate(udalt_00, udalt_01, udalt_02, udalt_03, udalt_04, udalt_05, udalt_06);
         darray::deallocate(udalt_lsqr_a, udalt_lsqr_b);
      }
      polpred = UPred::NONE;
      maxualt = 0;
      nualt = 0;
   }

   if (op & RcOp::ALLOC) {
      darray::allocate(n, &polarity, &polarity_inv);

      nep = nullptr;
      ep = eng_buf_elec;
      vir_ep = vir_buf_elec;
      depx = gx_elec;
      depy = gy_elec;
      depz = gz_elec;
      if (rc_a) {
         bufferAllocate(rc_flag, &nep);
         bufferAllocate(rc_flag, &ep, &vir_ep, &depx, &depy, &depz);
      }

      if (rc_flag & calc::grad) {
         darray::allocate(n, &ufld, &dufld);
      } else {
         ufld = nullptr;
         dufld = nullptr;
      }

      darray::allocate(n, &work01_, &work02_, &work03_, &work04_, &work05_);
      if (uprior::use_pred) {
         FstrView predstr = uprior::polpred;
         if (predstr == "ASPC") {
            polpred = UPred::ASPC;
         } else if (predstr == "GEAR") {
            polpred = UPred::GEAR;
         } else {
            polpred = UPred::LSQR;
         }
      } else {
         polpred = UPred::NONE;
      }
      maxualt = 0;
      nualt = 0;

      if (polpred == UPred::ASPC) {
         maxualt = 16;
         darray::allocate(n, &udalt_00, &udalt_01, &udalt_02, &udalt_03, &udalt_04, &udalt_05, &udalt_06, &udalt_07,
            &udalt_08, &udalt_09, &udalt_10, &udalt_11, &udalt_12, &udalt_13, &udalt_14, &udalt_15);
         darray::zero(g::q0, n, udalt_00, udalt_01, udalt_02, udalt_03, udalt_04, udalt_05, udalt_06, udalt_07,
            udalt_08, udalt_09, udalt_10, udalt_11, udalt_12, udalt_13, udalt_14, udalt_15);
      } else if (polpred == UPred::GEAR) {
         maxualt = 6;
         darray::allocate(n, &udalt_00, &udalt_01, &udalt_02, &udalt_03, &udalt_04, &udalt_05);
         darray::zero(g::q0, n, udalt_00, udalt_01, udalt_02, udalt_03, udalt_04, udalt_05);
      } else if (polpred == UPred::LSQR) {
         maxualt = 7;
         darray::allocate(n, &udalt_00, &udalt_01, &udalt_02, &udalt_03, &udalt_04, &udalt_05, &udalt_06);
         int lenb = maxualt - 1;
         int lena = lenb * lenb; // lenb*(lenb+1)/2 should be plenty.
         darray::allocate(lena, &udalt_lsqr_a);
         darray::allocate(lenb, &udalt_lsqr_b);
         darray::zero(g::q0, n, udalt_00, udalt_01, udalt_02, udalt_03, udalt_04, udalt_05, udalt_06);
      }
   }

   if (op & RcOp::INIT) {
      udiag = polpot::uaccel;

      const double polmin = 1.0e-16;
      std::vector<double> pinvbuf(n);
      for (int i = 0; i < n; ++i) {
         pinvbuf[i] = 1.0 / std::max(polar::polarity[i], polmin);
      }
      darray::copyin(g::q0, n, polarity, polar::polarity);
      darray::copyin(g::q0, n, polarity_inv, pinvbuf.data());
      waitFor(g::q0);
   }
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, epolarChgpenNonEwald, int, int, const real (*)[3]);
static void epolarChgpenNonEwald(int vers, int use_cf)
{
   // v0: E_dot
   // v1: EGV = E_dot + GV
   // v3: EA = E_pair + A
   // v4: EG = E_dot + G
   // v5: G
   // v6: GV
   auto edot = vers & calc::energy; // if not do_e, edot = false
   if (vers & calc::energy && vers & calc::analyz)
      edot = 0; // if do_e and do_a, edot = false
   int ver2 = vers;
   if (edot)
      ver2 &= ~calc::energy; // toggle off the calc::energy flag

   induce2(uind);
   if (edot)
      epolar0DotProd(uind, udir);
   if (vers != calc::v0)
      TINKER_FCALL2(acc1, cu1, epolarChgpenNonEwald, ver2, use_cf, uind);
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, epolarChgpenEwaldReal, int, int, const real (*)[3]);
static void epolarChgpenEwaldReal(int vers, int use_cf)
{
   TINKER_FCALL2(acc1, cu1, epolarChgpenEwaldReal, vers, use_cf, uind);
}

TINKER_FVOID2(acc1, cu1, epolarChgpenEwaldRecipSelf, int, int, const real (*)[3]);
void epolarChgpenEwaldRecipSelf(int vers, int use_cf)
{
   TINKER_FCALL2(acc1, cu1, epolarChgpenEwaldRecipSelf, vers, use_cf, uind);
}

static void epolarChgpenEwald(int vers, int use_cf)
{
   // v0: E_dot
   // v1: EGV = E_dot + GV
   // v3: EA = E_pair + A
   // v4: EG = E_dot + G
   // v5: G
   // v6: GV
   auto edot = vers & calc::energy; // if not do_e, edot = false
   if (vers & calc::energy && vers & calc::analyz)
      edot = 0; // if do_e and do_a, edot = false
   int ver2 = vers;
   if (edot)
      ver2 &= ~calc::energy; // toggle off the calc::energy flag

   induce2(uind);
   if (edot)
      epolar0DotProd(uind, udir);
   if (vers != calc::v0) {
      epolarChgpenEwaldReal(ver2, use_cf);
      epolarChgpenEwaldRecipSelf(ver2, use_cf);
   }
}
}

namespace tinker {
void epolarChgpen(int vers)
{
   auto rc_a = rc_flag & calc::analyz;
   auto do_a = vers & calc::analyz;
   auto do_e = vers & calc::energy;
   auto do_v = vers & calc::virial;
   auto do_g = vers & calc::grad;
   auto use_cf = use(Potent::CHGFLX);
   int use_cfgrad = use_cf and do_g;

   zeroOnHost(energy_ep, virial_ep);
   size_t bsize = bufferSize();
   if (rc_a) {
      if (do_a)
         darray::zero(g::q0, bsize, nep);
      if (do_e)
         darray::zero(g::q0, bsize, ep);
      if (do_v)
         darray::zero(g::q0, bsize, vir_ep);
      if (do_g)
         darray::zero(g::q0, n, depx, depy, depz);
   }

   if (use_cf)
      alterchg();
   mpoleInit(vers);
   if (use_cfgrad) {
      cfluxZeroPot();
   }
   if (useEwald())
      epolarChgpenEwald(vers, use_cfgrad);
   else
      epolarChgpenNonEwald(vers, use_cfgrad);
   epolarPairwiseExtfield(vers, uind);
   torque(vers, depx, depy, depz);
   if (use_cfgrad)
      dcflux(vers, depx, depy, depz, vir_ep);
   if (polpot::use_expol and do_g)
      dexpol(vers);
   if (do_v) {
      VirialBuffer u2 = vir_trq;
      virial_prec v2[9];
      virialReduce(v2, u2);
      for (int iv = 0; iv < 9; ++iv) {
         virial_ep[iv] += v2[iv];
         virial_elec[iv] += v2[iv];
      }
   }

   if (rc_a) {
      if (do_e) {
         EnergyBuffer u = ep;
         energy_prec e = energyReduce(u);
         energy_ep += e;
         energy_elec += e;
      }
      if (do_v) {
         VirialBuffer u1 = vir_ep;
         virial_prec v1[9];
         virialReduce(v1, u1);
         for (int iv = 0; iv < 9; ++iv) {
            virial_ep[iv] = v1[iv];
            virial_elec[iv] += v1[iv];
         }
      }
      if (do_g)
         sumGradient(gx_elec, gy_elec, gz_elec, depx, depy, depz);
   }
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, epolarAplusEwaldReal, int, int, const real (*)[3]);
static void epolarAplusEwaldReal(int vers, int use_cf)
{
   TINKER_FCALL2(acc1, cu1, epolarAplusEwaldReal, vers, use_cf, uind);
}

static void epolarAplusEwaldRecipSelf(int vers, int use_cf)
{
   epolarChgpenEwaldRecipSelf(vers, use_cf);
}

void epolarAplusEwald(int vers, int use_cf)
{
   auto edot = vers & calc::energy; // if not do_e, edot = false
   if (vers & calc::energy and vers & calc::analyz)
      edot = 0; // if do_e and do_a, edot = false
   int ver2 = vers;
   if (edot)
      ver2 &= ~calc::energy; // toggle off the calc::energy flag

   induce2(uind);
   if (edot)
      epolar0DotProd(uind, udir);
   if (vers != calc::v0) {
      epolarAplusEwaldReal(ver2, use_cf);
      epolarAplusEwaldRecipSelf(ver2, use_cf);
   }
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, epolarAplusNonEwald, int, int, const real (*)[3]);
void epolarAplusNonEwald(int vers, int use_cf)
{
   auto edot = vers & calc::energy; // if not do_e, edot = false
   if (vers & calc::energy and vers & calc::analyz)
      edot = 0; // if do_e and do_a, edot = false
   int ver2 = vers;
   if (edot)
      ver2 &= ~calc::energy; // toggle off the calc::energy flag

   induce2(uind);
   if (edot)
      epolar0DotProd(uind, udir);
   if (vers != calc::v0)
      TINKER_FCALL2(acc1, cu1, epolarAplusNonEwald, ver2, use_cf, uind);
}
}
