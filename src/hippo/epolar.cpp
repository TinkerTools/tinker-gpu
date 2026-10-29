#include "ff/amoeba/epolar.h"
#include "ff/amoeba/empole.h"
#include "ff/amoeba/mpolestate.h"
#include "ff/dlmda.h"
#include "ff/elec.h"
#include "ff/energy.h"
#include "ff/hippo/cflux.h"
#include "ff/hippo/expol.h"
#include "ff/amoeba/induce.h"
#include "ff/hippo/induce.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
#include "ff/potent.h"
#include "math/zero.h"
#include "tool/externfunc.h"
#include "tool/iofortstr.h"
#include <tinker/detail/dlmda.hh>
#include <tinker/detail/mplpot.hh>
#include <tinker/detail/mutant.hh>
#include <tinker/detail/polar.hh>
#include <tinker/detail/polpot.hh>
#include <tinker/detail/uprior.hh>

namespace tinker {
// HIPPO polarization owns ep_buf here, since epolarData leaves it to us.
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
      darray::deallocate(polarityorig);

      if (rc_a)
         bufferDeallocate(rc_flag, nep);
      ep_buf.manage(op, rc_flag, {}, {}, false);
      nep = nullptr;

      darray::deallocate(ufld, dufld);
      darray::deallocate(work01_, work02_, work03_, work04_, work05_);

      induceHistoryData(RcOp::DEALLOC, false);
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
      // the polarization parameter state is built from the unscaled values
      if (use_prst or use_epdt)
         darray::allocate(n, &polarityorig);
      else
         polarityorig = nullptr;

      nep = nullptr;
      ep_buf.manage(op, rc_flag, {&ep, &vir_ep, &depx, &depy, &depz},
         {eng_buf_elec, vir_buf_elec, gx_elec, gy_elec, gz_elec}, rc_a, //
         {&energy_ep, &virial_ep}, {&energy_elec, &virial_elec});
      if (rc_a)
         bufferAllocate(rc_flag, &nep);

      if (rc_flag & calc::grad) {
         darray::allocate(n, &ufld, &dufld);
      } else {
         ufld = nullptr;
         dufld = nullptr;
      }

      darray::allocate(n, &work01_, &work02_, &work03_, &work04_, &work05_);
      polpred = polarPredictor();
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
      induceHistoryData(RcOp::ALLOC, false);
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
      if (polarityorig)
         darray::copyin(g::q0, n, polarityorig, dlmda::polarityorig);
      waitFor(g::q0);
      epolarSetActiveMask();
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
      epolar0DotProd(uind, udir, ep);
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
      epolar0DotProd(uind, udir, ep);
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

   zeroOnHost(energy_ep, virial_ep);
   if (rc_a and do_a)
      darray::zero(g::q0, bufferSize(), nep);
   ep_buf.zero(vers);
   polarLmdaZero(vers);

   // Dual topology solves at each endpoint and weights them (epolar4.f:
   // epolar4f); single topology solves once at the polarization lambda.
   if (use_epdt)
      epolarDtPasses(vers);
   else
      epolarSingleTopology(vers, [](int vers, bool use_cfgrad) {
         if (useEwald())
            epolarChgpenEwald(vers, use_cfgrad);
         else
            epolarChgpenNonEwald(vers, use_cfgrad);
      });

   ep_buf.flush(vers);
   polarLmdaFlush(vers);
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
      epolar0DotProd(uind, udir, ep);
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
      epolar0DotProd(uind, udir, ep);
   if (vers != calc::v0)
      TINKER_FCALL2(acc1, cu1, epolarAplusNonEwald, ver2, use_cf, uind);
}
}
