#include "tool/fft.h"
#include "ff/elec.h"
#include "ff/hippo/edisp.h"
#include "ff/pme.h"
#include "tool/accasync.h"
#include "tool/cudalib.h"
#include "tool/error.h"
#include <cassert>
#include <cufft.h>

namespace tinker {
struct FFTPlanCUFFT : public FFTPlan
{
   cufftHandle h;
   bool has_real = false; ///< The real plans are made on the first one-set FFT.
   cufftHandle hr2c;      ///< Real-to-complex plan.
   cufftHandle hc2r;      ///< Complex-to-real plan.
   int nbatch = 0;        ///< Number of grids of the batched real plans, made on first use.
   int sbatch = 0;        ///< Their stride in reals.
   cufftHandle hr2cb;     ///< Batched real-to-complex plan.
   cufftHandle hc2rb;     ///< Batched complex-to-real plan.
};

void fftData(RcOp op)
{
   if (not useEwald() && not useDEwald())
      return;

   if (op & RcOp::DEALLOC) {
      int idx = 0;
      while (idx < FFTPlanUnit::size()) {
         FFTPlanUnit u = idx;
         auto& ps = u->castTo<FFTPlanCUFFT>();
         cufftDestroy(ps.h);
         if (ps.has_real) {
            cufftDestroy(ps.hr2c);
            cufftDestroy(ps.hc2r);
         }
         if (ps.nbatch) {
            cufftDestroy(ps.hr2cb);
            cufftDestroy(ps.hc2rb);
         }
         ++idx;
      }
      FFTPlanUnit::clear();
   }

   if (op & RcOp::ALLOC) {
      assert(FFTPlanUnit::size() == 0);

      const size_t size = PMEUnit::size();
      FFTPlanUnit::resize<FFTPlanCUFFT>(size);
   }

   if (op & RcOp::INIT) {
#if TINKER_REAL_SIZE == 4
      const cufftType typ = CUFFT_C2C;
#elif TINKER_REAL_SIZE == 8
      const cufftType typ = CUFFT_Z2Z;
#else
      static_assert(false, "");
#endif

      int idx = 0;
      while (idx < FFTPlanUnit::size()) {
         FFTPlanUnit plan_u = idx;
         PMEUnit pme_u = idx;
         auto& iplan = plan_u->castTo<FFTPlanCUFFT>().h;
         auto& st = *pme_u;

         int nfast = st.nfft1;
         int nslow = st.nfft3;
         // different from FFTW Fortran API
         check_rt(cufftPlan3d(&iplan, nslow, st.nfft2, nfast, typ));
         auto stream = g::s0;
         if (use_pme_stream)
            stream = g::spme;
         check_rt(cufftSetStream(iplan, stream));
         ++idx;
      }
   }
}

void fftfront(PMEUnit pme_u)
{
   FFTPlanUnit iplan_u = static_cast<int>(pme_u);
   auto& iplan = iplan_u->castTo<FFTPlanCUFFT>().h;
   auto& st = *pme_u;

#if TINKER_REAL_SIZE == 4
   cufftExecC2C(iplan, reinterpret_cast<cufftComplex*>(st.qgrid), reinterpret_cast<cufftComplex*>(st.qgrid),
      CUFFT_FORWARD);
#elif TINKER_REAL_SIZE == 8
   cufftExecZ2Z(iplan, reinterpret_cast<cufftDoubleComplex*>(st.qgrid), reinterpret_cast<cufftDoubleComplex*>(st.qgrid),
      CUFFT_FORWARD);
#else
   static_assert(false, "");
#endif
}

void fftback(PMEUnit pme_u)
{
   FFTPlanUnit iplan_u = static_cast<int>(pme_u);
   auto& iplan = iplan_u->castTo<FFTPlanCUFFT>().h;
   auto& st = *pme_u;

#if TINKER_REAL_SIZE == 4
   cufftExecC2C(iplan, reinterpret_cast<cufftComplex*>(st.qgrid), reinterpret_cast<cufftComplex*>(st.qgrid),
      CUFFT_INVERSE);
#elif TINKER_REAL_SIZE == 8
   cufftExecZ2Z(iplan, reinterpret_cast<cufftDoubleComplex*>(st.qgrid), reinterpret_cast<cufftDoubleComplex*>(st.qgrid),
      CUFFT_INVERSE);
#else
   static_assert(false, "");
#endif
}

static FFTPlanCUFFT& fftRealPlans(PMEUnit pme_u)
{
   FFTPlanUnit iplan_u = static_cast<int>(pme_u);
   auto& iplan = iplan_u->castTo<FFTPlanCUFFT>();
   if (not iplan.has_real) {
      auto& st = *pme_u;
#if TINKER_REAL_SIZE == 4
      const cufftType typf = CUFFT_R2C, typb = CUFFT_C2R;
#elif TINKER_REAL_SIZE == 8
      const cufftType typf = CUFFT_D2Z, typb = CUFFT_Z2D;
#else
      static_assert(false, "");
#endif
      // in place, so the real grid has the padded rows of the half spectrum
      check_rt(cufftPlan3d(&iplan.hr2c, st.nfft3, st.nfft2, st.nfft1, typf));
      check_rt(cufftPlan3d(&iplan.hc2r, st.nfft3, st.nfft2, st.nfft1, typb));
      auto stream = g::s0;
      if (use_pme_stream)
         stream = g::spme;
      check_rt(cufftSetStream(iplan.hr2c, stream));
      check_rt(cufftSetStream(iplan.hc2r, stream));
      iplan.has_real = true;
   }
   return iplan;
}

void fftfrontOneSet(PMEUnit pme_u, real* grid)
{
   auto& iplan = fftRealPlans(pme_u);
   real* q = grid ? grid : pme_u->qgrid;

#if TINKER_REAL_SIZE == 4
   cufftExecR2C(iplan.hr2c, reinterpret_cast<cufftReal*>(q), reinterpret_cast<cufftComplex*>(q));
#elif TINKER_REAL_SIZE == 8
   cufftExecD2Z(iplan.hr2c, reinterpret_cast<cufftDoubleReal*>(q),
      reinterpret_cast<cufftDoubleComplex*>(q));
#else
   static_assert(false, "");
#endif
}

void fftbackOneSet(PMEUnit pme_u, real* grid)
{
   auto& iplan = fftRealPlans(pme_u);
   real* q = grid ? grid : pme_u->qgrid;

#if TINKER_REAL_SIZE == 4
   cufftExecC2R(iplan.hc2r, reinterpret_cast<cufftComplex*>(q), reinterpret_cast<cufftReal*>(q));
#elif TINKER_REAL_SIZE == 8
   cufftExecZ2D(iplan.hc2r, reinterpret_cast<cufftDoubleComplex*>(q),
      reinterpret_cast<cufftDoubleReal*>(q));
#else
   static_assert(false, "");
#endif
}

// The plans of fftRealPlans for nset grids stride reals apart, remade when
// either changes.
static FFTPlanCUFFT& fftRealPlans(PMEUnit pme_u, int nset, int stride)
{
   FFTPlanUnit iplan_u = static_cast<int>(pme_u);
   auto& iplan = iplan_u->castTo<FFTPlanCUFFT>();
   if (iplan.nbatch != nset or iplan.sbatch != stride) {
      if (iplan.nbatch) {
         cufftDestroy(iplan.hr2cb);
         cufftDestroy(iplan.hc2rb);
      }
      auto& st = *pme_u;
#if TINKER_REAL_SIZE == 4
      const cufftType typf = CUFFT_R2C, typb = CUFFT_C2R;
#elif TINKER_REAL_SIZE == 8
      const cufftType typf = CUFFT_D2Z, typb = CUFFT_Z2D;
#else
      static_assert(false, "");
#endif
      // the padded rows of an in-place transform, as fftRealPlans takes them
      int nfft[3] = {st.nfft3, st.nfft2, st.nfft1};
      int rembed[3] = {st.nfft3, st.nfft2, 2 * (st.nfft1 / 2 + 1)};
      int cembed[3] = {st.nfft3, st.nfft2, st.nfft1 / 2 + 1};
      check_rt(cufftPlanMany(&iplan.hr2cb, 3, nfft, rembed, 1, stride, cembed, 1, stride / 2, typf, nset));
      check_rt(cufftPlanMany(&iplan.hc2rb, 3, nfft, cembed, 1, stride / 2, rembed, 1, stride, typb, nset));
      auto stream = g::s0;
      if (use_pme_stream)
         stream = g::spme;
      check_rt(cufftSetStream(iplan.hr2cb, stream));
      check_rt(cufftSetStream(iplan.hc2rb, stream));
      iplan.nbatch = nset;
      iplan.sbatch = stride;
   }
   return iplan;
}

void fftfrontOneSet(PMEUnit pme_u, real* grids, int nset, int stride)
{
   assert(stride % 2 == 0);
   auto& iplan = fftRealPlans(pme_u, nset, stride);
#if TINKER_REAL_SIZE == 4
   cufftExecR2C(iplan.hr2cb, reinterpret_cast<cufftReal*>(grids), reinterpret_cast<cufftComplex*>(grids));
#elif TINKER_REAL_SIZE == 8
   cufftExecD2Z(iplan.hr2cb, reinterpret_cast<cufftDoubleReal*>(grids), reinterpret_cast<cufftDoubleComplex*>(grids));
#else
   static_assert(false, "");
#endif
}

void fftbackOneSet(PMEUnit pme_u, real* grids, int nset, int stride)
{
   assert(stride % 2 == 0);
   auto& iplan = fftRealPlans(pme_u, nset, stride);
#if TINKER_REAL_SIZE == 4
   cufftExecC2R(iplan.hc2rb, reinterpret_cast<cufftComplex*>(grids), reinterpret_cast<cufftReal*>(grids));
#elif TINKER_REAL_SIZE == 8
   cufftExecZ2D(iplan.hc2rb, reinterpret_cast<cufftDoubleComplex*>(grids), reinterpret_cast<cufftDoubleReal*>(grids));
#else
   static_assert(false, "");
#endif
}
}
