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
   int sbatch = 0;        ///< Stride in reals of the batched real plans.
   bool has_batch[2] = {}; ///< The batched real plans of 2 and 3 grids, made on first use.
   cufftHandle hr2cb[2];   ///< Batched real-to-complex plans.
   cufftHandle hc2rb[2];   ///< Batched complex-to-real plans.
};

#if TINKER_REAL_SIZE == 4
static const cufftType fftTypeC2C = CUFFT_C2C, fftTypeR2C = CUFFT_R2C, fftTypeC2R = CUFFT_C2R;
#elif TINKER_REAL_SIZE == 8
static const cufftType fftTypeC2C = CUFFT_Z2Z, fftTypeR2C = CUFFT_D2Z, fftTypeC2R = CUFFT_Z2D;
#else
static_assert(false, "");
#endif

static void fftSetStream(cufftHandle h)
{
   check_rt(cufftSetStream(h, use_pme_stream ? g::spme : g::s0));
}

// in place on the padded rows of the half spectrum
static void fftExecR2C(cufftHandle h, real* q)
{
#if TINKER_REAL_SIZE == 4
   cufftExecR2C(h, reinterpret_cast<cufftReal*>(q), reinterpret_cast<cufftComplex*>(q));
#else
   cufftExecD2Z(h, reinterpret_cast<cufftDoubleReal*>(q), reinterpret_cast<cufftDoubleComplex*>(q));
#endif
}

static void fftExecC2R(cufftHandle h, real* q)
{
#if TINKER_REAL_SIZE == 4
   cufftExecC2R(h, reinterpret_cast<cufftComplex*>(q), reinterpret_cast<cufftReal*>(q));
#else
   cufftExecZ2D(h, reinterpret_cast<cufftDoubleComplex*>(q), reinterpret_cast<cufftDoubleReal*>(q));
#endif
}

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
         for (int k = 0; k < 2; ++k) {
            if (ps.has_batch[k]) {
               cufftDestroy(ps.hr2cb[k]);
               cufftDestroy(ps.hc2rb[k]);
            }
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
      int idx = 0;
      while (idx < FFTPlanUnit::size()) {
         FFTPlanUnit plan_u = idx;
         PMEUnit pme_u = idx;
         auto& iplan = plan_u->castTo<FFTPlanCUFFT>().h;
         auto& st = *pme_u;

         int nfast = st.nfft1;
         int nslow = st.nfft3;
         // different from FFTW Fortran API
         check_rt(cufftPlan3d(&iplan, nslow, st.nfft2, nfast, fftTypeC2C));
         fftSetStream(iplan);
         ++idx;
      }
   }
}

static void fftExecC2C(PMEUnit pme_u, int dir)
{
   FFTPlanUnit iplan_u = static_cast<int>(pme_u);
   auto& iplan = iplan_u->castTo<FFTPlanCUFFT>().h;
   real* q = pme_u->qgrid;
#if TINKER_REAL_SIZE == 4
   cufftExecC2C(iplan, reinterpret_cast<cufftComplex*>(q), reinterpret_cast<cufftComplex*>(q), dir);
#else
   cufftExecZ2Z(iplan, reinterpret_cast<cufftDoubleComplex*>(q), reinterpret_cast<cufftDoubleComplex*>(q), dir);
#endif
}

void fftfront(PMEUnit pme_u)
{
   fftExecC2C(pme_u, CUFFT_FORWARD);
}

void fftback(PMEUnit pme_u)
{
   fftExecC2C(pme_u, CUFFT_INVERSE);
}

static FFTPlanCUFFT& fftRealPlans(PMEUnit pme_u)
{
   FFTPlanUnit iplan_u = static_cast<int>(pme_u);
   auto& iplan = iplan_u->castTo<FFTPlanCUFFT>();
   if (not iplan.has_real) {
      auto& st = *pme_u;
      // in place, so the real grid has the padded rows of the half spectrum
      check_rt(cufftPlan3d(&iplan.hr2c, st.nfft3, st.nfft2, st.nfft1, fftTypeR2C));
      check_rt(cufftPlan3d(&iplan.hc2r, st.nfft3, st.nfft2, st.nfft1, fftTypeC2R));
      fftSetStream(iplan.hr2c);
      fftSetStream(iplan.hc2r);
      iplan.has_real = true;
   }
   return iplan;
}

void fftfrontOneSet(PMEUnit pme_u, real* grid)
{
   fftExecR2C(fftRealPlans(pme_u).hr2c, grid ? grid : pme_u->qgrid);
}

void fftbackOneSet(PMEUnit pme_u, real* grid)
{
   fftExecC2R(fftRealPlans(pme_u).hc2r, grid ? grid : pme_u->qgrid);
}

// The plans of fftRealPlans for nset grids stride reals apart, one pair for
// each nset kept, since the number of dispersion slots changes with lambda
// and a new plan synchronizes the device; a single grid takes the plans of
// fftRealPlans. All are remade if the stride changes.
static void fftRealPlans(PMEUnit pme_u, int nset, int stride, cufftHandle& r2c, cufftHandle& c2r)
{
   assert(stride % 2 == 0);
   assert(1 <= nset and nset <= 3);
   auto& iplan = fftRealPlans(pme_u);
   if (nset == 1) {
      r2c = iplan.hr2c, c2r = iplan.hc2r;
      return;
   }
   if (iplan.sbatch != stride) {
      for (int k = 0; k < 2; ++k) {
         if (iplan.has_batch[k]) {
            cufftDestroy(iplan.hr2cb[k]);
            cufftDestroy(iplan.hc2rb[k]);
            iplan.has_batch[k] = false;
         }
      }
      iplan.sbatch = stride;
   }
   const int k = nset - 2;
   if (not iplan.has_batch[k]) {
      auto& st = *pme_u;
      // the padded rows of an in-place transform, as fftRealPlans takes them
      int nfft[3] = {st.nfft3, st.nfft2, st.nfft1};
      int rembed[3] = {st.nfft3, st.nfft2, 2 * (st.nfft1 / 2 + 1)};
      int cembed[3] = {st.nfft3, st.nfft2, st.nfft1 / 2 + 1};
      check_rt(cufftPlanMany(&iplan.hr2cb[k], 3, nfft, rembed, 1, stride, cembed, 1, stride / 2, fftTypeR2C, nset));
      check_rt(cufftPlanMany(&iplan.hc2rb[k], 3, nfft, cembed, 1, stride / 2, rembed, 1, stride, fftTypeC2R, nset));
      fftSetStream(iplan.hr2cb[k]);
      fftSetStream(iplan.hc2rb[k]);
      iplan.has_batch[k] = true;
   }
   r2c = iplan.hr2cb[k], c2r = iplan.hc2rb[k];
}

void fftfrontOneSet(PMEUnit pme_u, real* grids, int nset, int stride)
{
   cufftHandle r2c, c2r;
   fftRealPlans(pme_u, nset, stride, r2c, c2r);
   fftExecR2C(r2c, grids);
}

void fftbackOneSet(PMEUnit pme_u, real* grids, int nset, int stride)
{
   cufftHandle r2c, c2r;
   fftRealPlans(pme_u, nset, stride, r2c, c2r);
   fftExecC2R(c2r, grids);
}
}
