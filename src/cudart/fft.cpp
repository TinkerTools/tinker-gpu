#include "tool/fft.h"
#include "ff/elec.h"
#include "ff/hippo/edisp.h"
#include "ff/pme.h"
#include "tool/accasync.h"
#include "tool/cudalib.h"
#include "tool/error.h"
#include <cufft.h>

namespace tinker {
struct FFTPlanCUFFT : public FFTPlan
{
   cufftHandle h;
   bool has_real = false; ///< The real plans are made on the first one-set FFT.
   cufftHandle hr2c;      ///< Real-to-complex plan.
   cufftHandle hc2r;      ///< Complex-to-real plan.
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

void fftfrontOneSet(PMEUnit pme_u)
{
   auto& iplan = fftRealPlans(pme_u);
   auto& st = *pme_u;

#if TINKER_REAL_SIZE == 4
   cufftExecR2C(iplan.hr2c, reinterpret_cast<cufftReal*>(st.qgrid), reinterpret_cast<cufftComplex*>(st.qgrid));
#elif TINKER_REAL_SIZE == 8
   cufftExecD2Z(iplan.hr2c, reinterpret_cast<cufftDoubleReal*>(st.qgrid),
      reinterpret_cast<cufftDoubleComplex*>(st.qgrid));
#else
   static_assert(false, "");
#endif
}

void fftbackOneSet(PMEUnit pme_u)
{
   auto& iplan = fftRealPlans(pme_u);
   auto& st = *pme_u;

#if TINKER_REAL_SIZE == 4
   cufftExecC2R(iplan.hc2r, reinterpret_cast<cufftComplex*>(st.qgrid), reinterpret_cast<cufftReal*>(st.qgrid));
#elif TINKER_REAL_SIZE == 8
   cufftExecZ2D(iplan.hc2r, reinterpret_cast<cufftDoubleComplex*>(st.qgrid),
      reinterpret_cast<cufftDoubleReal*>(st.qgrid));
#else
   static_assert(false, "");
#endif
}
}
