#include "tool/fft.h"
#include "ff/elec.h"
#include "ff/hippo/edisp.h"
#include "ff/pme.h"
#include <fftw3.h>

namespace tinker {
struct FFTPlanFFTW : public FFTPlan
{
#if TINKER_REAL_SIZE == 4
   using type = fftwf_plan;
#elif TINKER_REAL_SIZE == 8
   using type = fftw_plan;
#else
   static_assert(false, "");
#endif

   type planf; ///< FFT front plan.
   type planb; ///< FFT back plan.

   bool has_real = false; ///< The real plans are made on the first one-set FFT.
   type planr2c;          ///< Real-to-complex plan.
   type planc2r;          ///< Complex-to-real plan.
};

void fftData(RcOp op)
{
   if (not useEwald() && not useDEwald())
      return;

   if (op & RcOp::DEALLOC) {
      int idx = 0;
      while (idx < FFTPlanUnit::size()) {
         FFTPlanUnit u = idx;
         auto& ps = u->castTo<FFTPlanFFTW>();
#if TINKER_REAL_SIZE == 4
         fftwf_destroy_plan(ps.planf);
         fftwf_destroy_plan(ps.planb);
         if (ps.has_real) {
            fftwf_destroy_plan(ps.planr2c);
            fftwf_destroy_plan(ps.planc2r);
         }
#elif TINKER_REAL_SIZE == 8
         fftw_destroy_plan(ps.planf);
         fftw_destroy_plan(ps.planb);
         if (ps.has_real) {
            fftw_destroy_plan(ps.planr2c);
            fftw_destroy_plan(ps.planc2r);
         }
#else
         static_assert(false, "");
#endif
         ++idx;
      }
      FFTPlanUnit::clear();
   }

   if (op & RcOp::ALLOC) {
      assert(FFTPlanUnit::size() == 0);

      const size_t size = PMEUnit::size();
      FFTPlanUnit::resize<FFTPlanFFTW>(size);
   }

   if (op & RcOp::INIT) {
      int idx = 0;
      while (idx < FFTPlanUnit::size()) {
         FFTPlanUnit plan_u = idx;
         PMEUnit pme_u = idx;
         auto& iplan = plan_u->castTo<FFTPlanFFTW>();
         auto& st = *pme_u;

         const int nfast = st.nfft1;
         const int nfft2 = st.nfft2;
         const int nslow = st.nfft3;
         const int ifront = -1;
         const int iback = 1;
         const unsigned int iguess = 0;
         // different from FFTW Fortran API
#if TINKER_REAL_SIZE == 4
         iplan.planf = fftwf_plan_dft_3d(nslow, nfft2, nfast, reinterpret_cast<fftwf_complex*>(st.qgrid),
            reinterpret_cast<fftwf_complex*>(st.qgrid), ifront, iguess);
         iplan.planb = fftwf_plan_dft_3d(nslow, nfft2, nfast, reinterpret_cast<fftwf_complex*>(st.qgrid),
            reinterpret_cast<fftwf_complex*>(st.qgrid), iback, iguess);
#elif TINKER_REAL_SIZE == 8
         iplan.planf = fftw_plan_dft_3d(nslow, nfft2, nfast, reinterpret_cast<fftw_complex*>(st.qgrid),
            reinterpret_cast<fftw_complex*>(st.qgrid), ifront, iguess);
         iplan.planb = fftw_plan_dft_3d(nslow, nfft2, nfast, reinterpret_cast<fftw_complex*>(st.qgrid),
            reinterpret_cast<fftw_complex*>(st.qgrid), iback, iguess);
#else
         static_assert(false, "");
#endif
         ++idx;
      }
   }
}

void fftfront(PMEUnit pme_u)
{
   FFTPlanUnit iplan_u = static_cast<int>(pme_u);
   auto& iplan = iplan_u->castTo<FFTPlanFFTW>();
   auto& st = *pme_u;

#if TINKER_REAL_SIZE == 4
   fftwf_execute_dft(iplan.planf, reinterpret_cast<fftwf_complex*>(st.qgrid),
      reinterpret_cast<fftwf_complex*>(st.qgrid));
#elif TINKER_REAL_SIZE == 8
   fftw_execute_dft(iplan.planf, reinterpret_cast<fftw_complex*>(st.qgrid), reinterpret_cast<fftw_complex*>(st.qgrid));
#else
   static_assert(false, "");
#endif
}

void fftback(PMEUnit pme_u)
{
   FFTPlanUnit iplan_u = static_cast<int>(pme_u);
   auto& iplan = iplan_u->castTo<FFTPlanFFTW>();
   auto& st = *pme_u;

#if TINKER_REAL_SIZE == 4
   fftwf_execute_dft(iplan.planb, reinterpret_cast<fftwf_complex*>(st.qgrid),
      reinterpret_cast<fftwf_complex*>(st.qgrid));
#elif TINKER_REAL_SIZE == 8
   fftw_execute_dft(iplan.planb, reinterpret_cast<fftw_complex*>(st.qgrid), reinterpret_cast<fftw_complex*>(st.qgrid));
#else
   static_assert(false, "");
#endif
}

static FFTPlanFFTW& fftRealPlans(PMEUnit pme_u)
{
   FFTPlanUnit iplan_u = static_cast<int>(pme_u);
   auto& iplan = iplan_u->castTo<FFTPlanFFTW>();
   if (not iplan.has_real) {
      auto& st = *pme_u;
      // in place, so the real grid has the padded rows of the half spectrum;
      // FFTW_ESTIMATE leaves the grid alone while planning
#if TINKER_REAL_SIZE == 4
      iplan.planr2c = fftwf_plan_dft_r2c_3d(st.nfft3, st.nfft2, st.nfft1, st.qgrid,
         reinterpret_cast<fftwf_complex*>(st.qgrid), FFTW_ESTIMATE);
      iplan.planc2r = fftwf_plan_dft_c2r_3d(st.nfft3, st.nfft2, st.nfft1, reinterpret_cast<fftwf_complex*>(st.qgrid),
         st.qgrid, FFTW_ESTIMATE);
#elif TINKER_REAL_SIZE == 8
      iplan.planr2c = fftw_plan_dft_r2c_3d(st.nfft3, st.nfft2, st.nfft1, st.qgrid,
         reinterpret_cast<fftw_complex*>(st.qgrid), FFTW_ESTIMATE);
      iplan.planc2r = fftw_plan_dft_c2r_3d(st.nfft3, st.nfft2, st.nfft1, reinterpret_cast<fftw_complex*>(st.qgrid),
         st.qgrid, FFTW_ESTIMATE);
#else
      static_assert(false, "");
#endif
      iplan.has_real = true;
   }
   return iplan;
}

void fftfrontOneSet(PMEUnit pme_u)
{
   auto& iplan = fftRealPlans(pme_u);
   auto& st = *pme_u;

#if TINKER_REAL_SIZE == 4
   fftwf_execute_dft_r2c(iplan.planr2c, st.qgrid, reinterpret_cast<fftwf_complex*>(st.qgrid));
#elif TINKER_REAL_SIZE == 8
   fftw_execute_dft_r2c(iplan.planr2c, st.qgrid, reinterpret_cast<fftw_complex*>(st.qgrid));
#else
   static_assert(false, "");
#endif
}

void fftbackOneSet(PMEUnit pme_u)
{
   auto& iplan = fftRealPlans(pme_u);
   auto& st = *pme_u;

#if TINKER_REAL_SIZE == 4
   fftwf_execute_dft_c2r(iplan.planc2r, reinterpret_cast<fftwf_complex*>(st.qgrid), st.qgrid);
#elif TINKER_REAL_SIZE == 8
   fftw_execute_dft_c2r(iplan.planc2r, reinterpret_cast<fftw_complex*>(st.qgrid), st.qgrid);
#else
   static_assert(false, "");
#endif
}
}
