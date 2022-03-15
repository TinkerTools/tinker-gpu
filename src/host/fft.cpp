#include "fft.h"
#include "edisp.h"
#include "elec.h"
#include "pme.h"
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
};

void fft_data(rc_op op)
{
   if (!use_ewald() && !use_dewald())
      return;

   if (op & rc_dealloc) {
      int idx = 0;
      while (idx < FFTPlanUnit::size()) {
         FFTPlanUnit u = idx;
         auto& ps = u->self<FFTPlanFFTW>();
#if TINKER_REAL_SIZE == 4
         fftwf_destroy_plan(ps.planf);
         fftwf_destroy_plan(ps.planb);
#elif TINKER_REAL_SIZE == 8
         fftw_destroy_plan(ps.planf);
         fftw_destroy_plan(ps.planb);
#else
         static_assert(false, "");
#endif
         ++idx;
      }
      FFTPlanUnit::clear();
   }

   if (op & rc_alloc) {
      assert(FFTPlanUnit::size() == 0);

      const size_t size = PMEUnit::size();
      FFTPlanUnit::resize<FFTPlanFFTW>(size);
   }

   if (op & rc_init) {
      int idx = 0;
      while (idx < FFTPlanUnit::size()) {
         FFTPlanUnit plan_u = idx;
         PMEUnit pme_u = idx;
         auto& iplan = plan_u->self<FFTPlanFFTW>();
         auto& st = *pme_u;

         const int nfast = st.nfft1;
         const int nfft2 = st.nfft2;
         const int nslow = st.nfft3;
         const int ifront = -1;
         const int iback = 1;
         const unsigned int iguess = 0;
         // different from FFTW Fortran API
#if TINKER_REAL_SIZE == 4
         iplan.planf =
            fftwf_plan_dft_3d(nslow, nfft2, nfast, reinterpret_cast<fftwf_complex*>(st.qgrid),
               reinterpret_cast<fftwf_complex*>(st.qgrid), ifront, iguess);
         iplan.planb =
            fftwf_plan_dft_3d(nslow, nfft2, nfast, reinterpret_cast<fftwf_complex*>(st.qgrid),
               reinterpret_cast<fftwf_complex*>(st.qgrid), iback, iguess);
#elif TINKER_REAL_SIZE == 8
         iplan.planf =
            fftw_plan_dft_3d(nslow, nfft2, nfast, reinterpret_cast<fftw_complex*>(st.qgrid),
               reinterpret_cast<fftw_complex*>(st.qgrid), ifront, iguess);
         iplan.planb =
            fftw_plan_dft_3d(nslow, nfft2, nfast, reinterpret_cast<fftw_complex*>(st.qgrid),
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
   auto& iplan = iplan_u->self<FFTPlanFFTW>();
   auto& st = *pme_u;

#if TINKER_REAL_SIZE == 4
   fftwf_execute_dft(iplan.planf, reinterpret_cast<fftwf_complex*>(st.qgrid),
      reinterpret_cast<fftwf_complex*>(st.qgrid));
#elif TINKER_REAL_SIZE == 8
   fftw_execute_dft(iplan.planf, reinterpret_cast<fftw_complex*>(st.qgrid),
      reinterpret_cast<fftw_complex*>(st.qgrid));
#else
   static_assert(false, "");
#endif
}

void fftback(PMEUnit pme_u)
{
   FFTPlanUnit iplan_u = static_cast<int>(pme_u);
   auto& iplan = iplan_u->self<FFTPlanFFTW>();
   auto& st = *pme_u;

#if TINKER_REAL_SIZE == 4
   fftwf_execute_dft(iplan.planb, reinterpret_cast<fftwf_complex*>(st.qgrid),
      reinterpret_cast<fftwf_complex*>(st.qgrid));
#elif TINKER_REAL_SIZE == 8
   fftw_execute_dft(iplan.planb, reinterpret_cast<fftw_complex*>(st.qgrid),
      reinterpret_cast<fftw_complex*>(st.qgrid));
#else
   static_assert(false, "");
#endif
}
}
