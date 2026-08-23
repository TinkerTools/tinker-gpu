#include "ff/dlmda.h"
#include "ff/energy.h"
#include "ff/eost.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
#include "ff/ost.h"
#include "ff/potent.h"
#include "ff/thermint.h"
#include "md/misc.h"
#include "md/pq.h"
#include "tool/cudalib.h"
#include "tool/iofortstr.h"
#include <condition_variable>
#include <future>
#include <mutex>
#include <tinker/detail/atomid.hh>
#include <tinker/detail/atoms.hh>
#include <tinker/detail/couple.hh>
#include <tinker/detail/deriv.hh>
#include <tinker/detail/dlmda.hh>
#include <tinker/detail/expol.hh>
#include <tinker/detail/files.hh>
#include <tinker/detail/moldyn.hh>
#include <tinker/detail/mpole.hh>
#include <tinker/detail/mutant.hh>
#include <tinker/detail/ost.hh>
#include <tinker/detail/output.hh>
#include <tinker/detail/polar.hh>
#include <tinker/detail/polpot.hh>
#include <tinker/detail/thrmint.hh>
#include <tinker/detail/titles.hh>
#include <tinker/detail/units.hh>
#include <tinker/routines.h>

#if TINKER_CUDART
#include "tool/error.h"
#include "tool/gpucard.h"
#include <cuda_runtime.h>
#endif

namespace tinker {
static std::mutex mtx_dup, mtx_write;
static std::condition_variable cv_dup, cv_write;
static bool idle_dup, idle_write;
static std::future<void> fut_dup_then_write;

static bool mdsaveUseUind()
{
   return (static_cast<bool>(output::uindsave) or static_cast<bool>(output::tefsave)
             or static_cast<bool>(output::usyssave))
      and use(Potent::POLAR);
}

static bool mdsaveUseUstc()
{
   return static_cast<bool>(output::ustcsave) or static_cast<bool>(output::uchgsave)
      or static_cast<bool>(output::usyssave);
}

static bool mdsaveUseUdir()
{
   return (static_cast<bool>(output::udirsave) or static_cast<bool>(output::defsave)) and use(Potent::POLAR);
}

static bool mdsaveUseExpolUdir()
{
   return static_cast<bool>(polpot::use_expol) and static_cast<bool>(output::udirsave) and use(Potent::POLAR);
}

static bool mdsaveUseExpolTef()
{
   return static_cast<bool>(polpot::use_expol) and static_cast<bool>(output::tefsave) and use(Potent::POLAR);
}

#if TINKER_CUDART
static cudaEvent_t mdsave_begin_event, mdsave_end_event;
#endif
static real (*dup_buf_uind)[3];
static real (*dup_buf_udir)[3];
static real (*dup_buf_rpole)[MPL_TOTAL];
static real (*dup_buf_polinv)[3][3];
static real (*dup_buf_polscale)[3][3];
static energy_prec dup_buf_esum;
static Box dup_buf_box;
static pos_prec *dup_buf_x, *dup_buf_y, *dup_buf_z;
static vel_prec *dup_buf_vx, *dup_buf_vy, *dup_buf_vz;
static grad_prec *dup_buf_gx, *dup_buf_gy, *dup_buf_gz;
static int lmda_snap_mask;
static double s_lambda, s_dedl;
static bool ost_snap_active;
static double s_ostdgdl, s_ostddgdl, s_eosttot;
static double s_osttheta, s_ostvtheta;
static int s_iost, s_nflmda, s_fli0;
static int s_nosthist, s_sizeosthist, s_ost_first;
static std::vector<int> s_khist, s_ihist;
static std::vector<double> s_lhist, s_fhist, s_hhist, s_wlhist, s_wfhist;
static int s_nmetahist, s_sizemetahist, s_meta_first;
static std::vector<double> s_mlhist, s_mhhist, s_mwhist;
static std::vector<int> s_mihist;
static bool ti_snap_active;
static int s_tinbcount, s_ti_first;
static std::vector<double> s_tihist, s_tidedl, s_tisd;

static void mdsaveDupLmda()
{
   lmda_snap_mask = lmdaDerivMask(rc_flag, use_dlmda);
   if (not lmda_snap_mask)
      return;

   s_lambda = lambda;
   if (lmda_snap_mask & calc::energy_dlmda1)
      s_dedl = dedl;
}

static void mdsaveWriteLmda()
{
   if (not lmda_snap_mask)
      return;

   mutant::lambda = s_lambda;
   if (lmda_snap_mask & calc::energy_dlmda1)
      dlmda::dedl = s_dedl;
}

static void mdsaveDupOst(int istep)
{
   ost_snap_active = use_ost or use_meta;
   if (not ost_snap_active)
      return;

   s_iost = istep;
   s_nflmda = nflmda;
   s_fli0 = fli0;
   s_ostdgdl = ostdgdl;
   s_ostddgdl = ostddgdl;
   s_eosttot = eosttot;
   s_osttheta = osttheta;
   s_ostvtheta = ostvtheta;

   s_khist.clear(), s_ihist.clear(), s_lhist.clear(), s_fhist.clear();
   s_hhist.clear(), s_wlhist.clear(), s_wfhist.clear();
   s_nosthist = nosthist;
   s_sizeosthist = sizeosthist;
   s_ost_first = ost::nosthistsave + 1; // saves are serialized, so this is stable
   if (use_ost) {
      for (int k = s_ost_first; k <= s_nosthist; ++k) {
         s_khist.push_back(osthist[k]);
         s_ihist.push_back(ostihist[k]);
         s_lhist.push_back(ostlhist[k]);
         s_fhist.push_back(ostfhist[k]);
         s_hhist.push_back(osthhist[k]);
         s_wlhist.push_back(ostwlhist[k]);
         s_wfhist.push_back(ostwfhist[k]);
      }
   }

   s_mlhist.clear(), s_mhhist.clear(), s_mwhist.clear(), s_mihist.clear();
   s_nmetahist = nmetahist;
   s_sizemetahist = sizemetahist;
   s_meta_first = ost::nmethistsave + 1; // saves are serialized, so this is stable
   if (use_meta) {
      for (int k = s_meta_first; k <= s_nmetahist; ++k) {
         s_mlhist.push_back(metalhist[k]);
         s_mhhist.push_back(metahhist[k]);
         s_mwhist.push_back(metawhist[k]);
         s_mihist.push_back(metaihist[k]);
      }
   }
}

static void mdsaveWriteOst()
{
   if (not ost_snap_active)
      return;

   ost::iost = s_iost;
   ost::nflmda = s_nflmda;
   ost::fli0 = s_fli0;
   ost::ostdgdl = s_ostdgdl;
   ost::ostddgdl = s_ostddgdl;
   ost::eosttot = s_eosttot;
   ost::osttheta = s_osttheta;
   ost::ostvtheta = s_ostvtheta;

   if (use_ost) {
      // grow the Fortran arrays to match the engine, preserving existing entries
      while (ost::sizeosthist < s_sizeosthist)
         tinker_f_resizeosthist();
      ost::nosthist = s_nosthist;
      for (int i = 0; i < (int)s_khist.size(); ++i) {
         int j = s_ost_first - 1 + i;
         ost::osthist[j] = s_khist[i];
         ost::ostihist[j] = s_ihist[i];
         ost::ostlhist[j] = s_lhist[i];
         ost::ostfhist[j] = s_fhist[i];
         ost::osthhist[j] = s_hhist[i];
         ost::ostwlhist[j] = s_wlhist[i];
         ost::ostwfhist[j] = s_wfhist[i];
      }
      // Fortran saveost advances nosthistsave after appending this slice.
   }

   if (use_meta) {
      while (ost::sizemetahist < s_sizemetahist)
         tinker_f_resizemeta();
      ost::nmetahist = s_nmetahist;
      for (int i = 0; i < (int)s_mlhist.size(); ++i) {
         int j = s_meta_first - 1 + i;
         ost::metalhist[j] = s_mlhist[i];
         ost::metahhist[j] = s_mhhist[i];
         ost::metawhist[j] = s_mwhist[i];
         ost::metaihist[j] = s_mihist[i];
      }
   }
}

static void mdsaveDupTi()
{
   ti_snap_active = use_ti and not tilmdadedl.empty();
   if (not ti_snap_active)
      return;

   s_tihist.clear(), s_tidedl.clear(), s_tisd.clear();
   s_tinbcount = tinbcount;
   s_ti_first = thrmint::tinbsave; // saves are serialized, so this is stable
   for (int i = s_ti_first; i < s_tinbcount; ++i) {
      s_tihist.push_back(tilmdahist[i]);
      s_tidedl.push_back(tilmdadedl[i]);
      s_tisd.push_back(tilmdadedlstd[i]);
   }
}

static void mdsaveWriteTi()
{
   if (not ti_snap_active)
      return;

   for (int i = 0; i < (int)s_tihist.size(); ++i) {
      int j = s_ti_first + i;
      thrmint::tilmdahist[j] = s_tihist[i];
      thrmint::tilmdadedl[j] = s_tidedl[i];
      thrmint::tilmdadedlstd[j] = s_tisd[i];
   }
   thrmint::tinbcount = s_tinbcount;
}

static void mdsaveDupThenWrite(int istep, time_prec dt)
{
#if TINKER_CUDART
   // This function (mdsaveDupThenWrite) will run in another CPU thread.
   // There is no guarantee that the CUDA runtime will use the same GPU card as
   // the main thread, unless cudaSetDevice() is called explicitly.
   //
   // Of course this is not a problem if the computer has only one GPU card.
   check_rt(cudaSetDevice(idevice));
#endif

   // duplicate

   dup_buf_esum = esum;
   boxGetCurrent(dup_buf_box);
   darray::copy(g::q0, n, dup_buf_x, xpos);
   darray::copy(g::q0, n, dup_buf_y, ypos);
   darray::copy(g::q0, n, dup_buf_z, zpos);

   darray::copy(g::q0, n, dup_buf_vx, vx);
   darray::copy(g::q0, n, dup_buf_vy, vy);
   darray::copy(g::q0, n, dup_buf_vz, vz);

   darray::copy(g::q0, n, dup_buf_gx, gx);
   darray::copy(g::q0, n, dup_buf_gy, gy);
   darray::copy(g::q0, n, dup_buf_gz, gz);

   if (mdsaveUseUind())
      darray::copy(g::q0, 3 * n, &dup_buf_uind[0][0], &uind[0][0]);

   if (mdsaveUseUstc())
      darray::copy(g::q0, MPL_TOTAL * n, &dup_buf_rpole[0][0], &rpole[0][0]);

   if (mdsaveUseUdir())
      darray::copy(g::q0, 3 * n, &dup_buf_udir[0][0], &udir[0][0]);

   if (mdsaveUseExpolUdir())
      darray::copy(g::q0, 9 * n, &dup_buf_polinv[0][0][0], &polinv[0][0][0]);

   if (mdsaveUseExpolTef())
      darray::copy(g::q0, 9 * n, &dup_buf_polscale[0][0][0], &polscale[0][0][0]);

   mdsaveDupLmda();
   mdsaveDupOst(istep);
   mdsaveDupTi();

      // Record mdsave_begin_event when g::s0 is available.
      // g::s1 will wait until mdsave_begin_event is recorded.
#if TINKER_CUDART
   check_rt(cudaEventRecord(mdsave_begin_event, g::s0));
   check_rt(cudaStreamWaitEvent(g::s1, mdsave_begin_event, 0));
#endif

   mtx_dup.lock();
   idle_dup = true;
   cv_dup.notify_all();
   mtx_dup.unlock();

   // get gpu buffer and write to external files

   energy_prec epot = dup_buf_esum;
   boxSetTinker(dup_buf_box);
   if (sizeof(pos_prec) == sizeof(double)) {
      darray::copyout(g::q1, n, atoms::x, dup_buf_x);
      darray::copyout(g::q1, n, atoms::y, dup_buf_y);
      darray::copyout(g::q1, n, atoms::z, dup_buf_z);
      waitFor(g::q1);
   } else {
      std::vector<pos_prec> arrx(n), arry(n), arrz(n);
      darray::copyout(g::q1, n, arrx.data(), dup_buf_x);
      darray::copyout(g::q1, n, arry.data(), dup_buf_y);
      darray::copyout(g::q1, n, arrz.data(), dup_buf_z);
      waitFor(g::q1);
      for (int i = 0; i < n; ++i) {
         atoms::x[i] = arrx[i];
         atoms::y[i] = arry[i];
         atoms::z[i] = arrz[i];
      }
   }

   {
      std::vector<vel_prec> arrx(n), arry(n), arrz(n);
      darray::copyout(g::q1, n, arrx.data(), dup_buf_vx);
      darray::copyout(g::q1, n, arry.data(), dup_buf_vy);
      darray::copyout(g::q1, n, arrz.data(), dup_buf_vz);
      waitFor(g::q1);
      for (int i = 0; i < n; ++i) {
         int j = 3 * i;
         moldyn::v[j] = arrx[i];
         moldyn::v[j + 1] = arry[i];
         moldyn::v[j + 2] = arrz[i];
      }
   }

   {
      std::vector<double> arrx(n), arry(n), arrz(n);
      copyGradientSync(calc::grad, arrx.data(), arry.data(), arrz.data(), dup_buf_gx, dup_buf_gy, dup_buf_gz, g::q1);
      // convert gradient to acceleration
      const double ekcal = units::ekcal;
      for (int i = 0; i < n; ++i) {
         int j = 3 * i;
         deriv::desum[j + 0] = arrx[i];
         deriv::desum[j + 1] = arry[i];
         deriv::desum[j + 2] = arrz[i];
         double invmass = 1.0 / atomid::mass[i];
         moldyn::a[j + 0] = -ekcal * arrx[i] * invmass;
         moldyn::a[j + 1] = -ekcal * arry[i] * invmass;
         moldyn::a[j + 2] = -ekcal * arrz[i] * invmass;
         moldyn::aalt[j + 0] = 0;
         moldyn::aalt[j + 1] = 0;
         moldyn::aalt[j + 2] = 0;
      }
   }

   if (mdsaveUseUind()) {
      darray::copyout(g::q1, n, polar::uind, dup_buf_uind);
      waitFor(g::q1);
   }

   if (mdsaveUseUstc()) {
      std::vector<real> rpolev(n * MPL_TOTAL);
      darray::copyout(g::q1, n * MPL_TOTAL, rpolev.data(), &dup_buf_rpole[0][0]);
      waitFor(g::q1);
      for (int i = 0; i < n; ++i) {
         int c1 = 13 * i;
         int c2 = MPL_TOTAL * i;
         mpole::rpole[c1 + 0] = rpolev[c2 + MPL_PME_0];
         mpole::rpole[c1 + 1] = rpolev[c2 + MPL_PME_X];
         mpole::rpole[c1 + 2] = rpolev[c2 + MPL_PME_Y];
         mpole::rpole[c1 + 3] = rpolev[c2 + MPL_PME_Z];
      }
   }

   if (mdsaveUseUdir()) {
      darray::copyout(g::q1, n, polar::udir, dup_buf_udir);
      waitFor(g::q1);
   }

   if (mdsaveUseExpolUdir()) {
      darray::copyout(g::q1, 9 * n, &expol::polinv[0], &dup_buf_polinv[0][0][0]);
      waitFor(g::q1);
   }

   if (mdsaveUseExpolTef()) {
      darray::copyout(g::q1, 9 * n, &expol::polscale[0], &dup_buf_polscale[0][0][0]);
      waitFor(g::q1);
   }

   // Record mdsave_end_event when g::s1 is available.
   // g::s0 will wait until mdsave_end_event is recorded, so that the dup_
   // arrays are idle and ready to be written.
#if TINKER_CUDART
   check_rt(cudaEventRecord(mdsave_end_event, g::s1));
   check_rt(cudaStreamWaitEvent(g::s0, mdsave_end_event, 0));
#endif

   mdsaveWriteLmda();
   mdsaveWriteOst();
   mdsaveWriteTi();

   double dt1 = dt;
   double epot1 = epot;
   double eksum1 = eksum;
   tinker_f_mdsave(&istep, &dt1, &epot1, &eksum1);

   mtx_write.lock();
   idle_write = true;
   cv_write.notify_all();
   mtx_write.unlock();
}
}

namespace tinker {
void mdsaveAsync(int istep, time_prec dt)
{
   std::unique_lock<std::mutex> lck_write(mtx_write);
   cv_write.wait(lck_write, [=]() { return idle_write; });
   idle_write = false;

   fut_dup_then_write = std::async(std::launch::async, mdsaveDupThenWrite, istep, dt);

   std::unique_lock<std::mutex> lck_copy(mtx_dup);
   cv_dup.wait(lck_copy, [=]() { return idle_dup; });
   idle_dup = false;
}

void mdsaveSynchronize()
{
   if (fut_dup_then_write.valid())
      fut_dup_then_write.get();
}

void mdsaveLmdaFinal(int istep)
{
   mdsaveDupLmda();
   mdsaveWriteLmda();
   mdsaveDupOst(istep);
   mdsaveWriteOst();
   mdsaveDupTi();
   mdsaveWriteTi();

   if (dlmda::use_ostdyn)
      tinker_f_saveost();
   if (dlmda::use_metadyn)
      tinker_f_savemeta();
   if (dlmda::use_ti)
      tinker_f_saveti();
}

void mdsaveData(RcOp op)
{
   if (op & RcOp::DEALLOC) {
#if TINKER_CUDART
      check_rt(cudaEventDestroy(mdsave_begin_event));
      check_rt(cudaEventDestroy(mdsave_end_event));
#endif

      if (mdsaveUseUind())
         darray::deallocate(dup_buf_uind);

      if (mdsaveUseUstc())
         darray::deallocate(dup_buf_rpole);

      if (mdsaveUseUdir())
         darray::deallocate(dup_buf_udir);

      if (mdsaveUseExpolUdir())
         darray::deallocate(dup_buf_polinv);

      if (mdsaveUseExpolTef())
         darray::deallocate(dup_buf_polscale);

      darray::deallocate(dup_buf_x, dup_buf_y, dup_buf_z);
      darray::deallocate(dup_buf_vx, dup_buf_vy, dup_buf_vz);
      darray::deallocate(dup_buf_gx, dup_buf_gy, dup_buf_gz);

      ost_snap_active = false;
      lmda_snap_mask = 0;
      s_khist.clear(), s_ihist.clear(), s_lhist.clear(), s_fhist.clear();
      s_hhist.clear(), s_wlhist.clear(), s_wfhist.clear();
      s_mlhist.clear(), s_mhhist.clear(), s_mwhist.clear(), s_mihist.clear();
      ti_snap_active = false;
      s_tihist.clear(), s_tidedl.clear(), s_tisd.clear();
   }

   if (op & RcOp::ALLOC) {
#if TINKER_CUDART
      check_rt(cudaEventCreateWithFlags(&mdsave_begin_event, cudaEventDisableTiming));
      check_rt(cudaEventCreateWithFlags(&mdsave_end_event, cudaEventDisableTiming));
#endif

      if (mdsaveUseUind()) {
         darray::allocate(n, &dup_buf_uind);
      } else {
         dup_buf_uind = nullptr;
      }

      if (mdsaveUseUstc()) {
         darray::allocate(n, &dup_buf_rpole);
      } else {
         dup_buf_rpole = nullptr;
      }

      if (mdsaveUseUdir()) {
         darray::allocate(n, &dup_buf_udir);
      } else {
         dup_buf_udir = nullptr;
      }

      if (mdsaveUseExpolUdir()) {
         darray::allocate(n, &dup_buf_polinv);
      } else {
         dup_buf_polinv = nullptr;
      }

      if (mdsaveUseExpolTef()) {
         darray::allocate(n, &dup_buf_polscale);
      } else {
         dup_buf_polscale = nullptr;
      }

      darray::allocate(n, &dup_buf_x, &dup_buf_y, &dup_buf_z);
      darray::allocate(n, &dup_buf_vx, &dup_buf_vy, &dup_buf_vz);
      darray::allocate(n, &dup_buf_gx, &dup_buf_gy, &dup_buf_gz);
   }

   if (op & RcOp::INIT) {
      idle_dup = false;
      idle_write = true;
      lmda_snap_mask = 0;
      ost_snap_active = false;
      ti_snap_active = false;
   }
}
}
