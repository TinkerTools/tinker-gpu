#include "ff/energy.h"
#include "ff/modamoeba.h"
#include "ff/modhippo.h"
#include "ff/potent.h"
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
#include <tinker/detail/expol.hh>
#include <tinker/detail/files.hh>
#include <tinker/detail/moldyn.hh>
#include <tinker/detail/mpole.hh>
#include <tinker/detail/output.hh>
#include <tinker/detail/polar.hh>
#include <tinker/detail/polpot.hh>
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
   }
}
}
