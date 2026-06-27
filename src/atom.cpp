#include "ff/atom.h"
#include "ff/box.h"
#include "ff/energybuffer.h"
#include "ff/nblist.h"
#include "ff/potent.h"
#include "tool/darray.h"
#include "tool/error.h"
#include "tool/externfunc.h"
#include "tool/gpucard.h"
#include <tinker/detail/atomid.hh>
#include <tinker/detail/atoms.hh>
#include <tinker/detail/bound.hh>
#include <tinker/detail/usage.hh>

#include <cassert>

namespace tinker {
void nData(RcOp op)
{
   if (op & RcOp::DEALLOC) {
      trajn = -1;
      n = 0;
      padded_n = 0;
      nelem_buffer = 0;
   }

   if (op & RcOp::ALLOC) {
      n = atoms::n;
      padded_n = (n + WARP_SIZE - 1) / WARP_SIZE;
      padded_n *= WARP_SIZE;

      if (calc::traj & rc_flag) {
         // trajn must have been initialized by this point
         assert(trajn >= 0);
      }

#if TINKER_CUDART
      nelem_buffer = gpuMaxNParallel(idevice);
      nelem_buffer = pow2Ge(nelem_buffer);
#else
      nelem_buffer = 1;
#endif

      if (usage::nuse != n) {
         TINKER_THROW("All atoms must be active.");
      }
   }
}

void massData(RcOp op)
{
   if (not(calc::mass & rc_flag))
      return;

   if (op & RcOp::DEALLOC) {
      darray::deallocate(mass, massinv);
   }

   if (op & RcOp::ALLOC) {
      darray::allocate(n, &mass, &massinv);
   }

   if (op & RcOp::INIT) {
      std::vector<double> mbuf(n);
      for (int i = 0; i < n; ++i)
         mbuf[i] = 1 / atomid::mass[i];
      darray::copyin(g::q0, n, massinv, mbuf.data());
      darray::copyin(g::q0, n, mass, atomid::mass);
      waitFor(g::q0);
   }
}

void atomicData(RcOp op)
{
   if (not(use(Potent::CHGFLX) | use(Potent::NNVAL) | use(Potent::NNMET)))
      return;

   if (op & RcOp::DEALLOC) {
      darray::deallocate(atomic);
   }

   if (op & RcOp::ALLOC) {
      darray::allocate(n, &atomic);
   }

   if (op & RcOp::INIT) {
      darray::copyin(g::q0, n, atomic, atomid::atomic);
      waitFor(g::q0);
   }
}

void xyzData(RcOp op)
{
   if (not(calc::xyz & rc_flag))
      return;

   if (op & RcOp::DEALLOC) {
      if (calc::traj & rc_flag) {
         darray::deallocate(trajx, trajy, trajz);
         x = nullptr;
         y = nullptr;
         z = nullptr;
      } else {
         trajx = nullptr;
         trajy = nullptr;
         trajz = nullptr;
         darray::deallocate(x, y, z);
         if (sizeof(pos_prec) == sizeof(real)) {
            xpos = nullptr;
            ypos = nullptr;
            zpos = nullptr;
         } else {
            darray::deallocate(xpos, ypos, zpos);
         }
      }
   }

   if (op & RcOp::ALLOC) {
      if (calc::traj & rc_flag) {
         darray::allocate(n * trajn, &trajx, &trajy, &trajz);
         x = trajx;
         y = trajy;
         z = trajz;
      } else {
         darray::allocate(n, &x, &y, &z);
         if (sizeof(pos_prec) == sizeof(real)) {
            xpos = (pos_prec*)x;
            ypos = (pos_prec*)y;
            zpos = (pos_prec*)z;
         } else {
            darray::allocate(n, &xpos, &ypos, &zpos);
         }
      }
   }

   if (op & RcOp::INIT) {
      if (calc::traj & rc_flag) {
         darray::copyin(g::q0, n, x, atoms::x);
         darray::copyin(g::q0, n, y, atoms::y);
         darray::copyin(g::q0, n, z, atoms::z);
      } else {
         darray::copyin(g::q0, n, xpos, atoms::x);
         darray::copyin(g::q0, n, ypos, atoms::y);
         darray::copyin(g::q0, n, zpos, atoms::z);
         copyPosToXyz();
      }
   }
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, copyPosToXyz);
void copyPosToXyz()
{
   if CONSTEXPR (sizeof(pos_prec) == sizeof(real))
      return;

   TINKER_FCALL2(acc1, cu1, copyPosToXyz);
}

void copyPosToXyz(bool refreshNBList)
{
   copyPosToXyz();
   if (refreshNBList)
      nblistRefresh();
}

TINKER_FVOID2(acc1, cu1, bounds);
void bounds()
{
   if (not bound::use_bounds)
      return;

   TINKER_FCALL2(acc1, cu1, bounds);
   copyPosToXyz();
}
}
