#include "md/pq.h"
#include "tool/darray.h"
#include "tool/externfunc.h"
#include <tinker/detail/moldyn.hh>

namespace tinker {
TINKER_FVOID2(acc1, cu1, mdPos, time_prec, pos_prec*, pos_prec*, pos_prec*, const vel_prec*, const vel_prec*,
   const vel_prec*);
void mdPos(time_prec dt, pos_prec* qx, pos_prec* qy, pos_prec* qz, const vel_prec* vlx, const vel_prec* vly,
   const vel_prec* vlz)
{
   TINKER_FCALL2(acc1, cu1, mdPos, dt, qx, qy, qz, vlx, vly, vlz);
}

void mdPos(time_prec dt)
{
   TINKER_FCALL2(acc1, cu1, mdPos, dt, xpos, ypos, zpos, vx, vy, vz);
}

TINKER_FVOID2(acc1, cu1, mdPosAxbv, pos_prec, pos_prec);
void mdPosAxbv(pos_prec a, pos_prec b)
{
   TINKER_FCALL2(acc1, cu1, mdPosAxbv, a, b);
}

TINKER_FVOID2(acc1, cu1, mdPosAxbvAn, pos_prec (*)[3], pos_prec (*)[3]);
void mdPosAxbvAn(pos_prec (*a)[3], pos_prec (*b)[3])
{
   TINKER_FCALL2(acc1, cu1, mdPosAxbvAn, a, b);
}

TINKER_FVOID2(acc1, cu0, mdDebugPosNorm, pos_prec, time_prec, const vel_prec*, const vel_prec*, const vel_prec*);
void mdDebugPosNorm(pos_prec poseps, time_prec dt, //
   const vel_prec* vx, const vel_prec* vy, const vel_prec* vz)
{
   TINKER_FCALL2(acc1, cu0, mdDebugPosNorm, poseps, dt, vx, vy, vz);
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, mdVel, time_prec, vel_prec*, vel_prec*, vel_prec*, const grad_prec*, const grad_prec*,
   const grad_prec*);
void mdVel(time_prec dt, const grad_prec* grx, const grad_prec* gry, const grad_prec* grz)
{
   TINKER_FCALL2(acc1, cu1, mdVel, dt, vx, vy, vz, grx, gry, grz);
}

TINKER_FVOID2(acc1, cu1, mdVel2, time_prec, const grad_prec*, const grad_prec*, const grad_prec*, time_prec,
   const grad_prec*, const grad_prec*, const grad_prec*);
void mdVel2(time_prec dt, const grad_prec* grx, const grad_prec* gry, const grad_prec* grz, time_prec dt2,
   const grad_prec* grx2, const grad_prec* gry2, const grad_prec* grz2)
{
   TINKER_FCALL2(acc1, cu1, mdVel2, dt, grx, gry, grz, dt2, grx2, gry2, grz2);
}

TINKER_FVOID2(acc1, cu1, mdVelScale, vel_prec, int, vel_prec*, vel_prec*, vel_prec*);
void mdVelScale(vel_prec scal, int nelem, vel_prec* vx, vel_prec* vy, vel_prec* vz)
{
   TINKER_FCALL2(acc1, cu1, mdVelScale, scal, nelem, vx, vy, vz);
}

TINKER_FVOID2(acc1, cu1, mdVelAvbf, int, vel_prec, vel_prec, vel_prec*, vel_prec*, vel_prec*, const grad_prec*,
   const grad_prec*, const grad_prec*, const grad_prec*, const grad_prec*, const grad_prec*);
void mdVelAvbf(int nrespa, vel_prec a, vel_prec b, const grad_prec* gx1, const grad_prec* gy1, const grad_prec* gz1,
   const grad_prec* gx2, const grad_prec* gy2, const grad_prec* gz2)
{
   TINKER_FCALL2(acc1, cu1, mdVelAvbf, nrespa, a, b, vx, vy, vz, gx1, gy1, gz1, gx2, gy2, gz2);
}

TINKER_FVOID2(acc1, cu1, mdVelAvbfAn, int, vel_prec (*)[3], vel_prec (*)[3], vel_prec*, vel_prec*, vel_prec*,
   const grad_prec*, const grad_prec*, const grad_prec*, const grad_prec*, const grad_prec*, const grad_prec*);
void mdVelAvbfAn(int nrespa, vel_prec a[3][3], vel_prec b[3][3], const grad_prec* gx1, const grad_prec* gy1,
   const grad_prec* gz1, const grad_prec* gx2, const grad_prec* gy2, const grad_prec* gz2)
{
   TINKER_FCALL2(acc1, cu1, mdVelAvbfAn, nrespa, a, b, vx, vy, vz, gx1, gy1, gz1, gx2, gy2, gz2);
}
}

namespace tinker {
void mdVelData(RcOp op)
{
   if (not(calc::vel & rc_flag))
      return;

   if (op & RcOp::DEALLOC) {
      darray::deallocate(vx, vy, vz);
   }

   if (op & RcOp::ALLOC) {
      darray::allocate(n, &vx, &vy, &vz);
   }

   if (op & RcOp::INIT) {
      std::vector<vel_prec> vvx(n), vvy(n), vvz(n);
      for (int i = 0; i < n; ++i) {
         vvx[i] = moldyn::v[3 * i + 0];
         vvy[i] = moldyn::v[3 * i + 1];
         vvz[i] = moldyn::v[3 * i + 2];
      }
      darray::copyin(g::q0, n, vx, vvx.data());
      darray::copyin(g::q0, n, vy, vvy.data());
      darray::copyin(g::q0, n, vz, vvz.data());
      waitFor(g::q0);
   }
}
}
