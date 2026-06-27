#include "md/pq.h"
#include "seq/add.h"
#include "tool/error.h"
#include <tinker/detail/units.hh>

namespace tinker {
void mdPos_acc(time_prec dt, pos_prec* qx, pos_prec* qy, pos_prec* qz, const vel_prec* vlx, const vel_prec* vly,
   const vel_prec* vlz)
{
   #pragma acc parallel loop independent async deviceptr(qx,qy,qz,vlx,vly,vlz)
   for (int i = 0; i < n; ++i) {
      qx[i] += dt * vlx[i];
      qy[i] += dt * vly[i];
      qz[i] += dt * vlz[i];
   }
}

void mdPosAxbv_acc(pos_prec sa, pos_prec sb)
{
   #pragma acc parallel loop independent async deviceptr(xpos,ypos,zpos,vx,vy,vz)
   for (int i = 0; i < n; ++i) {
      xpos[i] = sa * xpos[i] + sb * vx[i];
      ypos[i] = sa * ypos[i] + sb * vy[i];
      zpos[i] = sa * zpos[i] + sb * vz[i];
   }
}

void mdPosAxbvAn_acc(pos_prec (*a)[3], pos_prec (*b)[3])
{
   auto a00 = a[0][0], a01 = a[0][1], a02 = a[0][2];
   auto a10 = a[1][0], a11 = a[1][1], a12 = a[1][2];
   auto a20 = a[2][0], a21 = a[2][1], a22 = a[2][2];
   auto b00 = b[0][0], b01 = b[0][1], b02 = b[0][2];
   auto b10 = b[1][0], b11 = b[1][1], b12 = b[1][2];
   auto b20 = b[2][0], b21 = b[2][1], b22 = b[2][2];
   #pragma acc parallel loop independent async\
               deviceptr(xpos,ypos,zpos,vx,vy,vz)
   for (int i = 0; i < n; ++i) {
      auto o = xpos[i], p = ypos[i], q = zpos[i];
      auto r = vx[i], s = vy[i], t = vz[i];
      xpos[i] = (a00 * o + a01 * p + a02 * q) + (b00 * r + b01 * s + b02 * t);
      ypos[i] = (a10 * o + a11 * p + a12 * q) + (b10 * r + b11 * s + b12 * t);
      zpos[i] = (a20 * o + a21 * p + a22 * q) + (b20 * r + b21 * s + b22 * t);
   }
}

void mdDebugPosNorm_acc(pos_prec poseps, time_prec dt, //
   const vel_prec* vlx, const vel_prec* vly, const vel_prec* vlz)
{
   int which = -1;
   pos_prec tol2 = poseps * poseps;
   #pragma acc parallel loop independent async\
               copy(which) reduction(max:which)\
               deviceptr(vlx,vly,vlz)
   for (int i = 0; i < n; ++i) {
      pos_prec x1, y1, z1, norm2;
      x1 = dt * vlx[i];
      y1 = dt * vly[i];
      z1 = dt * vlz[i];
      norm2 = x1 * x1 + y1 * y1 + z1 * z1;
      bool big = (norm2 > tol2) or (norm2 != norm2); // big or NaN
      int flag = big ? i : -1;
      which = which > flag ? which : flag;
   }
   #pragma acc wait

   if (which >= 0) {
      printError();
      TINKER_THROW(format("MD-DEBUG POS UPDATE  --  Atom %d Tried to Move More"
                          " than %.4lf Angstroms",
         which + 1, poseps));
   }
}
}

namespace tinker {
void mdVel_acc(time_prec dt, vel_prec* vlx, vel_prec* vly, vel_prec* vlz, const grad_prec* grx, const grad_prec* gry,
   const grad_prec* grz)
{
   const vel_prec ekcal = units::ekcal;
   #pragma acc parallel loop independent async\
               deviceptr(massinv,vlx,vly,vlz,grx,gry,grz)
   for (int i = 0; i < n; ++i) {
      vel_prec coef = -ekcal * massinv[i] * dt;
      vlx[i] += coef * toFloatGrad<vel_prec>(grx[i]);
      vly[i] += coef * toFloatGrad<vel_prec>(gry[i]);
      vlz[i] += coef * toFloatGrad<vel_prec>(grz[i]);
   }
}

void mdVel2_acc(time_prec dt, const grad_prec* grx, const grad_prec* gry, const grad_prec* grz, time_prec dt2,
   const grad_prec* grx2, const grad_prec* gry2, const grad_prec* grz2)
{
   const vel_prec ekcal = units::ekcal;
   #pragma acc parallel loop independent async\
               deviceptr(massinv,vx,vy,vz,grx,gry,grz,grx2,gry2,grz2)
   for (int i = 0; i < n; ++i) {
      vel_prec coef = -ekcal * massinv[i];
      // clang-format off
      vx[i] += coef*(toFloatGrad<vel_prec>(grx[i])*dt+toFloatGrad<vel_prec>(grx2[i])*dt2);
      vy[i] += coef*(toFloatGrad<vel_prec>(gry[i])*dt+toFloatGrad<vel_prec>(gry2[i])*dt2);
      vz[i] += coef*(toFloatGrad<vel_prec>(grz[i])*dt+toFloatGrad<vel_prec>(grz2[i])*dt2);
      // clang-format on
   }
}

void mdVelScale_acc(vel_prec sc, int nelem, vel_prec* vx0, vel_prec* vy0, vel_prec* vz0)
{
   #pragma acc parallel loop independent async deviceptr(vx0,vy0,vz0)
   for (int i = 0; i < nelem; ++i) {
      vx0[i] *= sc;
      vy0[i] *= sc;
      vz0[i] *= sc;
   }
}

void mdVelAvbf_acc(int nrespa, vel_prec a, vel_prec b, vel_prec* vx, vel_prec* vy, vel_prec* vz, const grad_prec* gx1,
   const grad_prec* gy1, const grad_prec* gz1, const grad_prec* gx2, const grad_prec* gy2, const grad_prec* gz2)
{
   auto ekcal = units::ekcal;
   if (nrespa == 1)
      goto label_nrespa1;
   else
      goto label_nrespa2;

label_nrespa1:
   #pragma acc parallel loop independent async\
               deviceptr(massinv,vx,vy,vz,gx1,gy1,gz1)
   for (int i = 0; i < n; ++i) {
      auto coef = -ekcal * massinv[i];
      auto v0x = vx[i];
      auto v0y = vy[i];
      auto v0z = vz[i];
      auto grx = toFloatGrad<vel_prec>(gx1[i]);
      auto gry = toFloatGrad<vel_prec>(gy1[i]);
      auto grz = toFloatGrad<vel_prec>(gz1[i]);
      vx[i] = a * v0x + coef * b * grx;
      vy[i] = a * v0y + coef * b * gry;
      vz[i] = a * v0z + coef * b * grz;
   }
   return;

label_nrespa2:
   #pragma acc parallel loop independent async\
               deviceptr(massinv,vx,vy,vz,gx1,gy1,gz1,gx2,gy2,gz2)
   for (int i = 0; i < n; ++i) {
      auto coef = -ekcal * massinv[i];
      auto v0x = vx[i];
      auto v0y = vy[i];
      auto v0z = vz[i];
      auto grx = toFloatGrad<vel_prec>(gx1[i]) / nrespa + toFloatGrad<vel_prec>(gx2[i]);
      auto gry = toFloatGrad<vel_prec>(gy1[i]) / nrespa + toFloatGrad<vel_prec>(gy2[i]);
      auto grz = toFloatGrad<vel_prec>(gz1[i]) / nrespa + toFloatGrad<vel_prec>(gz2[i]);
      vx[i] = a * v0x + coef * b * grx;
      vy[i] = a * v0y + coef * b * gry;
      vz[i] = a * v0z + coef * b * grz;
   }
   return;
}

void mdVelAvbfAn_acc(int nrespa, vel_prec (*a)[3], vel_prec (*b)[3], vel_prec* vx, vel_prec* vy, vel_prec* vz,
   const grad_prec* gx1, const grad_prec* gy1, const grad_prec* gz1, const grad_prec* gx2, const grad_prec* gy2,
   const grad_prec* gz2)
{
   auto ekcal = units::ekcal;
   auto a00 = a[0][0], a01 = a[0][1], a02 = a[0][2];
   auto a10 = a[1][0], a11 = a[1][1], a12 = a[1][2];
   auto a20 = a[2][0], a21 = a[2][1], a22 = a[2][2];
   auto b00 = b[0][0], b01 = b[0][1], b02 = b[0][2];
   auto b10 = b[1][0], b11 = b[1][1], b12 = b[1][2];
   auto b20 = b[2][0], b21 = b[2][1], b22 = b[2][2];

   if (nrespa == 1)
      goto label_nrespa1;
   else
      goto label_nrespa2;

label_nrespa1:
   #pragma acc parallel loop independent async\
               deviceptr(massinv,vx,vy,vz,gx1,gy1,gz1)
   for (int i = 0; i < n; ++i) {
      auto coef = -ekcal * massinv[i];
      auto v0x = vx[i];
      auto v0y = vy[i];
      auto v0z = vz[i];
      auto grx = toFloatGrad<vel_prec>(gx1[i]);
      auto gry = toFloatGrad<vel_prec>(gy1[i]);
      auto grz = toFloatGrad<vel_prec>(gz1[i]);
      // clang-format off
      vx[i] = a00*v0x + a01*v0y + a02*v0z + coef*(b00*grx+b01*gry+b02*grz);
      vy[i] = a10*v0x + a11*v0y + a12*v0z + coef*(b10*grx+b11*gry+b12*grz);
      vz[i] = a20*v0x + a21*v0y + a22*v0z + coef*(b20*grx+b21*gry+b22*grz);
      // clang-format on
   }
   return;

label_nrespa2:
   #pragma acc parallel loop independent async\
               deviceptr(massinv,vx,vy,vz,gx1,gy1,gz1,gx2,gy2,gz2)
   for (int i = 0; i < n; ++i) {
      auto coef = -ekcal * massinv[i];
      auto v0x = vx[i];
      auto v0y = vy[i];
      auto v0z = vz[i];
      auto grx = toFloatGrad<vel_prec>(gx1[i]) / nrespa + toFloatGrad<vel_prec>(gx2[i]);
      auto gry = toFloatGrad<vel_prec>(gy1[i]) / nrespa + toFloatGrad<vel_prec>(gy2[i]);
      auto grz = toFloatGrad<vel_prec>(gz1[i]) / nrespa + toFloatGrad<vel_prec>(gz2[i]);
      // clang-format off
      vx[i] = a00*v0x + a01*v0y + a02*v0z + coef*(b00*grx+b01*gry+b02*grz);
      vy[i] = a10*v0x + a11*v0y + a12*v0z + coef*(b10*grx+b11*gry+b12*grz);
      vz[i] = a20*v0x + a21*v0y + a22*v0z + coef*(b20*grx+b21*gry+b22*grz);
      // clang-format on
   }
   return;
}
}
