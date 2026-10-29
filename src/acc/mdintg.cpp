#include "md/misc.h"
#include "md/pq.h"
#include "tool/externfunc.h"
#include "tool/ioprint.h"
#include <tinker/detail/bound.hh>
#include <tinker/detail/inform.hh>
#include <tinker/detail/mdstuf.hh>
#include <tinker/detail/molcul.hh>
#include <tinker/detail/units.hh>
#include <tinker/routines.h>

namespace tinker {
TINKER_FVOID2(acc0, cu1, mdrestRemovePbcMomentum, bool, vel_prec&, vel_prec&, vel_prec&);
void mdrest_acc(int istep)
{
   if (not mdstuf::dorest)
      return;
   if ((istep % mdstuf::irest) != 0)
      return;

   const energy_prec ekcal = units::ekcal;

   // zero out the total mass and overall linear velocity

   auto totmass = molcul::totmass;
   vel_prec vtot1 = 0, vtot2 = 0, vtot3 = 0;

#if TINKER_CUDART
   if (pltfm_config & Platform::CUDA) {
      bool copyout = inform::debug or not bound::use_bounds;
      TINKER_FCALL2(acc0, cu1, mdrestRemovePbcMomentum, copyout, vtot1, vtot2, vtot3);
   } else
#endif
   {
      // compute linear velocity of the system center of mass

      vtot1 = 0, vtot2 = 0, vtot3 = 0;
      #pragma acc parallel loop independent async\
                  copy(vtot1,vtot2,vtot3) reduction(+:vtot1,vtot2,vtot3)\
                  deviceptr(mass,vx,vy,vz)
      for (int i = 0; i < n; ++i) {
         auto weigh = mass[i];
         vtot1 += vx[i] * weigh;
         vtot2 += vy[i] * weigh;
         vtot3 += vz[i] * weigh;
      }
      #pragma acc wait

      vtot1 /= totmass;
      vtot2 /= totmass;
      vtot3 /= totmass;

      // eliminate any translation of the overall system

      #pragma acc parallel loop independent async deviceptr(vx,vy,vz)
      for (int i = 0; i < n; ++i) {
         vx[i] -= vtot1;
         vy[i] -= vtot2;
         vz[i] -= vtot3;
      }
   }

   // print the translational velocity of the overall system

   mdrestPrintP1(static_cast<bool>(inform::debug), vtot1, vtot2, vtot3, totmass);

   if (not bound::use_bounds) {
      energy_prec erot = 0;
      pos_prec xtot = 0, ytot = 0, ztot = 0;
      vel_prec vang[3] = {0}; // angular momentum

      // find the center of mass coordinates of the overall system
      // compute the angular momentum of the overall system

      vel_prec mang1 = 0, mang2 = 0, mang3 = 0;

      #pragma acc parallel loop independent async\
                  copy(xtot,ytot,ztot,mang1,mang2,mang3)\
                  reduction(+:xtot,ytot,ztot,mang1,mang2,mang3)\
                  deviceptr(mass,xpos,ypos,zpos,vx,vy,vz)
      for (int i = 0; i < n; ++i) {
         auto weigh = mass[i];
         xtot += xpos[i] * weigh;
         ytot += ypos[i] * weigh;
         ztot += zpos[i] * weigh;
         mang1 += (ypos[i] * vz[i] - zpos[i] * vy[i]) * weigh;
         mang2 += (zpos[i] * vx[i] - xpos[i] * vz[i]) * weigh;
         mang3 += (xpos[i] * vy[i] - ypos[i] * vx[i]) * weigh;
      }
      #pragma acc wait
      xtot /= totmass;
      ytot /= totmass;
      ztot /= totmass;

      // calculate the moment of inertia tensor

      pos_prec xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;

      #pragma acc parallel loop independent async\
                  copy(xx,xy,xz,yy,yz,zz) reduction(+:xx,xy,xz,yy,yz,zz)\
                  deviceptr(mass,xpos,ypos,zpos)
      for (int i = 0; i < n; ++i) {
         auto weigh = mass[i];
         pos_prec xdel = xpos[i] - xtot;
         pos_prec ydel = ypos[i] - ytot;
         pos_prec zdel = zpos[i] - ztot;
         xx += xdel * xdel * weigh;
         xy += xdel * ydel * weigh;
         xz += xdel * zdel * weigh;
         yy += ydel * ydel * weigh;
         yz += ydel * zdel * weigh;
         zz += zdel * zdel * weigh;
      }
      #pragma acc wait

      double tensor[3][3];
      double eps = (n <= 2 ? 0.000001 : 0);
      tensor[0][0] = yy + zz + eps;
      tensor[0][1] = -xy;
      tensor[0][2] = -xz;
      tensor[1][0] = -xy;
      tensor[1][1] = xx + zz + eps;
      tensor[1][2] = -yz;
      tensor[2][0] = -xz;
      tensor[2][1] = -yz;
      tensor[2][2] = xx + yy + eps;

      int ndim = 3;
      tinker_f_invert(&ndim, &tensor[0][0]);

      // compute angular velocity and rotational kinetic energy

      for (int i = 0; i < 3; ++i) {
         vang[i] = tensor[0][i] * mang1 + tensor[1][i] * mang2 + tensor[2][i] * mang3;
      }
      erot = vang[0] * mang1 + vang[1] * mang2 + vang[2] * mang3;
      erot *= (0.5f / ekcal);

      // eliminate any rotation about the system center of mass

      #pragma acc parallel loop independent async\
                  deviceptr(xpos,ypos,zpos,vx,vy,vz)\
                  copyin(vang[0:3])
      for (int i = 0; i < n; ++i) {
         pos_prec xdel = xpos[i] - xtot;
         pos_prec ydel = ypos[i] - ytot;
         pos_prec zdel = zpos[i] - ztot;
         vx[i] = vx[i] - vang[1] * zdel + vang[2] * ydel;
         vy[i] = vy[i] - vang[2] * xdel + vang[0] * zdel;
         vz[i] = vz[i] - vang[0] * ydel + vang[1] * xdel;
      }

      // print the angular velocity of the overall system

      if (inform::debug) {
         print(stdout,
            " System Angular Velocity : %12.2e%12.2e%12.2e\n Rotational "
            "Kinetic Energy :%13s%12.4f Kcal/mole\n",
            vang[0], vang[1], vang[2], "", erot);
      }
   }
}
}
