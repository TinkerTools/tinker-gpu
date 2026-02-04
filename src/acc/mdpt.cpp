#include "ff/energy.h"
#include "ff/molecule.h"
#include "md/misc.h"
#include <tinker/detail/bath.hh>
#include <tinker/detail/bound.hh>
#include <tinker/detail/units.hh>

namespace tinker {
void kineticEnergy_acc(energy_prec& eksum_out, energy_prec (&ekin_out)[3][3], int n,
   const double* mass, const vel_prec* vx, const vel_prec* vy, const vel_prec* vz)
{
   const energy_prec ekcal_inv = 1.0 / units::ekcal;
   energy_prec exx = 0, eyy = 0, ezz = 0, exy = 0, eyz = 0, ezx = 0;
   #pragma acc parallel loop independent async\
               copy(exx,eyy,ezz,exy,eyz,ezx)\
               reduction(+:exx,eyy,ezz,exy,eyz,ezx)\
               deviceptr(mass,vx,vy,vz)
   for (int i = 0; i < n; ++i) {
      energy_prec term = 0.5 * mass[i] * ekcal_inv;
      exx += term * vx[i] * vx[i];
      eyy += term * vy[i] * vy[i];
      ezz += term * vz[i] * vz[i];
      exy += term * vx[i] * vy[i];
      eyz += term * vy[i] * vz[i];
      ezx += term * vz[i] * vx[i];
   }
   #pragma acc wait

   ekin_out[0][0] = exx;
   ekin_out[0][1] = exy;
   ekin_out[0][2] = ezx;
   ekin_out[1][0] = exy;
   ekin_out[1][1] = eyy;
   ekin_out[1][2] = eyz;
   ekin_out[2][0] = ezx;
   ekin_out[2][1] = eyz;
   ekin_out[2][2] = ezz;
   eksum_out = exx + eyy + ezz;
}

//====================================================================//

void berendsenBarostat_acc(time_prec dt, bool aniso)
{
   if (not bound::use_bounds)
      return;

   double vol = boxVolume();
   double factor = units::prescon / vol;
   double stress[3][3];
   for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
         int iv = 3 * i + j;
         stress[i][j] = factor * (2 * ekin[i][j] - vir[iv]);
      }
   }
   const double third = 1.0 / 3;
   double pres = (stress[0][0] + stress[1][1] + stress[2][2]) * third;

   if (aniso) {
      double scale = third * dt * bath::compress / bath::taupres;
      double ascale[3][3];
      for (int i = 0; i < 3; ++i) {
         for (int j = 0; j < 3; ++j) {
            if (j == i) {
               ascale[i][j] = 1 + scale * (stress[i][i] - bath::atmsph);
            } else {
               ascale[i][j] = scale * stress[i][j];
            }
         }
      }

      double temp[3][3], hbox[3][3];
      temp[0][0] = lvec1.x;
      temp[0][1] = 0;
      temp[0][2] = 0;
      temp[1][0] = lvec1.y;
      temp[1][1] = lvec2.y;
      temp[1][2] = 0;
      temp[2][0] = lvec1.z;
      temp[2][1] = lvec2.z;
      temp[2][2] = lvec3.z;
      for (int i = 0; i < 3; ++i) {
         for (int j = 0; j < 3; ++j) {
            hbox[i][j] = 0;
            for (int k = 0; k < 3; ++k) {
               hbox[i][j] += ascale[k][j] * temp[i][k];
            }
         }
      }
      double l0, l1, l2, a0 = 90, a1 = 90, a2 = 90;
      l0 = std::sqrt(hbox[0][0] * hbox[0][0] + hbox[0][1] * hbox[0][1] + hbox[0][2] * hbox[0][2]);
      l1 = std::sqrt(hbox[1][0] * hbox[1][0] + hbox[1][1] * hbox[1][1] + hbox[1][2] * hbox[1][2]);
      l2 = std::sqrt(hbox[2][0] * hbox[2][0] + hbox[2][1] * hbox[2][1] + hbox[2][2] * hbox[2][2]);
      if (box_shape == BoxShape::MONO or box_shape == BoxShape::TRI) {
         // beta
         a1 = (hbox[0][0] * hbox[2][0] + hbox[0][1] * hbox[2][1] + hbox[0][2] * hbox[2][2]) /
            (l0 * l2);
         a1 = radian * std::acos(a1);
      }
      if (box_shape == BoxShape::TRI) {
         // alpha
         a0 = (hbox[1][0] * hbox[2][0] + hbox[1][1] * hbox[2][1] + hbox[1][2] * hbox[2][2]) /
            (l1 * l2);
         a0 = radian * std::acos(a0);
         // gamma
         a2 = (hbox[0][0] * hbox[1][0] + hbox[0][1] * hbox[1][1] + hbox[0][2] * hbox[1][2]) /
            (l0 * l1);
         a2 = radian * std::acos(a2);
      }
      Box newbox;
      boxLattice(newbox, box_shape, l0, l1, l2, a0, a1, a2);
      boxSetCurrent(newbox);

      #pragma acc parallel loop independent async\
                  deviceptr(xpos,ypos,zpos)\
                  firstprivate(ascale[0:3][0:3])
      for (int i = 0; i < n; ++i) {
         pos_prec xk = xpos[i];
         pos_prec yk = ypos[i];
         pos_prec zk = zpos[i];
         xpos[i] = xk * ascale[0][0] + yk * ascale[0][1] + zk * ascale[0][2];
         ypos[i] = xk * ascale[1][0] + yk * ascale[1][1] + zk * ascale[1][2];
         zpos[i] = xk * ascale[2][0] + yk * ascale[2][1] + zk * ascale[2][2];
      }
      copyPosToXyz();
   } else {
      double scale = 1 + (dt * bath::compress / bath::taupres) * (pres - bath::atmsph);
      scale = std::pow(scale, third);

      lvec1 *= scale;
      lvec2 *= scale;
      lvec3 *= scale;
      boxSetCurrentRecip();

      #pragma acc parallel loop independent async deviceptr(xpos,ypos,zpos)
      for (int i = 0; i < n; ++i) {
         xpos[i] *= scale;
         ypos[i] *= scale;
         zpos[i] *= scale;
      }
      copyPosToXyz();
   }
}

void monteCarloMolMove_acc(double scale)
{
   int nmol = molecule.nmol;
   const auto* imol = molecule.imol;
   const auto* kmol = molecule.kmol;
   const auto* molmass = molecule.molmass;
   pos_prec pos_scale = scale - 1;
   #pragma acc parallel loop independent async\
               deviceptr(imol,kmol,mass,molmass,xpos,ypos,zpos)
   for (int i = 0; i < nmol; ++i) {
      pos_prec xcm = 0, ycm = 0, zcm = 0;
      int start = imol[i][0];
      int stop = imol[i][1];
      #pragma acc loop seq
      for (int j = start; j < stop; ++j) {
         int k = kmol[j];
         auto weigh = mass[k];
         xcm += xpos[k] * weigh;
         ycm += ypos[k] * weigh;
         zcm += zpos[k] * weigh;
      }
      pos_prec term = pos_scale / molmass[i];
      pos_prec xmove, ymove, zmove;
      xmove = term * xcm;
      ymove = term * ycm;
      zmove = term * zcm;
      #pragma acc loop seq
      for (int j = start; j < stop; ++j) {
         int k = kmol[j];
         xpos[k] += xmove;
         ypos[k] += ymove;
         zpos[k] += zmove;
      }
   }
}
}
