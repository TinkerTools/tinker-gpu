#include "md/stochastic.h"
#include "ff/atom.h"
#include "ff/energy.h"
#include "md/pq.h"
#include "seq/add.h"
#include "seq/launch.h"
#include "seq/philox.h"
#include <cmath>
#include <tinker/detail/units.hh>

namespace tinker {
__global__
static void sdTerm_cu1(int n, unsigned int seed, unsigned int step,      //
   pos_prec psig, vel_prec vsig, vel_prec rho, vel_prec rhoc,            //
   const double* restrict massinv,                                       //
   pos_prec (*restrict prand)[3], vel_prec (*restrict vrand)[3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      // massinv is a double array, so the root is taken there and narrowed on
      // assignment, the same way mdVel lets massinv into a vel_prec expression
      auto rtminv = sqrt(massinv[i]);
      pos_prec ps = psig * rtminv;
      vel_prec vs = vsig * rtminv;
      vel_prec pn0, vn0, pn1, vn1, pn2, vn2;
      philoxNormal2(pn0, vn0, seed, step, i, 0);
      philoxNormal2(pn1, vn1, seed, step, i, 1);
      philoxNormal2(pn2, vn2, seed, step, i, 2);
      prand[i][0] = ps * pn0;
      prand[i][1] = ps * pn1;
      prand[i][2] = ps * pn2;
      vrand[i][0] = vs * (rho * pn0 + rhoc * vn0);
      vrand[i][1] = vs * (rho * pn1 + rhoc * vn1);
      vrand[i][2] = vs * (rho * pn2 + rhoc * vn2);
   }
}

void sdTerm_cu(int istep, unsigned int seed, pos_prec psig, vel_prec vsig, vel_prec rho, pos_prec (*prand)[3],
   vel_prec (*vrand)[3])
{
   vel_prec rhoc = std::sqrt(1 - (double)rho * rho);
   launch_k1s(g::s0, n, sdTerm_cu1, //
      n, seed, (unsigned int)istep, psig, vsig, rho, rhoc, massinv, prand, vrand);
}

__global__
static void sdPos_cu1(int n, vel_prec pfric, time_prec vfric, time_prec vfric2, time_prec afric,  //
   vel_prec ekcal,                                                                                //
   pos_prec* restrict xpos, pos_prec* restrict ypos, pos_prec* restrict zpos,                     //
   vel_prec* restrict vx, vel_prec* restrict vy, vel_prec* restrict vz,                           //
   const grad_prec* restrict gx, const grad_prec* restrict gy, const grad_prec* restrict gz,      //
   const double* restrict massinv, const pos_prec (*restrict prand)[3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      // a(t), from the gradient left over from the previous step
      vel_prec coef = -ekcal * massinv[i];
      vel_prec ax = coef * toFloatGrad<vel_prec>(gx[i]);
      vel_prec ay = coef * toFloatGrad<vel_prec>(gy[i]);
      vel_prec az = coef * toFloatGrad<vel_prec>(gz[i]);

      xpos[i] += vx[i] * vfric + ax * afric + prand[i][0];
      ypos[i] += vy[i] * vfric + ay * afric + prand[i][1];
      zpos[i] += vz[i] * vfric + az * afric + prand[i][2];

      vx[i] = vx[i] * pfric + ax * vfric2;
      vy[i] = vy[i] * pfric + ay * vfric2;
      vz[i] = vz[i] * pfric + az * vfric2;
   }
}

void sdPos_cu(vel_prec pfric, time_prec vfric, time_prec afric, const pos_prec (*prand)[3])
{
   launch_k1s(g::s0, n, sdPos_cu1, //
      n, pfric, vfric, (time_prec)(0.5 * vfric), afric, (vel_prec)units::ekcal, xpos, ypos, zpos, vx, vy, vz, gx, gy,
      gz, massinv, prand);
}

__global__
static void sdVel2_cu1(int n, time_prec vfric2, vel_prec ekcal,                                //
   vel_prec* restrict vx, vel_prec* restrict vy, vel_prec* restrict vz,                        //
   const grad_prec* restrict gx, const grad_prec* restrict gy, const grad_prec* restrict gz,   //
   const double* restrict massinv, const vel_prec (*restrict vrand)[3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      vel_prec coef = -ekcal * massinv[i] * vfric2;
      vx[i] += coef * toFloatGrad<vel_prec>(gx[i]) + vrand[i][0];
      vy[i] += coef * toFloatGrad<vel_prec>(gy[i]) + vrand[i][1];
      vz[i] += coef * toFloatGrad<vel_prec>(gz[i]) + vrand[i][2];
   }
}

void sdVel2_cu(time_prec vfric, const vel_prec (*vrand)[3])
{
   launch_k1s(g::s0, n, sdVel2_cu1, //
      n, (time_prec)(0.5 * vfric), (vel_prec)units::ekcal, vx, vy, vz, gx, gy, gz, massinv, vrand);
}
}
