#include "md/stochastic.h"
#include "ff/atom.h"
#include "ff/energy.h"
#include "md/pq.h"
#include "seq/add.h"
#include "seq/philox.h"
#include <cmath>
#include <tinker/detail/units.hh>

namespace tinker {
void sdTerm_acc(int istep, unsigned int seed, pos_prec psig, vel_prec vsig, vel_prec rho, pos_prec (*prand)[3],
   vel_prec (*vrand)[3])
{
   vel_prec rhoc = std::sqrt(1 - (double)rho * rho);
   unsigned int step = istep;
   #pragma acc parallel loop independent async\
               deviceptr(massinv,prand,vrand)
   for (int i = 0; i < n; ++i) {
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

void sdPos_acc(vel_prec pfric, time_prec vfric, time_prec afric, const pos_prec (*prand)[3])
{
   const vel_prec ekcal = units::ekcal;
   const time_prec vfric2 = 0.5 * vfric;
   #pragma acc parallel loop independent async\
               deviceptr(xpos,ypos,zpos,vx,vy,vz,gx,gy,gz,massinv,prand)
   for (int i = 0; i < n; ++i) {
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

void sdVel2_acc(time_prec vfric, const vel_prec (*vrand)[3])
{
   const vel_prec ekcal = units::ekcal;
   const time_prec vfric2 = 0.5 * vfric;
   #pragma acc parallel loop independent async\
               deviceptr(vx,vy,vz,gx,gy,gz,massinv,vrand)
   for (int i = 0; i < n; ++i) {
      vel_prec coef = -ekcal * massinv[i] * vfric2;
      vx[i] += coef * toFloatGrad<vel_prec>(gx[i]) + vrand[i][0];
      vy[i] += coef * toFloatGrad<vel_prec>(gy[i]) + vrand[i][1];
      vz[i] += coef * toFloatGrad<vel_prec>(gz[i]) + vrand[i][2];
   }
}
}
