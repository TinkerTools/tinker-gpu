#include "ff/energy.h"
#include "ff/molecule.h"
#include "ff/nblist.h"
#include "math/random.h"
#include "md/misc.h"
#include "md/pq.h"
#include "tool/externfunc.h"
#include "tool/iofortstr.h"
#include <tinker/detail/bath.hh>
#include <tinker/detail/bound.hh>
#include <tinker/detail/mdstuf.hh>
#include <tinker/detail/units.hh>

#include <cmath>

namespace tinker {
TINKER_FVOID2(acc1, cu1, kineticEnergy, energy_prec&, energy_prec (&)[3][3], int n, const double*,
   const vel_prec*, const vel_prec*, const vel_prec*);
void kineticEnergy(energy_prec& eksum_out, energy_prec (&ekin_out)[3][3], int n, const double* mass,
   const vel_prec* vx, const vel_prec* vy, const vel_prec* vz)
{
   TINKER_FCALL2(acc1, cu1, kineticEnergy, eksum_out, ekin_out, n, mass, vx, vy, vz);
}

void kineticExplicit(T_prec& temp_out, energy_prec& eksum_out, energy_prec (&ekin_out)[3][3],
   const vel_prec* vx, const vel_prec* vy, const vel_prec* vz)
{
   kineticEnergy(eksum_out, ekin_out, n, mass, vx, vy, vz);
   temp_out = 2 * eksum_out / (mdstuf::nfree * units::gasconst);
}

void kinetic(T_prec& temp)
{
   kineticExplicit(temp, eksum, ekin, vx, vy, vz);
}
}

namespace tinker {
void bussiThermostat(time_prec dt_prec, T_prec temp_prec)
{
   double dt = dt_prec;
   double temp = temp_prec;

   double tautemp = bath::tautemp;
   double kelvin = bath::kelvin;
   int nfree = mdstuf::nfree;
   // double& eta = bath::eta;

   if (temp == 0)
      temp = 0.1;

   double c = std::exp(-dt / tautemp);
   double d = (1 - c) * (kelvin / temp) / nfree;
   double r = normal<double>();
   double s = chiSquared<double>(nfree - 1);
   double scale = c + (s + r * r) * d + 2 * r * std::sqrt(c * d);
   scale = std::sqrt(scale);
   if (r + std::sqrt(c / d) < 0)
      scale = -scale;
   // eta *= scale;

   vel_prec sc = scale;
   mdVelScale(sc, n, vx, vy, vz);
   for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
         ekin[i][j] *= scale * scale;
      }
   }
}

TINKER_FVOID2(acc1, cu1, monteCarloMolMove, double);
void monteCarloBarostat(energy_prec epot, T_prec temp)
{
   if (not bound::use_bounds)
      return;
   if (bath::isothermal)
      temp = bath::kelvin;

   FstrView volscale = bath::volscale;
   double third = 1.0 / 3.0;
   double volmove = bath::volmove;
   double kt = units::gasconst * temp;
   if (bath::isothermal)
      kt = units::gasconst * bath::kelvin;
   bool isotropic = true;
   // double aniso_rdm = random<double>();
   // if (bath::anisotrop && aniso_rdm > 0.5)
   //    isotropic = false;

   // save the system state prior to trial box size change
   Box boxold;
   boxGetCurrent(boxold);
   double volold = boxVolume();
   double volnew = 0;
   double eold = epot;
   darray::copy(g::q0, n, x_pmonte, xpos);
   darray::copy(g::q0, n, y_pmonte, ypos);
   darray::copy(g::q0, n, z_pmonte, zpos);

   if (isotropic) {
      double step_rdm = 2 * random<double>() - 1;
      double step = volmove * step_rdm;
      volnew = volold + step;
      double scale = std::pow(volnew / volold, third);

      lvec1 *= scale;
      lvec2 *= scale;
      lvec3 *= scale;
      boxSetCurrentRecip();

      if (volscale == "MOLECULAR") {
         TINKER_FCALL2(acc1, cu1, monteCarloMolMove, scale);
      }

      copyPosToXyz();
   }

   // get the potential energy and PV work changes for trial move
   nblistRefresh();
   energy(calc::energy);
   energy_prec enew;
   copyEnergy(calc::energy, &enew);
   double dpot = enew - eold;
   double dpv = bath::atmsph * (volnew - volold) / units::prescon;

   // estimate the kinetic energy change as an ideal gas term
   double dkin = 0;
   if (volscale == "MOLECULAR") {
      dkin = molecule.nmol * kt * std::log(volold / volnew);
   }

   // acceptance ratio from Epot change, Ekin change and PV work
   double term = -(dpot + dpv + dkin) / kt;
   double expterm = std::exp(term);

   // reject the step, and restore values prior to trial change
   double exp_rdm = random<double>();
   if (exp_rdm > expterm) {
      esum = eold;
      boxSetCurrent(boxold);
      darray::copy(g::q0, n, xpos, x_pmonte);
      darray::copy(g::q0, n, ypos, y_pmonte);
      darray::copy(g::q0, n, zpos, z_pmonte);
      copyPosToXyz();
      nblistRefresh();
   }
}

TINKER_FVOID2(acc1, cu0, berendsenBarostat, time_prec, bool);
void berendsenBarostat(time_prec dt, bool aniso)
{
   TINKER_FCALL2(acc1, cu0, berendsenBarostat, dt, aniso);
}
}
