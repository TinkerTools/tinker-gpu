#include "md/integrator.h"
#include "ff/energy.h"
#include "md/lflpiston.h"
#include "md/misc.h"
#include "md/pq.h"
#include "md/rattle.h"
#include "tool/error.h"
#include "tool/ioprint.h"
#include <tinker/detail/inform.hh>
#include <tinker/detail/mdstuf.hh>
#include <tinker/detail/units.hh>

namespace tinker {
void BasicIntegrator::plan(int istep)
{
   int vers0 = rc_flag & calc::vmask;
   vers1 = vers0;

   save = m_prop->ifSave(istep);
   bool mcbaro = false;
   m_baro->setPrintPressure(istep);
   if (m_baro->ifApply(istep))
      if (m_baro->getBarostatEnum() == BarostatEnum::MONTECARLO)
         mcbaro = true;
   // toggle off virial if !applyBaro
   if (not applyBaro)
      vers1 &= ~calc::virial;
   // toggle off virial for MC barostat
   if (mcbaro)
      vers1 &= ~calc::virial;
   // toggle off energy if neither save nor mcbaro
   if (not save and not mcbaro)
      vers1 &= ~calc::energy;
}

BasicIntegrator::BasicIntegrator(int nRespaLogV, PropagatorEnum pe, //
   ThermostatEnum te, BarostatEnum be)
   : m_prop(BasicPropagator::create(nRespaLogV, pe))
   , m_thermo(BasicThermostat::create(te))
   , m_baro(BasicBarostat::create(be))
{
   this->plan(0);
}

BasicIntegrator::BasicIntegrator()
   : m_prop(new VerletDevice)
   , m_thermo(new BasicThermostat)
   , m_baro(new BasicBarostat)
{
   this->plan(0);
}

BasicIntegrator::~BasicIntegrator()
{
   delete m_prop;
   delete m_thermo;
   delete m_baro;
}

void BasicIntegrator::printDetail(FILE* o)
{
   m_thermo->printDetail(o);
   m_baro->printDetail(o);
   print(o, "\n");
   print(o, " NRespa             %12d\n", nrespa);
   print(o, "\n");
   print(o, " %s\n", this->name());
}

void BasicIntegrator::dynamic(int istep, time_prec dt)
{
   time_prec dt2 = dt * 0.5;

   m_baro->control1(dt);
   m_thermo->control1(dt);

   if (nrespa == 1)
      m_prop->vel1(dt2);
   else
      m_prop->velR1(dt2, nrespa);
   m_prop->rattleSave();

   m_baro->control3(dt);

   this->plan(istep);
   if (nrespa == 1) {
      m_prop->pos(dt);
      m_prop->rattle(dt);
      copyPosToXyz(true);
      energy(vers1);
      if (vers1 & calc::virial)
         if (not atomic)
            hcVirial();
   } else {
      virial_prec vir_fast[9] = {0};
      energy_prec esum_f;
      time_prec dta = dt / nrespa;

      for (int ifast = 1; ifast < nrespa; ++ifast) {
         m_prop->pos(dta);
         copyPosToXyz(false);
         energy(vers1, RESPA_FAST, respaTSConfig());
         m_prop->velR0(dta);
         if (vers1 & calc::virial) {
            if (atomic) {
               for (int iv = 0; iv < 9; ++iv)
                  vir_fast[iv] += vir[iv];
            } else {
               hcVirial();
               for (int iv = 0; iv < 9; ++iv)
                  vir_fast[iv] += hc_vir[iv];
            }
         }
      }
      m_prop->pos(dta);
      m_prop->rattle(dt);
      copyPosToXyz(true);

      // fast force
      energy(vers1, RESPA_FAST, respaTSConfig());
      darray::copy(g::q0, n, gx1, gx);
      darray::copy(g::q0, n, gy1, gy);
      darray::copy(g::q0, n, gz1, gz);
      copyEnergy(vers1, &esum_f);
      if (vers1 & calc::virial) {
         if (atomic) {
            for (int iv = 0; iv < 9; ++iv)
               vir_fast[iv] += vir[iv];
         } else {
            hcVirial();
            for (int iv = 0; iv < 9; ++iv)
               vir_fast[iv] += hc_vir[iv];
         }
      }

      // slow force
      energy(vers1, RESPA_SLOW, respaTSConfig());
      darray::copy(g::q0, n, gx2, gx);
      darray::copy(g::q0, n, gy2, gy);
      darray::copy(g::q0, n, gz2, gz);
      if (vers1 & calc::energy)
         esum += esum_f;
      if (vers1 & calc::virial) {
         if (atomic) {
            for (int iv = 0; iv < 9; ++iv)
               vir[iv] += vir_fast[iv] / nrespa;
         } else {
            hcVirial();
            for (int iv = 0; iv < 9; ++iv)
               hc_vir[iv] += vir_fast[iv] / nrespa;
         }
      }
   }

   m_baro->control4(dt);

   if (nrespa == 1)
      m_prop->vel2(dt2);
   else
      m_prop->velR2(dt2, nrespa);
   // rattle2 does not change the molecular virial
   m_prop->rattle2(dt, static_cast<bool>(vers1 & calc::virial) and atomic);

   m_thermo->control2(dt, save);
   m_baro->control2(dt);
}
}

namespace tinker {
const char* VerletIntegrator::name() const
{
   return "Molecular Dynamics Trajectory via Velocity Verlet Algorithm";
}

void VerletIntegrator::kickoff()
{
   VerletIntegrator::KickOff();
}

VerletIntegrator::VerletIntegrator(ThermostatEnum te, BarostatEnum be)
   : BasicIntegrator(0, PropagatorEnum::VERLET, te, be)
{
   this->kickoff();
}

void VerletIntegrator::KickOff()
{
   energy((calc::grad | calc::virial) & rc_flag);
}
}

namespace tinker {
const char* RespaIntegrator::name() const
{
   return "Molecular Dynamics Trajectory via r-RESPA MTS Algorithm";
}

void RespaIntegrator::kickoff()
{
   RespaIntegrator::KickOff();
}

RespaIntegrator::RespaIntegrator(ThermostatEnum te, BarostatEnum be)
   : BasicIntegrator(0, PropagatorEnum::RESPA, te, be)
{
   this->kickoff();
}

void RespaIntegrator::KickOff()
{
   // save fast gradients to gx1 etc.
   energy(calc::grad, RESPA_FAST, respaTSConfig());
   darray::copy(g::q0, n, gx1, gx);
   darray::copy(g::q0, n, gy1, gy);
   darray::copy(g::q0, n, gz1, gz);

   // save slow gradients to gx2 etc.
   energy(calc::grad, RESPA_SLOW, respaTSConfig());
   darray::copy(g::q0, n, gx2, gx);
   darray::copy(g::q0, n, gy2, gy);
   darray::copy(g::q0, n, gz2, gz);

   energy((calc::grad | calc::virial) & rc_flag);
}
}

namespace tinker {
static double gbar;
static double vnh[maxnose];
static double qnh[maxnose];
static double gnh[maxnose];
static double press;

static void hoover(time_prec dt, virial_prec press)
{
   constexpr int nc = 5;
   constexpr int ns = 3;
   // w[0] = 1/(2 - 2**(1/3))
   // w[1] = 1 - w[0] - w[2]
   // w[2] = w[0]
   constexpr double w[3] = {1.351207191959657634047687808971460826921999376217144828328,
      -1.70241438391931526809537561794292165384399875243428965665,
      1.351207191959657634047687808971460826921999376217144828328};

   const double vbox = boxVolume();
   T_prec temp;
   kinetic(temp);
   const double ekt = units::gasconst * bath::kelvin;
   const int df = mdstuf::nfree;
   const double odnf = 1 + 3.0 / df;
   const double gn1kt = (1 + df) * ekt;
   const double dpress = (press - bath::atmsph) / units::prescon;
   const time_prec dtc = dt / nc;

   double scale = 1.0;
   for (int k = 0; k < nc; ++k) {
      for (int j = 0; j < ns; ++j) {
         const time_prec dts = w[j] * dtc;
         const time_prec dt2 = 0.5 * dts;
         const time_prec dt4 = 0.25 * dts;
         const time_prec dt8 = 0.125 * dts;
         double expterm;

         // update barostat and thermostat velocities and forces
         // eq. 41 first half
         for (int i = maxnose - 1; i > -1; --i) {
            if (i == 0)
               gnh[i] = (2 * eksum + qbar * vbar * vbar - gn1kt) / qnh[i];
            else
               gnh[i] = (qnh[i - 1] * vnh[i - 1] * vnh[i - 1] - ekt) / qnh[i];

            if (i == maxnose - 1)
               vnh[i] += gnh[i] * dt4;
            else {
               double exptm = std::exp(-vnh[i + 1] * dt8);
               vnh[i] = (vnh[i] * exptm + gnh[i] * dt4) * exptm;
            }
         }
         gbar = (2 * eksum * odnf + 3 * vbox * dpress) / qbar;
         expterm = std::exp(-vnh[0] * dt8);
         vbar = (vbar * expterm + gbar * dt4) * expterm;

         // find velocity scale factor and update kinetic energy
         // eq. 41 velocities
         expterm = std::exp(-(vnh[0] + vbar * odnf) * dt2);
         scale *= expterm;
         double exptm2 = expterm * expterm;
         eksum *= exptm2;
         for (int ii = 0; ii < 3; ++ii)
            for (int jj = 0; jj < 3; ++jj)
               ekin[ii][jj] *= exptm2;

         // update barostat and thermostat velocities and forces
         // eq. 41 second half
         gbar = (2 * eksum * odnf + 3 * vbox * dpress) / qbar;
         expterm = std::exp(-vnh[0] * dt8);
         vbar = (vbar * expterm + gbar * dt4) * expterm;
         for (int i = 0; i < maxnose; ++i) {
            if (i == 0)
               gnh[i] = (2 * eksum + qbar * vbar * vbar - gn1kt) / qnh[i];
            else
               gnh[i] = (qnh[i - 1] * vnh[i - 1] * vnh[i - 1] - ekt) / qnh[i];

            if (i == maxnose - 1)
               vnh[i] += gnh[i] * dt4;
            else {
               double exptm = std::exp(-vnh[i + 1] * dt8);
               vnh[i] = (vnh[i] * exptm + gnh[i] * dt4) * exptm;
            }
         }
      }
   }

   // use scale factor to update the atomic velocities
   // eq. 41 velocities
   mdVelScale(scale, n, vx, vy, vz);
}

static void nhc_npt(int istep, time_prec dt)
{
   int vers1 = rc_flag & calc::vmask;
   bool save = 0 == (istep % inform::iwrite);
   if (!save)
      vers1 &= ~calc::energy;

   // set some time values for the dynamics integration
   const time_prec dt_2 = 0.5f * dt;
   // This initialization is intentially kept the same as the Fortran code.
   // Yes, the real press is available here, but when the Fortran code was
   // written, virial may not be available if simulation was restarted from
   // a ".dyn" file.
   if (istep == 1)
      press = bath::atmsph;

   // update thermostat and barostat values, scale atomic velocities
   hoover(dt, press);

   mdVel(dt_2, gx, gy, gz);

   double term = vbar * dt_2;
   double term2 = term * term;
   double expterm = std::exp(term);
   double eterm2 = expterm * expterm;

   // update the periodic box size and total volume
   // eq. 42 volume
   lvec1 *= eterm2;
   lvec2 *= eterm2;
   lvec3 *= eterm2;
   boxSetCurrentRecip();

   // update atomic positions via coupling to barostat
   // eq. 42 coordinates
   constexpr double e2 = 1.0 / 6;
   constexpr double e4 = e2 / 20;
   constexpr double e6 = e4 / 42;
   constexpr double e8 = e6 / 72;
   // sinh(x)/x: Taylor series
   double poly = 1 + term2 * (e2 + term2 * (e4 + term2 * (e6 + term2 * e8)));
   poly *= expterm * dt;
   mdPosAxbv(eterm2, poly);
   copyPosToXyz(true);

   energy(vers1);

   mdVel(dt_2, gx, gy, gz);

   // update thermostat and barostat values, scale atomic velocities
   hoover(dt, press);

   // set isotropic pressure to the average of tensor diagonal
   double vbox = boxVolume();
   double factor = units::prescon / vbox;
   double stress[3][3];
   for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
         stress[i][j] = factor * (-vir[3 * i + j]);
      }
   }
   press = (stress[0][0] + stress[1][1] + stress[2][2]) / 3;
}

const char* Nhc96Integrator::name() const
{
   return "Molecular Dynamics Trajectory via Nose-Hoover NPT Algorithm";
}

void Nhc96Integrator::kickoff()
{
   double ekt = units::gasconst * bath::kelvin;
   vbar = 0;
   qbar = (mdstuf::nfree + 1) * ekt * bath::taupres * bath::taupres;
   gbar = 0;
   for (int i = 0; i < maxnose; ++i) {
      vnh[i] = 0;
      qnh[i] = ekt * bath::tautemp * bath::tautemp;
      gnh[i] = 0;
   }
   qnh[0] *= mdstuf::nfree;
   energy(calc::grad | calc::virial);
}

Nhc96Integrator::Nhc96Integrator()
   : BasicIntegrator()
{
   if (useRattle())
      TINKER_THROW("Constraints under NH-NPT require the ROLL algorithm.");

   this->kickoff();
}

void Nhc96Integrator::dynamic(int istep, time_prec dt)
{
   nhc_npt(istep, dt);
}
}

namespace tinker {
const char* Nhc06Integrator::name() const
{
   return "NHC2006";
}

void Nhc06Integrator::kickoff()
{
   if (nrespa == 1)
      VerletIntegrator::KickOff();
   else
      RespaIntegrator::KickOff();
}

Nhc06Integrator::Nhc06Integrator(int nrspa)
   : BasicIntegrator(nrspa, PropagatorEnum::m_LOGV, //
        ThermostatEnum::m_NHC2006, BarostatEnum::NHC2006)
{
   if (useRattle())
      TINKER_THROW("Constraints under NH-NPT require the ROLL algorithm.");

   if (aniso)
      TINKER_THROW("Cannot use ANISO-PRESSURE in Nhc06Integrator.");

   this->kickoff();
}
}

namespace tinker {
const char* LP22Integrator::name() const
{
   return "Langevin Piston (2022)";
}

void LP22Integrator::kickoff()
{
   if (nrespa == 1)
      VerletIntegrator::KickOff();
   else
      RespaIntegrator::KickOff();

   if (not atomic)
      hcVirial();
}

LP22Integrator::LP22Integrator(int nrspa)
   : BasicIntegrator(nrspa, PropagatorEnum::m_LOGV, //
        ThermostatEnum::m_LP2022, BarostatEnum::LP2022)
{
   this->kickoff();
}
}

namespace tinker {
const char* LeapFrogLPIntegrator::name() const
{
   return "Molecular Dynamics Trajectory via Langevin Piston Algorithm";
}

void LeapFrogLPIntegrator::kickoff()
{
   energy(calc::energy | calc::grad | calc::virial);
}

LeapFrogLPIntegrator::LeapFrogLPIntegrator()
   : BasicIntegrator()
{
   darray::allocate(n, &leapfrog_x, &leapfrog_y, &leapfrog_z);
   darray::allocate(n, &leapfrog_vx, &leapfrog_vy, &leapfrog_vz, &leapfrog_vxold, &leapfrog_vyold,
      &leapfrog_vzold);
   this->kickoff();
}

LeapFrogLPIntegrator::~LeapFrogLPIntegrator()
{
   darray::deallocate(leapfrog_x, leapfrog_y, leapfrog_z);
   darray::deallocate(
      leapfrog_vx, leapfrog_vy, leapfrog_vz, leapfrog_vxold, leapfrog_vyold, leapfrog_vzold);
}

void LeapFrogLPIntegrator::dynamic(int istep, time_prec dt)
{
   lf_lpiston_npt(istep, dt);
}
}
