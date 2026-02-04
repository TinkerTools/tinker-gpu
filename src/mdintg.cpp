#include "math/pow2.h"
#include "md/integrator.h"
#include "md/misc.h"
#include "md/pq.h"
#include "tool/argkey.h"
#include "tool/darray.h"
#include "tool/error.h"
#include "tool/externfunc.h"
#include "tool/iofortstr.h"
#include "tool/tinkersuppl.h"
#include <tinker/detail/files.hh>
#include <tinker/detail/inform.hh>
#include <tinker/detail/mdstuf.hh>
#include <tinker/detail/units.hh>

#include <cassert>

namespace tinker {
void mdData(RcOp op)
{
   if (not(calc::md & rc_flag))
      return;

   RcMan intg42{mdIntegrateData, op};
   RcMan save42{mdsaveData, op};
}

static BasicIntegrator* intg;

void mdIntegrateData(RcOp op)
{
   if (op & RcOp::DEALLOC) {
      delete intg;
      intg = nullptr;
   }

   if (op & RcOp::INIT) {
      ThermostatEnum thermostat = ThermostatEnum::NONE;
      if (bath::isothermal) {
         FstrView th = bath::thermostat;
         if (th == "ANDERSEN")
            thermostat = ThermostatEnum::ANDERSEN;
         else if (th == "BERENDSEN")
            thermostat = ThermostatEnum::BERENDSEN;
         else if (th == "BUSSI")
            thermostat = ThermostatEnum::BUSSI;
         else if (th == "NOSE-HOOVER")
            thermostat = ThermostatEnum::NHC;
         else if (th == "LPISTON")
            thermostat = ThermostatEnum::m_LEAPFROGLP;
         else
            assert(false);
      }

      BarostatEnum barostat = BarostatEnum::NONE;
      if (bath::isobaric) {
         FstrView br = bath::barostat;
         if (br == "BERENDSEN")
            barostat = BarostatEnum::BERENDSEN;
         else if (br == "BUSSI")
            barostat = BarostatEnum::BUSSI;
         else if (br == "LANGEVIN")
            barostat = BarostatEnum::LP2022;
         else if (br == "MONTECARLO")
            barostat = BarostatEnum::MONTECARLO;
         else if (br == "NHC06")
            barostat = BarostatEnum::NHC2006;
         else if (br == "LPISTON")
            barostat = BarostatEnum::m_LEAPFROGLP;
         else if (br == "NOSE-HOOVER")
            barostat = BarostatEnum::m_NHC1996;
         else
            assert(false);
      }

      IntegratorEnum integrator = IntegratorEnum::BEEMAN;
      if (thermostat == ThermostatEnum::m_LEAPFROGLP and barostat == BarostatEnum::m_LEAPFROGLP)
         integrator = IntegratorEnum::LEAPFROGLP;
      else if (thermostat == ThermostatEnum::NHC and barostat == BarostatEnum::m_NHC1996)
         integrator = IntegratorEnum::NHC1996;

      FstrView itg = mdstuf::integrate;
      if (itg == "BRESPA" or itg == "SRESPA" or itg == "VRESPA") {
         integrator = IntegratorEnum::RESPA;
      } else if (itg == "VERLET") {
         integrator = IntegratorEnum::VERLET;
      } else if (itg == "LPISTON") {
         integrator = IntegratorEnum::LEAPFROGLP;
         thermostat = ThermostatEnum::m_LEAPFROGLP;
         barostat = BarostatEnum::m_LEAPFROGLP;
      } else if (itg == "NOSE-HOOVER") {
         integrator = IntegratorEnum::NHC1996;
         thermostat = ThermostatEnum::m_NHC1996;
         barostat = BarostatEnum::m_NHC1996;
      }

      int nrspa = 1;
      if (integrator == IntegratorEnum::VERLET or integrator == IntegratorEnum::RESPA) {
         if (barostat == BarostatEnum::LP2022) {
            if (integrator == IntegratorEnum::RESPA)
               nrspa = mdstuf::nrespa;
            integrator = IntegratorEnum::LP2022;
            thermostat = ThermostatEnum::m_LP2022;
         } else if (barostat == BarostatEnum::NHC2006) {
            if (integrator == IntegratorEnum::RESPA)
               nrspa = mdstuf::nrespa;
            integrator = IntegratorEnum::NHC2006;
            thermostat = ThermostatEnum::m_NHC2006;
         }
      }

      intg = nullptr;
      if (integrator == IntegratorEnum::RESPA)
         intg = new RespaIntegrator(thermostat, barostat);
      else if (integrator == IntegratorEnum::VERLET)
         intg = new VerletIntegrator(thermostat, barostat);
      else if (integrator == IntegratorEnum::LEAPFROGLP)
         intg = new LeapFrogLPIntegrator;
      else if (integrator == IntegratorEnum::LP2022)
         intg = new LP22Integrator(nrspa);
      else if (integrator == IntegratorEnum::NHC1996)
         intg = new Nhc96Integrator;
      else if (integrator == IntegratorEnum::NHC2006)
         intg = new Nhc06Integrator(nrspa);
      else if (integrator == IntegratorEnum::BEEMAN)
         TINKER_THROW("Beeman integrator is not available.");
      intg->printDetail(stdout);
   }
}
}

namespace tinker {
TINKER_FVOID2(acc1, cu1, mdrest, int);
void mdrest(int istep)
{
   TINKER_FCALL2(acc1, cu1, mdrest, istep);
}

void mdrestPrintP1(bool prints, double vtot1, double vtot2, double vtot3, double totmass)
{
   if (prints) {
      // compute translational kinetic energy of overall system
      auto etrans = vtot1 * vtot1 + vtot2 * vtot2 + vtot3 * vtot3;
      etrans *= 0.5 * totmass / units::ekcal;

      print(stdout,
         " System Linear Velocity :  %12.2e%12.2e%12.2e\n"
         " Translational Kinetic Energy :%10s%12.4f Kcal/mole\n",
         vtot1, vtot2, vtot3, "", etrans);
   }
}

void mdPropagate(int nsteps, time_prec dt_ps)
{
   for (int istep = 1; istep <= nsteps; ++istep) {
      intg->dynamic(istep, dt_ps);

      // mdstat
      bool save = (istep % inform::iwrite == 0);
      if (save || (istep % BOUNDS_EVERY_X_STEPS) == 0)
         bounds();
      if (save) {
         T_prec temp;
         kinetic(temp);
         mdsaveAsync(istep, dt_ps);
      }
      mdrest(istep);
   }
   mdsaveSynchronize();
}

const TimeScaleConfig& respaTSConfig()
{
   constexpr int fast = floorLog2_constexpr(RESPA_FAST); // short-range
   constexpr int slow = floorLog2_constexpr(RESPA_SLOW); // long-range
   static TimeScaleConfig tsconfig{
      {"ebond", fast},
      {"eangle", fast},
      {"estrbnd", fast},
      {"eurey", fast},
      {"eopbend", fast},
      {"etors", fast},
      {"eimprop", fast},
      {"eimptor", fast},
      {"epitors", fast},
      {"estrtor", fast},
      {"eangtor", fast},
      {"etortor", fast},
      {"egeom", fast},

      {"evalence", fast},

      {"evdw", slow},

      {"echarge", slow},
      {"echglj", slow},

      {"emplar", slow},
      {"empole", slow},
      {"epolar", slow},

      {"empoleChgpen", slow},
      {"epolarChgpen", slow},

      {"echgtrn", slow},
      {"edisp", slow},
      {"erepel", slow},
      {"ehippo", slow},
   };
   return tsconfig;
}
}
