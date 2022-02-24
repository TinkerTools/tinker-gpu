#include "itgiLP22.h"
#include "itgiRespa.h"
#include "itgiVerlet.h"
#include "itgpLogV.h"
#include "tinker_rt.h"

namespace tinker {
const char* LP22Integrator::name() const
{
   return "Langevin Piston (2022)";
}

void LP22Integrator::kickoff()
{
   if (m_isNRespa1)
      VerletIntegrator::KickOff();
   else
      RespaIntegrator::KickOff();
}

LP22Integrator::LP22Integrator(bool isNRespa1)
   : BasicIntegrator(PropagatorEnum::Verlet, ThermostatEnum::m_LP2022,
                     BarostatEnum::LP2022)
   , m_isNRespa1(isNRespa1)
{
   delete m_prop;
   m_prop = new LogVPropagator(isNRespa1);

   this->kickoff();
}
}
