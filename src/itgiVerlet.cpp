#include "itgiVerlet.h"
#include "energy.h"
#include "mdcalc.h"
#include "mdpq.h"

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
   : BasicIntegrator(PropagatorEnum::Verlet, te, be)
{
   this->kickoff();
}

void VerletIntegrator::KickOff()
{
   energy((calc::grad | calc::virial) & rc_flag);
}
}
