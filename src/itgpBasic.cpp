#include "itgpBasic.h"
#include "mdegv.h"
#include "mdpq.h"
#include "rattle.h"
#include "tool/darray.h"
#include <tinker/detail/bath.hh>
#include <tinker/detail/inform.hh>

namespace tinker {
bool BasicPropagator::ifSave(int istep) const
{
   return (istep % inform::iwrite) == 0;
}

BasicPropagator::BasicPropagator()
{
   nrespa = 1;

   atomic = not useRattle();
   aniso = bath::anisotrop;
}

BasicPropagator::~BasicPropagator() {}

void BasicPropagator::updateVelocity0(time_prec t)
{
   propagate_velocity(t, gx, gy, gz);
}

void BasicPropagator::updateVelocity1(time_prec t)
{
   this->updateVelocity0(t);
}

void BasicPropagator::updateVelocity2(time_prec t)
{
   this->updateVelocity1(t);
}

void BasicPropagator::updateVelocityR1(time_prec tfast, time_prec t) {}

void BasicPropagator::updateVelocityR2(time_prec tfast, time_prec t)
{
   this->updateVelocityR1(tfast, t);
}

void BasicPropagator::updatePosition(time_prec t)
{
   propagate_pos(t);
}

void BasicPropagator::rattleSave()
{
   if (not useRattle())
      return;

   darray::copy(g::q0, n, rattle_xold, xpos);
   darray::copy(g::q0, n, rattle_yold, ypos);
   darray::copy(g::q0, n, rattle_zold, zpos);
}

void BasicPropagator::rattle(time_prec timeStep)
{
   if (not useRattle())
      return;

   tinker::rattle(timeStep, rattle_xold, rattle_yold, rattle_zold);
}

void BasicPropagator::rattle2(time_prec timeStep, bool doVirial)
{
   if (not useRattle())
      return;

   tinker::rattle2(timeStep, doVirial);
}

bool BasicPropagator::useRattle()
{
   return use_rattle();
}
}
