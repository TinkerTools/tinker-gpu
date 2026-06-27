#include "ff/switch.h"
#include "math/maxmin.h"
#include "nn/nn.h"
#include <cmath>
#include <tinker/detail/limits.hh>
#include <tinker/detail/nonpol.hh>

namespace tinker {
real switchCut(Switch mode)
{
   real cut;
   using namespace limits;
   switch (mode) {
   case Switch::VDW:
      cut = vdwtaper;
      break;
   case Switch::REPULS:
      cut = reptaper;
      break;
   case Switch::DISP:
      cut = disptaper;
      break;
   case Switch::CHARGE:
      cut = chgtaper;
      break;
   case Switch::CHGDPL:
      cut = std::sqrt(chgtaper * dpltaper);
      break;
   case Switch::DIPOLE:
      cut = dpltaper;
      break;
   case Switch::MPOLE:
      cut = mpoletaper;
      break;
   case Switch::CHGTRN:
      cut = ctrntaper;
      break;
   case Switch::EWALD:
      cut = ewaldcut;
      break;
   case Switch::DEWALD:
      cut = dewaldcut;
      break;
   case Switch::USOLVE:
      cut = usolvcut;
      break;
   case Switch::GKV:
      cut = nonpol::spcut;
      break;
   case Switch::GKSA:
      cut = nonpol::stoff;
      break;
   default:
      cut = minOf(vdwtaper, reptaper, disptaper, chgtaper, dpltaper, mpoletaper, ctrntaper);
      break;
   }
   return cut;
}

real switchOff(Switch mode)
{
   real off;
   using namespace limits;
   switch (mode) {
   case Switch::VDW:
      off = vdwcut;
      break;
   case Switch::REPULS:
      off = repcut;
      break;
   case Switch::DISP:
      off = dispcut;
      break;
   case Switch::CHARGE:
      off = chgcut;
      break;
   case Switch::CHGDPL:
      off = std::sqrt(chgcut * dplcut);
      break;
   case Switch::DIPOLE:
      off = dplcut;
      break;
   case Switch::MPOLE:
      off = mpolecut;
      break;
   case Switch::CHGTRN:
      off = ctrncut;
      break;
   case Switch::EWALD:
      off = ewaldcut;
      break;
   case Switch::DEWALD:
      off = dewaldcut;
      break;
   case Switch::USOLVE:
      off = usolvcut;
      break;
   case Switch::GKV:
      off = nonpol::spoff;
      break;
   case Switch::GKSA:
      off = nonpol::stcut;
      break;
   case Switch::NN:
      off = nncut;
      break;
   default:
      off = minOf(vdwcut, repcut, dispcut, chgcut, dplcut, mpolecut, ctrncut);
      break;
   }
   return off;
}
}
